"""模型註冊表：啟動時載入一次模型（FastAPI lifespan），請求之間共用。

使用中的分類模型以資料庫 model_versions.is_active 為準：API 與 Celery worker 是不同的 process，
各自載入模型；每次處理請求前呼叫 sync()，發現資料庫的使用中版本變了就重新載入，
所以 PUT /models/active 切換後不用重啟服務（驗收 3.6）。

vision 物件只依賴下面的 Protocol，測試時可以換成假的模型（CI 沒有 ONNX 權重）。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import ModelVersion
from vision.classifier import Classification
from vision.detector import Box
from vision.io import Image
from vision.shelf import GapClassifier

log = logging.getLogger("app.registry")
RETRY_SECONDS = 30


class ClassifierLike(Protocol):
    # 用唯讀 property 宣告：OnnxClassifier 的 version 是 property，假模型用一般屬性也相容
    @property
    def version(self) -> str: ...

    def predict(self, img: Image, k: int = 5) -> Classification: ...

    def embed(self, img: Image) -> NDArray[np.float32]: ...


class DetectorLike(Protocol):
    @property
    def version(self) -> str: ...

    def detect(self, img: Image) -> list[Box]: ...


ClassifierLoader = Callable[[str], ClassifierLike]


def onnx_classifier_loader(settings: Settings) -> ClassifierLoader:
    def load(name: str) -> ClassifierLike:
        from vision.classifier import OnnxClassifier

        path = settings.models_dir / f"{name}.onnx"
        if not path.exists():
            raise FileNotFoundError(path)
        return OnnxClassifier(path, threads=settings.onnx_threads)

    return load


class ModelRegistry:
    def __init__(
        self,
        classifier_loader: ClassifierLoader,
        detector: DetectorLike | None,
        gap_classifier: GapClassifier | None,
    ) -> None:
        self._load_classifier = classifier_loader
        self._classifier: ClassifierLike | None = None
        self.detector = detector
        self.gap_classifier = gap_classifier
        self._lock = threading.Lock()
        self._failed: tuple[str, float] | None = None  # （載入失敗的版本, 時間）

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelRegistry:
        """任何模型檔不存在或損壞都不會讓服務啟動失敗（Codex 第 5 輪 L1）。

        權重檔不進 git（.onnx、.joblib 都在 .gitignore），乾淨的 checkout 或部署漏放檔案很常見：
        - 分類 / 偵測模型讀不到：設為 None，/healthz 回 503，相關端點回 MODEL_NOT_READY。
        - 空位分類器是選配的（.json 有 commit、.joblib 沒有）：讀不到就設為 None，
          貨架分析退回 v1 的紋理規則，只記 warning。
        """
        from vision.detector import OnnxDetector

        det_path = settings.models_dir / f"{settings.detector_version}.onnx"
        gap_path = settings.models_dir / settings.gap_classifier_file
        detector: DetectorLike | None = None
        if det_path.exists():
            try:
                detector = OnnxDetector(det_path, threads=settings.onnx_threads)
            except Exception:
                log.warning("偵測模型 %s 無法載入，/healthz 會回報 503", det_path, exc_info=True)
        else:
            log.warning("找不到偵測模型 %s，/healthz 會回報 503", det_path)
        gap: GapClassifier | None = None
        if gap_path.exists():
            try:
                gap = GapClassifier.load(gap_path, allowed_dir=settings.models_dir)
            except (FileNotFoundError, ValueError, OSError):
                log.warning("空位分類器無法載入，貨架分析改用紋理規則（v1）", exc_info=True)
        reg = cls(onnx_classifier_loader(settings), detector, gap)
        try:
            reg.load_classifier(settings.classifier_version)
        except Exception:
            log.warning(
                "分類模型 %s 無法載入，/healthz 會回報 503",
                settings.classifier_version,
                exc_info=True,
            )
        return reg

    @property
    def classifier(self) -> ClassifierLike | None:
        return self._classifier

    def load_classifier(self, name: str) -> ClassifierLike:
        clf = self._load_classifier(name)
        with self._lock:
            self._classifier = clf
        log.info("分類模型 %s 已載入", name)
        return clf

    def build_classifier(self, name: str) -> ClassifierLike:
        """只建立、不切換：切換前要先用新模型重新計算 catalog embedding。"""
        return self._load_classifier(name)

    def set_classifier(self, clf: ClassifierLike) -> None:
        with self._lock:
            self._classifier = clf

    def sync(self, db: Session) -> None:
        """資料庫的使用中分類模型和記憶體中的不同時，重新載入（其他 process 切換過模型）。

        載入失敗（檔案不存在或損壞）不會讓請求變成 500（Codex 第 6 輪 L1）：分類模型設為不可用，
        需要它的端點回 MODEL_NOT_READY，貨架分析（只用偵測模型）不受影響。
        同一個壞掉的版本 RETRY_SECONDS 內不重試，避免每個請求都去解析一次壞檔。
        """
        active = (
            db.query(ModelVersion.name)
            .filter(ModelVersion.task == "classification", ModelVersion.is_active.is_(True))
            .scalar()
        )
        current = self._classifier.version if self._classifier else None
        if not active or active == current:
            return
        failed = self._failed
        if failed and failed[0] == active and time.monotonic() - failed[1] < RETRY_SECONDS:
            return
        log.info("偵測到使用中模型改為 %s，重新載入", active)
        try:
            self.load_classifier(active)
            self._failed = None
        except Exception:
            log.error(
                "使用中模型 %s 無法載入，分類相關端點回 MODEL_NOT_READY", active, exc_info=True
            )
            with self._lock:
                self._classifier = None
            self._failed = (active, time.monotonic())

    def status(self) -> dict[str, str | None]:
        return {
            "classifier": self._classifier.version if self._classifier else None,
            "detector": self.detector.version if self.detector else None,
            "gap_classifier": self.gap_classifier.kind if self.gap_classifier else None,
        }

    @property
    def ready(self) -> bool:
        return self._classifier is not None and self.detector is not None


def model_file(settings: Settings, name: str) -> Path:
    return settings.models_dir / f"{name}.onnx"
