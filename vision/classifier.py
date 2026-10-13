"""ONNX 商品分類（細類 + 粗類），同時輸出 embedding 給以圖搜圖。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vision.io import Image
from vision.onnx_utils import create_session
from vision.preprocess import to_classifier_input


@dataclass(frozen=True)
class LabelScore:
    label: str
    confidence: float


@dataclass(frozen=True)
class Classification:
    fine: LabelScore
    coarse: LabelScore
    top_k: list[LabelScore]
    embedding: NDArray[np.float32] = field(repr=False)


def softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


@dataclass(frozen=True)
class ClassifierMeta:
    version: str
    fine_names: list[str]
    coarse_names: list[str]
    fine_to_coarse: list[int]
    input_size: int = 224

    @classmethod
    def load(cls, path: str | Path) -> ClassifierMeta:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=d["version"],
            fine_names=d["fine_names"],
            coarse_names=d["coarse_names"],
            fine_to_coarse=d["fine_to_coarse"],
            input_size=d.get("input_size", 224),
        )


class OnnxClassifier:
    """模型檔旁邊要有同名 .json（類別名稱、版本），由 training/export_onnx.py 產生。"""

    def __init__(self, onnx_path: str | Path, threads: int | None = None) -> None:
        onnx_path = Path(onnx_path)
        self.meta = ClassifierMeta.load(onnx_path.with_suffix(".json"))
        self.session = create_session(onnx_path, threads)
        self.input_name = self.session.get_inputs()[0].name

    @property
    def version(self) -> str:
        return self.meta.version

    def logits(
        self, batch: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        fine, coarse, emb = self.session.run(None, {self.input_name: batch})
        return fine, coarse, emb

    def embed(self, img: Image) -> NDArray[np.float32]:
        return self.logits(to_classifier_input(img, self.meta.input_size))[2][0]

    def predict(self, img: Image, k: int = 5) -> Classification:
        fine_l, coarse_l, emb = self.logits(to_classifier_input(img, self.meta.input_size))
        pf, pc = softmax(fine_l[0]), softmax(coarse_l[0])
        order = np.argsort(-pf)[:k]
        top_k = [LabelScore(self.meta.fine_names[i], round(float(pf[i]), 4)) for i in order]
        c = int(np.argmax(pc))
        return Classification(
            fine=top_k[0],
            coarse=LabelScore(self.meta.coarse_names[c], round(float(pc[c]), 4)),
            top_k=top_k,
            embedding=emb[0],
        )
