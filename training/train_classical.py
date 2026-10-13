"""訓練 SVM baseline（HSV 直方圖 + HOG），結果寫入 models/metrics.json 的 "svm" 區塊。

超參數 C、gamma 只用 val 選，test 只在最後評估一次，避免拿測試集調參。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from training.common import MODELS, SEED, Sample, grocery_meta, grocery_split, update_metrics
from vision.classical import ClassicalClassifier, extract_features
from vision.io import load_image

GRID = [(c, g) for c in (1.0, 3.0, 10.0, 30.0) for g in ("scale", 0.5e-3, 1e-3)]


def _features_of(path: object) -> np.ndarray:
    return extract_features(load_image(str(path)))


def featurize(samples: list[Sample]) -> np.ndarray:
    return np.stack(Parallel(n_jobs=-1)(delayed(_features_of)(s.path) for s in samples))


def fit(X: np.ndarray, y: np.ndarray, C: float, gamma: Any) -> Any:
    # 先標準化：HOG 與直方圖的數值尺度不同，不標準化的話 RBF 距離會被某一組特徵主導
    model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf", random_state=SEED))
    return model.fit(X, y)


def select(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, name: str) -> Any:
    best: tuple[float, Any] = (-1.0, None)
    for C, gamma in GRID:
        model = fit(Xtr, ytr, C, gamma)
        acc = float((model.predict(Xva) == yva).mean())
        print(f"  {name} C={C} gamma={gamma}: val acc={acc:.4f}")
        if acc > best[0]:
            best = (acc, model)
    return best[1]


def main() -> None:
    meta = grocery_meta()
    train, val, test = (grocery_split(s) for s in ("train", "val", "test"))
    t0 = time.perf_counter()
    Xtr, Xva, Xte = featurize(train), featurize(val), featurize(test)
    print(f"特徵擷取 {time.perf_counter() - t0:.1f}s，維度 {Xtr.shape[1]}")

    def labels(samples: list[Sample], attr: str) -> np.ndarray:
        return np.array([getattr(s, attr) for s in samples])

    fine = select(Xtr, labels(train, "fine_id"), Xva, labels(val, "fine_id"), "fine")
    coarse = select(Xtr, labels(train, "coarse_id"), Xva, labels(val, "coarse_id"), "coarse")
    clf = ClassicalClassifier(
        fine, coarse, {"fine_names": meta["fine_names"], "coarse_names": meta["coarse_names"]}
    )
    MODELS.mkdir(exist_ok=True)
    clf.save(MODELS / "classical_svm.joblib")

    t1 = time.perf_counter()
    pf, pc = clf.predict_features(Xte)
    infer_ms = (time.perf_counter() - t1) / len(test) * 1000
    yf, yc = labels(test, "fine_id"), labels(test, "coarse_id")
    f2c = np.array(meta["fine_to_coarse"])
    metrics = {
        "fine_top1": round(float((pf == yf).mean()), 4),
        "coarse_top1": round(float((pc == yc).mean()), 4),
        # 由細類預測換算的粗類，和獨立訓練的粗類 SVM 對照
        "coarse_top1_from_fine": round(float((f2c[pf] == yc).mean()), 4),
        "fine_params": {k: str(v) for k, v in fine[-1].get_params().items() if k in ("C", "gamma")},
        "coarse_params": {
            k: str(v) for k, v in coarse[-1].get_params().items() if k in ("C", "gamma")
        },
        "feature_dim": int(Xtr.shape[1]),
        "svm_predict_ms_per_image_cpu": round(infer_ms, 3),
        "n_test": len(test),
    }
    update_metrics("svm", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
