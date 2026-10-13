from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from vision.classical import ClassicalClassifier, color_histogram, extract_features


def test_features_fixed_length(sharp_image: np.ndarray) -> None:
    a = extract_features(sharp_image)
    b = extract_features(sharp_image[:200, :300])
    assert a.shape == b.shape and a.dtype == np.float32
    assert np.isfinite(a).all()


def test_color_histogram_distinguishes_colors() -> None:
    red = np.zeros((64, 64, 3), np.uint8)
    red[:] = (0, 0, 255)
    green = np.zeros((64, 64, 3), np.uint8)
    green[:] = (0, 255, 0)
    hr, hg = color_histogram(red), color_histogram(green)
    assert np.isclose((hr**2).sum(), 1)  # 開根號後平方和為 1
    assert float(hr @ hg) < 0.1


def test_classifier_save_load(tmp_path) -> None:
    rng = np.random.default_rng(0)
    imgs = []
    for color in [(0, 0, 255), (0, 255, 0)] * 6:
        img = np.zeros((96, 96, 3), np.uint8)
        img[:] = color
        imgs.append(np.clip(img + rng.integers(0, 30, img.shape), 0, 255).astype(np.uint8))
    X = np.stack([extract_features(i) for i in imgs])
    y = np.array([0, 1] * 6)
    clf = ClassicalClassifier(
        make_pipeline(StandardScaler(), SVC()).fit(X, y),
        make_pipeline(StandardScaler(), SVC()).fit(X, y),
        {"fine_names": ["red", "green"]},
    )
    clf.save(tmp_path / "m.joblib")
    loaded = ClassicalClassifier.load(tmp_path / "m.joblib")
    assert loaded.predict(imgs[0]).fine_id == 0
    assert loaded.predict(imgs[1]).coarse_id == 1
