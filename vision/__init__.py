"""RetailVision 純影像與推論模組。

這個套件刻意不依賴 FastAPI 或 PyTorch：API、Celery worker、訓練腳本都能直接重用，
也能單獨做單元測試。推論一律走 ONNX Runtime。
"""
