"""ONNX Runtime session 建立。"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort


def create_session(path: str | Path, threads: int | None = None) -> ort.InferenceSession:
    """建立 CPU session。

    為什麼部署用 ONNX Runtime：API 映像檔不需要 PyTorch（省下 1 GB 以上），
    CPU 推論也比 eager PyTorch 快，因為 ORT 會做運算子融合等圖最佳化。
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
