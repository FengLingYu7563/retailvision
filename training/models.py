"""PyTorch 模型定義（只在 training/ 使用，上線推論走 ONNX）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class HierarchicalClassifier(nn.Module):
    """MobileNetV3-Large backbone + 細類 / 粗類雙頭。

    為什麼做雙頭而不是只訓練細類再換算：粗類監督訊號提供額外的正則化，
    細類分錯時（例如兩種蘋果）粗類通常仍然對，API 可以回報「至少知道是蘋果」。

    為什麼選 MobileNetV3-Large：4 GB VRAM 能用 batch 32 + AMP 訓練，CPU 推論只要數十毫秒，
    符合 API 單張 p95 < 300 ms 的目標；ResNet50 留作細類不達標時的備案。

    輸出的 embedding 取兩個分類頭之前的 1280 維共用層（倒數第二層），
    做 L2 正規化後內積就是 cosine 相似度，給以圖搜圖用。
    """

    def __init__(self, n_fine: int, n_coarse: int, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        base = mobilenet_v3_large(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 沿用 ImageNet 預訓練的 960 -> 1280 投影層，比隨機初始化收斂快
        self.neck = nn.Sequential(base.classifier[0], base.classifier[1])
        self.dropout = nn.Dropout(dropout)
        self.fine_head = nn.Linear(1280, n_fine)
        self.coarse_head = nn.Linear(1280, n_coarse)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.neck(torch.flatten(self.pool(self.features(x)), 1))
        d = self.dropout(h)
        return self.fine_head(d), self.coarse_head(d), F.normalize(h, dim=1)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.features.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        return [
            *self.neck.parameters(),
            *self.fine_head.parameters(),
            *self.coarse_head.parameters(),
        ]
