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


class DinoClassifier(nn.Module):
    """DINOv2-small（ViT-S/14）backbone + 同樣的 neck 與雙頭，介面和 HierarchicalClassifier 相同。

    為什麼試 DINOv2：它是自監督預訓練，學到的是通用的物體外觀，不依賴 ImageNet 的類別標籤，
    文獻上對棚拍與實拍這類 domain gap 比較穩健（驗收 1.7 的 B 方案）。
    代價是 CPU 推論比 MobileNetV3 慢數倍。
    """

    def __init__(self, n_fine: int, n_coarse: int, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=pretrained, img_size=224, num_classes=0
        )
        dim = self.encoder.num_features
        self.neck = nn.Sequential(nn.Linear(dim, 1280), nn.Hardswish())
        self.dropout = nn.Dropout(dropout)
        self.fine_head = nn.Linear(1280, n_fine)
        self.coarse_head = nn.Linear(1280, n_coarse)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.neck(self.encoder(x))  # num_classes=0 時回傳 CLS token 特徵
        d = self.dropout(h)
        return self.fine_head(d), self.coarse_head(d), F.normalize(h, dim=1)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        return [
            *self.neck.parameters(),
            *self.fine_head.parameters(),
            *self.coarse_head.parameters(),
        ]


ARCHS: dict[str, type[nn.Module]] = {
    "mnv3": HierarchicalClassifier,
    "dinov2s": DinoClassifier,
}


def build_model(arch: str, n_fine: int, n_coarse: int, pretrained: bool = True) -> nn.Module:
    return ARCHS[arch](n_fine, n_coarse, pretrained=pretrained)
