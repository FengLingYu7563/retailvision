"""訓練 MobileNetV3 雙頭分類器（Grocery Store 細類 + 粗類）。

兩階段：
1. 凍結 backbone，只訓練 neck 與兩個分類頭（資料只有 2,640 張，一開始就全部解凍的話，
   隨機初始化的分類頭產生的大梯度會破壞預訓練特徵）。
2. 解凍全部，backbone 用較小學習率微調，cosine 衰減。

每個 epoch 存 last.pt，val 細類準確率最好的存 best.pt；加 --resume 可從 last.pt 接續。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from training.common import CHECKPOINTS, SEED, Sample, grocery_meta, grocery_split, seed_everything
from training.models import HierarchicalClassifier
from vision.io import load_image
from vision.preprocess import resize_short_side, to_classifier_input

CACHE_SHORT_SIDE = 288


class GroceryDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """一次把圖解碼並縮到短邊 288 放在記憶體。

    原圖解析度高，每個 epoch 重新解碼 JPEG 會讓 GPU 等 CPU；Windows 上 DataLoader 多進程
    又要 spawn 並複製資料，所以直接快取縮圖、單進程讀取最穩定。
    """

    def __init__(self, samples: list[Sample], train: bool):
        self.samples = samples
        self.train = train
        self.images = [resize_short_side(load_image(s.path), CACHE_SHORT_SIDE) for s in samples]
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(224, scale=(0.4, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
                # hue 只給很小的範圍：蘋果、甜椒這類商品主要靠顏色區分，色相變太多會改變類別
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        s, img = self.samples[i], self.images[i]
        if self.train:
            rgb = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1)
            x = self.aug(rgb)
        else:
            # 驗證與測試用和上線完全相同的 numpy 前處理
            x = torch.from_numpy(to_classifier_input(img)[0])
        return x, s.fine_id, s.coarse_id


def loss_fn(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    yf: torch.Tensor,
    yc: torch.Tensor,
    coarse_weight: float,
) -> torch.Tensor:
    # label smoothing：81 類裡有很多外觀幾乎一樣的品種，硬標籤容易讓模型過度自信
    return F.cross_entropy(fine, yf, label_smoothing=0.1) + coarse_weight * F.cross_entropy(
        coarse, yc, label_smoothing=0.1
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    n = top1 = top5 = ctop1 = 0
    for x, yf, yc in loader:
        x, yf, yc = x.to(device), yf.to(device), yc.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            fine, coarse, _ = model(x)
        t5 = fine.topk(5, dim=1).indices
        top1 += int((t5[:, 0] == yf).sum())
        top5 += int((t5 == yf[:, None]).any(1).sum())
        ctop1 += int((coarse.argmax(1) == yc).sum())
        n += len(yf)
    return {"fine_top1": top1 / n, "fine_top5": top5 / n, "coarse_top1": ctop1 / n}


def build_optimizer(model: HierarchicalClassifier, stage: int, lr: float) -> torch.optim.Optimizer:
    if stage == 1:
        for p in model.backbone_parameters():
            p.requires_grad = False
        return torch.optim.AdamW(model.head_parameters(), lr=lr, weight_decay=1e-4)
    for p in model.parameters():
        p.requires_grad = True
    # backbone 學習率是 head 的 1/10：保留預訓練特徵，只做小幅調整
    return torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": lr * 0.1},
            {"params": model.head_parameters(), "lr": lr},
        ],
        weight_decay=1e-4,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="classifier_mnv3")
    ap.add_argument("--epochs-frozen", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30, help="解凍後的 epoch 數")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--coarse-weight", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = grocery_meta()
    out = CHECKPOINTS / args.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train_ds = GroceryDataset(grocery_split("train"), train=True)
    val_ds = GroceryDataset(grocery_split("val"), train=False)
    print(f"載入資料 {time.perf_counter() - t0:.1f}s：train {len(train_ds)}、val {len(val_ds)}")
    g = torch.Generator().manual_seed(SEED)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, generator=g, drop_last=True
    )
    val_dl = DataLoader(val_ds, batch_size=64)

    model = HierarchicalClassifier(len(meta["fine_names"]), len(meta["coarse_names"])).to(device)
    total = args.epochs_frozen + args.epochs
    start_epoch, best, history = 0, -1.0, []
    state = None
    if args.resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        start_epoch, best, history = state["epoch"] + 1, state["best"], state["history"]
        print(f"從 epoch {start_epoch} 接續")

    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    stage = 0
    opt: torch.optim.Optimizer | None = None
    sched: torch.optim.lr_scheduler.LRScheduler | None = None
    for epoch in range(start_epoch, total):
        want_stage = 1 if epoch < args.epochs_frozen else 2
        if want_stage != stage:
            stage = want_stage
            opt = build_optimizer(model, stage, args.lr if stage == 1 else args.lr * 0.3)
            steps = (args.epochs_frozen if stage == 1 else args.epochs) * len(train_dl)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=[gr["lr"] for gr in opt.param_groups],
                total_steps=steps,
                pct_start=0.1 if stage == 2 else 0.3,
            )
            if state is not None and state["stage"] == stage:
                opt.load_state_dict(state["opt"])
                sched.load_state_dict(state["sched"])
                scaler.load_state_dict(state["scaler"])
        assert opt is not None and sched is not None

        model.train()
        t1, running, seen = time.perf_counter(), 0.0, 0
        for x, yf, yc in train_dl:
            x, yf, yc = x.to(device, non_blocking=True), yf.to(device), yc.to(device)
            opt.zero_grad(set_to_none=True)
            # AMP：4 GB VRAM 下 batch 32 才放得進去，速度也快約一倍
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                fine, coarse, _ = model(x)
                loss = loss_fn(fine, coarse, yf, yc, args.coarse_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss) * len(yf)
            seen += len(yf)
        val = evaluate(model, val_dl, device)
        rec = {
            "epoch": epoch,
            "stage": stage,
            "loss": running / seen,
            **val,
            "sec": round(time.perf_counter() - t1, 1),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if val["fine_top1"] > best:
            best = val["fine_top1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": val,
                    "n_fine": len(meta["fine_names"]),
                    "n_coarse": len(meta["coarse_names"]),
                },
                out / "best.pt",
            )
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "stage": stage,
                "best": best,
                "history": history,
            },
            out / "last.pt",
        )
    Path(out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"最佳 val 細類 top-1：{best:.4f}")


if __name__ == "__main__":
    main()
