"""MobileNetV3 雙頭分類 + 度量學習（SupCon）。

目標是讓標準圖與店內實拍的 embedding 對齊（驗收 1.7）。

和 train_classifier.py（v1）的差別：
1. 訓練資料加入 Grocery Store 的標準圖（每類 1 張白底棚拍），並做強擴增：
   換背景、排成一堆、裁切、色彩抖動，讓模型看過「標準圖長得像實拍」的各種變化。
2. PK 取樣：每個 batch 抽 P 個類別，每類 1 張標準圖 + K 張實拍。
   SupCon 需要同一個 batch 裡有同類別的正樣本，隨機抽 batch 的話 81 類幾乎湊不到。
3. 損失 = 細類 CE + 0.5 粗類 CE + supcon_weight x SupCon（在 L2 正規化的 1280 維 embedding 上）。
   CE 保住分類準確率（1.4），SupCon 把同類別的標準圖和實拍拉近、不同類別推遠。
4. 每個 epoch 在 val 上同時看分類 top-1 與「每類 1 張標準圖」的檢索 top-1，
   以兩者總和選 best.pt。test 完全不在這裡使用。

--holdout N：固定 seed 挑 N 個細類完全不參與訓練（實拍與標準圖都排除），
用來測量模型對「沒看過的新商品」的檢索能力，證明不是背答案。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Sampler
from torchvision.transforms import v2

from training.common import (
    CHECKPOINTS,
    GROCERY,
    SEED,
    Sample,
    grocery_meta,
    grocery_split,
    seed_everything,
)
from training.models import ARCHS, build_model
from training.train_classifier import CACHE_SHORT_SIDE, build_optimizer, loss_fn
from vision.io import load_image
from vision.preprocess import resize_short_side, to_classifier_input

HOLDOUT_SEED = 1234


def holdout_classes(n: int, n_fine: int) -> list[int]:
    return sorted(random.Random(HOLDOUT_SEED).sample(range(n_fine), n)) if n else []


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss（Khosla et al., 2020）。emb 必須已 L2 正規化。"""
    emb = emb.float()
    sim = emb @ emb.T / temperature
    n = len(labels)
    self_mask = torch.eye(n, dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = pos.sum(1)
    valid = n_pos > 0
    mean_log_pos = (log_prob.masked_fill(~pos, 0).sum(1)[valid]) / n_pos[valid]
    return -mean_log_pos.mean()


def white_mask(img: np.ndarray) -> np.ndarray:
    """標準圖是白底：亮度高且飽和度低的像素視為背景。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg = (hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 25)
    fg = (~bg).astype(np.uint8)
    return cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def tile(img: np.ndarray, n: int) -> np.ndarray:
    small = cv2.resize(img, (img.shape[1] // n, img.shape[0] // n), interpolation=cv2.INTER_AREA)
    return np.vstack([np.hstack([small] * n)] * n)


class MetricDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int, int]]):
    """索引 0..len(store)-1 是實拍照，之後每個類別一張標準圖。"""

    def __init__(
        self,
        store: list[Sample],
        iconic: dict[int, tuple[np.ndarray, int]],
        backgrounds: list[np.ndarray],
    ) -> None:
        self.store = store
        self.store_imgs = [resize_short_side(load_image(s.path), CACHE_SHORT_SIDE) for s in store]
        self.iconic_ids = sorted(iconic)
        self.iconic = iconic
        self.backgrounds = backgrounds
        self.rng = np.random.default_rng(SEED)
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(224, scale=(0.4, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
                # hue 範圍很小：蘋果、甜椒這類商品靠顏色區分
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.store) + len(self.iconic_ids)

    def iconic_view(self, img: np.ndarray) -> np.ndarray:
        """把白底標準圖變得比較像店內實拍：換背景、排成一堆。"""
        out = img
        if self.rng.random() < 0.6 and self.backgrounds:
            bg = self.backgrounds[int(self.rng.integers(len(self.backgrounds)))]
            bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
            out = np.where(white_mask(img)[:, :, None] > 0, img, bg)
        if self.rng.random() < 0.5:
            out = tile(out, int(self.rng.integers(2, 4)))
        return resize_short_side(out, CACHE_SHORT_SIDE)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        if i < len(self.store):
            s = self.store[i]
            img, fine, coarse = self.store_imgs[i], s.fine_id, s.coarse_id
        else:
            fine = self.iconic_ids[i - len(self.store)]
            base, coarse = self.iconic[fine]
            img = self.iconic_view(base)
        rgb = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1)
        return self.aug(rgb), fine, coarse


class PKSampler(Sampler[list[int]]):
    """每個 batch：P 個類別 x（1 張標準圖 + K 張實拍）。"""

    def __init__(self, ds: MetricDataset, p: int, k: int, seed: int) -> None:
        self.by_class: dict[int, list[int]] = {}
        for i, s in enumerate(ds.store):
            self.by_class.setdefault(s.fine_id, []).append(i)
        self.iconic_index = {f: len(ds.store) + j for j, f in enumerate(ds.iconic_ids)}
        self.classes = sorted(self.by_class)
        self.p, self.k = p, k
        self.n_batches = len(ds.store) // (p * k)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):  # type: ignore[no-untyped-def]
        for _ in range(self.n_batches):
            batch: list[int] = []
            for c in self.rng.sample(self.classes, self.p):
                pool = self.by_class[c]
                picks = (
                    self.rng.sample(pool, self.k)
                    if len(pool) >= self.k
                    else self.rng.choices(pool, k=self.k)
                )
                batch += picks
                if c in self.iconic_index:
                    batch.append(self.iconic_index[c])
            yield batch


@torch.no_grad()
def embed_all(model: torch.nn.Module, imgs: list[np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(imgs), 64):
        x = torch.from_numpy(np.concatenate([to_classifier_input(im) for im in imgs[i : i + 64]]))
        out.append(model(x.to(device))[2].float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def prepare_data(
    holdout: list[int],
) -> tuple[list[Sample], list[Sample], dict[int, np.ndarray], list[np.ndarray]]:
    """回傳 (train, val, 標準圖 {類別: 影像}, 背景素材)，全部排除保留類別。

    保留類別的任何圖（train、val、標準圖）在訓練程式裡都不讀取，只在 evaluate.py 的
    eval_holdout 最後評估時才用到；背景素材也只從排除後的 train 抽。
    tests/unit/test_holdout_isolation.py 會攔截 load_image 檢查這一點。
    """
    meta = grocery_meta()
    train = [s for s in grocery_split("train") if s.fine_id not in holdout]
    val = [s for s in grocery_split("val") if s.fine_id not in holdout]
    iconic = {
        f: load_image(GROCERY / rel)
        for f, rel in enumerate(meta["iconic_images"])
        if f not in holdout
    }
    # 換背景用的素材：實拍照模糊後當背景，只保留色調與明暗、不保留可辨識的商品
    bg_rng = random.Random(SEED)
    backgrounds = [
        cv2.GaussianBlur(resize_short_side(load_image(s.path), 256), (0, 0), 8)
        for s in bg_rng.sample(train, 60)
    ]
    return train, val, iconic, backgrounds


@torch.no_grad()
def val_metrics(
    model: torch.nn.Module,
    val: list[Sample],
    val_imgs: list[np.ndarray],
    iconic: dict[int, np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    """val 上的分類 top-1，以及「每類 1 張標準圖」gallery 的檢索 top-1（嚴格定義）。

    呼叫端只會傳入訓練範圍內的類別（--holdout 時已排除保留類別的 val 圖與標準圖）：
    保留類別的圖不能出現在 query，也不能出現在 gallery 當干擾項，
    否則 checkpoint 的選擇就間接用到了「沒看過」的類別（Codex 第 6 輪審查指出的洩漏）。
    """
    model.eval()
    ids = sorted(iconic)
    q = embed_all(model, val_imgs, device)
    g = embed_all(model, [iconic[i] for i in ids], device)
    y = np.array([s.fine_id for s in val])
    # gallery 的第 k 列是類別 ids[k]，用明確的 ID 對應，不依賴索引剛好等於類別編號
    pred_ret = np.array(ids)[(q @ g.T).argmax(1)]
    x = torch.from_numpy(np.concatenate([to_classifier_input(im) for im in val_imgs]))
    fine_logits = torch.cat(
        [model(x[i : i + 64].to(device))[0].float().cpu() for i in range(0, len(x), 64)]
    )
    pred_cls = fine_logits.argmax(1).numpy()
    return {
        "fine_top1": float((pred_cls == y).mean()),
        "retrieval_top1": float((pred_ret == y).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="classifier_mnv3_metric")
    ap.add_argument("--epochs-frozen", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--p", type=int, default=12, help="每個 batch 的類別數")
    ap.add_argument("--k", type=int, default=3, help="每類實拍張數")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--coarse-weight", type=float, default=0.5)
    ap.add_argument("--supcon-weight", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="mnv3")
    args = ap.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = grocery_meta()
    n_fine, n_coarse = len(meta["fine_names"]), len(meta["coarse_names"])
    holdout = holdout_classes(args.holdout, n_fine)
    out = CHECKPOINTS / args.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train, val, iconic, backgrounds = prepare_data(holdout)
    iconic_train = {f: (img, meta["fine_to_coarse"][f]) for f, img in iconic.items()}
    ds = MetricDataset(train, iconic_train, backgrounds)
    val_imgs = [resize_short_side(load_image(s.path), CACHE_SHORT_SIDE) for s in val]
    sampler = PKSampler(ds, args.p, args.k, SEED)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler)
    print(
        f"載入 {time.perf_counter() - t0:.1f}s：train {len(train)}、標準圖 {len(iconic_train)}、"
        f"保留類別 {holdout}",
        flush=True,
    )

    model = build_model(args.arch, n_fine, n_coarse).to(device)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    total = args.epochs_frozen + args.epochs
    best, history, stage = -1.0, [], 0
    opt: torch.optim.Optimizer | None = None
    sched: Any = None
    for epoch in range(total):
        want = 1 if epoch < args.epochs_frozen else 2
        if want != stage:
            stage = want
            opt = build_optimizer(model, stage, args.lr if stage == 1 else args.lr * 0.3)
            steps = (args.epochs_frozen if stage == 1 else args.epochs) * len(dl)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=[gr["lr"] for gr in opt.param_groups],
                total_steps=steps,
                pct_start=0.1 if stage == 2 else 0.3,
            )
        assert opt is not None
        model.train()
        t1, ce_sum, sc_sum, seen = time.perf_counter(), 0.0, 0.0, 0
        for x, yf, yc in dl:
            x, yf, yc = x.to(device, non_blocking=True), yf.to(device), yc.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                fine, coarse, emb = model(x)
                ce = loss_fn(fine, coarse, yf, yc, args.coarse_weight)
            # SupCon 在 fp32 算：除以溫度 0.1 後 exp 容易在 fp16 溢位
            sc = supcon_loss(emb, yf, args.temperature)
            loss = ce + args.supcon_weight * sc
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ce_sum += float(ce) * len(yf)
            sc_sum += float(sc) * len(yf)
            seen += len(yf)
        m = val_metrics(model, val, val_imgs, iconic, device)
        rec = {
            "epoch": epoch,
            "stage": stage,
            "ce": ce_sum / seen,
            "supcon": sc_sum / seen,
            **m,
            "sec": round(time.perf_counter() - t1, 1),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        score = m["fine_top1"] + m["retrieval_top1"]
        if score > best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": m,
                    "arch": args.arch,
                    "n_fine": n_fine,
                    "n_coarse": n_coarse,
                    "holdout": holdout,
                    "args": vars(args),
                },
                out / "best.pt",
            )
    Path(out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"最佳 val（細類 + 檢索）：{best:.4f}")


if __name__ == "__main__":
    main()
