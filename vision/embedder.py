"""以圖搜圖：embedding 最近鄰搜尋。

embedding 來自分類模型的倒數第二層（見 training/models.py），已經 L2 正規化，
所以內積就是 cosine 相似度。資料量只有數百到數千筆，numpy 矩陣乘法在 1 ms 內完成，
不需要 FAISS 或向量資料庫；超過十萬筆時再換。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def l2_normalize(v: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(norm, 1e-12)).astype(np.float32)


@dataclass(frozen=True)
class Match:
    item_id: int
    similarity: float


class EmbeddingIndex:
    """每個商品可以有多張參考圖（多個向量），查詢時取該商品所有向量中最高的相似度。

    為什麼取最大值而不是平均向量：參考圖可能是不同角度或包裝正反面，
    平均後的向量會落在中間、哪一面都不像；取最大值則只要有一張夠像就能找到。
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._ids = np.zeros((0,), dtype=np.int64)

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def item_ids(self) -> list[int]:
        return sorted({int(i) for i in self._ids})

    def add(self, item_id: int, vectors: NDArray[np.float32]) -> None:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量維度 {vectors.shape[1]} 不等於 {self.dim}")
        self._vectors = np.vstack([self._vectors, l2_normalize(vectors)])
        self._ids = np.concatenate([self._ids, np.full(len(vectors), item_id, dtype=np.int64)])

    def remove(self, item_id: int) -> None:
        keep = self._ids != item_id
        self._vectors, self._ids = self._vectors[keep], self._ids[keep]

    def search(self, query: NDArray[np.float32], k: int = 5) -> list[Match]:
        if len(self._ids) == 0:
            return []
        sims = self._vectors @ l2_normalize(np.asarray(query, dtype=np.float32).ravel())
        best: dict[int, float] = {}
        for item_id, s in zip(self._ids.tolist(), sims.tolist(), strict=True):
            if s > best.get(item_id, -2.0):
                best[item_id] = s
        ranked = sorted(best.items(), key=lambda t: (-t[1], t[0]))[:k]
        return [Match(i, round(float(s), 4)) for i, s in ranked]

    def search_batch(self, queries: NDArray[np.float32], k: int = 5) -> list[list[Match]]:
        return [self.search(q, k) for q in queries]
