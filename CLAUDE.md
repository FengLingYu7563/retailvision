# Project Memory

## Project Overview

**Name**: RetailVision 零售商品影像辨識服務
**Purpose**: 作品集專案。單品辨識（Grocery Store）、貨架偵測與缺貨空位（SKU-110K）、新品登錄與以圖搜圖，之後以 RESTful API + DB + 儀表板提供。
**Main directory**: .
**Spec**: `RetailVision_SPEC.md`（需求、架構、API、驗收標準）
**GitHub**: not configured（尚未建 repo，未 push）
**Obsidian vault**: not used
**Obsidian dashboard**: not used

## Workflow

開工：讀本檔、`docs/ACCEPTANCE_PHASE1.md` 的狀態表、`git status`，不自動 pull/commit/push。
收工：在此總結完成工作、待辦、決策；commit 或 push 前先問。

規格要求的流程：大任務先寫驗收標準給使用者審核；拆解 -> 產出 -> 驗證 -> 修改。
停止條件：同一問題 5 次未過、錯誤無進步、超過成本，停下回報，不假裝完成。
每個 Phase 結束：先過機械檢查，再送 Codex 審查（L1 / L2 FAIL 必修，L3 參考）。

## 專案慣例

- 環境：`.venv`（Python 3.12），RTX 3050 Laptop 4 GB，CUDA 12.6。
- 機械檢查：
  ```bash
  .venv/Scripts/python -m ruff check vision training tests
  .venv/Scripts/python -m ruff format --check vision training tests
  .venv/Scripts/python -m mypy
  .venv/Scripts/python -m pytest tests -q
  ```
- `vision/` 是純函式模組，不可 import torch / torchvision / ultralytics / fastapi（有測試把關）。推論只用 ONNX Runtime。
- 資料集、checkpoint、`.onnx`、`.joblib` 不 commit；只 commit 腳本、切分清單、`models/metrics.json`、模型旁的 `.json` 中繼資料、圖表、人工標註 JSON。
- 參數與方法只能在 val / 調參集上選，test / 評估集只在最後跑一次。
- 安裝套件：一次只跑一個 pip；ultralytics 會從 PyPI 換掉 CUDA 版 torch，torch 一律最後用 `--force-reinstall --no-deps` 從 `download.pytorch.org/whl/cu126` 裝；`timm` 用 `--no-deps`。
- 跑 Ultralytics 前設 `YOLO_AUTOINSTALL=false`，避免它自動安裝未經同意的套件。
- 所有輸出繁體中文、不用 emoji；commit 訊息不加 AI 共同作者署名。
- PowerShell 5.1 的 `Set-Content -Encoding utf8` 會加 BOM，改用 `[IO.File]::WriteAllText(..., UTF8Encoding($false))`。

## 主要檔案

| 用途 | 位置 |
|---|---|
| 影像與推論模組 | `vision/`（quality、preprocess、classical、classifier、embedder、detector、shelf、draw） |
| 訓練與評估 | `training/`（train_classical、train_classifier、train_metric、train_detector、export_onnx、evaluate、fit_gap_classifier、annotate_gaps） |
| 切分清單與標註 | `training/data/splits/`、`training/data/gap_annotations*.json` |
| 指標與圖表 | `models/metrics.json`、`models/confusion_*.png` |
| 驗收狀態 | `docs/ACCEPTANCE_PHASE1.md` |
| 實驗紀錄 | `docs/experiments/retrieval_1_7.md`、`docs/experiments/gaps_1_8.md` |

## Last Progress

**Phase 1 完成**；**Phase 2 實作完成，機械檢查與實測 2.1 到 2.12 全部通過，等 Codex 審查**。
狀態表在 `docs/ACCEPTANCE_PHASE1.md`、`docs/ACCEPTANCE_PHASE2.md`。

- 上線模型：分類 / embedding `cls-dinov2-v2`、偵測 `det-yolov8n-v1`、空位分類器 `models/gap_classifier.*`（GBT）。
- API：`uvicorn app.main:create_app --factory`；PostgreSQL / Redis 用 `docker compose up -d db redis`；
  測試資料庫 `retailvision_test`（API 測試會自動 migrate 與清空）。
- 1.8（L3）未達標並已記錄。15 張評估集已用掉，改方法要另標新的評估集。

## Todos

1. Phase 2 送 Codex 審查（request 檔要含原始需求、`docs/ACCEPTANCE_PHASE2.md`、審查範圍 `app/`、`migrations/`、`tests/api/`、`tools/`）。
2. Phase 2 PASS 後，跟使用者確認再開始 Phase 3（Docker、CI、模擬相機、Streamlit、README），先寫驗收標準給使用者審核。
3. L3 待辦：分類器中斷後接續訓練的自動化測試（要先重構訓練迴圈）。
4. 1.8 延伸方向（未排程）：YOLO 加「空位」類別，需要數百張空位標註。

## Important Decisions

- Grocery Store 用官方切分；SKU-110K 用 seed 42 子集（train 3000 / val 500 / 完整 test 2936）。
- 偵測 `max_det=1000`；`cv2.dnn.NMSBoxes` 不可傳 `top_k`（會在 NMS 前截斷候選）。
- 模糊門檻 30（用訓練集校正，不用常見的 100）。
- 1.7 嚴格定義：每類 1 張標準圖、1 個向量。A（MobileNetV3+SupCon）與 B（DINOv2+SupCon）在 val 比較後選 B；保留 20 類實驗顯示沒看過的商品 top-1 47.0%、top-5 92.7%，新品登錄 API 要回傳前 k 名。
- 1.8 標註：調參集來自 SKU val、評估集來自 SKU test，分開使用；預填寬鬆框，避免高估召回率。
  判斷原則是「正面看過去那個位置有沒有商品」：上下堆疊、上層空掉要框；前排有、後排空（往前推）不框，
  那是庫存偏低，正面照片看不到深度（列為延伸方向）。
- SKU-110K 下載用 HTTPS 並驗證釘選的 SHA-256（trust on first use）。

## Do Not Do

- 未經明確同意不刪檔。
- 不改寫 Git 歷史。
- 未經明確同意不 push、不建 GitHub repo。
- 下載資料集或安裝大型套件前先問（列出名稱、來源、大小）。
- 不用評估集 / test 調參數；不默默降低驗收門檻。
