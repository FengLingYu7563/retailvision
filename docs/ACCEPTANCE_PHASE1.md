# Phase 1 驗收標準（使用者已核准，2026-10-13）

依據 `RetailVision_SPEC.md` 第 9 節 Phase 1，外加以下 5 點使用者核准的調整。

## 原始標準

| # | 標準 | 等級 | 驗證方式 |
|---|---|---|---|
| 1.1 | 資料下載與切分腳本可重現：切分寫成檔案、固定 seed，重跑結果完全相同 | L1 | 跑兩次後 diff 切分檔 |
| 1.2 | `vision/quality.py` 能分辨模糊與清晰的圖：GaussianBlur 模糊圖判為未通過，原圖通過 | L1 | pytest |
| 1.3 | SVM baseline 完成，在測試集回報細類與粗類 top-1 準確率 | L1 | metrics.json |
| 1.4 | CNN 分類：粗類 top-1 >= 85%，細類 top-1 >= 75%、top-5 >= 95%，且每一項都高於 SVM | L1 | evaluate.py |
| 1.5 | YOLO 貨架偵測：SKU-110K 測試集（或固定測試子集）mAP50 >= 0.75 | L1 | evaluate.py |
| 1.6 | ONNX 與 PyTorch 輸出比對：分類 logits 最大誤差 < 1e-3，top-1 一致率 100% | L1 | pytest |
| 1.7 | 以圖搜圖：Grocery Store 標準圖建 gallery，測試圖 top-1 檢索準確率 >= 70% | L1 | evaluate.py |
| 1.8 | 缺貨空位偵測：至少 10 張人工標註空位的貨架圖，空位召回率 >= 70% | L3 | evaluate.py |
| 1.9 | 產出混淆矩陣圖，README 寫出最常混淆的 3 組類別與可能原因 | L3 | 人工檢查 |

## 核准的調整

1. **1.1**：Grocery Store 使用官方 train/val/test 切分（與論文可比），切分腳本把官方清單正規化後寫成檔案；固定 seed 的抽樣只用於 SKU-110K 子集清單（約 3,000 張訓練圖，以及固定的 val/test 子集）。
2. **1.5**：SKU-110K 單張圖物件可達約 700 個，訓練與評估一律設 `max_det=1000`，避免預設 300 造成 mAP 被低估。
3. **1.6**：一致性測試需要 torch 與本機權重；缺少時以 `needs_weights` marker 自動 skip（CI 不跑），本機必須跑過並通過。
4. **1.7**：gallery 每個細類只有 1 張棚拍去背標準圖，與店內實拍有明顯 domain gap；若未達 70%，依第 9 節規則在 README 附實驗紀錄說明，不默默降標。
5. **1.8**：空位標註由程式產生候選框，再由使用者人工確認或修正；未完成人工標註前此項記為「待人工標註」。

## Codex 審查補充的檢查項目（使用者已核准納入，2026-10-15）

| 項目 | 等級 | 驗證方式 |
|---|---|---|
| `vision/` 不 import torch、torchvision、ultralytics、fastapi、starlette | L1 | `tests/unit/test_repo_hygiene.py` |
| repo 不追蹤資料集、checkpoint、權重、log、`.env` | L2 | `tests/unit/test_repo_hygiene.py` |
| 資料集用 HTTPS 下載，解壓前驗證 SHA-256，解壓防路徑穿越 | L2 | `training/data/download_sku110k.py` |
| 1.7 驗收值採嚴格定義（每類 1 張標準圖、1 個向量），增強結果另列 | L1 | `training/evaluate.py` |
| 1.8 召回率用一對一配對；「10 張」只算有標註空位的圖 | L1 | `tests/unit/test_shelf.py`、`training/evaluate.py` |
| 自寫的 ONNX YOLO 解碼與 Ultralytics 逐框一致 | L1 | `tests/unit/test_detector_parity.py` |
| 分類器中斷後接續訓練的自動化測試 | L3 | **未實作**（訓練迴圈需先重構） |
| 空位分類器（joblib / pickle）只載入專案產生的檔案：先驗 SHA-256 再反序列化，可限定資料夾 | L2 | `tests/unit/test_shelf.py` |
| 偵測框被判成空位時，item_count、density、gap_ratio 一致且 NMS 去重 | L1 | `tests/unit/test_shelf.py` |
| README 與 docs 的數字和 `models/metrics.json` 一致（每次評估更新後核對） | L3 | 人工檢查 |

## 重做 1.7 / 1.8 的方法規定（使用者 2026-10-15 決定：不接受第一版進 Phase 2）

門檻不變（1.7 top-1 >= 70%、1.8 召回率 >= 70%），新增以下方法規定：

- **1.7** 試兩種做法並比較：A 度量學習重訓 MobileNetV3（SupCon + 標準圖擴增）、B DINOv2-small 特徵。
  - 方法與超參數只用 Grocery val 比較；test 只在選定後跑一次。
  - A 另做「保留類別」實驗：固定 seed 挑 20 個細類完全不參與訓練（store 照片與標準圖都不用），
    獨立報告這 20 類的 test 檢索準確率，證明不是背答案。正式模型仍用 81 類訓練。
  - 換掉分類模型的話，1.4 與 1.6 必須重新驗證並通過。
- **1.8** 另標 10 張調參集（`gap_annotations_tune.json`，來自 SKU-110K val）。演算法修改與參數選擇只看調參集；
  原本 15 張（來自 test）只用來報告最後成績。

## Phase 1 狀態（2026-10-15）

| # | 結果 | 狀態 |
|---|---|---|
| 1.1 | 兩次產出逐位元組相同 | 通過 |
| 1.2 | 合成圖與 50 張真實圖 | 通過 |
| 1.3 | SVM 細類 31.3%、粗類 48.0% | 通過 |
| 1.4 | v2 細類 95.1% / top-5 99.8% / 粗類 96.7%（v1：85.7% / 98.2% / 90.5%） | 通過 |
| 1.5 | mAP50 0.903（pt）/ 0.900（ONNX） | 通過 |
| 1.6 | v1、v2 logits 誤差 < 1e-3，top-1 一致 100% | 通過 |
| 1.7 | v2 top-1 95.3%（v1：52.8%）；沒看過的 20 類 48.5% / top-5 92.9% | 通過，見 `docs/experiments/retrieval_1_7.md` |
| 1.8 | v2 召回率 56.4%、精確率 38.7%（15 張、94 個空位；v1 為 26.6%） | **未達標（L3）**，見 `docs/experiments/gaps_1_8.md` |
| 1.9 | 混淆矩陣與前 3 組分析 | 通過（待人工檢查） |
