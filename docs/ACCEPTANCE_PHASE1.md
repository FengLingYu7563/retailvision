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
