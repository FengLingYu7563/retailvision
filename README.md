# RetailVision：零售商品影像辨識服務

作品集專案。用公開資料集模擬超市情境，做三件事：

1. **單品辨識**：拍一個商品，回答它是什麼（細類 81 種 + 粗類 43 種）。
2. **貨架分析**：拍一排貨架，算出商品數量、缺貨空位、陳列密度。
3. **新品登錄與以圖搜圖**：新商品不用重新訓練，上傳幾張參考圖就能辨識。

> 目前進度：**Phase 1（影像處理與模型，離線）**。API、資料庫、部署為 Phase 2 / 3。

## Phase 1 結果

指標全部來自 `models/metrics.json`，由 `training/evaluate.py` 用**匯出後的 ONNX 模型**計算（也就是上線時實際使用的檔案）。

### 單品分類（Grocery Store 測試集 2,485 張）

| 模型 | 細類 top-1 | 細類 top-5 | 粗類 top-1 | CPU 延遲 p95 |
|---|---|---|---|---|
| HSV 直方圖 + HOG + SVM（baseline） | 31.3% | 無 | 48.0% | 約 8 ms（只算 SVM） |
| MobileNetV3-Large 雙頭（ONNX） | **85.7%** | **98.2%** | **90.5%** | 7.7 ms（含前處理） |
| 門檻（驗收 1.4） | 75% | 95% | 85% | |

- 延遲機器：RTX 3050 Laptop 4 GB 的筆電，ONNX Runtime CPU 執行，單張圖片。
- 驗收 1.6：ONNX 與 PyTorch 在 296 張 val 圖上 logits 最大誤差 < 1e-3，top-1 一致率 100%（`tests/unit/test_onnx_parity.py`）。

### 最常混淆的類別（驗收 1.9）

混淆矩陣：`models/confusion_fine.png`、`models/confusion_coarse.png`

| 排名 | 類別組 | 次數 | 可能原因 |
|---|---|---|---|
| 1 | Orange 與 Satsumas | 27 | 都是橘色、圓形、表皮紋理相近；差別主要在大小，但照片沒有尺度參考 |
| 2 | Golden-Delicious 被判成 Granny-Smith | 20（單向） | 黃綠色調在賣場暖色燈光下接近；單向混淆表示模型偏向 Granny-Smith（訓練張數 59 對 45） |
| 3 | Lemon 被判成 Red-Grapefruit / Orange 與 Red-Grapefruit | 各 14 | 柑橘類形狀相同，顏色在黃到橘之間連續變化，裝在網袋裡時表皮細節也被遮住 |

共同點：容易混淆的都是**同一粗類裡的不同品種**，所以粗類準確率（90.5%）明顯高於細類（85.7%）。這也是做雙頭輸出的理由：細類不確定時，API 至少能回報可信的粗類。

### 以圖搜圖（驗收 1.7，未達標）

| Gallery | test top-1 | test top-5 |
|---|---|---|
| 每類 1 張官方標準圖、1 個向量（驗收值） | 52.8% | 83.2% |
| 同上，加翻轉與 2x2 / 3x3 排列增強（實驗） | 54.5% | 83.4% |
| 每類 3 張店內實拍照（模擬新品登錄） | 79.7% | 97.0% |
| 門檻（驗收 1.7，標準圖 gallery） | 70% | |

標準圖是白底棚拍，和店內實拍的差距太大。試了 4 種不用重新訓練的方法，都沒辦法接近 70%，觸發「沒有進步」的停止條件。
完整實驗紀錄與原因分析：[docs/experiments/retrieval_1_7.md](docs/experiments/retrieval_1_7.md)。
新品登錄 API 的設計是上傳實拍參考圖，對應的是第二列的情境。

### 影像品質檢查

`vision/quality.py` 用 Laplacian 變異數判斷模糊。門檻 30 是用訓練集 2,640 張校正的：原圖 99.8% 通過，`GaussianBlur(9x9, sigma=2)` 之後最高只有 26.5。
常見的門檻 100 是給高解析度文件照用的，蔬果表面平滑，用 100 會誤判六成原圖是模糊的。

### 貨架偵測（驗收 1.5，SKU-110K 完整測試集 2,936 張）

| 模型 | mAP50 | mAP50-95 | CPU 延遲 p95（偵測 + 貨架分析） |
|---|---|---|---|
| YOLOv8n，PyTorch 權重 | **0.903** | 0.546 | 無 |
| YOLOv8n，ONNX（上線版本） | **0.900** | 0.544 | 161 ms |
| 門檻 | 0.75 | | |

- 訓練：3,000 張固定子集，640 輸入，batch 8，AMP，60 epochs（RTX 3050 4 GB，約 1.7 小時）。
- `max_det=1000`：SKU-110K 一張圖可能有 700 個商品，預設上限 300 會讓 mAP 被低估。
- **抓到的錯誤**：一開始把 `max_det` 傳給 `cv2.dnn.NMSBoxes` 的 `top_k`。OpenCV 的 `top_k` 是在 NMS **之前**只保留前 k 個候選框，密集貨架的候選框有好幾千個，一張圖會少偵測 10% 以上的商品。`tests/unit/test_detector_parity.py` 比對 Ultralytics 用同一個 ONNX 檔的推論結果時發現這個問題，修正後兩邊的框數量與位置完全一致。

### 缺貨空位（驗收 1.8，未達標）

| 評估集 | 召回率 | 精確率 | 門檻 |
|---|---|---|---|
| 15 張人工標註貨架圖，94 個空位（IoU >= 0.3，一對一配對） | 25.5% | 60.0% | 召回率 70% |

主要原因是紋理驗證太嚴格，剔除了 47% 的真實空位；就算完全不做紋理驗證，召回率上限也只有 72%（精確率降到 40%）。各階段的漏失分析和門檻取捨見 [docs/experiments/gaps_1_8.md](docs/experiments/gaps_1_8.md)。這項是 L3，依規格記錄原因，不調整門檻。

偵測模型只框出有商品的地方，空位要從框之間的空隙推算（`vision/shelf.py`）：依框的高度分層、找出層內的水平空隙，再做**紋理驗證**：比較空隙和同層商品的邊緣密度（Canny 加積分影像）。偵測器漏掉的商品還是有包裝的邊緣和文字，真正缺貨的地方通常只看得到背板。加上這一步之後，一張測試圖的空位誤判從 32 個降到 5 個。

人工標註流程與避免高估召回率的設計見 `training/annotate_gaps.py`。

## 重現步驟

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -e ".[train,dev]"

python training/data/download_grocery.py
python training/data/download_sku110k.py        # 約 12.2 GB，多連線續傳
python training/data/make_splits.py             # 切分檔已 commit，重跑結果相同

python -m training.train_classical
python -m training.train_classifier
python -m training.train_detector
python -m training.export_onnx
python -m training.evaluate
pytest
```

## 資料集出處與授權

本專案只做非商用展示，不重新散布任何原始資料；資料集由腳本從官方來源下載，不包含在 repo 裡。

- **Grocery Store Dataset**：M. Klasson, C. Zhang, H. Kjellström. *A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels.* WACV 2019. https://github.com/marcusklasson/GroceryStoreDataset
  - 使用官方 train / val / test 切分。粗類實際有 43 個（論文寫 42，以資料中的 `classes.csv` 為準）。
- **SKU-110K**：E. Goldman et al. *Precise Detection in Densely Packed Scenes.* CVPR 2019. https://github.com/eg4000/SKU110K_CVPR19
  - 訓練用固定 seed 抽樣的 3,000 張子集，評估用完整官方測試集 2,936 張。

## 已知限制

- 以圖搜圖在「白底標準圖對店內實拍」的情境下準確率不足（見上方）。
- 缺貨空位判斷依賴偵測結果：偵測器表現差的畫面（例如吊掛式小包裝、透視角度很大的棧板堆疊），分層和空位都會不準；表面平滑、沒有圖案的包裝也可能被誤判成空位。
- 細類只有 81 種，都是瑞典超市的商品；換到其他國家的商品需要重新訓練或登錄參考圖。
