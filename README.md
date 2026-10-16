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
| v1：MobileNetV3-Large 雙頭（ONNX） | 85.7% | 98.2% | 90.5% | 7.7 ms |
| **v2：DINOv2-small + SupCon 雙頭（ONNX，上線版本）** | **95.1%** | **99.8%** | **96.7%** | 55.8 ms |
| 門檻（驗收 1.4） | 75% | 95% | 85% | |

- 延遲機器：RTX 3050 Laptop 4 GB 的筆電，ONNX Runtime CPU 執行，單張圖片，含前處理。
- 驗收 1.6：v1、v2 的 ONNX 與 PyTorch 在 296 張 val 圖上 logits 最大誤差都 < 1e-3，top-1 一致率 100%（`tests/unit/test_onnx_parity.py`）。
- v1 是第一版模型，因為以圖搜圖（1.7）沒達標而改成 v2，保留下來做對照（見下方以圖搜圖段落）。

### 最常混淆的類別（驗收 1.9，v2 模型）

混淆矩陣：`models/confusion_fine.png`、`models/confusion_coarse.png`

| 排名 | 類別組 | 次數 | 可能原因 |
|---|---|---|---|
| 1 | Satsumas 被判成 Orange | 20（反方向只有 1） | 都是橘色、圓形、表皮紋理相近，主要差別在大小，但照片沒有尺度參考；v1 時兩個方向都會錯，v2 剩下單向 |
| 2 | Golden-Delicious 與 Granny-Smith | 13（9 對 4） | 黃綠色調在賣場暖色燈光下很接近；v1 時有 20 次 |
| 3 | Royal-Gala 被判成 Pink-Lady | 10（單向） | 兩者都是紅黃漸層的表皮，差別在顏色比例和斑點，堆成一堆時很難分辨 |

共同點：容易混淆的都是**同一粗類裡的不同品種**（柑橘、蘋果），所以粗類準確率（96.7%）高於細類（95.1%）。這也是做雙頭輸出的理由：細類不確定時，API 至少能回報可信的粗類。

### 以圖搜圖（驗收 1.7）

Gallery 是**每類 1 張官方白底標準圖、1 個向量**（嚴格定義），查詢是 2,485 張店內實拍測試圖。

| 模型 | test top-1 | test top-5 |
|---|---|---|
| v1：MobileNetV3（只用分類損失訓練） | 52.8% | 83.2% |
| **v2：DINOv2-small + SupCon（上線版本）** | **95.3%** | **99.9%** |
| 門檻 | 70% | |

**新商品泛化**（另外訓練一個實驗模型，固定 seed 挑出的 20 個細類完全不參與訓練；gallery 仍然是 81 類各 1 張標準圖）：

| 類別 | test top-1 | test top-5 | 張數 |
|---|---|---|---|
| 訓練時看過的 61 類 | 95.6% | 99.9% | 1,823 |
| **訓練時沒看過的 20 類** | **47.0%** | **92.7%** | 662 |

v2 能達到 95% 有一部分原因是**訓練時看過這些商品的標準圖**。對完全沒看過的新商品，第一名的準確率只有大約一半，但正確答案有 93% 會出現在前 5 名。所以新品登錄的 API 應該回傳前 k 名讓人確認，不要直接採用第一名。
v1 的失敗分析、A 和 B 兩種方案的比較，以及選擇的理由，都記在 [docs/experiments/retrieval_1_7.md](docs/experiments/retrieval_1_7.md)。

### 影像品質檢查

`vision/quality.py` 用 Laplacian 變異數判斷模糊。門檻 30 是用訓練集 2,640 張校正的：原圖 99.8% 通過，`GaussianBlur(9x9, sigma=2)` 之後最高只有 26.5。
常見的門檻 100 是給高解析度文件照用的，蔬果表面平滑，用 100 會誤判六成原圖是模糊的。

### 貨架偵測（驗收 1.5，SKU-110K 完整測試集 2,936 張）

| 模型 | mAP50 | mAP50-95 | CPU 延遲 p95（偵測 + 貨架分析） |
|---|---|---|---|
| YOLOv8n，PyTorch 權重 | **0.903** | 0.546 | 無 |
| YOLOv8n，ONNX（上線版本） | **0.900** | 0.544 | 212 ms（含空位分類器） |
| 門檻 | 0.75 | | |

- 訓練：3,000 張固定子集，640 輸入，batch 8，AMP，60 epochs（RTX 3050 4 GB，約 1.7 小時）。
- `max_det=1000`：SKU-110K 一張圖可能有 700 個商品，預設上限 300 會讓 mAP 被低估。
- **抓到的錯誤**：一開始把 `max_det` 傳給 `cv2.dnn.NMSBoxes` 的 `top_k`。OpenCV 的 `top_k` 是在 NMS **之前**只保留前 k 個候選框，密集貨架的候選框有好幾千個，一張圖會少偵測 10% 以上的商品。`tests/unit/test_detector_parity.py` 比對 Ultralytics 用同一個 ONNX 檔的推論結果時發現這個問題，修正後兩邊的框數量與位置完全一致。

### 缺貨空位（驗收 1.8，未達標）

評估集：15 張人工標註貨架圖（SKU-110K test）、94 個空位；IoU >= 0.3、一對一配對。

| 方法 | 召回率 | 精確率 |
|---|---|---|
| v1：空隙寬度 + 紋理規則 | 26.6% | 61.0% |
| **v2：候選（空隙 + 偵測框）+ 梯度提升樹** | **56.4%** | **38.7%** |
| 門檻 | 70% | |

v2 用另外 40 張調參集（SKU-110K val、236 個空位）訓練，門檻在調參集的交叉驗證上預先選好，評估集只跑一次。
召回率比 v1 提高一倍，但仍未達 70%：交叉驗證的召回率剛好壓在 70%，換到評估集就掉下來；
偵測器把空格誤框成商品、候選框形狀和人工標註對不上、深色商品和空背板外觀相近，是主要的限制。
CNN 特徵、多種形狀的候選都試過，沒有幫助。完整紀錄見 [docs/experiments/gaps_1_8.md](docs/experiments/gaps_1_8.md)。這項是 L3，依規格記錄原因，不調整門檻。

貨架分析完整流程（偵測 + 空位分類器）的 CPU 延遲 p95 為 212 ms。

偵測模型只框出有商品的地方，空位要從框之間的空隙推算（`vision/shelf.py`），先依框的高度分層：

- **v2（上線版本）**：候選有兩種，一種是層內框與框之間的空隙（0.4 個商品寬以上），另一種是**偵測框本身**（偵測器會把暗色的空格誤框成商品）。對每個候選算出和同層商品比較的外觀特徵（亮度、飽和度、邊緣密度與其變異、寬度、偵測信心、是否在貨架邊緣），用梯度提升樹判斷，最後用 NMS 去掉重複。被判成空位的偵測框不算進商品數量。
- **v1（對照）**：只看空隙寬度，再做**紋理驗證**，也就是比較空隙和同層商品的邊緣密度（Canny 加積分影像）。偵測器漏掉的商品還是有包裝的邊緣和文字，真正缺貨的地方通常只看得到背板。

空位分類器的模型檔（`.joblib`）不 commit；載入前會比對 `.json` 裡記錄的 SHA-256，而且只從模型資料夾載入（pickle 反序列化會執行程式碼，絕不載入使用者提供的檔案）。

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

- 以圖搜圖對訓練時沒看過的新商品，第一名準確率約 47.0%（前 5 名約 92.7%），應該回傳前 k 名讓人確認。
- 缺貨空位判斷依賴偵測結果：偵測器表現差的畫面（例如吊掛式小包裝、透視角度很大的棧板堆疊），分層和空位都會不準；表面平滑、沒有圖案的包裝也可能被誤判成空位。
- 細類只有 81 種，都是瑞典超市的商品；換到其他國家的商品需要重新訓練或登錄參考圖。
- 只偵測「正面看得到的缺貨空位」。前排有商品、後排已經空了（商品被往前推）的「庫存偏低」，正面照片看不到深度，無法偵測。

## 延伸方向

- 庫存偏低偵測：需要從上方或斜角拍攝、深度相機，或層板重量感測，才看得到後排是否還有商品。
