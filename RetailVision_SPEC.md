# RetailVision：零售商品影像辨識服務 專案規格

> 版本：v1.0（2026-10-13）
> 用途：作品集專案。本文件包含需求、架構、API 規格、驗收標準，可以直接交給 AI agent 依照 workflow 實作。

## 0. 給執行 agent 的說明

- 這是**大任務**。本文件的驗收標準（第 9 節）就是要交給使用者審核的版本。開工前請使用者再確認一次，之後才能動手。
- 依照「拆解 → 產出 → 驗證 → 修改」的迴圈進行。驗證只認真實的工具結果（pytest、實際打 API、實際訓練出來的指標）。
- 停止條件：同一個問題修改 5 次仍未通過、錯誤和上次相同、或超過成本上限。停下時要回報卡在哪裡，不可以假裝已完成。
- 每個 Phase 結束時，機械檢查（lint、測試）要先通過，再交給 Codex 做判斷型審查（L1 正確性、L2 安全性 FAIL 必須修正，L3 僅供參考）。
- **以下動作一律先問使用者**：下載資料集（列出名稱、來源、大小）、安裝 CUDA 或 PyTorch 這類大型套件、建立 GitHub repo、push。
- 資料集和模型權重**不可以 commit 進 repo**（寫進 `.gitignore`）。只 commit 下載腳本和 `metrics.json`。
- 所有文字輸出都用繁體中文，不使用 emoji。commit 訊息不加 AI 共同作者署名。
- 程式碼的關鍵決策處要寫註解說明「為什麼這樣做」，例如為什麼用 CLAHE、為什麼轉 ONNX。

## 1. 資料集調查結果

| 題目 | 資料集 | 規模 | 標註類型 | 授權 | 適合度 |
|---|---|---|---|---|---|
| **零售（單品分類）** | Grocery Store Dataset (Klasson et al., WACV 2019) | 5,125 張，81 細類 / 42 粗類，附商品標準圖與文字描述 | 分類（階層式標籤） | 未標準化（Other），研究用途 | 高。小、好訓練、有階層標籤，還有標準圖可做以圖搜圖 |
| **零售（貨架偵測）** | SKU-110K (Goldman et al., CVPR 2019) | 11,762 張，170 萬個框 | 偵測（只有一類「商品」） | Other，研究用途 | 高。Ultralytics 有內建設定檔。檔案大，可以只用子集 |
| 零售（結帳台） | RPC (Wei et al., 2019) | 83,739 張，200 種 SKU | 偵測加分類 | Other，研究用途 | 中。最貼近自助結帳情境，但資料量大，4 GB VRAM 訓練吃力。列為選配 |
| 零售（分類備案） | Freiburg Groceries | 5,000 張，25 類 | 分類 | Other | 中。Grocery Store 有問題時才用 |
| 車牌 | CCPD | 25 萬張以上 | 偵測加字元 | MIT | 中。是中國車牌，有隱私疑慮 |
| 安全帽 | Hard Hat Workers (Roboflow) | 約 5,000 張 | 偵測 | CC BY 4.0 / Public Domain（依版本而異） | 中 |
| 口罩 | Face Mask Detection (Kaggle, andrewmvd) | 853 張 | 偵測 | CC0 | 低。資料太少，題目也過時 |
| 鋼材瑕疵 | NEU-DET | 1,800 張，6 類 | 分類加偵測 | 研究用途 | 中。適合 AOI 職缺 |

**結論**：零售題目有足夠的資料集，**採用 Grocery Store（分類 + 以圖搜圖）加 SKU-110K（貨架偵測）**。這兩個都不是商用授權，作品集屬於非商用展示，可以使用。但要在 README 註明出處與引用，而且不能重新散布原始資料。

## 2. 專案概述

**情境**：零售業者（超市、便利商店）想用影像做三件事：
1. **單品辨識**：拍一個商品，告訴我它是什麼（可用於自助結帳、庫存登錄）。
2. **貨架分析**：拍一整排貨架，算出有幾個商品、哪裡有缺貨空位、陳列密度多少。
3. **新品登錄與以圖搜圖**：新商品上架時，**不需要重新訓練**，只要上傳幾張參考圖就能辨識。

上面三件事全部透過 RESTful API 提供，結果存進資料庫，並用儀表板呈現統計。

**影像來源**：用公開資料集的照片代替相機。另外做一個「模擬相機」程式，監看資料夾、有新圖就自動送出辨識，用來展示即時流程。

## 3. 技術棧與技能對照

| 職缺關鍵字 | 在本專案的實作 |
|---|---|
| OpenCV 影像處理 | 影像品質檢查（Laplacian 模糊偵測、亮度與曝光檢查）、CLAHE、去雜訊、透視校正（貨架拉正）、色彩直方圖、ORB 特徵比對、缺貨空位偵測（偵測框之間的空隙分析）、標註繪圖 |
| 傳統機器學習 | Baseline：HSV 色彩直方圖加 HOG 特徵，用 SVM 分類（scikit-learn），與 CNN 比較 |
| 影像辨識（分類） | PyTorch 遷移學習：MobileNetV3-Large 或 ResNet50，做細類加粗類的雙頭輸出（階層標籤） |
| 物件偵測 | Ultralytics YOLOv8n/s，用 SKU-110K 訓練 |
| 度量學習與以圖搜圖 | CNN 倒數第二層的 embedding，用 cosine 相似度做最近鄰搜尋（先用 numpy，資料量大時再換 FAISS） |
| 模型部署 | 匯出 ONNX，用 ONNX Runtime 推論（API 映像檔不含 PyTorch） |
| RESTful API | FastAPI、Pydantic v2、OpenAPI 文件、版本化路由、分頁、統一錯誤格式 |
| 資料庫 | PostgreSQL 16、SQLAlchemy 2.0、Alembic 資料庫遷移 |
| 非同步任務 | Celery 搭配 Redis 處理批次辨識 |
| 驗證與安全 | JWT（access token）、密碼雜湊、上傳檔案驗證（大小、副檔名、magic bytes 實際格式）、rate limit |
| 測試與 CI | pytest、httpx TestClient、ruff、mypy（vision 模組）、GitHub Actions |
| 容器化 | Docker 多階段建置、docker-compose（api、worker、db、redis、dashboard） |
| 實驗管理 | 固定 random seed、資料切分檔、`metrics.json`、混淆矩陣圖。可選用 MLflow |
| 前端 | Streamlit 儀表板 |

**Python 版本**：3.11 或 3.12。**PyTorch** 安裝 CUDA 12.x 版本（只用在訓練環境）。

## 4. 系統架構

```
                        +-------------------+
 Client / 模擬相機 ---->|  FastAPI (api)    |---- JWT 驗證、檔案驗證
 Streamlit 儀表板 ----->|  /api/v1/...      |
                        +---------+---------+
                                  |
                 同步（單張）     |     非同步（批次）
                +-----------------+------------------+
                v                                    v
        +---------------+                    +---------------+
        | vision 套件    |<-------------------| Celery worker |<-- Redis（broker）
        +---------------+                    +---------------+
        | quality.py    | 模糊、曝光檢查
        | preprocess.py | CLAHE、resize、透視校正
        | classical.py  | HSV 直方圖 + HOG + SVM（baseline）
        | classifier.py | ONNX 分類（細類 + 粗類）
        | embedder.py   | ONNX embedding + 最近鄰搜尋
        | detector.py   | ONNX YOLO 貨架偵測
        | shelf.py      | 數量、空位、密度分析（OpenCV）
        | draw.py       | 畫框與標籤
        +-------+-------+
                |
                v
        +---------------+       +----------------+
        | PostgreSQL    |       | 檔案儲存       |
        | 紀錄、統計    |       | storage/ 目錄   |
        +---------------+       +----------------+
```

**設計原則**：
- `vision/` 是純函式模組，不依賴 FastAPI，可以單獨測試和重用。
- 模型在啟動時載入一次（FastAPI lifespan），不要每個請求都重新載入。
- 推論只用 ONNX Runtime。PyTorch 只出現在 `training/` 目錄。

## 5. 硬體限制與訓練設定（RTX 3050 Laptop 4 GB）

| 任務 | 模型 | 輸入大小 | Batch | 其他 |
|---|---|---|---|---|
| 分類 | MobileNetV3-Large（優先）/ ResNet50 | 224 | 32 / 16 | AMP 混合精度、凍結前段 → 解凍微調 |
| 偵測 | YOLOv8n（優先）/ YOLOv8s | 640 | 8 / 4 | AMP、`workers=2`、OOM 時降 batch |
| Embedding | 共用分類模型的 backbone | 224 | — | 只做推論 |

- SKU-110K 檔案大。先用約 3,000 張訓練圖的子集跑通流程，指標達標後再決定要不要用完整資料。
- 筆電長時間訓練要注意散熱。每個 epoch 都要存 checkpoint，中斷後可以接續訓練。
- 推論延遲**以 CPU 為準**來測量（展示環境不一定有 GPU）。GPU 數字另外列出。

## 6. RESTful API 規格

基底路徑 `/api/v1`。除了 `/healthz` 和 `/auth/token`，其他端點都需要 `Authorization: Bearer <token>`。

| 方法 | 路徑 | 說明 | 成功回應 |
|---|---|---|---|
| POST | `/auth/token` | 帳號密碼換 JWT | 200 |
| POST | `/recognitions/product` | 單張商品辨識（multipart 上傳），同步回傳 | 201 |
| POST | `/recognitions/shelf` | 單張貨架分析，同步回傳 | 201 |
| POST | `/recognitions/batch` | 多張批次辨識 | 202 + `job_id` |
| GET | `/jobs/{job_id}` | 查詢批次任務狀態與結果 | 200 |
| GET | `/recognitions` | 歷史查詢：`type`、`label`、`from`、`to`、`page`、`size` | 200 |
| GET | `/recognitions/{id}` | 單筆結果 | 200 / 404 |
| GET | `/recognitions/{id}/annotated` | 已標註的圖片（image/jpeg） | 200 |
| POST | `/catalog/products` | 新品登錄：名稱加 1 到 5 張參考圖，產生 embedding | 201 |
| GET | `/catalog/products` | 列出已登錄的商品 | 200 |
| DELETE | `/catalog/products/{id}` | 刪除商品 | 204 |
| POST | `/search/similar` | 以圖搜圖，回傳前 k 個相似商品 | 200 |
| GET | `/models` | 模型版本與指標 | 200 |
| PUT | `/models/active` | 切換使用中的模型版本（限 admin） | 200 |
| GET | `/stats/summary` | 各類別數量、平均信心度、缺貨率、每日趨勢 | 200 |
| GET | `/healthz` | 健康檢查（含 DB、Redis、模型載入狀態） | 200 / 503 |

**單品辨識回應範例**：
```json
{
  "id": "6f1c...",
  "type": "product",
  "quality": {"blur_score": 182.4, "brightness": 0.53, "passed": true, "warnings": []},
  "prediction": {
    "fine":   {"label": "Granny-Smith", "confidence": 0.91},
    "coarse": {"label": "Apple",        "confidence": 0.98},
    "top_k":  [{"label": "Granny-Smith", "confidence": 0.91}, {"label": "Golden-Delicious", "confidence": 0.05}]
  },
  "catalog_match": {"product_id": 12, "name": "青蘋果", "similarity": 0.87},
  "model_version": "cls-mnv3-v1",
  "latency_ms": 38,
  "created_at": "2026-10-13T10:00:00Z"
}
```

**貨架分析回應重點欄位**：`item_count`、`boxes[]`（x1、y1、x2、y2、confidence）、`gaps[]`（缺貨空位框）、`gap_ratio`、`density`。

**統一錯誤格式**（所有 4xx、5xx 都用這個格式）：
```json
{"error": {"code": "INVALID_IMAGE", "message": "檔案不是有效的圖片", "request_id": "..."}}
```
錯誤碼至少包含：`INVALID_IMAGE`、`FILE_TOO_LARGE`（上限 10 MB）、`UNSUPPORTED_FORMAT`、`LOW_QUALITY_IMAGE`（只是警告，不會擋下請求，除非帶 `strict=true`）、`UNAUTHORIZED`、`FORBIDDEN`、`NOT_FOUND`、`RATE_LIMITED`、`MODEL_NOT_READY`。

## 7. 資料庫設計

- `users`：id、username、password_hash、role（admin / user）、created_at
- `recognitions`：id (UUID)、type（product / shelf）、user_id、image_path、annotated_path、result (JSONB)、top_label、confidence、model_version、latency_ms、quality_passed、created_at；`(type, created_at)` 和 `top_label` 要建索引
- `jobs`：id、status（pending / running / done / failed）、total、done_count、error、created_at、finished_at
- `catalog_products`：id、name、sku、created_at
- `catalog_embeddings`：id、product_id (FK)、vector（float32 bytes，或之後改用 pgvector）、image_path
- `model_versions`：id、name、task、onnx_path、metrics (JSONB)、is_active、created_at

## 8. 目錄結構

```
retailvision/
├── app/
│   ├── main.py              # FastAPI app、lifespan 載入模型
│   ├── api/v1/              # auth, recognitions, jobs, catalog, search, models, stats, health
│   ├── core/                # config (pydantic-settings), security, logging, errors
│   ├── db/                  # SQLAlchemy models, session
│   ├── schemas/             # Pydantic
│   ├── services/            # 業務邏輯（API 和 worker 共用）
│   └── worker.py            # Celery tasks
├── vision/                  # 純影像與推論模組，不依賴 FastAPI
├── training/
│   ├── data/                # 下載與切分腳本（資料本身不 commit）
│   ├── train_classical.py
│   ├── train_classifier.py
│   ├── train_detector.py
│   ├── export_onnx.py
│   └── evaluate.py          # 產出 metrics.json 與混淆矩陣
├── models/                  # .onnx（gitignore）+ metrics.json（commit）
├── tools/camera_sim.py      # 模擬相機：監看資料夾並自動上傳
├── dashboard/app.py         # Streamlit
├── migrations/              # Alembic
├── tests/
│   ├── unit/                # vision 模組
│   └── api/                 # 端點測試
├── docker/                  # Dockerfile.api, Dockerfile.worker
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml
└── README.md
```

## 9. 驗收標準

等級說明：**L1** 正確性、**L2** 安全性（這兩級 FAIL 一律打回修正）、**L3** 建議（未達成要記錄原因）。
指標門檻是預估值。若訓練後確定達不到，可以調整，但必須在 README 說明原因並附上實驗紀錄，不可以默默降低標準。

### Phase 1：影像處理與模型（離線）

| # | 標準 | 等級 | 驗證方式 |
|---|---|---|---|
| 1.1 | 資料下載與切分腳本可重現：切分寫成檔案、固定 seed，重跑結果完全相同 | L1 | 跑兩次後 diff 切分檔 |
| 1.2 | `vision/quality.py` 能分辨模糊與清晰的圖：人工做的模糊圖（GaussianBlur）被判為未通過，原圖通過 | L1 | pytest |
| 1.3 | SVM baseline 完成，並在測試集上回報細類與粗類的 top-1 準確率 | L1 | metrics.json |
| 1.4 | CNN 分類：粗類 top-1 至少 85%，細類 top-1 至少 75%、top-5 至少 95%，而且每一項都比 SVM 高 | L1 | evaluate.py 輸出 |
| 1.5 | YOLO 貨架偵測：SKU-110K 測試集（或固定的測試子集）mAP50 至少 0.75 | L1 | evaluate.py 輸出 |
| 1.6 | ONNX 匯出後與 PyTorch 輸出比對：分類 logits 最大誤差小於 1e-3，top-1 一致率 100% | L1 | pytest |
| 1.7 | 以圖搜圖：用 Grocery Store 標準圖建 gallery，測試圖的 top-1 檢索準確率至少 70% | L1 | evaluate.py |
| 1.8 | 缺貨空位偵測：準備至少 10 張人工標註空位的貨架圖，空位召回率至少 70% | L3 | evaluate.py |
| 1.9 | 產出混淆矩陣圖，並在 README 寫出最常混淆的 3 組類別和可能原因 | L3 | 人工檢查 |

### Phase 2：RESTful API 與資料庫

| # | 標準 | 等級 | 驗證方式 |
|---|---|---|---|
| 2.1 | 第 6 節列出的所有端點都已實作，`/docs` 可以看到，schema 與實際回應一致 | L1 | pytest 加 OpenAPI schema 測試 |
| 2.2 | 每個端點都有成功與失敗的測試；整體測試覆蓋率至少 80%（`vision/` 和 `app/`） | L1 | pytest-cov |
| 2.3 | 上傳非圖片檔、改副檔名的假圖片、超過 10 MB 的檔案，都回 4xx 和統一錯誤格式，服務不會崩潰 | L2 | pytest |
| 2.4 | 沒帶 token、token 過期、一般使用者呼叫 admin 端點，分別回 401、401、403 | L2 | pytest |
| 2.5 | 密碼用 bcrypt 或 argon2 雜湊；JWT secret 從環境變數讀取，repo 裡沒有任何真實的 secret | L2 | 程式檢查加 grep |
| 2.6 | 檔案存檔時用伺服器產生的 UUID 檔名，不使用使用者提供的檔名（防止路徑穿越） | L2 | pytest 加程式檢查 |
| 2.7 | SQL 全部透過 ORM 或參數化查詢，沒有字串拼接 SQL | L2 | 程式檢查 |
| 2.8 | 批次任務：送出 20 張圖，回 202；之後輪詢 `/jobs/{id}` 會到 done，結果數量正確；其中一張壞圖不會讓整批失敗 | L1 | 整合測試 |
| 2.9 | 新品登錄後，不重新訓練就能用 `/search/similar` 找到該商品（用登錄時沒用過的同商品照片測試） | L1 | 整合測試 |
| 2.10 | Alembic 能從空資料庫 upgrade 到最新版，也能 downgrade 回去 | L1 | 指令實測 |
| 2.11 | rate limit 生效：超過限制回 429 | L3 | pytest |
| 2.12 | 單品辨識 API 在 CPU 上的 p95 延遲小於 300 ms，貨架分析小於 1.5 秒（README 記錄機器規格） | L3 | 壓測腳本（locust 或自寫） |

### Phase 3：部署、CI、展示

| # | 標準 | 等級 | 驗證方式 |
|---|---|---|---|
| 3.1 | 乾淨的環境只要 `cp .env.example .env` 加 `docker compose up`，就能啟動全部服務，`/healthz` 回 200 | L1 | 實測 |
| 3.2 | API 映像檔不含 PyTorch，大小小於 1.5 GB | L3 | docker images |
| 3.3 | GitHub Actions：ruff、mypy（vision）、pytest 全部通過；CI 不需要 GPU，也不用下載完整資料集（用小型 fixture 圖片） | L1 | CI 綠燈 |
| 3.4 | 模擬相機程式：把圖片丟進監看資料夾，5 秒內出現在辨識紀錄裡 | L1 | 實測 |
| 3.5 | Streamlit 儀表板能顯示：上傳辨識、歷史列表、類別分布圖、每日趨勢、缺貨率 | L1 | 實測截圖 |
| 3.6 | 切換使用中的模型版本後，新的請求會使用新版本（回應中的 model_version 改變），且不需要重啟服務 | L1 | 整合測試 |
| 3.7 | README 包含：架構圖、快速開始、curl 範例、模型比較表（SVM / CNN、各項指標、延遲）、資料集出處與授權聲明、已知限制、Demo GIF 或截圖 | L1 | 人工檢查 |
| 3.8 | 結構化日誌（JSON），每個請求都有 request_id，錯誤回應裡也看得到 | L3 | 實測 |

## 10. 實作階段與任務拆解

**Phase 1：影像處理與模型**
1. 建立專案骨架、pyproject、ruff/mypy/pytest 設定、`.gitignore`
2. 資料下載腳本（先問使用者）與切分腳本
3. `vision/quality.py`、`preprocess.py`，加上單元測試
4. `classical.py` 的 SVM baseline
5. `train_classifier.py`：MobileNetV3 雙頭（細類加粗類）
6. `train_detector.py`：YOLOv8n，SKU-110K 子集
7. `export_onnx.py` 和一致性測試
8. `embedder.py` 和檢索評估
9. `shelf.py`：數量、空位、密度
10. `evaluate.py`：產出 metrics.json 和混淆矩陣

**Phase 2：API 與資料庫**
1. FastAPI 骨架、設定、統一錯誤處理、request_id middleware
2. 資料庫 models 和 Alembic
3. 認證（JWT、角色）
4. 辨識端點（product / shelf / annotated）
5. 歷史查詢、統計
6. Catalog 與以圖搜圖
7. Celery 批次任務
8. 模型版本管理
9. 檔案驗證、rate limit
10. API 測試補齊到覆蓋率門檻

**Phase 3：部署與展示**
1. Dockerfile（多階段）、docker-compose
2. GitHub Actions
3. 模擬相機程式
4. Streamlit 儀表板
5. 延遲壓測
6. README、架構圖、Demo 截圖

## 11. 不做的事（Non-goals）

- 不接真實相機或 PLC 硬體
- 不做前端帳號註冊流程（admin 用 CLI 或 seed 建立帳號）
- 不做雲端部署（可在 README 寫「延伸方向」）
- 不做 Kubernetes、多 agent、向量資料庫服務（資料量小，用不到）
- 不做模型線上再訓練

## 12. 風險與對策

| 風險 | 對策 |
|---|---|
| 4 GB VRAM OOM | 降 batch、開 AMP、改用 YOLOv8n、縮小 imgsz |
| SKU-110K 下載太大、訓練太久 | 先用子集，固定子集清單檔讓結果可重現 |
| 細類準確率不達標（81 類，有些很相似） | 加強資料擴增、解凍更多層、改用 ResNet50；仍不達標就照第 9 節的規則調整並說明 |
| 資料集授權 | 只做非商用展示，README 註明引用，不重新散布原始資料 |
| Windows 上跑 Celery 不穩定 | 開發和測試一律在 docker-compose 裡跑 worker |

## 13. 參考資料

- Grocery Store Dataset：https://github.com/marcusklasson/GroceryStoreDataset
- SKU-110K：https://github.com/eg4000/SKU110K_CVPR19 、Ultralytics 說明 https://docs.ultralytics.com/datasets/detect/sku-110k
- RPC：https://rpc-dataset.github.io
- CCPD：https://github.com/detectRecog/CCPD
