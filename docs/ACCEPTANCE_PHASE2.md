# Phase 2 驗收標準（使用者已核准，含 9 點調整）

依據 `RetailVision_SPEC.md` 第 6、7、9、10 節 Phase 2，加上 Phase 1 實際成果帶來的調整。
等級：L1 正確性、L2 安全性（FAIL 一律修正）、L3 建議（未達成要記錄原因）。

## 規格原文

| # | 標準 | 等級 | 驗證方式 |
|---|---|---|---|
| 2.1 | 第 6 節所有端點都已實作，`/docs` 看得到，schema 與實際回應一致 | L1 | pytest + OpenAPI schema 測試 |
| 2.2 | 每個端點都有成功與失敗的測試；`vision/` 和 `app/` 整體覆蓋率 >= 80% | L1 | pytest-cov |
| 2.3 | 非圖片檔、改副檔名的假圖片、超過 10 MB 的檔案，都回 4xx 與統一錯誤格式，服務不崩潰 | L2 | pytest |
| 2.4 | 沒帶 token、token 過期、一般使用者呼叫 admin 端點，分別回 401、401、403 | L2 | pytest |
| 2.5 | 密碼用 bcrypt 或 argon2；JWT secret 從環境變數讀，repo 沒有任何真實 secret | L2 | 程式檢查 + grep |
| 2.6 | 存檔用伺服器產生的 UUID 檔名，不用使用者提供的檔名（防路徑穿越） | L2 | pytest + 程式檢查 |
| 2.7 | SQL 全部透過 ORM 或參數化查詢，沒有字串拼接 SQL | L2 | 程式檢查 |
| 2.8 | 批次：送 20 張回 202；輪詢 `/jobs/{id}` 會到 done、結果數量正確；一張壞圖不會讓整批失敗 | L1 | 整合測試 |
| 2.9 | 新品登錄後不重新訓練，就能用 `/search/similar` 找到該商品（用登錄時沒用過的同商品照片） | L1 | 整合測試 |
| 2.10 | Alembic 能從空資料庫 upgrade 到最新，也能 downgrade 回去 | L1 | 指令實測 |
| 2.11 | rate limit 生效，超過回 429 | L3 | pytest |
| 2.12 | 單品辨識 CPU p95 < 300 ms、貨架分析 < 1.5 秒（README 記錄機器規格） | L3 | 壓測腳本 |

## 調整與補充（使用者已核准）

1. **測試分兩層**（對應 2.2 與 Phase 3 的 CI 不需權重）：
   - API 測試用**假的 vision 模型**（依賴注入替換），CI 不需要 ONNX 權重也能跑完所有端點的成功與失敗案例。
   - 另有**真實模型整合測試**（`needs_weights`，只在本機跑）：2.8、2.9、3.6 用真的 ONNX 模型驗證。
2. **資料庫與佇列在測試中都用真的**：PostgreSQL 16 與 Redis 用 Docker 跑（JSONB、Alembic 必須在 PostgreSQL 上驗證，不用 SQLite 代替）。
   一般 API 測試讓 Celery 用 eager 模式（同步執行）；2.8 另外用真的 worker 跑一次。
3. **2.9 的判定**：用 Grocery Store 某一類的 3 張 train 實拍照登錄成新商品，查詢用 test 集同類照片，
   該商品必須出現在 `/search/similar` 的**前 5 名**，並記錄 top-1。
   理由：Phase 1 量到「完全沒看過的商品」top-1 只有 47.0%、top-5 92.7%；上線模型看過全部 81 類，
   所以這個測試驗證的是「不重新訓練就能登錄並搜到」的流程，不代表新商品的辨識率（README 照實寫）。
4. **切換模型與 catalog 的一致性**（2.9、3.6）：不同模型的 embedding 不在同一個空間，
   catalog 每個向量都記錄 `model_version`；`PUT /models/active` 切換分類模型時，
   先用新模型對所有參考圖重新計算 embedding，完成後才生效。參考圖要保存在 `storage/`。
5. **單品辨識的 `catalog_match`**：相似度低於門檻時回傳 `null`，門檻在 Grocery val 上選定（不用 test）。
6. **上傳檔案驗證再加一項（L2）**：除了大小、副檔名、magic bytes，還要限制**解碼後的像素數**
   （例如 40 MP），防止 decompression bomb 這種小檔案解碼後吃光記憶體的攻擊。
7. **認證細節**：密碼用 argon2；JWT access token 30 分鐘過期；admin 帳號用 CLI 腳本建立（不做註冊流程）；
   `.env.example` 只放範例值，真實 secret 不進 repo。
8. **歷史查詢分頁**：`size` 上限 100；`from` / `to` 用 ISO 8601。
9. **2.12 壓測**：用自寫的 httpx 腳本對本機 uvicorn 打請求（CPU only），README 記錄 CPU 型號、執行緒數、樣本數。
   參考 Phase 1 單機量測：單品辨識（DINOv2）約 56 ms、貨架分析約 212 ms。

## 安裝與下載（使用者已同意）

| 項目 | 來源 | 大小（估計） |
|---|---|---|
| 啟動 Docker Desktop（已安裝、目前停止） | 本機 | 無 |
| `postgres:16-alpine` 映像檔 | Docker Hub 官方 | 約 100 MB |
| `redis:7-alpine` 映像檔 | Docker Hub 官方 | 約 15 MB |
| fastapi、uvicorn、pydantic-settings、sqlalchemy、alembic、psycopg、celery、redis、argon2-cffi、pyjwt、python-multipart、slowapi | PyPI，裝在 `.venv`（不含 torch 依賴，不會動到 CUDA torch） | 約 40 MB |

## 實作順序（規格第 10 節）

1. FastAPI 骨架、設定、統一錯誤處理、request_id middleware
2. 資料庫 models 與 Alembic
3. 認證（JWT、角色）
4. 辨識端點（product / shelf / annotated）
5. 歷史查詢、統計
6. Catalog 與以圖搜圖
7. Celery 批次任務
8. 模型版本管理
9. 檔案驗證、rate limit
10. 測試補齊到覆蓋率門檻；機械檢查通過後送 Codex 審查

## Phase 2 狀態（機械檢查與實測結果）

| # | 結果 | 驗證 | 狀態 |
|---|---|---|---|
| 2.1 | 規格第 6 節 16 個端點都在 OpenAPI，需登入的都標示 Bearer；回應用 response_model 驗證 | `tests/api/test_misc.py`、各端點測試用 schema 驗證回應 | 通過 |
| 2.2 | 全部測試 108 個（API 55 個用真 PostgreSQL / Redis + 假模型，另有真實模型整合 3 個）；`vision/` + `app/` 覆蓋率 96% | pytest-cov | 通過 |
| 2.3 | 文字檔改 .jpg、PDF 改 .png、截斷 JPEG、不支援副檔名、超過 10 MB、壞檔後服務仍正常 | `tests/api/test_uploads.py` | 通過 |
| 2.4 | 沒 token 401、過期 401、錯誤簽章 / alg=none 401、一般使用者打 admin 端點 403；角色以資料庫為準 | `tests/api/test_auth.py` | 通過 |
| 2.5 | argon2id；JWT_SECRET 必填、至少 32 字元；git 追蹤的檔案沒有真實 secret | `test_auth.py`、`test_misc.py` | 通過 |
| 2.6 | 存檔用伺服器產生的 UUID 檔名（含路徑穿越檔名的測試）；讀檔時再檢查路徑不出 storage | `test_uploads.py` | 通過 |
| 2.7 | 全部 ORM 或參數化；AST 靜態檢查沒有字串組 SQL | `test_misc.py` | 通過 |
| 2.8 | 20 張（含 1 張壞圖）回 202，job 到 done、19 筆紀錄、壞圖記在 item_errors。eager 測試與**真的 Celery worker**（Windows solo pool）都跑過 | `test_batch_catalog_models.py`、`tools/smoke_batch.py`（約 6.5 秒完成） | 通過 |
| 2.9 | 真實模型：10 類各用 3 張 train 實拍登錄，test 照片查詢，top-1 10/10、top-5 10/10 | `tests/api/test_real_models.py` | 通過 |
| 2.10 | 從空資料庫 upgrade 到最新、downgrade 回空、再 upgrade | `test_misc.py`、指令實測 | 通過 |
| 2.11 | 超過限制回 429 + Retry-After；壓測時實際伺服器也觸發 | `test_auth.py` | 通過 |
| 2.12 | 單品 p95 104.7 ms、貨架 p95 454.7 ms（AMD Ryzen 7 4800H、16 邏輯 CPU、ONNX Runtime CPU、client 端端到端） | `tools/bench_latency.py` | 通過 |
| 調整 5 | catalog_match 門檻 0.675（Grocery val：正確率 91.5%） | `training/tune_catalog_threshold.py` | 完成 |
| 調整 6 | decompression bomb：Pillow 內建防護與自訂 4000 萬像素上限都有測試 | `test_uploads.py` | 通過 |

### 和規格的差異（實作時的決定）

- `recognitions` 加 `job_id`；`jobs` 加 `type`、`failed_count`、`item_errors`；`catalog_embeddings` 加 `model_version`。
- 一般使用者只看得到自己的辨識紀錄、批次任務與統計；別人的紀錄回 404（不是 403），避免用猜 ID 確認紀錄存在。
- `DELETE /catalog/products/{id}` 限 admin（會影響所有使用者）。
- `PUT /models/active` 目前只切換分類模型（偵測模型只有一個版本）。
- `/healthz` 同時提供 `/api/v1/healthz` 與根路徑 `/healthz`（給 docker healthcheck）。
