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

## 建議的調整與補充（請審核）

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

## 需要使用者同意的安裝與下載

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
