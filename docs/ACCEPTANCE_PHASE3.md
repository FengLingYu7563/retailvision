# Phase 3 驗收標準（使用者已核准，含 9 點調整）

依據 `RetailVision_SPEC.md` 第 9 節 Phase 3，加上 Phase 1、2 實際狀況帶來的調整。
等級：L1 正確性、L2 安全性（FAIL 一律修正）、L3 建議（未達成要記錄原因）。

## 規格原文

| # | 標準 | 等級 | 驗證方式 |
|---|---|---|---|
| 3.1 | 乾淨的環境只要 `cp .env.example .env` 加 `docker compose up`，就能啟動全部服務，`/healthz` 回 200 | L1 | 實測 |
| 3.2 | API 映像檔不含 PyTorch，大小小於 1.5 GB | L3 | docker images |
| 3.3 | GitHub Actions：ruff、mypy、pytest 全部通過；CI 不需要 GPU、不下載完整資料集（用小型 fixture 圖片） | L1 | CI 綠燈 |
| 3.4 | 模擬相機：把圖片丟進監看資料夾，5 秒內出現在辨識紀錄裡 | L1 | 實測 |
| 3.5 | Streamlit 儀表板：上傳辨識、歷史列表、類別分布圖、每日趨勢、缺貨率 | L1 | 實測截圖 |
| 3.6 | 切換使用中的模型版本後，新的請求使用新版本（model_version 改變），不需重啟服務 | L1 | 整合測試 |
| 3.7 | README：架構圖、快速開始、curl 範例、模型比較表、資料集出處與授權、已知限制、Demo GIF 或截圖 | L1 | 人工檢查 |
| 3.8 | 結構化日誌（JSON），每個請求都有 request_id，錯誤回應裡也看得到 | L3 | 實測 |

## 調整與補充（使用者已核准）

1. **3.1 的權重檔**：權重不進 git（規格要求），所以「乾淨環境」指的是乾淨的 clone **加上** `models/` 底下放好權重檔
   （訓練產出，或從其他地方複製進來）。README 會寫清楚需要哪些檔案、怎麼重新產生。
   **不建議公開散布權重**：模型是用研究用途授權的資料集（SKU-110K、Grocery Store）訓練的，公開權重可能違反資料集條款。
   另外驗證：沒放權重時服務仍能啟動，`/healthz` 回 503 並指出缺哪個模型（Phase 2 已實作）。
2. **啟動自動化**：API 容器啟動時自動 `alembic upgrade head` 與 `seed-models`；
   `.env` 有設定 `ADMIN_USERNAME` / `ADMIN_PASSWORD` 時自動建立 admin（已存在就略過），
   讓 `cp .env.example .env` + `docker compose up` 真的能直接用。
3. **API 與 worker 映像檔**：多階段建置，只裝主要依賴（不含 torch，Phase 2 已有測試把關）；worker 用同一個映像檔、
   不同的啟動指令，在 Linux 容器裡用 prefork pool（不再用 Windows 的 solo pool）。
4. **3.3 CI**：GitHub Actions 用 PostgreSQL / Redis service container 跑完整 API 測試；需要權重或資料集的測試自動 skip
   （已有 marker）。**需要建立 GitHub repo 並 push**（見下方，需要你同意）。
5. **3.4 模擬相機**：輪詢資料夾（不加 watchdog 依賴），新檔案寫完才送出（檔案大小連續兩次相同），送出後移到
   `processed/` 或 `failed/`；帳密從環境變數讀。以 docker compose 中的一個服務或本機指令執行皆可。
6. **3.5 儀表板**：Streamlit 只呼叫 API（不直接連資料庫），用帳號密碼登入取得 JWT；一般使用者只看到自己的資料。
7. **3.6**：Phase 2 已有整合測試；Phase 3 另外在 docker compose 環境實測 API 與 worker 兩個 process 都跟著切換。
8. **反向代理**（新增，L2 第二道防線）：加一個 nginx 服務在 API 前面，設定 `client_max_body_size`（Phase 2 Codex 建議），
   對外只開 nginx 的埠；API、資料庫、Redis 不直接對外。
9. **3.7 Demo**：用截圖（儀表板各頁、`/docs`、標註圖），GIF 列為選配。

## 使用者同意的事項（含建立公開 GitHub repo 並 push）

| 項目 | 來源 | 大小 / 影響 |
|---|---|---|
| `python:3.12-slim` 映像檔 | Docker Hub 官方 | 約 50 MB |
| `nginx:alpine` 映像檔（若同意第 8 點） | Docker Hub 官方 | 約 20 MB |
| 映像檔建置時從 PyPI 安裝主要依賴 | PyPI | 映像內約 400 到 600 MB |
| `streamlit`（裝在 `.venv` 與 dashboard 映像檔） | PyPI | 約 100 MB（含 pyarrow、altair） |
| **建立 GitHub repo 並 push**（CI 需要） | GitHub | 對外動作：repo 名稱、公開或私有要你決定 |

## 實作順序（規格第 10 節）

1. Dockerfile（多階段）、docker-compose、啟動自動化、nginx
2. GitHub Actions
3. 模擬相機程式
4. Streamlit 儀表板
5. 延遲壓測（在 docker 環境重跑一次）
6. README、架構圖、Demo 截圖
7. 機械檢查通過後送 Codex 審查

## 驗收狀態

| # | 狀態 | 證據 |
|---|---|---|
| 3.1 | 通過 | `cp .env.example .env`、填入三個密鑰後 `docker compose up -d --build`：API 啟動時自動 migrate、登錄模型、建立 admin；經 nginx 的 `/healthz` 回 200。只有 nginx 對 host 開埠 |
| 3.2 | 通過 | `retailvision/api` 822 MB；`pip list` 不含 torch、torchvision、ultralytics、timm（CI 的 docker job 也會檢查） |
| 3.3 | 通過 | GitHub Actions run 39097838674：lint、test（PostgreSQL / Redis service container，125 passed、12 skipped 為需要權重或資料集的測試，覆蓋率 95.5%）、docker（不含 torch、< 1.5 GB）全部綠燈 |
| 3.4 | 通過 | `docker compose --profile camera up -d camera`，丟 5 張圖，最慢 2.55 秒出現在 `/recognitions`，檔案全部移到 `processed/` |
| 3.5 | 通過 | 儀表板四頁實測：上傳辨識（Granny-Smith 77.8%、粗類 Apple 91.0%、129 ms）、歷史列表與標註圖、類別分布、每日趨勢、缺貨率；截圖在 `docs/screenshots/` |
| 3.6 | 通過 | Phase 2 整合測試；docker 環境另跑 `tools/check_model_switch.py`：API 與 worker 都從 cls-dinov2-v2 切到 cls-mnv3-v1 再切回，沒有重啟容器 |
| 3.7 | 通過 | README：架構圖（mermaid）、快速開始、需要的模型檔、curl 範例、模型比較表、資料集出處與授權、已知限制、截圖 |
| 3.8 | 通過 | API 與 uvicorn 日誌都是 JSON；錯誤回應的 `request_id` 與日誌一致，request_id 由 nginx 產生 |

補充：docker 環境的 API 延遲（經 nginx）單品 p95 139.2 ms、貨架 p95 504.7 ms，仍在 2.12 的門檻內
（`models/metrics.json` 的 `api_latency_docker`）；docker 環境的批次辨識（2.8）由 prefork worker 處理，約 4.8 秒完成。

## Codex 審查補充的檢查項目（第 1 輪 criteria_gaps）

| # | 項目 | 等級 | 狀態 | 證據 |
|---|---|---|---|---|
| 3.9 | 正式部署（`docker compose up`）中 db、redis 不對 host 開埠；本機開發另外疊加 `docker-compose.dev.yml` | L2 | 通過 | `docker compose ps` 只有 nginx 有埠；127.0.0.1:5432 / 6379 連不上；backend 是 internal 網路 |
| 3.10 | `.env.example` 不含可用的密鑰或 admin 密碼；範例值會被拒絕 | L2 | 通過 | 三個密鑰留空，compose 報「請在 .env 設定 JWT_SECRET」；`Settings` 拒絕 `replace-with...` / `change-me...`；`create_user` 拒絕範例密碼（測試） |
| 3.11 | X-Forwarded-For 只信任 nginx；非 nginx 來源偽造的 header 繞不過 IP 限流 | L2 | 通過 | uvicorn `--forwarded-allow-ips 172.30.57.10`（nginx 固定 IP）；儀表板 / 相機只在 frontend 網路、連不到 api；經 nginx 與從內部直連 api、每次換 XFF，都在第 61 次回 429 |
| 3.12 | 沒放權重時所有服務仍會啟動，`/healthz` 回 503（liveness 與 readiness 分開） | L1 | 通過 | healthcheck 改用 `/livez`；空的模型資料夾實測：所有容器起來、api healthy、經 nginx 的 `/healthz` 回 503 並列出缺少的模型 |

3.1 的步驟因此多了「填入三個密鑰」：照抄範例檔就能啟動，等於所有部署共用公開的 JWT 金鑰與 admin 密碼（Codex L2）。

## Codex 審查補充的檢查項目（第 2 輪 criteria_gaps）

| # | 項目 | 等級 | 狀態 | 證據 |
|---|---|---|---|---|
| 3.13 | 模擬相機只處理監看資料夾直接底下的一般圖片檔；符號連結、特殊檔案、跑出資料夾的路徑、超過大小上限（和 API 一致，10 MB）的檔案一律拒絕，且不先讀進記憶體（第 3 輪補充） | L2 | 通過 | `tools/camera_sim.py` 的 `scan()`（lstat）與 `read_regular_file()`（O_NOFOLLOW + fstat）；Linux 容器實測：符號連結與 FIFO 移到 `failed/`、O_NOFOLLOW 擋下競爭、資料夾外的檔案沒被讀取；`tests/unit/test_camera_sim.py`（CI 在 Linux 跑符號連結測試）；12 MB 檔案在讀取前被拒絕，相機容器記憶體 17.5 MiB；列出後被刪掉的檔案略過、不中止迴圈（L3） |
| 3.14 | 自動化服務（模擬相機）用最小權限帳號，不 fallback 到 admin | L2 | 通過 | compose 移除 ADMIN_* fallback；沒設 CAMERA_* 時 camera_sim 直接結束並提示建一般使用者；實測以 role=user 的 camera 帳號送辨識成功 |
