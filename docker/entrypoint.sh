#!/bin/sh
# API / worker 共用映像檔的進入點。
#   api：先 migrate、登錄模型版本、（有設定時）建立 admin，再啟動 uvicorn
#   worker：啟動 Celery worker（Linux 容器用 prefork pool）
set -e

case "$1" in
  api)
    alembic upgrade head
    python -m app.cli seed-models
    python -m app.cli bootstrap-admin
    # --proxy-headers：API 在 nginx 後面，rate limit 要用 X-Forwarded-For 的真實 client IP。
    # forwarded-allow-ips 設 *：API 不對外開放埠，只有 nginx 連得到，才能信任這個 header。
    # --log-config：uvicorn 自己的 log 也輸出成 JSON；--no-access-log：存取紀錄由 app 的
    # request_id middleware 輸出（含 request_id 與耗時），不要 uvicorn 再印一份純文字的
    exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 \
      --proxy-headers --forwarded-allow-ips '*' --no-server-header \
      --log-config /srv/uvicorn-log.json --no-access-log
    ;;
  worker)
    exec celery -A app.worker worker --loglevel=info --concurrency="${WORKER_CONCURRENCY:-2}"
    ;;
  *)
    exec "$@"
    ;;
esac
