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
    # 只信任 nginx 的固定 IP（FORWARDED_ALLOW_IPS，docker-compose.yml 設定）；
    # 其他來源帶的 X-Forwarded-For 一律忽略。沒設定時用 uvicorn 的預設（只信任 127.0.0.1）。
    # --log-config：uvicorn 自己的 log 也輸出成 JSON；--no-access-log：存取紀錄由 app 的
    # request_id middleware 輸出（含 request_id 與耗時），不要 uvicorn 再印一份純文字的
    exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 \
      --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}" --no-server-header \
      --log-config /srv/uvicorn-log.json --no-access-log
    ;;
  worker)
    exec celery -A app.worker worker --loglevel=info --concurrency="${WORKER_CONCURRENCY:-2}"
    ;;
  *)
    exec "$@"
    ;;
esac
