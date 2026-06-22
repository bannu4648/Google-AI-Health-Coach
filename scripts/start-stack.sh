#!/usr/bin/env bash
# Start the WhatsApp coach backend + ngrok tunnel (required for Meta webhooks).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
NGROK_LOG="${NGROK_LOG:-/tmp/health_coach_ngrok.log}"
UVICORN_LOG="${UVICORN_LOG:-/tmp/health_coach_uvicorn.log}"

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok not found. Install: brew install ngrok/ngrok/ngrok"
  exit 1
fi

if lsof -ti ":${PORT}" >/dev/null 2>&1; then
  echo "Backend already listening on port ${PORT}"
else
  echo "Starting FastAPI on http://localhost:${PORT}"
  nohup python3 -m uvicorn backend.health_coach.app:app \
    --host 0.0.0.0 --port "${PORT}" >"${UVICORN_LOG}" 2>&1 &
  sleep 2
fi

if ! curl -sf "http://localhost:${PORT}/docs" >/dev/null; then
  echo "Backend failed to start. See ${UVICORN_LOG}"
  exit 1
fi
echo "Backend OK (http://localhost:${PORT}/docs)"

if pgrep -f "ngrok http ${PORT}" >/dev/null 2>&1; then
  echo "ngrok already running for port ${PORT}"
else
  echo "Starting ngrok http ${PORT}"
  nohup ngrok http "${PORT}" --log=stdout >"${NGROK_LOG}" 2>&1 &
  sleep 3
fi

PUBLIC_URL=""
for _ in 1 2 3 4 5; do
  PUBLIC_URL="$(curl -sf http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | python3 -c "import sys,json; t=json.load(sys.stdin).get('tunnels',[]); print(t[0]['public_url'] if t else '')" 2>/dev/null || true)"
  [[ -n "${PUBLIC_URL}" ]] && break
  sleep 1
done

if [[ -z "${PUBLIC_URL}" ]]; then
  echo "ngrok started but public URL not ready yet. Check ${NGROK_LOG}"
  exit 1
fi

echo ""
echo "Public URL: ${PUBLIC_URL}"
echo "Webhook:    ${PUBLIC_URL}/webhook"
echo ""
echo "Set in .env (no trailing slash):"
echo "  PUBLIC_BASE_URL=${PUBLIC_URL}"
echo ""
echo "Quick checks:"
echo "  curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/docs"
echo "  curl -s -o /dev/null -w '%{http_code}' ${PUBLIC_URL}/docs"
echo ""
echo "Logs: ${UVICORN_LOG}  ${NGROK_LOG}"
