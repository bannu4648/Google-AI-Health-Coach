#!/usr/bin/env bash
# Start backend + frontend for local development.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Starting FastAPI on http://localhost:8000"
python3 -m uvicorn backend.health_coach.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting dashboard on http://localhost:5173"
(cd frontend && npm run dev) &
FRONTEND_PID=$!

trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null' EXIT
wait
