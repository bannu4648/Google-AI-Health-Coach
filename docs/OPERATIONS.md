# Operations

## Run The Backend

Primary packaged entrypoint:

```bash
uvicorn backend.health_coach.app:app --host 0.0.0.0 --port 8000 --reload
```

Compatibility entrypoint:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Run The Dashboard

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Scheduler

Set `ENABLE_SCHEDULER=true` in `.env` to start morning and evening jobs with FastAPI.

Defaults:

```text
MORNING_SUMMARY_TIME=08:00
EVENING_SUMMARY_TIME=21:30
```

## Local Data

SQLite lives at `data/health_coach.sqlite3` by default. Keep `data/`, `.env`, `credentials.json`, and `token.json` local-only.
