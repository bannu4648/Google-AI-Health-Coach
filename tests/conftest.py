from __future__ import annotations

import os
import tempfile
from pathlib import Path


_TEST_DIR = Path(tempfile.mkdtemp(prefix="health-coach-tests-"))
os.environ["HEALTH_COACH_DB_PATH"] = str(_TEST_DIR / "health_coach_test.sqlite3")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
