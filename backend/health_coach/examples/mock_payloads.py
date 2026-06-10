"""
Reference payloads for Google Health API v4 data types.

Use these as templates when building LangGraph tools or testing the client locally.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


NOW = _utc_now()
NOW_PLUS_15_MINUTES = (datetime.now(timezone.utc) + timedelta(minutes=15)).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)
NOW_PLUS_1_MINUTE = (datetime.now(timezone.utc) + timedelta(minutes=1)).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)
HKT_OFFSET = "28800s"


MOCK_NUTRITION_LOG_DATA_POINT = {
    "nutritionLog": {
        "interval": {
            "startTime": NOW,
            "startUtcOffset": HKT_OFFSET,
            "endTime": NOW_PLUS_15_MINUTES,
            "endUtcOffset": HKT_OFFSET,
        },
        "foodDisplayName": "Greek yogurt with berries",
        "mealType": "BREAKFAST",
        "energy": {"kcal": 220},
        "totalCarbohydrate": {"grams": 28},
        "totalFat": {"grams": 6},
        "nutrients": [
            {"nutrient": "PROTEIN", "quantity": {"grams": 18}},
            {"nutrient": "DIETARY_FIBER", "quantity": {"grams": 4}},
        ],
    }
}

MOCK_HYDRATION_LOG_DATA_POINT = {
    "hydrationLog": {
        "interval": {
            "startTime": NOW,
            "startUtcOffset": HKT_OFFSET,
            "endTime": NOW_PLUS_1_MINUTE,
            "endUtcOffset": HKT_OFFSET,
        },
        "amountConsumed": {
            "milliliters": 500,
            "userProvidedUnit": "MILLILITER",
        },
    }
}

MOCK_WEIGHT_DATA_POINT = {
    "weight": {
        "sampleTime": {
            "physicalTime": NOW,
        },
        "weightGrams": 75500,
        "notes": "Morning weigh-in",
    }
}

MOCK_STEPS_DATA_POINT = {
    "steps": {
        "interval": {
            "startTime": "2026-06-09T08:00:00Z",
            "endTime": "2026-06-09T08:15:00Z",
        },
        "count": 1200,
    }
}

MOCK_SLEEP_DATA_POINT = {
    "sleep": {
        "interval": {
            "startTime": "2026-06-08T23:30:00Z",
            "endTime": "2026-06-09T07:15:00Z",
        },
        "type": "CLASSIC",
        "summary": {
            "minutesAsleep": 420,
            "minutesAwake": 25,
        },
    }
}

MOCK_HEART_RATE_DATA_POINT = {
    "heartRate": {
        "sampleTime": {
            "physicalTime": NOW,
        },
        "beatsPerMinute": 62,
    }
}

MOCK_ACTIVE_ZONE_MINUTES_DATA_POINT = {
    "activeZoneMinutes": {
        "interval": {
            "startTime": "2026-06-09T07:00:00Z",
            "endTime": "2026-06-09T07:30:00Z",
        },
        "minutesInFatBurnZone": 12,
        "minutesInCardioZone": 8,
        "minutesInPeakZone": 2,
    }
}

MOCK_LIST_QUERY = {
    "data_type": "steps",
    "start_time": "2026-06-02T00:00:00Z",
    "end_time": "2026-06-09T00:00:00Z",
    "page_size": 100,
}

MOCK_RECONCILE_QUERY = {
    "data_type": "steps",
    "start_time": "2026-06-02T00:00:00Z",
    "end_time": "2026-06-09T00:00:00Z",
    "data_source_family": "users/me/dataSourceFamilies/all-sources",
}

MOCK_DAILY_ROLLUP_QUERY = {
    "data_type": "steps",
    "start_time": "2026-06-02T00:00:00Z",
    "end_time": "2026-06-09T00:00:00Z",
    "window_size_days": 1,
}

# dailyRollUp request body shape (CivilTimeInterval)
MOCK_DAILY_ROLLUP_BODY = {
    "range": {
        "start": {"date": {"year": 2026, "month": 6, "day": 2}},
        "end": {"date": {"year": 2026, "month": 6, "day": 9}},
    },
    "windowSizeDays": 1,
    "dataSourceFamily": "users/me/dataSourceFamilies/all-sources",
}

MOCK_ROLLUP_QUERY = {
    "data_type": "heart-rate",
    "start_time": "2026-06-08T00:00:00Z",
    "end_time": "2026-06-09T00:00:00Z",
    "window_size": "3600s",
}
