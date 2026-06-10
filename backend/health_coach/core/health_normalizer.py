"""Compact Google Health API responses for LLM summaries and dashboards.

Google Health payloads can be large, especially sleep stage timelines and high
frequency heart-rate samples. This module keeps raw responses in SQLite while
passing compact, type-aware summaries to the LLM.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Callable

from .timezone import format_local, get_user_tz, parse_to_utc

MAX_RECORDS_FOR_LLM = 80


def _minutes_between(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        seconds = (parse_to_utc(end) - parse_to_utc(start)).total_seconds()
    except (TypeError, ValueError):
        return None
    return int(round(seconds / 60)) if seconds >= 0 else None


def _local(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return format_local(parse_to_utc(value))
    except (TypeError, ValueError):
        return value


def _civil_date_time(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    date = value.get("date", {})
    time = value.get("time", {})
    if not date:
        return None
    text = f"{date.get('year'):04d}-{date.get('month'):02d}-{date.get('day'):02d}"
    if time:
        text += f" {time.get('hours', 0):02d}:{time.get('minutes', 0):02d}"
    return text


def _interval_record(interval: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_hkt": _local(interval.get("startTime")),
        "end_hkt": _local(interval.get("endTime")),
        "civil_start_hkt": _civil_date_time(interval.get("civilStartTime")),
        "civil_end_hkt": _civil_date_time(interval.get("civilEndTime")),
        "duration_minutes": _minutes_between(interval.get("startTime"), interval.get("endTime")),
    }


def _source(point: dict[str, Any]) -> str | None:
    return point.get("dataSource", {}).get("platform") or point.get("dataSource", {}).get("recordingMethod")


def _quantity(value: dict[str, Any] | None) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("countSum", "minutesSum", "grams", "milliliters", "kcal", "beatsPerMinute", "weightGrams"):
        if key in value:
            return value[key]
    return value


def _normalize_rollups(data_type: str, result: dict[str, Any]) -> dict[str, Any] | None:
    points = result.get("rollupDataPoints")
    if not isinstance(points, list):
        return None

    records = []
    total = 0.0
    for point in points:
        metric = point.get(data_type.replace("-", "")) or point.get(_camel_data_type(data_type)) or {}
        if data_type == "steps":
            value = int(metric.get("countSum", 0) or 0)
            total += value
        elif data_type == "active-zone-minutes":
            value = int(metric.get("minutesSum", 0) or 0)
            total += value
        elif data_type == "hydration-log":
            value = _quantity(metric)
            try:
                total += float(value or 0)
            except (TypeError, ValueError):
                pass
        else:
            value = _quantity(metric)
        records.append(
            {
                "period_start_hkt": _civil_date_time(point.get("civilStartTime")),
                "period_end_hkt": _civil_date_time(point.get("civilEndTime")),
                "value": value,
            }
        )

    return _base(data_type, result, records, {"total": total})


def _camel_data_type(data_type: str) -> str:
    parts = data_type.split("-")
    return parts[0] + "".join(part.title() for part in parts[1:])


def _sleep(result: dict[str, Any], data_type: str) -> dict[str, Any]:
    records = []
    totals = Counter()
    for point in result.get("dataPoints", []):
        sleep = point.get("sleep", {})
        interval = sleep.get("interval", {})
        stages = sleep.get("stages", [])
        stage_minutes = Counter()
        for stage in stages:
            minutes = _minutes_between(stage.get("startTime"), stage.get("endTime")) or 0
            stage_minutes[str(stage.get("type", "UNKNOWN")).lower()] += minutes
        duration = _minutes_between(interval.get("startTime"), interval.get("endTime"))
        if duration:
            totals["duration_minutes"] += duration
        for key, value in stage_minutes.items():
            totals[f"{key}_minutes"] += value
        end = interval.get("endTime")
        wake_date = parse_to_utc(end).astimezone(get_user_tz()).date().isoformat() if end else None
        records.append(
            {
                "wake_date": wake_date,
                **_interval_record(interval),
                "sleep_type": sleep.get("type"),
                "stage_minutes": dict(stage_minutes),
                "source": _source(point),
            }
        )
    return _base(data_type, result, records, dict(totals))


def _exercise(result: dict[str, Any], data_type: str) -> dict[str, Any]:
    records = []
    for point in result.get("dataPoints", []):
        exercise = point.get("exercise", {})
        metrics = exercise.get("metricsSummary", {})
        records.append(
            {
                **_interval_record(exercise.get("interval", {})),
                "exercise_type": exercise.get("exerciseType"),
                "calories_kcal": metrics.get("caloriesKcal"),
                "distance_m": _safe_div(metrics.get("distanceMillimeters"), 1000),
                "steps": _safe_int(metrics.get("steps")),
                "avg_heart_rate": _safe_int(metrics.get("averageHeartRateBeatsPerMinute")),
                "active_zone_minutes": _safe_int(metrics.get("activeZoneMinutes")),
                "source": _source(point),
            }
        )
    totals = {
        "workouts": len(records),
        "calories_kcal": sum(_safe_float(row.get("calories_kcal")) for row in records),
        "active_zone_minutes": sum(_safe_float(row.get("active_zone_minutes")) for row in records),
    }
    return _base(data_type, result, records, totals)


def _nutrition(result: dict[str, Any], data_type: str) -> dict[str, Any]:
    records = []
    totals = Counter()
    for point in result.get("dataPoints", []):
        nutrition = point.get("nutritionLog", {})
        nutrients = {
            item.get("nutrient"): item.get("quantity", {})
            for item in nutrition.get("nutrients", [])
        }
        kcal = _safe_float(nutrition.get("energy", {}).get("kcal"))
        protein = _safe_float(nutrients.get("PROTEIN", {}).get("grams"))
        carbs = _safe_float(nutrition.get("totalCarbohydrate", {}).get("grams"))
        fat = _safe_float(nutrition.get("totalFat", {}).get("grams"))
        totals["calories_kcal"] += kcal
        totals["protein_grams"] += protein
        totals["carbs_grams"] += carbs
        totals["fat_grams"] += fat
        records.append(
            {
                **_interval_record(nutrition.get("interval", {})),
                "food": nutrition.get("foodDisplayName"),
                "meal_type": nutrition.get("mealType"),
                "calories_kcal": kcal,
                "protein_grams": protein,
                "carbs_grams": carbs,
                "fat_grams": fat,
                "source": _source(point),
            }
        )
    return _base(data_type, result, records, dict(totals))


def _hydration(result: dict[str, Any], data_type: str) -> dict[str, Any]:
    records = []
    total_ml = 0.0
    for point in result.get("dataPoints", []):
        hydration = point.get("hydrationLog", {})
        amount = _safe_float(hydration.get("volume", {}).get("milliliters"))
        total_ml += amount
        records.append(
            {
                **_interval_record(hydration.get("interval", {})),
                "milliliters": amount,
                "source": _source(point),
            }
        )
    return _base(data_type, result, records, {"milliliters": total_ml})


def _weight(result: dict[str, Any], data_type: str) -> dict[str, Any]:
    records = []
    for point in result.get("dataPoints", []):
        weight = point.get("weight", {})
        grams = _safe_float(weight.get("weightGrams"))
        sample = weight.get("sampleTime", {})
        records.append(
            {
                "time_hkt": _local(sample.get("physicalTime")),
                "weight_kg": round(grams / 1000, 2) if grams else 0,
                "source": _source(point),
            }
        )
    return _base(data_type, result, records, {"latest_weight_kg": records[0]["weight_kg"] if records else None})


def _heart_rate(result: dict[str, Any], data_type: str) -> dict[str, Any]:
    records = []
    values = []
    for point in result.get("dataPoints", []):
        body = point.get("heartRate", {}) or point.get("dailyRestingHeartRate", {})
        bpm = _safe_float(body.get("beatsPerMinute") or body.get("bpm"))
        if bpm:
            values.append(bpm)
        sample = body.get("sampleTime", {})
        records.append(
            {
                "time_hkt": _local(sample.get("physicalTime")) or _civil_date_time(body.get("date")),
                "bpm": bpm,
                "source": _source(point),
            }
        )
    totals = {
        "sample_count": len(values),
        "min_bpm": min(values) if values else None,
        "max_bpm": max(values) if values else None,
        "avg_bpm": round(sum(values) / len(values), 1) if values else None,
    }
    return _base(data_type, result, records, totals)


def _generic(result: dict[str, Any], data_type: str) -> dict[str, Any]:
    records = []
    for point in result.get("dataPoints", []):
        body = point.get(_camel_data_type(data_type), {})
        records.append({"source": _source(point), "body_keys": sorted(body.keys()) if isinstance(body, dict) else []})
    return _base(data_type, result, records, {})


def _base(data_type: str, result: dict[str, Any], records: list[dict[str, Any]], totals: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    if result.get("_pagination", {}).get("truncated"):
        warnings.append("Google Health response was truncated by max_pages.")
    if len(records) > MAX_RECORDS_FOR_LLM:
        warnings.append(f"Records truncated for LLM from {len(records)} to {MAX_RECORDS_FOR_LLM}.")
    return {
        "normalized": True,
        "data_type": data_type,
        "record_count": len(records),
        "records": records[:MAX_RECORDS_FOR_LLM],
        "totals": totals,
        "warnings": warnings,
        "source_payload_shape": {
            "dataPoints": len(result.get("dataPoints", [])),
            "rollupDataPoints": len(result.get("rollupDataPoints", [])),
            "has_next_page": bool(result.get("nextPageToken")),
            "pages": result.get("_pagination", {}).get("pages"),
        },
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_div(value: Any, divisor: float) -> float:
    return round(_safe_float(value) / divisor, 2) if value else 0.0


NORMALIZERS: dict[str, Callable[[dict[str, Any], str], dict[str, Any]]] = {
    "sleep": _sleep,
    "exercise": _exercise,
    "nutrition-log": _nutrition,
    "hydration-log": _hydration,
    "weight": _weight,
    "heart-rate": _heart_rate,
    "daily-resting-heart-rate": _heart_rate,
}


def normalize_health_result(data_type: str, result: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, type-aware representation of a Google Health result."""
    rollup = _normalize_rollups(data_type, result)
    if rollup is not None:
        return rollup
    normalizer = NORMALIZERS.get(data_type, _generic)
    return normalizer(result, data_type)
