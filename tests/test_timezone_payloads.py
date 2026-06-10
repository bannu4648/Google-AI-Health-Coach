from backend.health_coach.core.payloads import normalize_router_payload, resolve_session_time_utc
from backend.health_coach.core.timezone import format_utc_iso, parse_to_utc, to_civil_filter_literal


def test_logged_at_hkt_converts_to_utc_for_google():
    assert resolve_session_time_utc({"logged_at_hkt": "2026-06-08T22:30:00"}) == "2026-06-08T14:30:00Z"


def test_parse_to_utc_accepts_hkt_naive_timestamp():
    assert format_utc_iso(parse_to_utc("2026-06-08T22:30:00")) == "2026-06-08T14:30:00Z"


def test_civil_filter_literal_has_no_z_suffix():
    literal = to_civil_filter_literal("2026-06-08T14:30:00Z")
    assert literal == "2026-06-08T22:30:00"
    assert "Z" not in literal


def test_normalize_router_payload_builds_nutrition_data_point():
    payload = normalize_router_payload(
        "LOG_NUTRITION",
        {
            "food_display_name": "chapati",
            "logged_at_hkt": "2026-06-08T22:30:00",
            "calories_kcal": 100,
        },
    )
    assert payload["data_type"] == "nutrition-log"
    assert payload["data_point"]["nutritionLog"]["foodDisplayName"] == "chapati"
