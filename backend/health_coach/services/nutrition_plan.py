"""Personal nutrition targets for cut phase — injected into every coach LLM turn."""

from __future__ import annotations

import json
from typing import Any

from ..core.database import connect, init_db, utc_now_iso
from .user_goals import fetch_active_goals
from .user_profile import fetch_user_profile_snapshot

_PREF_KEY = "default"


def _default_plan(*, weight_kg: float | None = None) -> dict[str, Any]:
    w = float(weight_kg or 76)
    protein_min = int(round(w * 1.5))
    protein_max = int(round(w * 1.65))
    daily_kcal = 1900
    return {
        "daily_calories_target": daily_kcal,
        "daily_calories_range": [1750, 2100],
        "protein_grams_min": protein_min,
        "protein_grams_max": protein_max,
        "calories_per_meal_3": 630,
        "calories_per_meal_4": 475,
        "bmr_kcal_estimate": 1710,
        "tdee_light_kcal": 2350,
        "tdee_moderate_kcal": 2650,
        "weight_kg_start": w,
        "weight_kg_target": 69.5,
        "deadline_hkt": "2026-08-31",
        "coaching_notes": (
            "Cut phase: prioritize protein (~1.5–1.6 g/kg), ~1900 kcal/day average, "
            "~500–650 kcal deficit vs maintenance. Adjust ±150–200 kcal based on weekly scale trend."
        ),
    }


def _plan_from_goals(goals: list[dict[str, Any]], *, weight_kg: float | None) -> dict[str, Any]:
    plan = _default_plan(weight_kg=weight_kg)
    for goal in goals:
        cat = (goal.get("category") or "").lower()
        target = goal.get("target") or {}
        if cat == "nutrition":
            if target.get("protein_grams_min"):
                plan["protein_grams_min"] = int(target["protein_grams_min"])
            if target.get("protein_grams_max"):
                plan["protein_grams_max"] = int(target["protein_grams_max"])
            if target.get("daily_calories_target"):
                plan["daily_calories_target"] = int(target["daily_calories_target"])
            if target.get("daily_calories_range"):
                plan["daily_calories_range"] = target["daily_calories_range"]
        if cat == "weight":
            if target.get("start_weight_kg"):
                plan["weight_kg_start"] = float(target["start_weight_kg"])
            if target.get("target_weight_kg"):
                plan["weight_kg_target"] = float(target["target_weight_kg"])
            if goal.get("deadline_hkt"):
                plan["deadline_hkt"] = goal["deadline_hkt"]
    return plan


def get_nutrition_plan_settings(*, key: str = _PREF_KEY) -> dict[str, Any]:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT settings_json FROM coaching_preferences WHERE pref_key = ?",
            (key,),
        ).fetchone()
    if not row:
        return {}
    try:
        settings = json.loads(row["settings_json"] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return settings.get("nutrition_plan") or {}


def save_nutrition_plan_settings(plan: dict[str, Any], *, key: str = _PREF_KEY) -> dict[str, Any]:
    init_db()
    now = utc_now_iso()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM coaching_preferences WHERE pref_key = ?",
            (key,),
        ).fetchone()
        if row:
            try:
                settings = json.loads(row["settings_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                settings = {}
            settings["nutrition_plan"] = plan
            conn.execute(
                """
                UPDATE coaching_preferences
                SET settings_json = ?, updated_at = ?
                WHERE pref_key = ?
                """,
                (json.dumps(settings, ensure_ascii=False), now, key),
            )
        else:
            conn.execute(
                """
                INSERT INTO coaching_preferences (pref_key, coaching_focus, settings_json, created_at, updated_at)
                VALUES (?, '', ?, ?, ?)
                """,
                (key, json.dumps({"nutrition_plan": plan}, ensure_ascii=False), now, now),
            )
    return plan


def build_nutrition_plan(
    *,
    weight_kg: float | None = None,
    goals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge saved plan, active goals, and profile weight."""
    profile = fetch_user_profile_snapshot()
    w = weight_kg if weight_kg is not None else profile.get("weight_kg")
    active = goals if goals is not None else fetch_active_goals(limit=10)
    plan = _plan_from_goals(active, weight_kg=w)
    saved = get_nutrition_plan_settings()
    if saved:
        plan = {**plan, **saved}
    plan["current_weight_kg"] = w
    return plan


def sum_today_nutrition(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Aggregate kcal and protein from today's Google Health nutrition items."""
    total_kcal = 0.0
    total_protein = 0.0
    meal_count = 0
    for item in snapshot.get("nutrition", {}).get("items", []):
        nutrition = item.get("nutritionLog") or item.get("nutrition_log") or {}
        if not nutrition:
            continue
        meal_count += 1
        energy = nutrition.get("energy") or {}
        total_kcal += float(energy.get("kcal") or 0)
        for nutrient in nutrition.get("nutrients") or []:
            if str(nutrient.get("nutrient")).upper() == "PROTEIN":
                qty = nutrient.get("quantity") or {}
                total_protein += float(qty.get("grams") or 0)
    return {
        "meals_logged": meal_count,
        "calories_kcal": int(round(total_kcal)),
        "protein_grams": int(round(total_protein)),
    }


def format_nutrition_plan_for_prompt(
    *,
    snapshot: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> str:
    """Block for LLM: targets + today's intake vs plan."""
    if plan is None:
        plan = build_nutrition_plan()
    lines = [
        "NUTRITION PLAN (use when user logs food or asks about calories/protein/meals):",
        f"- Cut target: ~{plan['daily_calories_target']} kcal/day "
        f"(range {plan['daily_calories_range'][0]}–{plan['daily_calories_range'][1]}; "
        f"~{plan['calories_per_meal_4']} kcal × 4 meals or ~{plan['calories_per_meal_3']} × 3)",
        f"- Protein: {plan['protein_grams_min']}–{plan['protein_grams_max']} g/day (~30–40 g per meal)",
        f"- Weight goal: {plan.get('weight_kg_start', '?')} → {plan.get('weight_kg_target', '?')} kg "
        f"by {plan.get('deadline_hkt', 'deadline')}",
        f"- Metabolism estimates (formula, not lab): BMR ~{plan.get('bmr_kcal_estimate')} kcal; "
        f"maintenance ~{plan.get('tdee_light_kcal')} (light) to ~{plan.get('tdee_moderate_kcal')} (moderate activity)",
        f"- {plan.get('coaching_notes', '')}",
    ]
    if snapshot:
        intake = sum_today_nutrition(snapshot)
        kcal_left = plan["daily_calories_target"] - intake["calories_kcal"]
        protein_left = plan["protein_grams_min"] - intake["protein_grams"]
        lines.append(
            f"- Today so far: {intake['calories_kcal']} kcal, {intake['protein_grams']} g protein "
            f"({intake['meals_logged']} meal(s) logged)"
        )
        if intake["meals_logged"]:
            if kcal_left > 0:
                lines.append(f"- Remaining vs target: ~{max(0, kcal_left)} kcal, ~{max(0, protein_left)} g protein")
            else:
                lines.append(
                    f"- Over daily kcal target by ~{abs(kcal_left)} kcal — suggest lighter next meal, not guilt"
                )
            if protein_left > 0:
                lines.append(
                    f"- After each log, briefly note protein/calorie budget left and suggest how to hit protein"
                )
    lines.append(
        "- When logging meals: confirm the log, then add one short line on progress vs this plan (if data exists)."
    )
    return "\n".join(lines)


def format_brief_progress_line(snapshot: dict[str, Any], *, plan: dict[str, Any] | None = None) -> str:
    """One-line WhatsApp nudge after a meal log."""
    plan = plan or build_nutrition_plan()
    intake = sum_today_nutrition(snapshot)
    if intake["meals_logged"] == 0:
        return ""
    kcal_left = int(plan["daily_calories_target"]) - intake["calories_kcal"]
    protein_left = int(plan["protein_grams_min"]) - intake["protein_grams"]
    parts = [
        f"Today ({snapshot.get('date_hkt', 'HKT')}): {intake['calories_kcal']}/{plan['daily_calories_target']} kcal",
        f"{intake['protein_grams']}/{plan['protein_grams_min']} g protein",
    ]
    if protein_left > 0:
        parts.append(f"~{protein_left} g protein left to hit target")
    elif protein_left <= 0:
        parts.append("protein target hit — nice")
    if kcal_left > 0:
        parts.append(f"~{kcal_left} kcal budget left")
    elif kcal_left < -100:
        parts.append("over kcal target — go lighter next meal")
    return "📊 " + ", ".join(parts) + "."
