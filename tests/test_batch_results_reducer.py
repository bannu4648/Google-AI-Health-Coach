from backend.health_coach.agent.actions import _foods_match_for_dedupe
from backend.health_coach.agent.graph import _merge_batch_results


def test_merge_batch_results_clears_stale_checkpoint_data():
    stale = [{"item": {"food_display_name": "pizza"}, "reply": "old"}]
    assert _merge_batch_results(stale, []) == []


def test_merge_batch_results_accumulates_parallel_items():
    first = [{"item": {"food_display_name": "kebab"}, "reply": "one"}]
    second = [{"item": {"food_display_name": "shake"}, "reply": "two"}]
    merged = _merge_batch_results(first, second)
    assert len(merged) == 2


def test_foods_match_for_dedupe_spicy_mcchicken_variants():
    assert _foods_match_for_dedupe(
        "spicy mcchicken burger",
        "McDonald's Hot 'N Spicy McChicken burger",
    )
