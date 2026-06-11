from unittest.mock import MagicMock

from backend.health_coach.agent.vision import VisionAgent, VisionAnalysis


def test_vision_agent_defaults_to_lookup_without_log_caption():
    client = MagicMock()
    client.generate_structured.return_value = None
    agent = VisionAgent(client=client)
    result = agent.analyze_food_image(
        image_bytes=b"fake-image",
        mime_type="image/jpeg",
        caption="",
    )
    assert result["lookup_only"] is True
    assert result["wants_to_log"] is False


def test_vision_agent_returns_structured_meal_fields():
    client = MagicMock()
    client.generate_structured.return_value = VisionAnalysis(
        food_display_name="Chicken rice",
        portion_description="1 plate",
        meal_type="LUNCH",
        wants_to_log=True,
        lookup_only=False,
        conversational_reply="Looks like chicken rice on your plate — I'll look up nutrition and log it.",
    )
    agent = VisionAgent(client=client)
    result = agent.analyze_food_image(
        image_bytes=b"fake-image",
        mime_type="image/jpeg",
        caption="log this lunch",
    )
    assert result["food_display_name"] == "Chicken rice"
    assert result["wants_to_log"] is True
    client.generate_structured.assert_called_once()
