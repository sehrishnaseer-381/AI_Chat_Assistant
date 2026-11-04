import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services import temperature_service
from app.services.temperature_service import (
    ResponseCategory,
    classify_response_category,
    select_temperature,
)


@pytest.mark.parametrize(
    ("prompt", "category", "temperature"),
    [
        ("What is photosynthesis?", ResponseCategory.FACTUAL, 0.3),
        ("Debug this Python function.", ResponseCategory.CODING, 0.2),
        ("Calculate 12 * 8.", ResponseCategory.MATHEMATICAL, 0.1),
        (
            "Compare the trade-offs of these strategies.",
            ResponseCategory.REASONING,
            0.45,
        ),
        (
            "Write a poem about the ocean.",
            ResponseCategory.CREATIVE,
            1.0,
        ),
        ("Hi, how are you?", ResponseCategory.CONVERSATIONAL, 0.7),
    ],
)
def test_select_temperature_for_response_categories(
    prompt, category, temperature
):
    selection = select_temperature(prompt, default_temperature=0.7)

    assert selection.category is category
    assert selection.temperature == temperature


def test_manual_temperature_override_takes_precedence():
    selection = select_temperature(
        "Calculate 12 * 8.",
        default_temperature=0.7,
        override_temperature=1.4,
    )

    assert selection.category is ResponseCategory.MATHEMATICAL
    assert selection.temperature == 1.4


def test_configured_default_is_used_for_conversation_and_classifier_fallback(
    monkeypatch,
):
    assert (
        select_temperature(
            "Hi, how are you?",
            default_temperature=0.62,
        ).temperature
        == 0.62
    )

    def fail_classification(_message):
        raise RuntimeError("classifier unavailable")

    monkeypatch.setattr(
        temperature_service,
        "classify_response_category",
        fail_classification,
    )
    selection = select_temperature(
        "A prompt",
        default_temperature=0.62,
    )

    assert selection.temperature == 0.62


@pytest.mark.parametrize(
    "setting",
    ["groq_default_temperature", "groq_temperature_override"],
)
@pytest.mark.parametrize("value", [-0.01, 2.01])
def test_temperature_settings_reject_values_outside_supported_range(
    setting, value
):
    with pytest.raises(ValidationError):
        Settings(
            database_url="sqlite://",
            jwt_secret_key="test-key",
            **{setting: value},
        )


def test_classifier_rejects_non_text_input():
    with pytest.raises(TypeError, match="must be text"):
        classify_response_category(None)
