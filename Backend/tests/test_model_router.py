from types import SimpleNamespace

import pytest

from app.services import model_router
from app.services.model_router import (
    ModelPreference,
    build_model_registry,
    estimate_cost_usd,
    route_model,
)


@pytest.mark.parametrize(
    ("prompt", "expected_model"),
    [
        ("Hello there!", ModelPreference.GPT_OSS_20B),
        ("What is the capital of France?", ModelPreference.GPT_OSS_20B),
        ("Summarize this document in three bullets.", ModelPreference.GPT_OSS_20B),
        ("Write a simple Python function to add two numbers.", ModelPreference.GPT_OSS_20B),
        (
            "Analyze this complex system architecture and its trade-offs.",
            ModelPreference.GPT_OSS_120B,
        ),
        (
            "Debug this production traceback and find the root cause.",
            ModelPreference.GPT_OSS_120B,
        ),
        (
            "Search and then fetch each result using the available tools.",
            ModelPreference.GPT_OSS_120B,
        ),
    ],
)
def test_auto_routes_request_categories(prompt, expected_model):
    assert route_model(
        prompt,
        tools_available="available tools" in prompt,
    ).model_id is expected_model


@pytest.mark.parametrize(
    "preference",
    [
        ModelPreference.GPT_OSS_20B,
        ModelPreference.GPT_OSS_120B,
    ],
)
def test_manual_model_selection_overrides_auto(preference):
    selected = route_model(
        "Analyze this complex architecture.",
        preference,
    )

    assert selected.model_id is preference
    assert selected.reason == "manual model selection"


def test_multi_operation_requests_use_large_model_when_tools_are_available():
    assert route_model(
        "Calculate the total and summarize the result.",
        tools_available=True,
    ).model_id is ModelPreference.GPT_OSS_120B


def test_invalid_model_preference_is_rejected():
    with pytest.raises(ValueError):
        route_model("Hello", "not-a-supported-model")


def test_model_registry_builds_all_clients_with_a_bounded_token_limit(
    monkeypatch,
):
    created = []

    class FakeChatGroq:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def bind_tools(self, tools):
            return self

    monkeypatch.setattr(model_router, "ChatGroq", FakeChatGroq)
    registry = build_model_registry(
        SimpleNamespace(
            groq_api_key="test-key",
            groq_vision_model="qwen/test-vision",
            groq_default_temperature=0.7,
            groq_max_completion_tokens=8192,
        ),
        tools=[object()],
    )

    assert set(registry) == {
        ModelPreference.GPT_OSS_20B,
        ModelPreference.GPT_OSS_120B,
        ModelPreference.QWEN_VISION,
    }
    assert {kwargs["model"] for kwargs in created} == {
        ModelPreference.GPT_OSS_20B.value,
        ModelPreference.GPT_OSS_120B.value,
        "qwen/test-vision",
    }
    assert all(
        kwargs["model_kwargs"]["max_completion_tokens"] == 8192
        for kwargs in created
    )


def test_estimated_cost_uses_model_specific_rates():
    small_cost = estimate_cost_usd(ModelPreference.GPT_OSS_20B, 1000, 1000)
    large_cost = estimate_cost_usd(ModelPreference.GPT_OSS_120B, 1000, 1000)

    assert small_cost == pytest.approx(0.000375)
    assert large_cost == pytest.approx(0.00075)
    assert estimate_cost_usd(ModelPreference.QWEN_VISION, 1000, 1000) is None
