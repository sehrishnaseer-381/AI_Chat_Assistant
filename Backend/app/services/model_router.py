import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_groq import ChatGroq

from app.core.config import Settings


class ModelPreference(str, Enum):
    AUTO = "auto"
    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    QWEN_VISION = "qwen_vision"


@dataclass(frozen=True)
class ModelSpec:
    model_id: ModelPreference
    max_completion_tokens: int
    input_cost_per_million: float | None
    output_cost_per_million: float | None


@dataclass(frozen=True)
class ModelSelection:
    model_id: ModelPreference
    reason: str


MODEL_SPECS = {
    ModelPreference.GPT_OSS_20B: ModelSpec(
        model_id=ModelPreference.GPT_OSS_20B,
        max_completion_tokens=65536,
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
    ),
    ModelPreference.GPT_OSS_120B: ModelSpec(
        model_id=ModelPreference.GPT_OSS_120B,
        max_completion_tokens=65536,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    ModelPreference.QWEN_VISION: ModelSpec(
        model_id=ModelPreference.QWEN_VISION,
        max_completion_tokens=65536,
        input_cost_per_million=None,
        output_cost_per_million=None,
    ),
}


def configured_model_name(
    settings: Settings,
    model_id: ModelPreference,
) -> str:
    if model_id is ModelPreference.QWEN_VISION:
        return settings.groq_vision_model
    return model_id.value


_COMPLEXITY_PATTERN = re.compile(
    r"\b(complex|advanced|architecture|architect|debug|debugging|"
    r"traceback|production|concurrency|concurrent|race condition|"
    r"optimi[sz]e|refactor|security|trade-?offs?|"
    r"step[- ]by[- ]step|multi[- ]step|multiple stages|"
    r"deep analysis|in[- ]depth|reason through|reasoning|"
    r"problem[- ]solving|analy[sz]e|analysis|evaluate|compare|prove|"
    r"design a system|system design|root cause|"
    r"orchestrat(?:e|ion)|several tools|multiple tools)\b",
    re.IGNORECASE,
)
_SIMPLE_TASK_PATTERN = re.compile(
    r"\b(summar(?:y|ize|ise)|extract|key points|in one sentence|"
    r"briefly|simple explanation|basic explanation|"
    r"what does .+ mean|define|hello|hi|hey|thanks|thank you)\b",
    re.IGNORECASE,
)
_CODING_PATTERN = re.compile(
    r"\b(code|coding|program|programming|debug|bug|"
    r"function|class|python|javascript|typescript|react|"
    r"fastapi|sql|api|compiler|test case|unit test)\b",
    re.IGNORECASE,
)
_MULTI_OPERATION_PATTERN = re.compile(
    r"\b(and then|after that|multiple|several|all of|each of)\b|"
    r"\b(compare|calculate|search|fetch|retrieve)\b.{0,80}\band\b",
    re.IGNORECASE,
)


def route_model(
    prompt: str,
    preference: ModelPreference | str = ModelPreference.AUTO,
    *,
    tools_available: bool = False,
) -> ModelSelection:
    """Choose one of the configured GPT-OSS models without another API call."""
    normalized_preference = ModelPreference(preference)
    if normalized_preference is not ModelPreference.AUTO:
        return ModelSelection(
            model_id=normalized_preference,
            reason="manual model selection",
        )

    if not isinstance(prompt, str):
        raise TypeError("The user prompt must be text.")

    complex_request = bool(_COMPLEXITY_PATTERN.search(prompt))
    coding_request = bool(_CODING_PATTERN.search(prompt))
    multi_operation = bool(_MULTI_OPERATION_PATTERN.search(prompt))

    if complex_request or (coding_request and multi_operation):
        return ModelSelection(
            model_id=ModelPreference.GPT_OSS_120B,
            reason="complex reasoning, coding, or orchestration signals",
        )

    if tools_available and multi_operation:
        return ModelSelection(
            model_id=ModelPreference.GPT_OSS_120B,
            reason="multi-operation request with tools available",
        )

    if _SIMPLE_TASK_PATTERN.search(prompt) or coding_request:
        reason = (
            "simple coding request"
            if coding_request
            else "greeting, summary, extraction, or basic explanation"
        )
        return ModelSelection(
            model_id=ModelPreference.GPT_OSS_20B,
            reason=reason,
        )

    return ModelSelection(
        model_id=ModelPreference.GPT_OSS_20B,
        reason="default economical route for general and factual requests",
    )


def build_model_registry(
    settings: Settings,
    tools: list[Any] | None = None,
) -> dict[ModelPreference, Any]:
    """Create reusable Groq model clients once during graph startup."""
    tools = tools or []
    registry = {}
    for model_id, spec in MODEL_SPECS.items():
        configured_limit = getattr(settings, "groq_max_completion_tokens", 4096)
        model = ChatGroq(
            api_key=settings.groq_api_key,
            model=configured_model_name(settings, model_id),
            temperature=settings.groq_default_temperature,
            model_kwargs={
                "max_completion_tokens": min(
                    configured_limit,
                    spec.max_completion_tokens,
                )
            },
        )
        registry[model_id] = model.bind_tools(tools) if tools else model
    return registry


def estimate_cost_usd(
    model_id: ModelPreference,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    spec = MODEL_SPECS[model_id]
    if (
        spec.input_cost_per_million is None
        or spec.output_cost_per_million is None
    ):
        return None
    return (
        input_tokens * spec.input_cost_per_million
        + output_tokens * spec.output_cost_per_million
    ) / 1_000_000
