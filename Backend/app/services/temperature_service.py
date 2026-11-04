import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseCategory(str, Enum):
    FACTUAL = "factual"
    CODING = "coding"
    MATHEMATICAL = "mathematical"
    REASONING = "reasoning"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"


@dataclass(frozen=True)
class TemperatureSelection:
    category: ResponseCategory
    temperature: float


_CATEGORY_TEMPERATURES = {
    ResponseCategory.FACTUAL: 0.3,
    ResponseCategory.CODING: 0.2,
    ResponseCategory.MATHEMATICAL: 0.1,
    ResponseCategory.REASONING: 0.45,
    ResponseCategory.CREATIVE: 1.0,
}

_MATH_PATTERN = re.compile(
    r"\b(calculate|compute|what is \d|how much is \d|solve for|"
    r"equation|arithmetic|integral|derivative|multiply|divide|"
    r"plus|minus|sum of|square root|factorial)\b|[0-9]\s*[+\-*/=]\s*[0-9]",
    re.IGNORECASE,
)
_CODING_PATTERN = re.compile(
    r"\b(code|coding|program|programming|debug|bug|exception|"
    r"traceback|function|class|api|sql|python|javascript|typescript|"
    r"react|fastapi|langgraph|compiler|compile|test case|unit test)\b",
    re.IGNORECASE,
)
_CREATIVE_PATTERN = re.compile(
    r"\b(write a (story|poem|song|script)|creative writing|"
    r"brainstorm|imagine|fiction|poem|story|lyrics|slogan|"
    r"come up with ideas|give me ideas)\b",
    re.IGNORECASE,
)
_REASONING_PATTERN = re.compile(
    r"\b(analy[sz]e|reason|reasoning|compare|contrast|trade-?offs?|"
    r"step by step|work through|evaluate|strategy|pros and cons|"
    r"why does|why is|how would|design an approach|"
    r"problem[- ]solving)\b",
    re.IGNORECASE,
)
_CONVERSATIONAL_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|good morning|good afternoon|good evening|"
    r"how are you|what's up|thanks|thank you|goodbye|bye|"
    r"nice to meet you)\b",
    re.IGNORECASE,
)


def classify_response_category(message: str) -> ResponseCategory:
    """Classify a user prompt without sending it to another model."""
    if not isinstance(message, str):
        raise TypeError("The user prompt must be text.")

    if _MATH_PATTERN.search(message):
        return ResponseCategory.MATHEMATICAL
    if _CODING_PATTERN.search(message):
        return ResponseCategory.CODING
    if _CREATIVE_PATTERN.search(message):
        return ResponseCategory.CREATIVE
    if _REASONING_PATTERN.search(message):
        return ResponseCategory.REASONING
    if _CONVERSATIONAL_PATTERN.search(message):
        return ResponseCategory.CONVERSATIONAL
    return ResponseCategory.FACTUAL


def select_temperature(
    message: str,
    default_temperature: float,
    override_temperature: float | None = None,
) -> TemperatureSelection:
    """Select a validated effective temperature for one model invocation."""
    try:
        category = classify_response_category(message)
    except Exception as exc:
        logger.warning(
            "Temperature classification failed (%s); using configured default.",
            type(exc).__name__,
        )
        return TemperatureSelection(
            category=ResponseCategory.CONVERSATIONAL,
            temperature=default_temperature,
        )

    temperature = (
        override_temperature
        if override_temperature is not None
        else _CATEGORY_TEMPERATURES.get(category, default_temperature)
    )
    return TemperatureSelection(category=category, temperature=temperature)
