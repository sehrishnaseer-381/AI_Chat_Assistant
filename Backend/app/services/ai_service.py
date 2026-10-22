
from groq import Groq

from app.core.config import get_settings


def generate_reply(messages: list[dict[str, str]]) -> str:
    settings = get_settings()

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=settings.groq_api_key)

    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=1024,
    )

    reply = completion.choices[0].message.content

    if not reply:
        raise RuntimeError("The AI model returned an empty response.")

    return reply
