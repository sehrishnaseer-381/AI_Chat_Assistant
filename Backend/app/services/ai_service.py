
from app.services.graph_service import (
    generate_graph_reply,
    stream_graph_reply,
    synchronize_graph_history as synchronize_graph_checkpoint,
)
from app.schemas.model import ModelPreference


async def generate_reply(
    messages: list[dict[str, object]],
    graph,
    conversation_id: int,
    model_preference: ModelPreference = "auto",
) -> str:

    return await generate_graph_reply(
        messages=messages,
        graph=graph,
        conversation_id=conversation_id,
        model_preference=model_preference,
    )


async def synchronize_graph_history(
    messages: list[dict[str, object]],
    graph,
    conversation_id: int,
) -> None:
    await synchronize_graph_checkpoint(
        messages=messages,
        graph=graph,
        conversation_id=conversation_id,
    )


async def stream_reply(
    messages: list[dict[str, object]],
    graph,
    conversation_id: int,
    model_preference: ModelPreference = "auto",
):
    async for event in stream_graph_reply(
        messages=messages,
        graph=graph,
        conversation_id=conversation_id,
        model_preference=model_preference,
    ):
        yield event