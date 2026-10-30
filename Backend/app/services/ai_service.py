
from app.services.graph_service import (
    generate_graph_reply,
    stream_graph_reply,
    synchronize_graph_history as synchronize_graph_checkpoint,
)


async def generate_reply(
    messages: list[dict[str, str]],
    graph,
    conversation_id: int,
) -> str:

    return await generate_graph_reply(
        messages=messages,
        graph=graph,
        conversation_id=conversation_id,
    )


async def synchronize_graph_history(
    messages: list[dict[str, str]],
    graph,
    conversation_id: int,
) -> None:
    await synchronize_graph_checkpoint(
        messages=messages,
        graph=graph,
        conversation_id=conversation_id,
    )


async def stream_reply(
    messages: list[dict[str, str]],
    graph,
    conversation_id: int,
):
    async for event in stream_graph_reply(
        messages=messages,
        graph=graph,
        conversation_id=conversation_id,
    ):
        yield event