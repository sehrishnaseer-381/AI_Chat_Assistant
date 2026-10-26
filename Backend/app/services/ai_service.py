
from app.services.graph_service import generate_graph_reply


async def generate_reply(
    messages: list[dict[str, str]],
    graph,
    conversation_id: int,
) -> str:
    from app.services.graph_service import generate_graph_reply

    return await generate_graph_reply(
        messages=messages,
        graph=graph,
        conversation_id=conversation_id,
    )