
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import get_settings

import logging

logger = logging.getLogger(__name__)

from langchain_core.callbacks import AsyncCallbackHandler

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Use the available calculator tools for arithmetic when appropriate. "
    "After receiving tool results, explain the answer clearly."
)

logger = logging.getLogger(__name__)


class ToolExecutionLogger(AsyncCallbackHandler):
    """Log MCP tool execution without exposing tool inputs or secrets."""

    async def on_tool_start(
        self, serialized, input_str, *, run_id, **kwargs
    ):
        logger.info(
            "Tool execution started: %s",
            serialized.get("name", "unknown"),
        )

    async def on_tool_end(self, output, *, run_id, **kwargs):
        logger.info("Tool execution completed successfully.")

    async def on_tool_error(self, error, *, run_id, **kwargs):
        logger.error(
            "Tool execution failed: %s",
            type(error).__name__,
        )

class AssistantState(MessagesState):
    """State passed between nodes in the assistant workflow."""


def build_graph(checkpointer=None, tools=None):
    """Build a LangGraph workflow with optional MCP-discovered tools."""
    tools = tools or []
    settings = get_settings()

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    model = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.7,
        max_tokens=4096,
    )

    model_with_tools = model.bind_tools(tools) if tools else model


    async def call_model(state: AssistantState) -> dict:
        messages = state["messages"]

        response = await model_with_tools.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *messages]
        )

        logger.info(
            "Model finish reason: %s",
            response.response_metadata.get("finish_reason"),
        )
        logger.info(
            "Model usage: %s",
            response.usage_metadata,
        )
        logger.info(
            "Model response: tool_calls=%s",
            [
                {
                    "name": tool_call.get("name"),
                    "id": tool_call.get("id"),
                }
                for tool_call in getattr(response, "tool_calls", [])
            ],
        )

        return {"messages": [response]}


    builder = StateGraph(AssistantState)
    builder.add_node("assistant", call_model)
    builder.add_edge(START, "assistant")

    if tools:
        builder.add_node(
    "tools",
    ToolNode(tools),)
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
    else:
        builder.add_edge("assistant", END)

    return builder.compile(checkpointer=checkpointer)


from langchain_core.callbacks import AsyncCallbackHandler


async def generate_graph_reply(
    messages: list[dict[str, str]],
    graph,
    conversation_id: int,
) -> str:
    """Run the checkpointed graph asynchronously for MCP tool support."""

    latest_user_message = next(
        (
            message
            for message in reversed(messages)
            if message["role"] == "user"
        ),
        None,
    )

    if latest_user_message is None:
        raise ValueError("No user message found.")

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=latest_user_message["content"])
            ]
        },
        config={
            "configurable": {
                "thread_id": str(conversation_id),
            },
            "callbacks": [ToolExecutionLogger()],
        },
    )

    reply = result["messages"][-1].content

    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("The AI model returned an empty response.")

    return reply
