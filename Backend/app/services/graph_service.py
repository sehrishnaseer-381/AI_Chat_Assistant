import asyncio
import json
import logging
from collections import OrderedDict

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import get_settings

from langgraph.types import Overwrite

logger = logging.getLogger(__name__)

MCP_TOOL_TIMEOUT_SECONDS = 10
MAX_TOOL_CALLS_PER_MODEL_RESPONSE = 4
MAX_TOOL_ARGUMENT_BYTES = 4096
MAX_TOOL_CACHE_SIZE = 128
GRAPH_RECURSION_LIMIT = 16

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


def _to_graph_history(messages: list[dict[str, str]]):
    history = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if not isinstance(content, str) or not content.strip():
            continue

        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))

    history = history[-10:]
    while history and not isinstance(history[0], HumanMessage):
        history.pop(0)

    if not history:
        raise ValueError("No recent user message found.")

    return history


async def synchronize_graph_history(
    messages: list[dict[str, str]],
    graph,
    conversation_id: int,
) -> None:
    """Replace a conversation's checkpoint with its persisted message history."""
    history = _to_graph_history(messages)
    config = {
        "configurable": {
            "thread_id": f"{conversation_id}:generation",
        },
        "callbacks": [ToolExecutionLogger()],
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    await graph.aupdate_state(
        config,
        {"messages": Overwrite(history)},
    )


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
        max_tokens=2048,
    )

    model_with_tools = model.bind_tools(tools) if tools else model
    available_tools = {tool.name: tool for tool in tools}
    tool_cache: OrderedDict[tuple, ToolMessage] = OrderedDict()
    tool_inflight: dict[tuple, asyncio.Task] = {}
    tool_cache_lock = asyncio.Lock()


    async def execute_tool_safely(request, handler):
        tool_call = request.tool_call
        tool = request.tool
        args = tool_call.get("args")

        if tool is None or tool_call.get("name") not in available_tools:
            raise ValueError("The requested tool is not available.")
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be a JSON object.")

        try:
            serialized_args = json.dumps(
                args,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Tool arguments must be valid JSON.") from exc
        if len(serialized_args.encode("utf-8")) > MAX_TOOL_ARGUMENT_BYTES:
            raise ValueError("Tool arguments exceed the supported size.")

        args_schema = tool.args_schema
        if hasattr(args_schema, "model_validate"):
            args_schema.model_validate(args)

        runtime_config = request.runtime.config
        configurable = runtime_config.get("configurable", {})
        metadata = runtime_config.get("metadata", {})
        cache_key = (
            configurable.get("thread_id"),
            metadata.get("langgraph_step"),
            tool_call["name"],
            serialized_args,
        )
        if cache_key[1] is None:
            cache_key = (*cache_key[:1], tool_call.get("id"), *cache_key[2:])

        async with tool_cache_lock:
            cached = tool_cache.get(cache_key)
            if cached is not None:
                tool_cache.move_to_end(cache_key)
                return ToolMessage(
                    content=cached.content,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                    status=cached.status,
                )

            task = tool_inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    asyncio.wait_for(
                        handler(request),
                        timeout=MCP_TOOL_TIMEOUT_SECONDS,
                    )
                )
                tool_inflight[cache_key] = task

        try:
            result = await task
        except BaseException:
            async with tool_cache_lock:
                if tool_inflight.get(cache_key) is task:
                    tool_inflight.pop(cache_key, None)
            raise

        if isinstance(result, ToolMessage):
            async with tool_cache_lock:
                tool_cache[cache_key] = result
                tool_cache.move_to_end(cache_key)
                while len(tool_cache) > MAX_TOOL_CACHE_SIZE:
                    tool_cache.popitem(last=False)
                if tool_inflight.get(cache_key) is task:
                    tool_inflight.pop(cache_key, None)
            if result.tool_call_id == tool_call["id"]:
                return result
            return ToolMessage(
                content=result.content,
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                status=result.status,
            )
        return result


    async def call_model(state: AssistantState) -> dict:
        messages = state["messages"]

        logger.info("Messages sent to model: %s", len(messages))
        logger.info(
            "Approximate input characters: %s",
            sum(len(str(message.content)) for message in messages),
        )

        response = None
        async for chunk in model_with_tools.astream(
            [SystemMessage(content=SYSTEM_PROMPT), *messages]
        ):
            response = chunk if response is None else response + chunk

        if response is None:
            raise RuntimeError("The AI model returned no response.")

        invalid_calls = getattr(response, "invalid_tool_calls", [])
        if invalid_calls:
            logger.warning(
                "Groq returned %s malformed tool call(s).",
                len(invalid_calls),
            )
            raise ValueError("The model returned malformed tool arguments.")

        tool_calls = getattr(response, "tool_calls", [])
        if len(tool_calls) > MAX_TOOL_CALLS_PER_MODEL_RESPONSE:
            raise ValueError("The model requested too many tool executions.")
        for tool_call in tool_calls:
            if tool_call.get("name") not in available_tools:
                raise ValueError("The model requested an unavailable tool.")
            if not isinstance(tool_call.get("args"), dict):
                raise ValueError("The model returned malformed tool arguments.")

        logger.info(
            "Model finish reason: %s",
            response.response_metadata.get("finish_reason"),
        )
        logger.info("Model usage: %s", response.usage_metadata)
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
            ToolNode(
                tools,
                handle_tool_errors=True,
                awrap_tool_call=execute_tool_safely,
            ),
        )
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
    """Generate a reply using the current database conversation history."""

    config = {
        "configurable": {
            "thread_id": f"{conversation_id}:generation",
        },
        "callbacks": [ToolExecutionLogger()],
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    history = _to_graph_history(messages)

    # Rebuild graph state from the authoritative conversation history.
    result = await graph.ainvoke(
        {"messages": Overwrite(history)},
        config=config,
    )

    reply = result["messages"][-1].content

    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("The AI model returned an empty response.")

    return reply


async def stream_graph_reply(
    messages: list[dict[str, str]],
    graph,
    conversation_id: int,
):
    """Yield assistant text chunks while preserving LangGraph tool execution."""
    config = {
        "configurable": {
            "thread_id": f"{conversation_id}:generation",
        },
        "callbacks": [ToolExecutionLogger()],
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    history = _to_graph_history(messages)
    received_text = False

    async for event in graph.astream(
        {"messages": Overwrite(history)},
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if event.get("type") == "messages":
            chunk, metadata = event["data"]
            if metadata.get("langgraph_node") != "assistant":
                continue

            content = chunk.content
            if isinstance(content, str) and content:
                received_text = True
                yield {"type": "token", "content": content}
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and isinstance(block.get("text"), str)
                    ):
                        text = block["text"]
                        if text:
                            received_text = True
                            yield {"type": "token", "content": text}
        elif event.get("type") == "updates":
            for node_name, update in event["data"].items():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):
                    if node_name == "assistant":
                        for tool_call in getattr(message, "tool_calls", []):
                            yield {
                                "type": "tool_start",
                                "tool": tool_call.get("name", "unknown"),
                                "tool_call_id": tool_call.get("id"),
                            }
                    elif node_name == "tools" and isinstance(message, ToolMessage):
                        result = str(message.content)
                        yield {
                            "type": "tool_result",
                            "tool": message.name or "unknown",
                            "tool_call_id": message.tool_call_id,
                            "content": result[:500],
                            "status": message.status or "success",
                        }

    if not received_text:
        raise RuntimeError("The AI model returned an empty response.")