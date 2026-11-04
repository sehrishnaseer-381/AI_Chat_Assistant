import asyncio
import json
import logging
import time
from collections import OrderedDict

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import get_settings
from app.services.model_router import (
    ModelPreference,
    ModelSelection,
    build_model_registry,
    configured_model_name,
    estimate_cost_usd,
    route_model,
)
from app.services.temperature_service import select_temperature

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

    model_preference: str
    selected_model: str | None
    routing_reason: str | None
    fallback_used: bool


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return ""


def _message_has_images(message: object) -> bool:
    content = getattr(message, "content", None)
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for block in content
    )


def _to_graph_history(messages: list[dict[str, object]]):
    latest_image_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
            and messages[index].get("attachments")
        ),
        None,
    )
    selected_indices = set(range(max(0, len(messages) - 10), len(messages)))
    if latest_image_index is not None:
        selected_indices.add(latest_image_index)

    history = []
    for index in sorted(selected_indices):
        message = messages[index]
        role = message.get("role")
        content = message.get("content", "")
        text = _message_text(content)

        if not text.strip() and index != latest_image_index:
            continue

        if role == "user":
            if index == latest_image_index:
                attachments = message.get("attachments", [])
                if not isinstance(attachments, list) or len(attachments) > 3:
                    raise ValueError("The stored image attachments are invalid.")
                content_blocks: list[dict[str, object]] = []
                if text.strip():
                    content_blocks.append({"type": "text", "text": text})
                for attachment in attachments:
                    if not isinstance(attachment, dict):
                        raise ValueError(
                            "The stored image attachment is invalid."
                        )
                    data_url = attachment.get("data_url")
                    content_type = attachment.get("content_type")
                    if (
                        not isinstance(data_url, str)
                        or content_type
                        not in {"image/png", "image/jpeg", "image/webp"}
                        or not data_url.startswith(
                            f"data:{content_type};base64,"
                        )
                    ):
                        raise ValueError(
                            "The stored image attachment is invalid."
                        )
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                history.append(HumanMessage(content=content_blocks))
            else:
                history.append(HumanMessage(content=text))
        elif role == "assistant":
            if text.strip():
                history.append(AIMessage(content=text))

    history = history[-11:]
    while history and not isinstance(history[0], HumanMessage):
        history.pop(0)

    if not history:
        raise ValueError("No recent user message found.")

    return history


async def synchronize_graph_history(
    messages: list[dict[str, object]],
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

    model_registry = build_model_registry(settings, tools)
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


    async def call_model(
        state: AssistantState,
        config: RunnableConfig,
    ) -> dict:
        messages = state["messages"]
        latest_user_content = next(
            (
                message.content
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        latest_user_message = _message_text(latest_user_content)
        configurable = config.get("configurable", {})
        request_override = configurable.get("temperature_override")
        if request_override is not None and not 0.0 <= request_override <= 2.0:
            raise ValueError("Temperature override must be between 0 and 2.")

        selection = select_temperature(
            latest_user_message,
            default_temperature=settings.groq_default_temperature,
            override_temperature=(
                request_override
                if request_override is not None
                else settings.groq_temperature_override
            ),
        )
        temperature = selection.temperature

        preference = state.get(
            "model_preference",
            config.get("configurable", {}).get(
                "model_preference",
                ModelPreference.AUTO.value,
            ),
        )
        selected_model = state.get("selected_model")
        routing_reason = state.get("routing_reason")
        if selected_model is None:
            has_images = any(_message_has_images(message) for message in messages)
            model_selection = (
                ModelSelection(
                    model_id=ModelPreference.QWEN_VISION,
                    reason="image attachment requires vision model",
                )
                if has_images
                else route_model(
                    latest_user_message,
                    preference,
                    tools_available=bool(tools),
                )
            )
            selected_model = model_selection.model_id.value
            routing_reason = model_selection.reason

        logger.info(
            "Selected response category=%s temperature=%.2f",
            selection.category.value,
            temperature,
        )
        logger.info(
            "Selected model=%s reason=%s",
            configured_model_name(settings, ModelPreference(selected_model)),
            routing_reason,
        )

        logger.info("Messages sent to model: %s", len(messages))
        logger.info(
            "Approximate input text characters: %s",
            sum(
                len(_message_text(message.content))
                for message in messages
            ),
        )

        prompt_messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        fallback_used = state.get("fallback_used", False)

        async def invoke_model(model_id: ModelPreference):
            response = None
            request_model = model_registry[model_id].bind(
                temperature=temperature,
            )
            started_at = time.perf_counter()
            async for chunk in request_model.astream(prompt_messages):
                response = chunk if response is None else response + chunk

            if response is None:
                raise RuntimeError("The AI model returned no response.")

            invalid_calls = getattr(response, "invalid_tool_calls", [])
            if invalid_calls:
                raise ValueError("The model returned malformed tool arguments.")

            tool_calls = getattr(response, "tool_calls", [])
            if len(tool_calls) > MAX_TOOL_CALLS_PER_MODEL_RESPONSE:
                raise ValueError("The model requested too many tool executions.")
            for tool_call in tool_calls:
                if tool_call.get("name") not in available_tools:
                    raise ValueError("The model requested an unavailable tool.")
                if not isinstance(tool_call.get("args"), dict):
                    raise ValueError("The model returned malformed tool arguments.")

            response_metadata = response.response_metadata or {}
            finish_reason = response_metadata.get("finish_reason")
            if finish_reason == "length":
                raise RuntimeError("The model response reached its output limit.")

            content = response.content
            has_text = (
                isinstance(content, str) and bool(content.strip())
            ) or (
                isinstance(content, list)
                and any(
                    isinstance(block, dict)
                    and isinstance(block.get("text"), str)
                    and block["text"].strip()
                    for block in content
                )
            )
            if not has_text and not tool_calls:
                raise RuntimeError("The AI model returned an empty response.")

            usage = response.usage_metadata or {}
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Model usage model=%s latency_ms=%.2f input_tokens=%s "
                "output_tokens=%s estimated_cost_usd=%s "
                "tool_calls=%s finish_reason=%s",
                configured_model_name(settings, model_id),
                elapsed_ms,
                input_tokens,
                output_tokens,
                estimate_cost_usd(model_id, input_tokens, output_tokens),
                len(tool_calls),
                finish_reason,
            )
            return response

        active_model = ModelPreference(selected_model)
        model_attempt_started_at = time.perf_counter()
        try:
            response = await invoke_model(active_model)
        except Exception as exc:
            elapsed_ms = (
                time.perf_counter() - model_attempt_started_at
            ) * 1000
            tool_already_executed = any(
                isinstance(message, ToolMessage)
                for message in messages
            )
            if (
                active_model is not ModelPreference.GPT_OSS_20B
                or fallback_used
                or tool_already_executed
            ):
                logger.error(
                    "Model attempt failed model=%s latency_ms=%.2f reason=%s",
                    active_model.value,
                    elapsed_ms,
                    type(exc).__name__,
                )
                raise

            fallback_model = ModelPreference.GPT_OSS_120B
            logger.warning(
                "Model fallback from=%s to=%s latency_ms=%.2f reason=%s",
                active_model.value,
                fallback_model.value,
                elapsed_ms,
                type(exc).__name__,
            )
            get_stream_writer()({"type": "reset"})
            fallback_started_at = time.perf_counter()
            try:
                response = await invoke_model(fallback_model)
            except Exception as fallback_error:
                logger.error(
                    "Fallback model failed model=%s latency_ms=%.2f "
                    "reason=%s",
                    fallback_model.value,
                    (time.perf_counter() - fallback_started_at) * 1000,
                    type(fallback_error).__name__,
                )
                raise
            selected_model = fallback_model.value
            routing_reason = "fallback after economical model failure"
            fallback_used = True

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

        return {
            "messages": [response],
            "selected_model": selected_model,
            "routing_reason": routing_reason,
            "fallback_used": fallback_used,
        }


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
    messages: list[dict[str, object]],
    graph,
    conversation_id: int,
    model_preference: str = ModelPreference.AUTO.value,
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
        {
            "messages": Overwrite(history),
            "model_preference": model_preference,
            "selected_model": None,
            "routing_reason": None,
            "fallback_used": False,
        },
        config=config,
    )

    reply = _message_text(result["messages"][-1].content)

    if not reply.strip():
        raise RuntimeError("The AI model returned an empty response.")

    return reply


async def stream_graph_reply(
    messages: list[dict[str, object]],
    graph,
    conversation_id: int,
    model_preference: str = ModelPreference.AUTO.value,
):
    """Yield assistant text chunks while preserving LangGraph tool execution."""
    config = {
        "configurable": {
            "thread_id": f"{conversation_id}:generation",
            "model_preference": model_preference,
        },
        "callbacks": [ToolExecutionLogger()],
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    history = _to_graph_history(messages)
    received_text = False

    async for event in graph.astream(
        {
            "messages": Overwrite(history),
            "model_preference": model_preference,
            "selected_model": None,
            "routing_reason": None,
            "fallback_used": False,
        },
        config=config,
        stream_mode=["messages", "updates", "custom"],
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
        elif event.get("type") == "custom":
            if event.get("data") == {"type": "reset"}:
                yield {"type": "reset"}
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
                        result = message.content
                        if isinstance(result, str):
                            result = result[:500]
                        yield {
                            "type": "tool_result",
                            "tool": message.name or "unknown",
                            "tool_call_id": message.tool_call_id,
                            "content": result[:500],
                            "status": message.status or "success",
                        }

    if not received_text:
        raise RuntimeError("The AI model returned an empty response.")