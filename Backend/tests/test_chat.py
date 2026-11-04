
import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base, get_db
from app.models import Conversation, Message, User
from app.api.routes import chat as chat_routes
from app.api.routes import messages as messages_routes
from app.core.security import create_access_token
from app.services import graph_service
from app.services import model_router
from app.services.model_router import ModelPreference
from app.services.graph_service import stream_graph_reply


@pytest.fixture
def client():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestSession = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture
def conversation_id(client):
    # Create a test user and conversation.
    db = next(app.dependency_overrides[get_db]())

    try:
        user = User(email="chat-test@example.com")
        db.add(user)
        db.commit()
        db.refresh(user)

        user_id = user.id

        conversation = Conversation(
            user_id=user_id,
            title="Test conversation",
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return {
            "id": conversation.id,
            "user_id": user_id,
        }

    finally:
        db.close()


@pytest.fixture
def auth_headers(conversation_id):
    # Generate a valid JWT for the test user.
    token = create_access_token(conversation_id["user_id"])

    return {
        "Authorization": f"Bearer {token}"
    }


def fake_generate_reply(messages, graph, conversation_id):
    assert messages[-1]["content"] == "Explain embeddings."
    assert graph is not None
    assert conversation_id == 1
    return "Embeddings represent data as numbers."


def _patch_graph_model_registry(monkeypatch, model_factory):
    registry = {
        model_id: model_factory()
        for model_id in (
            ModelPreference.GPT_OSS_20B,
            ModelPreference.GPT_OSS_120B,
            ModelPreference.QWEN_VISION,
        )
    }

    def build_registry(settings, tools):
        if tools:
            return {
                model_id: model.bind_tools(tools)
                for model_id, model in registry.items()
            }
        return registry

    monkeypatch.setattr(
        graph_service,
        "build_model_registry",
        build_registry,
    )


def test_chat_conversation_not_found(client, auth_headers):
    response = client.post(
        "/api/v1/conversations/999999/chat",
        json={"content": "Hello"},
        headers=auth_headers,
    )

    assert response.status_code == 404


def test_chat_openapi_documents_json_and_image_uploads(client):
    schema = client.get("/openapi.json").json()

    for path in (
        "/api/v1/conversations/{conversation_id}/chat",
        "/api/v1/conversations/{conversation_id}/chat/stream",
    ):
        content = schema["paths"][path]["post"]["requestBody"]["content"]
        assert "application/json" in content
        assert "multipart/form-data" in content



def test_chat_ai_failure(client, conversation_id, auth_headers, monkeypatch):
    async def fake_generate_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        raise RuntimeError("Simulated AI failure")

    monkeypatch.setattr(chat_routes, "generate_reply", fake_generate_reply)

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        json={"content": "Hello"},
        headers=auth_headers,
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Could not generate an AI response. "
        "Check the backend configuration and logs."
    )


def test_chat_success_and_message_persistence(
    client, conversation_id, auth_headers, monkeypatch
):
    async def fake_generate_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Explain embeddings."
        assert graph is not None
        return "Embeddings represent data as numerical vectors."

    monkeypatch.setattr(chat_routes, "generate_reply", fake_generate_reply)

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        json={"content": "Explain embeddings."},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["user_message"]["content"] == "Explain embeddings."
    assert data["user_message"]["role"] == "user"
    assert (
        data["assistant_message"]["content"]
        == "Embeddings represent data as numerical vectors."
    )
    assert data["assistant_message"]["role"] == "assistant"

    history_response = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    )

    assert history_response.status_code == 200
    history = history_response.json()

    assert len(history) == 2
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[0]["content"] == "Explain embeddings."
    assert (
        history[1]["content"]
        == "Embeddings represent data as numerical vectors."
    )


def test_chat_passes_manual_model_preference_to_generation(
    client, conversation_id, auth_headers, monkeypatch
):
    observed_preferences = []

    async def fake_generate_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        observed_preferences.append(model_preference)
        return "Generated with the selected model."

    monkeypatch.setattr(chat_routes, "generate_reply", fake_generate_reply)

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        json={
            "content": "Explain a complex architecture.",
            "model_preference": "openai/gpt-oss-120b",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert observed_preferences == ["openai/gpt-oss-120b"]


def test_chat_rejects_another_users_conversation(
    client, conversation_id, auth_headers
):
    # Create a real second user in the test database.
    db = next(app.dependency_overrides[get_db]())
    try:
        other_user = User(email="other-chat-test@example.com")
        db.add(other_user)
        db.commit()
        db.refresh(other_user)
        other_user_id = other_user.id
    finally:
        db.close()

    other_user_token = create_access_token(other_user_id)
    other_user_headers = {
        "Authorization": f"Bearer {other_user_token}"
    }

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        data={"content": "Access someone else's conversation"},
        files=[("images", ("private.png", b"invalid", "image/png"))],
        headers=other_user_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_search_conversations_matches_titles_and_saved_messages(
    client, conversation_id, auth_headers
):
    db = next(app.dependency_overrides[get_db]())
    try:
        message_conversation = Conversation(
            user_id=conversation_id["user_id"],
            title="An unrelated title",
        )
        title_conversation = Conversation(
            user_id=conversation_id["user_id"],
            title="A Partial FIND in the title",
        )
        db.add_all([message_conversation, title_conversation])
        db.flush()
        db.add(
            Message(
                conversation_id=message_conversation.id,
                role="user",
                content="This saved message mentions AncientVioletBird.",
            )
        )
        db.commit()
        message_conversation_id = message_conversation.id
        title_conversation_id = title_conversation.id
    finally:
        db.close()

    title_response = client.get(
        "/api/v1/conversations/search",
        params={"q": "partial find"},
        headers=auth_headers,
    )
    message_response = client.get(
        "/api/v1/conversations/search",
        params={"q": "violet"},
        headers=auth_headers,
    )

    assert title_response.status_code == 200
    assert [item["id"] for item in title_response.json()] == [
        title_conversation_id
    ]
    assert message_response.status_code == 200
    assert [item["id"] for item in message_response.json()] == [
        message_conversation_id
    ]


def test_search_conversations_blank_query_returns_empty_list(
    client, auth_headers
):
    response = client.get(
        "/api/v1/conversations/search",
        params={"q": "   "},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == []


def test_search_conversations_only_returns_current_users_results(
    client, conversation_id, auth_headers
):
    db = next(app.dependency_overrides[get_db]())
    try:
        other_user = User(email="other-search-test@example.com")
        db.add(other_user)
        db.flush()
        private_conversation = Conversation(
            user_id=other_user.id,
            title="Secret shared keyword",
        )
        db.add(private_conversation)
        db.commit()
    finally:
        db.close()

    response = client.get(
        "/api/v1/conversations/search",
        params={"q": "shared keyword"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == []


def test_search_conversations_requires_authentication(client):
    response = client.get(
        "/api/v1/conversations/search",
        params={"q": "search"},
    )

    assert response.status_code == 401



def test_chat_requires_authentication(client, conversation_id):
    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        json={"content": "Hello"},
    )

    assert response.status_code == 401


def _read_sse_events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_chat_streams_and_persists_a_single_assistant_message(
    client, conversation_id, auth_headers, monkeypatch
):
    generated_messages = []
    synchronized_history = []

    async def fake_stream_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        generated_messages.extend(messages)
        assert graph is not None
        yield "A streamed "
        yield "response."

    async def fake_synchronize_graph_history(
        messages, graph, conversation_id
    ):
        synchronized_history.extend(messages)

    monkeypatch.setattr(chat_routes, "stream_reply", fake_stream_reply)
    monkeypatch.setattr(
        chat_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat/stream",
        json={"content": "Stream this answer"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _read_sse_events(response)
    assert [event["type"] for event in events] == [
        "user",
        "token",
        "token",
        "complete",
    ]
    assert generated_messages[-1] == {
        "role": "user",
        "content": "Stream this answer",
    }
    assert events[1]["content"] + events[2]["content"] == "A streamed response."
    assert events[0]["user_message"]["content"] == "Stream this answer"
    assert events[3]["assistant_message"]["content"] == "A streamed response."
    assert synchronized_history == [
        {"role": "user", "content": "Stream this answer"},
        {"role": "assistant", "content": "A streamed response."},
    ]

    history_response = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    )
    history = history_response.json()
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[1]["content"] == "A streamed response."


def test_chat_stream_forwards_tool_events_but_persists_only_assistant_text(
    client, conversation_id, auth_headers, monkeypatch
):
    async def fake_stream_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        yield {
            "type": "tool_start",
            "tool": "calculator",
            "tool_call_id": "call-1",
        }
        yield {"type": "token", "content": "The answer is "}
        yield {
            "type": "tool_result",
            "tool": "calculator",
            "tool_call_id": "call-1",
            "content": {"value": 42},
            "status": "success",
        }
        yield {"type": "token", "content": "42."}

    async def fake_synchronize_graph_history(
        messages, graph, conversation_id
    ):
        return None

    monkeypatch.setattr(chat_routes, "stream_reply", fake_stream_reply)
    monkeypatch.setattr(
        chat_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat/stream",
        json={"content": "What is the answer?"},
        headers=auth_headers,
    )

    events = _read_sse_events(response)
    assert [event["type"] for event in events] == [
        "user",
        "tool_start",
        "token",
        "tool_result",
        "token",
        "complete",
    ]
    assert events[1]["tool"] == "calculator"
    assert events[3]["content"] == {"value": 42}
    assert events[-1]["assistant_message"]["content"] == "The answer is 42."

    history_response = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    )
    history = history_response.json()
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[-1]["content"] == "The answer is 42."


def test_chat_stream_reset_discards_failed_attempt_before_persistence(
    client, conversation_id, auth_headers, monkeypatch
):
    async def fake_stream_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        yield {"type": "token", "content": "Partial failed answer"}
        yield {"type": "reset"}
        yield {"type": "token", "content": "Successful fallback answer"}

    async def fake_synchronize_graph_history(
        messages, graph, conversation_id
    ):
        return None

    monkeypatch.setattr(chat_routes, "stream_reply", fake_stream_reply)
    monkeypatch.setattr(
        chat_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat/stream",
        json={
            "content": "Use a fallback model",
            "model_preference": "openai/gpt-oss-20b",
        },
        headers=auth_headers,
    )

    events = _read_sse_events(response)
    assert [event["type"] for event in events] == [
        "user",
        "token",
        "reset",
        "token",
        "complete",
    ]
    assert events[-1]["assistant_message"]["content"] == (
        "Successful fallback answer"
    )

    history = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    ).json()
    assert [item["content"] for item in history] == [
        "Use a fallback model",
        "Successful fallback answer",
    ]


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [""],
        [{"type": "token", "content": {"text": "not a string"}}],
    ],
)
def test_chat_stream_rejects_empty_or_malformed_chunks_without_persisting(
    client, conversation_id, auth_headers, monkeypatch, chunks
):
    async def fake_stream_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        for chunk in chunks:
            yield chunk

    async def fake_synchronize_graph_history(
        messages, graph, conversation_id
    ):
        return None

    monkeypatch.setattr(chat_routes, "stream_reply", fake_stream_reply)
    monkeypatch.setattr(
        chat_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat/stream",
        json={"content": "Handle an empty response"},
        headers=auth_headers,
    )
    events = _read_sse_events(response)
    assert events[-1]["type"] == "error"

    history_response = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    )
    history = history_response.json()
    assert [message["role"] for message in history] == ["user"]


def test_chat_stream_failure_does_not_persist_partial_assistant_message(
    client, conversation_id, auth_headers, monkeypatch
):
    restored_history = []

    async def failing_stream_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        yield "Incomplete "
        raise RuntimeError("Simulated provider failure")

    async def fake_synchronize_graph_history(
        messages, graph, conversation_id
    ):
        restored_history.extend(messages)

    monkeypatch.setattr(chat_routes, "stream_reply", failing_stream_reply)
    monkeypatch.setattr(
        chat_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat/stream",
        json={"content": "Fail after a partial response"},
        headers=auth_headers,
    )

    events = _read_sse_events(response)
    assert [event["type"] for event in events] == ["user", "token", "error"]
    assert events[1]["content"] == "Incomplete "
    assert restored_history == [
        {
            "role": "user",
            "content": "Fail after a partial response",
        }
    ]

    history_response = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    )
    history = history_response.json()
    assert len(history) == 1
    assert history[0]["role"] == "user"


def test_chat_stream_disconnect_restores_checkpoint_without_persisting_partial(
    client, conversation_id, monkeypatch
):
    db = next(app.dependency_overrides[get_db]())
    try:
        user_message = Message(
            conversation_id=conversation_id["id"],
            role="user",
            content="Disconnect during generation",
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)

        restored_history = []

        async def fake_stream_reply(
            *, messages, graph, conversation_id, model_preference="auto"
        ):
            yield "Partial answer"
            yield " must not be persisted"

        async def fake_synchronize_graph_history(
            messages, graph, conversation_id
        ):
            restored_history.extend(messages)

        checks = 0

        async def is_disconnected():
            nonlocal checks
            checks += 1
            return checks > 1

        monkeypatch.setattr(chat_routes, "stream_reply", fake_stream_reply)
        monkeypatch.setattr(
            chat_routes,
            "synchronize_graph_history",
            fake_synchronize_graph_history,
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(graph=object()),
            ),
            is_disconnected=is_disconnected,
        )
        response = chat_routes.stream_response_for_message(
            request,
            db,
            conversation_id["id"],
            user_message,
            [user_message],
        )

        async def consume_until_disconnect():
            emitted = []
            with pytest.raises(asyncio.CancelledError):
                async for event in response.body_iterator:
                    emitted.append(event)
            return emitted

        emitted = asyncio.run(consume_until_disconnect())
        assert len(emitted) == 2
        assert restored_history == [
            {
                "role": "user",
                "content": "Disconnect during generation",
            }
        ]
        persisted = db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id["id"])
            .order_by(Message.id.asc())
        ).all()
        assert [message.role for message in persisted] == ["user"]
    finally:
        db.close()


def test_retry_stream_reuses_existing_user_message(
    client, conversation_id, auth_headers, monkeypatch
):
    db = next(app.dependency_overrides[get_db]())
    try:
        user_message = Message(
            conversation_id=conversation_id["id"],
            role="user",
            content="Retry this answer",
        )
        db.add(user_message)
        db.commit()
        user_message_id = user_message.id
    finally:
        db.close()

    async def fake_stream_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        assert messages[-1]["content"] == "Retry this answer"
        yield "Recovered response."

    async def fake_synchronize_graph_history(
        messages, graph, conversation_id
    ):
        return None

    monkeypatch.setattr(chat_routes, "stream_reply", fake_stream_reply)
    monkeypatch.setattr(
        chat_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/messages/"
        f"{user_message_id}/retry/stream",
        headers=auth_headers,
    )

    assert response.status_code == 200
    events = _read_sse_events(response)
    assert [event["type"] for event in events] == [
        "user",
        "token",
        "complete",
    ]
    assert events[0]["user_message"]["id"] == user_message_id
    assert events[-1]["assistant_message"]["content"] == "Recovered response."

    history_response = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    )
    history = history_response.json()
    assert len(history) == 2
    assert sum(item["role"] == "user" for item in history) == 1
    assert history[1]["role"] == "assistant"


def test_graph_stream_yields_assistant_text_and_ignores_tool_events():
    class FakeGraph:
        async def astream(
            self, input, config, stream_mode, version
        ):
            assert stream_mode == ["messages", "updates", "custom"]
            assert version == "v2"
            yield {
                "type": "messages",
                "data": (
                    SimpleNamespace(content="Partial "),
                    {"langgraph_node": "assistant"},
                ),
            }
            yield {
                "type": "messages",
                "data": (
                    SimpleNamespace(content="tool output"),
                    {"langgraph_node": "tools"},
                ),
            }
            yield {
                "type": "messages",
                "data": (
                    SimpleNamespace(content=[{"type": "text", "text": "answer"}]),
                    {"langgraph_node": "assistant"},
                ),
            }
            yield {
                "type": "updates",
                "data": {
                    "assistant": {
                        "messages": [
                            SimpleNamespace(
                                tool_calls=[
                                    {
                                        "name": "calculator",
                                        "id": "call-1",
                                    }
                                ]
                            )
                        ]
                    }
                },
            }
            yield {
                "type": "updates",
                "data": {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                content=[{"type": "text", "text": "42"}],
                                tool_call_id="call-1",
                                name="calculator",
                            )
                        ]
                    }
                },
            }

    async def collect():
        return [
            chunk
            async for chunk in stream_graph_reply(
                [{"role": "user", "content": "Question"}],
                FakeGraph(),
                conversation_id=11,
            )
        ]

    assert asyncio.run(collect()) == [
        {"type": "token", "content": "Partial "},
        {"type": "token", "content": "answer"},
        {
            "type": "tool_start",
            "tool": "calculator",
            "tool_call_id": "call-1",
        },
        {
            "type": "tool_result",
            "tool": "calculator",
            "tool_call_id": "call-1",
            "content": [{"type": "text", "text": "42"}],
            "status": "success",
        },
    ]


def test_graph_stream_rejects_an_empty_model_response():
    class EmptyGraph:
        async def astream(self, input, config, stream_mode, version):
            if False:
                yield None

    async def collect():
        return [
            chunk
            async for chunk in stream_graph_reply(
                [{"role": "user", "content": "Question"}],
                EmptyGraph(),
                conversation_id=12,
            )
        ]

    with pytest.raises(RuntimeError, match="empty response"):
        asyncio.run(collect())


@pytest.mark.parametrize(
    ("failure_case", "failed_attempt_content"),
    [
        ("provider_error", "Partial small answer"),
        ("empty_response", ""),
        ("output_limit", "Partial small answer"),
    ],
)
def test_graph_falls_back_to_large_model_and_resets_partial_stream(
    monkeypatch, failure_case, failed_attempt_content
):
    calls = []

    class FakeModel(BaseChatModel):
        model_id: ModelPreference

        @property
        def _llm_type(self):
            return "routing-test"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content="Complete large answer")
                    )
                ]
            )

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            calls.append(self.model_id)
            if self.model_id is ModelPreference.GPT_OSS_20B:
                if failure_case == "empty_response":
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content="")
                    )
                    return
                if failure_case == "output_limit":
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="Partial small answer",
                            response_metadata={"finish_reason": "length"},
                        )
                    )
                    return
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="Partial small answer")
                )
                raise RuntimeError("Simulated provider error")
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="Complete large answer")
            )

    monkeypatch.setattr(
        graph_service,
        "get_settings",
        lambda: SimpleNamespace(
            groq_api_key="test-key",
            groq_model="openai/gpt-oss-120b",
            groq_default_temperature=0.7,
            groq_temperature_override=None,
        ),
    )
    registry = {
        ModelPreference.GPT_OSS_20B: FakeModel(
            model_id=ModelPreference.GPT_OSS_20B
        ),
        ModelPreference.GPT_OSS_120B: FakeModel(
            model_id=ModelPreference.GPT_OSS_120B
        ),
    }
    monkeypatch.setattr(
        graph_service,
        "build_model_registry",
        lambda settings, tools: registry,
    )
    graph = graph_service.build_graph()

    async def collect():
        return [
            event
            async for event in stream_graph_reply(
                [{"role": "user", "content": "Hello"}],
                graph,
                conversation_id=61,
                model_preference=ModelPreference.GPT_OSS_20B.value,
            )
        ]

    events = asyncio.run(collect())

    assert calls == [
        ModelPreference.GPT_OSS_20B,
        ModelPreference.GPT_OSS_120B,
    ]
    expected_events = []
    if failed_attempt_content:
        expected_events.append(
            {"type": "token", "content": failed_attempt_content}
        )
    expected_events.extend(
        [
            {"type": "reset"},
            {"type": "token", "content": "Complete large answer"},
        ]
    )
    assert events == expected_events


def test_graph_does_not_fallback_after_tool_execution(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, model_id):
            self.model_id = model_id

        def bind(self, **kwargs):
            return self

        async def astream(self, messages):
            calls.append(self.model_id)
            if self.model_id is ModelPreference.GPT_OSS_20B:
                raise RuntimeError("Failure after a tool result")
            yield AIMessageChunk(content="Must not run")

    monkeypatch.setattr(
        graph_service,
        "get_settings",
        lambda: SimpleNamespace(
            groq_api_key="test-key",
            groq_default_temperature=0.7,
            groq_temperature_override=None,
        ),
    )
    registry = {
        ModelPreference.GPT_OSS_20B: FakeModel(
            ModelPreference.GPT_OSS_20B
        ),
        ModelPreference.GPT_OSS_120B: FakeModel(
            ModelPreference.GPT_OSS_120B
        ),
    }
    monkeypatch.setattr(
        graph_service,
        "build_model_registry",
        lambda settings, tools: registry,
    )
    graph = graph_service.build_graph()

    async def invoke():
        return await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Calculate the result."),
                    ToolMessage(
                        content="4",
                        tool_call_id="call-1",
                        name="calculator",
                    ),
                ],
                "model_preference": ModelPreference.GPT_OSS_20B.value,
                "selected_model": ModelPreference.GPT_OSS_20B.value,
                "routing_reason": "manual model selection",
                "fallback_used": False,
            }
        )

    with pytest.raises(RuntimeError, match="Failure after a tool result"):
        asyncio.run(invoke())

    assert calls == [ModelPreference.GPT_OSS_20B]


def test_non_stream_graph_generation_aggregates_groq_stream_chunks(monkeypatch):
    class FakeChatGroq:
        def __init__(self, **kwargs):
            assert kwargs["model"] in {
                ModelPreference.GPT_OSS_20B.value,
                ModelPreference.GPT_OSS_120B.value,
                ModelPreference.QWEN_VISION.value,
            }
            assert kwargs["temperature"] == 0.7
            assert kwargs["model_kwargs"] == {
                "max_completion_tokens": 4096,
            }
            self.temperature = None

        def bind(self, **kwargs):
            self.temperature = kwargs["temperature"]
            return self

        async def astream(self, messages):
            assert messages[-1].content == "Question"
            assert self.temperature == 0.3
            yield AIMessageChunk(content="Complete ")
            yield AIMessageChunk(content="answer.")

    monkeypatch.setattr(
        graph_service,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "groq_api_key": "test-key",
                "groq_model": "test-model",
                "groq_default_temperature": 0.7,
                "groq_temperature_override": None,
            },
        )(),
    )
    monkeypatch.setattr(model_router, "ChatGroq", FakeChatGroq)

    graph = graph_service.build_graph()

    async def invoke_graph():
        return await graph.ainvoke(
            {"messages": [HumanMessage(content="Question")]}
        )

    result = asyncio.run(invoke_graph())
    assert result["messages"][-1].content == "Complete answer."


def test_graph_uses_isolated_temperature_per_concurrent_request(monkeypatch):
    observed = []

    class FakeBoundModel:
        def __init__(self, temperature):
            self.temperature = temperature

        async def astream(self, messages):
            await asyncio.sleep(0)
            prompt = next(
                message.content
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            )
            observed.append((prompt, self.temperature))
            yield AIMessageChunk(content="response")

    class FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind(self, **kwargs):
            return FakeBoundModel(kwargs["temperature"])

    monkeypatch.setattr(
        graph_service,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "groq_api_key": "test-key",
                "groq_model": "test-model",
                "groq_default_temperature": 0.7,
                "groq_temperature_override": None,
            },
        )(),
    )
    _patch_graph_model_registry(monkeypatch, FakeChatGroq)
    graph = graph_service.build_graph()

    async def invoke(prompt, thread_id, override=None):
        config = {"configurable": {"thread_id": thread_id}}
        if override is not None:
            config["configurable"]["temperature_override"] = override
        return await graph.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )

    async def run_concurrently():
        return await asyncio.gather(
            invoke("Calculate 2 + 2.", "math"),
            invoke("Write a poem about rain.", "creative", 0.6),
        )

    asyncio.run(run_concurrently())

    assert sorted(observed) == sorted(
        [
            ("Calculate 2 + 2.", 0.1),
            ("Write a poem about rain.", 0.6),
        ]
    )


def test_graph_uses_configured_manual_temperature_override(monkeypatch):
    observed = []

    class FakeBoundModel:
        def __init__(self, temperature):
            self.temperature = temperature

        async def astream(self, messages):
            observed.append(self.temperature)
            yield AIMessageChunk(content="response")

    class FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind(self, **kwargs):
            return FakeBoundModel(kwargs["temperature"])

    monkeypatch.setattr(
        graph_service,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "groq_api_key": "test-key",
                "groq_model": "test-model",
                "groq_default_temperature": 0.7,
                "groq_temperature_override": 1.35,
            },
        )(),
    )
    _patch_graph_model_registry(monkeypatch, FakeChatGroq)
    graph = graph_service.build_graph()

    async def invoke():
        return await graph.ainvoke(
            {"messages": [HumanMessage(content="Calculate 2 + 2.")]},
        )

    asyncio.run(invoke())

    assert observed == [1.35]


def test_graph_stream_uses_selected_temperature(monkeypatch):
    selected_temperatures = []

    class FakeBoundModel:
        def __init__(self, temperature):
            self.temperature = temperature

        async def astream(self, messages):
            selected_temperatures.append(self.temperature)
            yield AIMessageChunk(content="The result is 4.")

    class FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind(self, **kwargs):
            return FakeBoundModel(kwargs["temperature"])

    monkeypatch.setattr(
        graph_service,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "groq_api_key": "test-key",
                "groq_model": "test-model",
                "groq_default_temperature": 0.7,
                "groq_temperature_override": None,
            },
        )(),
    )
    _patch_graph_model_registry(monkeypatch, FakeChatGroq)
    graph = graph_service.build_graph()

    async def collect():
        return [
            event
            async for event in stream_graph_reply(
                [{"role": "user", "content": "Calculate 2 + 2."}],
                graph,
                conversation_id=25,
            )
        ]

    events = asyncio.run(collect())

    assert selected_temperatures == [0.1]
    assert events == [{"type": "token", "content": "The result is 4."}]


def test_graph_applies_low_temperature_to_mcp_math_tool_calls(monkeypatch):
    from langchain_core.tools import tool

    selected_temperatures = []

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    class FakeBoundModel:
        def __init__(self, temperature):
            self.temperature = temperature

        async def astream(self, messages):
            selected_temperatures.append(self.temperature)
            if any(isinstance(message, ToolMessage) for message in messages):
                yield AIMessageChunk(content="4")
                return
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "add",
                        "args": '{"a": 2, "b": 2}',
                        "id": "call-add",
                        "index": 0,
                    }
                ],
            )

    class FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools):
            assert tools == [add]
            return self

        def bind(self, **kwargs):
            return FakeBoundModel(kwargs["temperature"])

    monkeypatch.setattr(
        graph_service,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "groq_api_key": "test-key",
                "groq_model": "test-model",
                "groq_default_temperature": 0.7,
                "groq_temperature_override": None,
            },
        )(),
    )
    _patch_graph_model_registry(monkeypatch, FakeChatGroq)
    graph = graph_service.build_graph(tools=[add])

    async def invoke():
        return await graph.ainvoke(
            {"messages": [HumanMessage(content="Calculate 2 + 2.")]},
            config={"configurable": {"thread_id": "mcp-math"}},
        )

    result = asyncio.run(invoke())

    assert result["messages"][-1].content == "4"
    assert selected_temperatures == [0.1, 0.1]


def test_chat_stream_requires_authentication(client, conversation_id):
    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat/stream",
        json={"content": "Unauthenticated"},
    )

    assert response.status_code == 401


def test_regenerate_updates_existing_assistant_message_and_checkpoint(
    client, conversation_id, auth_headers, monkeypatch
):
    db = next(app.dependency_overrides[get_db]())
    try:
        user_message = Message(
            conversation_id=conversation_id["id"],
            role="user",
            content="Original prompt",
        )
        assistant_message = Message(
            conversation_id=conversation_id["id"],
            role="assistant",
            content="Original answer",
        )
        later_user_message = Message(
            conversation_id=conversation_id["id"],
            role="user",
            content="A later prompt",
        )
        later_assistant_message = Message(
            conversation_id=conversation_id["id"],
            role="assistant",
            content="A later answer",
        )
        db.add_all(
            [
                user_message,
                assistant_message,
                later_user_message,
                later_assistant_message,
            ]
        )
        db.commit()
        assistant_message_id = assistant_message.id
    finally:
        db.close()

    generated_messages = []
    synchronized_history = []

    async def fake_generate_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        generated_messages.extend(messages)
        assert conversation_id == conversation_id_value
        assert graph is not None
        return "Regenerated answer"

    async def fake_synchronize_graph_history(
        messages, graph, conversation_id
    ):
        synchronized_history.extend(messages)

    conversation_id_value = conversation_id["id"]
    monkeypatch.setattr(messages_routes, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(
        messages_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id_value}/messages/"
        f"{assistant_message_id}/regenerate",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["id"] == assistant_message_id
    assert response.json()["content"] == "Regenerated answer"
    assert generated_messages == [{"role": "user", "content": "Original prompt"}]
    assert synchronized_history == [
        {"role": "user", "content": "Original prompt"},
        {"role": "assistant", "content": "Regenerated answer"},
        {"role": "user", "content": "A later prompt"},
        {"role": "assistant", "content": "A later answer"},
    ]

    history_response = client.get(
        f"/api/v1/conversations/{conversation_id_value}/messages",
        headers=auth_headers,
    )
    history = history_response.json()
    assert len(history) == 4
    assert history[1]["id"] == assistant_message_id
    assert history[1]["content"] == "Regenerated answer"
    assert [item["content"] for item in history[2:]] == [
        "A later prompt",
        "A later answer",
    ]


def test_regenerate_rejects_non_assistant_message(
    client, conversation_id, auth_headers
):
    db = next(app.dependency_overrides[get_db]())
    try:
        message = Message(
            conversation_id=conversation_id["id"],
            role="user",
            content="A user prompt",
        )
        db.add(message)
        db.commit()
        message_id = message.id
    finally:
        db.close()

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/messages/"
        f"{message_id}/regenerate",
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only assistant messages can be regenerated."
    )


def test_edit_message_replaces_response_and_removes_later_messages(
    client, conversation_id, auth_headers, monkeypatch
):
    db = next(app.dependency_overrides[get_db]())
    try:
        messages = [
            Message(
                conversation_id=conversation_id["id"],
                role="user",
                content="Original prompt",
            ),
            Message(
                conversation_id=conversation_id["id"],
                role="assistant",
                content="Original answer",
            ),
            Message(
                conversation_id=conversation_id["id"],
                role="user",
                content="Later prompt",
            ),
        ]
        db.add_all(messages)
        db.commit()
        edited_message_id = messages[0].id
    finally:
        db.close()

    generated_messages = []

    async def fake_generate_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        generated_messages.extend(messages)
        assert graph is not None
        assert conversation_id == conversation_id_value
        return "Updated answer"

    conversation_id_value = conversation_id["id"]
    monkeypatch.setattr(messages_routes, "generate_reply", fake_generate_reply)

    response = client.put(
        f"/api/v1/conversations/{conversation_id_value}/messages/"
        f"{edited_message_id}",
        json={"content": "Edited prompt"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["user_message"]["content"] == "Edited prompt"
    assert response.json()["assistant_message"]["content"] == "Updated answer"
    assert generated_messages[-1] == {
        "role": "user",
        "content": "Edited prompt",
    }

    history_response = client.get(
        f"/api/v1/conversations/{conversation_id_value}/messages",
        headers=auth_headers,
    )
    assert [
        (message["role"], message["content"])
        for message in history_response.json()
    ] == [
        ("user", "Edited prompt"),
        ("assistant", "Updated answer"),
    ]


def test_edit_message_failure_restores_database_and_graph_history(
    client, conversation_id, auth_headers, monkeypatch
):
    db = next(app.dependency_overrides[get_db]())
    try:
        messages = [
            Message(
                conversation_id=conversation_id["id"],
                role="user",
                content="Keep this prompt",
            ),
            Message(
                conversation_id=conversation_id["id"],
                role="assistant",
                content="Keep this answer",
            ),
            Message(
                conversation_id=conversation_id["id"],
                role="user",
                content="Keep this later prompt",
            ),
        ]
        db.add_all(messages)
        db.commit()
        edited_message_id = messages[0].id
    finally:
        db.close()

    restored_history = []

    async def fail_generate_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        raise RuntimeError("Simulated provider failure")

    async def fake_synchronize_graph_history(
        messages, graph, conversation_id
    ):
        restored_history.extend(messages)

    monkeypatch.setattr(messages_routes, "generate_reply", fail_generate_reply)
    monkeypatch.setattr(
        messages_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )

    response = client.put(
        f"/api/v1/conversations/{conversation_id['id']}/messages/"
        f"{edited_message_id}",
        json={"content": "Edited prompt"},
        headers=auth_headers,
    )

    assert response.status_code == 502
    assert restored_history == [
        {"role": "user", "content": "Keep this prompt"},
        {"role": "assistant", "content": "Keep this answer"},
        {"role": "user", "content": "Keep this later prompt"},
    ]

    history_response = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    )
    assert [message["content"] for message in history_response.json()] == [
        "Keep this prompt",
        "Keep this answer",
        "Keep this later prompt",
    ]


def test_regenerate_failure_preserves_existing_message(
    client, conversation_id, auth_headers, monkeypatch
):
    db = next(app.dependency_overrides[get_db]())
    try:
        user_message = Message(
            conversation_id=conversation_id["id"],
            role="user",
            content="Keep this prompt",
        )
        assistant_message = Message(
            conversation_id=conversation_id["id"],
            role="assistant",
            content="Keep this answer",
        )
        db.add_all([user_message, assistant_message])
        db.commit()
        assistant_message_id = assistant_message.id
    finally:
        db.close()

    restored_history = []

    async def fail_generate_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        raise RuntimeError("Simulated provider failure")

    async def fake_synchronize_graph_history(
        messages, graph, conversation_id
    ):
        restored_history.extend(messages)

    monkeypatch.setattr(messages_routes, "generate_reply", fail_generate_reply)
    monkeypatch.setattr(
        messages_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/messages/"
        f"{assistant_message_id}/regenerate",
        headers=auth_headers,
    )

    assert response.status_code == 502
    assert restored_history == [
        {"role": "user", "content": "Keep this prompt"},
        {"role": "assistant", "content": "Keep this answer"},
    ]

    db = next(app.dependency_overrides[get_db]())
    try:
        persisted = db.get(Message, assistant_message_id)
        assert persisted is not None
        assert persisted.content == "Keep this answer"
    finally:
        db.close()


def test_regenerate_checkpoint_failure_preserves_existing_message(
    client, conversation_id, auth_headers, monkeypatch
):
    db = next(app.dependency_overrides[get_db]())
    try:
        user_message = Message(
            conversation_id=conversation_id["id"],
            role="user",
            content="Keep this prompt",
        )
        assistant_message = Message(
            conversation_id=conversation_id["id"],
            role="assistant",
            content="Keep this answer",
        )
        db.add_all([user_message, assistant_message])
        db.commit()
        assistant_message_id = assistant_message.id
    finally:
        db.close()

    sync_attempts = []

    async def fake_generate_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        return "Unsaved regenerated answer"

    async def fail_once_then_restore(messages, graph, conversation_id):
        sync_attempts.append(messages)
        if len(sync_attempts) == 1:
            raise RuntimeError("Simulated checkpoint failure")

    monkeypatch.setattr(messages_routes, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(
        messages_routes,
        "synchronize_graph_history",
        fail_once_then_restore,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/messages/"
        f"{assistant_message_id}/regenerate",
        headers=auth_headers,
    )

    assert response.status_code == 502
    assert len(sync_attempts) == 2
    assert sync_attempts[0][1]["content"] == "Unsaved regenerated answer"
    assert sync_attempts[1][1]["content"] == "Keep this answer"

    db = next(app.dependency_overrides[get_db]())
    try:
        persisted = db.get(Message, assistant_message_id)
        assert persisted is not None
        assert persisted.content == "Keep this answer"
    finally:
        db.close()


def test_regenerate_rejects_another_users_conversation(
    client, conversation_id, auth_headers
):
    db = next(app.dependency_overrides[get_db]())
    try:
        other_user = User(email="other-regenerate-test@example.com")
        db.add(other_user)
        db.flush()
        private_conversation = Conversation(
            user_id=other_user.id,
            title="Private conversation",
        )
        db.add(private_conversation)
        db.flush()
        user_message = Message(
            conversation_id=private_conversation.id,
            role="user",
            content="Private prompt",
        )
        assistant_message = Message(
            conversation_id=private_conversation.id,
            role="assistant",
            content="Private answer",
        )
        db.add_all([user_message, assistant_message])
        db.commit()
        private_conversation_id = private_conversation.id
        assistant_message_id = assistant_message.id
    finally:
        db.close()

    response = client.post(
        f"/api/v1/conversations/{private_conversation_id}/messages/"
        f"{assistant_message_id}/regenerate",
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_chat_persists_attachment_and_sends_multimodal_history(
    client, conversation_id, auth_headers, monkeypatch
):
    observed_messages = []

    async def fake_generate_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        observed_messages.extend(messages)
        assert messages[-1]["content"] == "What is in this screenshot?"
        assert len(messages[-1]["attachments"]) == 2
        return "It is a red square."

    monkeypatch.setattr(chat_routes, "generate_reply", fake_generate_reply)
    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        data={"content": "What is in this screenshot?"},
        files=[
            ("images", ("screen.png", _png_bytes(), "image/png")),
            ("images", ("diagram.png", _png_bytes(), "image/png")),
        ],
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    attachment = data["user_message"]["attachments"][0]
    assert attachment["filename"] == "screen.png"
    assert attachment["content_type"] == "image/png"
    assert attachment["data_url"].startswith("data:image/png;base64,")
    assert len(data["user_message"]["attachments"]) == 2
    assert data["assistant_message"]["content"] == "It is a red square."

    history = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    ).json()
    assert history[0]["attachments"][0] == attachment
    assert len(history[0]["attachments"]) == 2
    assert history[1]["attachments"] == []


def test_image_chat_streams_and_persists_once(
    client, conversation_id, auth_headers, monkeypatch
):
    observed_messages = []
    synchronized_history = []

    async def fake_stream_reply(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        observed_messages.extend(messages)
        assert messages[-1]["content"] == "Please describe the attached image(s)."
        yield {"type": "token", "content": "A red "}
        yield {"type": "token", "content": "square."}

    async def fake_synchronize_graph_history(messages, graph, conversation_id):
        synchronized_history.extend(messages)

    monkeypatch.setattr(chat_routes, "stream_reply", fake_stream_reply)
    monkeypatch.setattr(
        chat_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )
    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat/stream",
        data={"content": ""},
        files=[("images", ("square.png", _png_bytes(), "image/png"))],
        headers=auth_headers,
    )

    events = _read_sse_events(response)
    assert [event["type"] for event in events] == [
        "user",
        "token",
        "token",
        "complete",
    ]
    assert events[0]["user_message"]["attachments"][0]["filename"] == "square.png"
    assert events[-1]["assistant_message"]["content"] == "A red square."
    assert observed_messages[-1]["attachments"][0]["content_type"] == "image/png"
    assert synchronized_history[-1] == {
        "role": "assistant",
        "content": "A red square.",
    }

    persisted = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    ).json()
    assert [message["role"] for message in persisted] == ["user", "assistant"]
    assert sum(message["role"] == "assistant" for message in persisted) == 1
    assert persisted[0]["attachments"][0]["filename"] == "square.png"


def test_interrupted_image_stream_can_retry_without_duplicate_messages(
    client, conversation_id, auth_headers, monkeypatch
):
    attempts = 0
    observed_images = []

    async def stream_with_one_failure(
        *, messages, graph, conversation_id, model_preference="auto"
    ):
        nonlocal attempts
        attempts += 1
        observed_images.append(messages[-1]["attachments"][0]["filename"])
        if attempts == 1:
            yield {"type": "token", "content": "Incomplete "}
            raise RuntimeError("Simulated interrupted vision response.")
        yield {"type": "token", "content": "Recovered response."}

    async def fake_synchronize_graph_history(messages, graph, conversation_id):
        return None

    monkeypatch.setattr(chat_routes, "stream_reply", stream_with_one_failure)
    monkeypatch.setattr(
        chat_routes,
        "synchronize_graph_history",
        fake_synchronize_graph_history,
    )
    initial = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat/stream",
        data={"content": "Read this page."},
        files=[("images", ("page.png", _png_bytes(), "image/png"))],
        headers=auth_headers,
    )
    initial_events = _read_sse_events(initial)
    user_message_id = initial_events[0]["user_message"]["id"]

    incomplete_history = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    ).json()
    assert [message["role"] for message in incomplete_history] == ["user"]
    assert incomplete_history[0]["attachments"][0]["filename"] == "page.png"
    assert initial_events[-1]["type"] == "error"

    retry = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/messages/"
        f"{user_message_id}/retry/stream",
        json={"model_preference": "auto"},
        headers=auth_headers,
    )
    retry_events = _read_sse_events(retry)

    assert retry.status_code == 200
    assert retry_events[0]["user_message"]["id"] == user_message_id
    assert retry_events[-1]["assistant_message"]["content"] == (
        "Recovered response."
    )
    assert observed_images == ["page.png", "page.png"]
    completed_history = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    ).json()
    assert [message["role"] for message in completed_history] == [
        "user",
        "assistant",
    ]


def test_image_chat_rejects_invalid_and_oversized_uploads(
    client, conversation_id, auth_headers
):
    invalid = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        data={"content": "Describe this."},
        files=[("images", ("invalid.png", b"not an image", "image/png"))],
        headers=auth_headers,
    )
    oversized = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        data={"content": "Describe this."},
        files=[
            (
                "images",
                ("large.png", b"x" * (4 * 1024 * 1024 + 1), "image/png"),
            )
        ],
        headers=auth_headers,
    )

    assert invalid.status_code == 422
    assert oversized.status_code == 413


def test_image_chat_failure_does_not_persist_an_assistant_response(
    client, conversation_id, auth_headers, monkeypatch
):
    async def fail_generation(**kwargs):
        raise RuntimeError("Simulated vision provider failure.")

    monkeypatch.setattr(chat_routes, "generate_reply", fail_generation)
    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        data={"content": "Read the text in this image."},
        files=[("images", ("document.png", _png_bytes(), "image/png"))],
        headers=auth_headers,
    )

    assert response.status_code == 502
    history = client.get(
        f"/api/v1/conversations/{conversation_id['id']}/messages",
        headers=auth_headers,
    ).json()
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["attachments"][0]["filename"] == "document.png"


def test_image_request_history_uses_multimodal_content_and_qwen(monkeypatch):
    selected_models = []

    class FakeBoundModel:
        def __init__(self, model_id):
            self.model_id = model_id

        async def astream(self, messages):
            selected_models.append(self.model_id)
            content = messages[-1].content
            assert content[0] == {
                "type": "text",
                "text": "Describe this image.",
            }
            assert content[1]["type"] == "image_url"
            yield AIMessageChunk(content="A red square.")

    class FakeModel:
        def __init__(self, model_id):
            self.model_id = model_id

        def bind(self, **kwargs):
            return FakeBoundModel(self.model_id)

    monkeypatch.setattr(
        graph_service,
        "get_settings",
        lambda: SimpleNamespace(
            groq_api_key="test-key",
            groq_vision_model="qwen/test-vision",
            groq_default_temperature=0.7,
            groq_temperature_override=None,
        ),
    )
    monkeypatch.setattr(
        graph_service,
        "build_model_registry",
        lambda settings, tools: {
            model_id: FakeModel(model_id)
            for model_id in (
                ModelPreference.GPT_OSS_20B,
                ModelPreference.GPT_OSS_120B,
                ModelPreference.QWEN_VISION,
            )
        },
    )
    graph = graph_service.build_graph()
    image = {
        "filename": "square.png",
        "content_type": "image/png",
        "data_url": "data:image/png;base64,aGVsbG8=",
    }

    reply = asyncio.run(
        graph_service.generate_graph_reply(
            [
                {
                    "role": "user",
                    "content": "Describe this image.",
                    "attachments": [image],
                }
            ],
            graph,
            conversation_id=27,
        )
    )

    assert reply == "A red square."
    assert selected_models == [ModelPreference.QWEN_VISION]
