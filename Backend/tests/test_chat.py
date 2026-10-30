
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base, get_db
from app.models import Conversation, Message, User
from app.api.routes import chat as chat_routes
from app.api.routes import messages as messages_routes
from app.core.security import create_access_token
from app.services import graph_service
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


def test_chat_conversation_not_found(client, auth_headers):
    response = client.post(
        "/api/v1/conversations/999999/chat",
        json={"content": "Hello"},
        headers=auth_headers,
    )

    assert response.status_code == 404



def test_chat_ai_failure(client, conversation_id, auth_headers, monkeypatch):
    async def fake_generate_reply(*, messages, graph, conversation_id):
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
    async def fake_generate_reply(*, messages, graph, conversation_id):
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



def test_chat_rejects_another_users_conversation(
    client, conversation_id, auth_headers, monkeypatch
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
        json={"content": "Access someone else's conversation"},
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

    async def fake_stream_reply(*, messages, graph, conversation_id):
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


def test_chat_stream_failure_does_not_persist_partial_assistant_message(
    client, conversation_id, auth_headers, monkeypatch
):
    restored_history = []

    async def failing_stream_reply(*, messages, graph, conversation_id):
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

    async def fake_stream_reply(*, messages, graph, conversation_id):
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
            assert stream_mode == "messages"
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

    async def collect():
        return [
            chunk
            async for chunk in stream_graph_reply(
                [{"role": "user", "content": "Question"}],
                FakeGraph(),
                conversation_id=11,
            )
        ]

    assert asyncio.run(collect()) == ["Partial ", "answer"]


def test_non_stream_graph_generation_aggregates_groq_stream_chunks(monkeypatch):
    class FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        async def astream(self, messages):
            assert messages[-1].content == "Question"
            yield AIMessageChunk(content="Complete ")
            yield AIMessageChunk(content="answer.")

    monkeypatch.setattr(
        graph_service,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {"groq_api_key": "test-key", "groq_model": "test-model"},
        )(),
    )
    monkeypatch.setattr(graph_service, "ChatGroq", FakeChatGroq)

    graph = graph_service.build_graph()

    async def invoke_graph():
        return await graph.ainvoke(
            {"messages": [HumanMessage(content="Question")]}
        )

    result = asyncio.run(invoke_graph())
    assert result["messages"][-1].content == "Complete answer."


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

    async def fake_generate_reply(*, messages, graph, conversation_id):
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

    async def fail_generate_reply(*, messages, graph, conversation_id):
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

    async def fake_generate_reply(*, messages, graph, conversation_id):
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
