
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base, get_db
from app.models import Conversation, User
from app.api.routes import chat as chat_routes
from app.core.security import create_access_token


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



def test_chat_requires_authentication(client, conversation_id):
    response = client.post(
        f"/api/v1/conversations/{conversation_id['id']}/chat",
        json={"content": "Hello"},
    )

    assert response.status_code == 401
