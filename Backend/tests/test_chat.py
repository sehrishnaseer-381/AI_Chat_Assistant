
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base, get_db
from app.models import Conversation, User
from app.api.routes import chat as chat_routes


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
    # Create a user and conversation in the test database.
    db = next(app.dependency_overrides[get_db]())

    try:
        user = User(email="chat-test@example.com")
        db.add(user)
        db.commit()
        db.refresh(user)

        conversation = Conversation(
            user_id=user.id,
            title="Test conversation",
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return conversation.id
    finally:
        db.close()


def test_chat_success(client, conversation_id, monkeypatch):
    def fake_generate_reply(messages):
        assert messages[-1]["content"] == "Explain embeddings."
        return "Embeddings represent data as numbers."

    monkeypatch.setattr(
        chat_routes,
        "generate_reply",
        fake_generate_reply,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/chat",
        json={"content": "Explain embeddings."},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["user_message"]["role"] == "user"
    assert data["user_message"]["content"] == "Explain embeddings."
    assert data["assistant_message"]["role"] == "assistant"
    assert (
        data["assistant_message"]["content"]
        == "Embeddings represent data as numbers."
    )


def test_chat_conversation_not_found(client):
    response = client.post(
        "/api/v1/conversations/999999/chat",
        json={"content": "Hello"},
    )

    assert response.status_code == 404


def test_chat_ai_failure(client, conversation_id, monkeypatch):
    def fake_generate_reply(messages):
        raise RuntimeError("Simulated AI failure")

    monkeypatch.setattr(
        chat_routes,
        "generate_reply",
        fake_generate_reply,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/chat",
        json={"content": "Hello"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Could not generate an AI response. "
        "Check the backend configuration and logs."
    )
