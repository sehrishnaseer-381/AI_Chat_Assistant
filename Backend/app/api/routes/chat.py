
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Conversation, Message
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.ai_service import generate_reply

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post(
    "/conversations/{conversation_id}/chat",
    response_model=ChatResponse,
)
def chat_with_ai(
    conversation_id: int,
    payload: ChatRequest,
    db: Session = Depends(get_db),
):
    conversation = db.get(Conversation, conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    # Save the user's message first.
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=payload.content,
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # Load the conversation history in chronological order.
    statement = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    history = list(db.scalars(statement).all())

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Give clear, accurate, and useful answers."
            ),
        }
    ]

    messages.extend(
        {"role": message.role, "content": message.content}
        for message in history
    )

    # Ask Groq to generate the assistant's response.
    try:
        reply = generate_reply(messages)
    except Exception as exc:
        logger.exception("AI response generation failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not generate an AI response. "
                "Check the backend configuration and logs."
            ),
        ) from exc

    # Save the assistant's response.
    assistant_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=reply,
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    return ChatResponse(
        user_message=user_message,
        assistant_message=assistant_message,
    )
