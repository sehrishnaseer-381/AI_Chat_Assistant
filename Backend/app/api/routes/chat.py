
import logging
from fastapi import Request
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import Conversation, Message, User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.ai_service import generate_reply

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post(
    "/conversations/{conversation_id}/chat",
    response_model=ChatResponse,
)
async def chat_with_ai(
    conversation_id: int,
    payload: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    ):
    # Verify that this conversation belongs to the logged-in user.
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    # Save the user's message.
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=payload.content,
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # Load conversation history chronologically.
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

    # Ask Groq to generate the response.
    try:
       reply = await generate_reply(
    messages=messages,
    graph=request.app.state.graph,
    conversation_id=conversation_id,
    )
    except Exception as exc:
        logger.exception("AI response generation failed")
        raise HTTPException(
            status_code=502,
            detail="Could not generate an AI response. "
                   "Check the backend configuration and logs.",
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
