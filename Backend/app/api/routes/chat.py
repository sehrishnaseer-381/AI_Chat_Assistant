
import asyncio
import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import Conversation, Message, User
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.message import MessageRead
from app.services.ai_service import (
    generate_reply,
    stream_reply,
    synchronize_graph_history,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


def generate_conversation_title(message: str) -> str:
    """Create a short title from the first user message."""
    title = re.sub(r"\s+", " ", message).strip()

    # Remove common punctuation from the ends.
    title = title.strip(" \t\n\r.,!?;:-")

    if not title:
        return "New chat"

    # Keep the title short and within the database's 200-character limit.
    max_length = 60
    if len(title) > max_length:
        title = title[:max_length].rsplit(" ", 1)[0].strip()

    return title or "New chat"


def sse_event(event_type: str, payload: dict) -> str:
    data = json.dumps(
        {"type": event_type, **jsonable_encoder(payload)},
        ensure_ascii=False,
    )
    return f"data: {data}\n\n"


def stream_response_for_message(
    request: Request,
    db: Session,
    conversation_id: int,
    user_message: Message,
    history: list[Message],
) -> StreamingResponse:
    graph = request.app.state.graph
    persisted_history = [
        {"role": item.role, "content": item.content}
        for item in history
    ]
    user_message_data = MessageRead.model_validate(
        user_message
    ).model_dump(mode="json")

    async def restore_checkpoint_from_database():
        db.rollback()
        current_history = list(
            db.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc(), Message.id.asc())
            ).all()
        )
        await synchronize_graph_history(
            [
                {"role": item.role, "content": item.content}
                for item in current_history
            ],
            graph,
            conversation_id,
        )

    async def event_stream():
        try:
            yield sse_event(
                "user",
                {"user_message": user_message_data},
            )

            response_parts = []
            messages = [
                {"role": item.role, "content": item.content}
                for item in history
            ]

            async for chunk in stream_reply(
                messages=messages,
                graph=graph,
                conversation_id=conversation_id,
            ):
                if await request.is_disconnected():
                    raise asyncio.CancelledError()

                response_parts.append(chunk)
                yield sse_event("token", {"content": chunk})

            reply = "".join(response_parts)
            if not reply.strip():
                raise RuntimeError("The AI model returned an empty response.")

            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=reply,
            )
            updated_history = persisted_history + [
                {"role": "assistant", "content": reply}
            ]

            db.add(assistant_message)
            db.flush()
            db.refresh(assistant_message)
            assistant_message_data = MessageRead.model_validate(
                assistant_message
            ).model_dump(mode="json")
            await synchronize_graph_history(
                updated_history,
                graph,
                conversation_id,
            )
            db.commit()

            yield sse_event(
                "complete",
                {"assistant_message": assistant_message_data},
            )
        except asyncio.CancelledError:
            try:
                await asyncio.shield(restore_checkpoint_from_database())
            except Exception:
                logger.exception(
                    "Could not restore graph history after disconnect for "
                    "conversation %s",
                    conversation_id,
                )
            raise
        except Exception:
            logger.exception(
                "Streaming AI response failed for conversation %s",
                conversation_id,
            )
            try:
                await restore_checkpoint_from_database()
            except Exception:
                logger.exception(
                    "Could not restore graph history for conversation %s",
                    conversation_id,
                )
            yield sse_event(
                "error",
                {"detail": "Could not complete the AI response. Please retry."},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


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
    # Verify conversation ownership.
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

    # Generate a title only for a new conversation.
    if conversation.title == "New chat":
        existing_user_message = db.scalar(
            select(Message.id)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == "user",
            )
            .limit(1)
        )

        if existing_user_message is None:
            conversation.title = generate_conversation_title(
                payload.content
            )
            db.commit()
            db.refresh(conversation)

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

    # Ask Groq/LangGraph to generate the response.
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


@router.post(
    "/conversations/{conversation_id}/chat/stream",
    response_class=StreamingResponse,
)
async def chat_with_ai_stream(
    conversation_id: int,
    payload: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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

    if conversation.title == "New chat":
        existing_user_message = db.scalar(
            select(Message.id)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == "user",
            )
            .limit(1)
        )
        if existing_user_message is None:
            conversation.title = generate_conversation_title(payload.content)
            db.commit()
            db.refresh(conversation)

    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=payload.content,
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    history = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        ).all()
    )

    return stream_response_for_message(
        request,
        db,
        conversation_id,
        user_message,
        history,
    )


@router.post(
    "/conversations/{conversation_id}/messages/{message_id}/retry/stream",
    response_class=StreamingResponse,
)
async def retry_chat_with_ai_stream(
    conversation_id: int,
    message_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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

    history = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        ).all()
    )
    if (
        not history
        or history[-1].id != message_id
        or history[-1].role != "user"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only the latest unanswered user prompt can be retried.",
        )

    return stream_response_for_message(
        request,
        db,
        conversation_id,
        history[-1],
        history,
    )
