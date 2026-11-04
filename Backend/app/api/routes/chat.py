
import asyncio
import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import Conversation, Message, User
from app.schemas.chat import ChatRequest, ChatResponse, ModelSelectionRequest
from app.schemas.message import MessageRead
from app.schemas.model import ModelPreference
from app.services.ai_service import (
    generate_reply,
    stream_reply,
    synchronize_graph_history,
)
from app.services.vision_service import (
    MAX_TOTAL_IMAGE_SIZE_BYTES,
    validate_image_uploads,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

MAX_CHAT_REQUEST_BYTES = MAX_TOTAL_IMAGE_SIZE_BYTES + 128 * 1024
CHAT_REQUEST_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": ChatRequest.model_json_schema(),
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "maxLength": 12000},
                        "model_preference": {
                            "type": "string",
                            "enum": [
                                "auto",
                                "openai/gpt-oss-20b",
                                "openai/gpt-oss-120b",
                            ],
                        },
                        "images": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {"type": "string", "format": "binary"},
                        },
                    },
                },
            },
        },
    },
}


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


async def parse_chat_request(request: Request) -> tuple[
    ChatRequest,
    list[dict[str, str]],
]:
    content_type = request.headers.get("content-type", "").lower()
    attachments: list[dict[str, str]] = []

    try:
        if content_type.startswith("application/json"):
            body = await request.json()
            payload = ChatRequest.model_validate(body)
        elif content_type.startswith("multipart/form-data"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_CHAT_REQUEST_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="The chat request exceeds the supported size.",
                )

            async with request.form(
                max_files=3,
                max_fields=2,
                max_part_size=64 * 1024,
            ) as form:
                raw_content = form.get("content", "")
                raw_preference = form.get("model_preference", "auto")
                uploads = form.getlist("images")
                if not isinstance(raw_content, str) or not isinstance(
                    raw_preference, str
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="The chat request fields are invalid.",
                    )
                if not all(isinstance(item, UploadFile) for item in uploads):
                    raise HTTPException(
                        status_code=422,
                        detail="Image attachments must be uploaded as files.",
                    )
                attachments = await validate_image_uploads(uploads)

            if not raw_content.strip() and attachments:
                raw_content = "Please describe the attached image(s)."
            payload = ChatRequest.model_validate(
                {
                    "content": raw_content,
                    "model_preference": raw_preference,
                }
            )
        else:
            raise HTTPException(
                status_code=415,
                detail="Send chat requests as JSON or multipart form data.",
            )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_input=False),
        ) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The chat request could not be parsed.",
        ) from exc

    if not isinstance(payload, ChatRequest):
        raise HTTPException(
            status_code=422,
            detail="The chat request must be a JSON object.",
        )
    return payload, attachments


def graph_history_message(message: Message) -> dict[str, object]:
    history_message: dict[str, object] = {
        "role": message.role,
        "content": message.content,
    }
    if message.attachments:
        history_message["attachments"] = message.attachments
    return history_message


def sse_event(event_type: str, payload: dict) -> str:
    data = json.dumps(
        {"type": event_type, **jsonable_encoder(payload)},
        ensure_ascii=False,
    )
    return f"data: {data}\n\n"


def _parse_stream_chunk(
    chunk: object,
) -> tuple[str, dict[str, object], str | None]:
    if isinstance(chunk, str):
        return "token", {"content": chunk}, chunk

    if not isinstance(chunk, dict):
        raise TypeError("The AI stream returned an unsupported chunk.")

    event_type = chunk.get("type")
    if event_type == "token":
        content = chunk.get("content")
        if not isinstance(content, str):
            raise TypeError("The AI stream returned a malformed token event.")
        return "token", {"content": content}, content

    if event_type == "tool_start":
        tool = chunk.get("tool")
        tool_call_id = chunk.get("tool_call_id")
        if not isinstance(tool, str) or (
            tool_call_id is not None and not isinstance(tool_call_id, str)
        ):
            raise TypeError("The AI stream returned a malformed tool event.")
        return "tool_start", {
            "tool": tool,
            "tool_call_id": tool_call_id,
        }, None

    if event_type == "tool_result":
        tool = chunk.get("tool")
        tool_call_id = chunk.get("tool_call_id")
        content = chunk.get("content")
        status = chunk.get("status")
        if (
            not isinstance(tool, str)
            or (tool_call_id is not None and not isinstance(tool_call_id, str))
            or not isinstance(content, (str, list, dict))
            or not isinstance(status, str)
        ):
            raise TypeError("The AI stream returned a malformed tool result.")
        return "tool_result", {
            "tool": tool,
            "tool_call_id": tool_call_id,
            "content": content,
            "status": status,
        }, None

    if event_type == "reset":
        return "reset", {}, None

    raise ValueError("The AI stream returned an unknown event.")


def stream_response_for_message(
    request: Request,
    db: Session,
    conversation_id: int,
    user_message: Message,
    history: list[Message],
    model_preference: ModelPreference = "auto",
) -> StreamingResponse:
    graph = request.app.state.graph
    persisted_history = [graph_history_message(item) for item in history]
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
            [graph_history_message(item) for item in current_history],
            graph,
            conversation_id,
        )

    async def event_stream():
        try:
            yield sse_event(
                "user",
                {"user_message": user_message_data},
            )

            response_parts: list[str] = []
            messages = [graph_history_message(item) for item in history]

            async for chunk in stream_reply(
                messages=messages,
                graph=graph,
                conversation_id=conversation_id,
                model_preference=model_preference,
            ):
                if await request.is_disconnected():
                    raise asyncio.CancelledError()

                event_type, payload, text = _parse_stream_chunk(chunk)
                if event_type == "token" and not text:
                    continue
                if event_type == "reset":
                    response_parts.clear()
                if text is not None:
                    response_parts.append(text)
                yield sse_event(event_type, payload)

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
    openapi_extra=CHAT_REQUEST_OPENAPI,
)
async def chat_with_ai(
    conversation_id: int,
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

    payload, attachments = await parse_chat_request(request)

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
        attachments=attachments,
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
        graph_history_message(message)
        for message in history
    )

    # Ask Groq/LangGraph to generate the response.
    try:
        reply = await generate_reply(
            messages=messages,
            graph=request.app.state.graph,
            conversation_id=conversation_id,
            model_preference=payload.model_preference,
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
    openapi_extra=CHAT_REQUEST_OPENAPI,
)
async def chat_with_ai_stream(
    conversation_id: int,
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

    payload, attachments = await parse_chat_request(request)

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
        attachments=attachments,
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
        payload.model_preference,
    )


@router.post(
    "/conversations/{conversation_id}/messages/{message_id}/retry/stream",
    response_class=StreamingResponse,
)
async def retry_chat_with_ai_stream(
    conversation_id: int,
    message_id: int,
    request: Request,
    payload: ModelSelectionRequest | None = None,
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
        (payload or ModelSelectionRequest()).model_preference,
    )
