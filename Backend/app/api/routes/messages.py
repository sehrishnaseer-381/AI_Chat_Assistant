
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import Conversation, Message, User
from app.schemas.chat import ChatResponse
from app.schemas.message import MessageCreate, MessageRead, MessageUpdate
from app.services.ai_service import generate_reply, synchronize_graph_history


logger = logging.getLogger(__name__)
router = APIRouter(tags=["Messages"])


def get_owned_conversation(
    conversation_id: int,
    current_user: User,
    db: Session,
) -> Conversation:
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return conversation


def get_owned_message(
    conversation_id: int,
    message_id: int,
    current_user: User,
    db: Session,
) -> Message:
    message = db.scalar(
        select(Message).where(
            Message.id == message_id,
            Message.conversation_id == conversation_id,
            Message.conversation.has(
                Conversation.user_id == current_user.id,
            ),
        )
    )

    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found.",
        )

    return message


def message_history(messages: list[Message]) -> list[dict[str, str]]:
    return [
        {"role": message.role, "content": message.content}
        for message in messages
    ]


async def restore_graph_history(
    history: list[dict[str, str]],
    graph,
    conversation_id: int,
) -> None:
    try:
        await synchronize_graph_history(history, graph, conversation_id)
    except Exception:
        logger.exception(
            "Could not restore LangGraph history for conversation %s",
            conversation_id,
        )


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageRead,
    status_code=status.HTTP_201_CREATED,
)
def create_message(
    conversation_id: int,
    payload: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_owned_conversation(conversation_id, current_user, db)

    message = Message(
        conversation_id=conversation_id,
        role=payload.role,
        content=payload.content,
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageRead],
)
def list_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_owned_conversation(conversation_id, current_user, db)

    statement = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )

    return list(db.scalars(statement).all())


@router.put(
    "/conversations/{conversation_id}/messages/{message_id}",
    response_model=ChatResponse,
)
async def edit_message(
    conversation_id: int,
    message_id: int,
    payload: MessageUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_owned_conversation(conversation_id, current_user, db)
    message = get_owned_message(conversation_id, message_id, current_user, db)

    if message.role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only user messages can be edited.",
        )

    new_content = payload.content.strip()
    if not new_content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message content cannot be empty.",
        )

    if message.content == new_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The message content is unchanged.",
        )

    message.content = new_content
    db.flush()

    history = list(
        db.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.id <= message.id,
            )
            .order_by(Message.id.asc())
        ).all()
    )

    response_messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    response_messages.extend(
        {"role": item.role, "content": item.content}
        for item in history
    )

    try:
        reply = await generate_reply(
            messages=response_messages,
            graph=request.app.state.graph,
            conversation_id=conversation_id,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not regenerate a response for the edited message.",
        ) from exc

    later_messages = db.scalars(
        select(Message).where(
            Message.conversation_id == conversation_id,
            Message.id > message.id,
        )
    ).all()
    for later_message in later_messages:
        db.delete(later_message)

    assistant_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=reply,
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(message)
    db.refresh(assistant_message)

    return ChatResponse(
        user_message=message,
        assistant_message=assistant_message,
    )


@router.post(
    "/conversations/{conversation_id}/messages/{message_id}/regenerate",
    response_model=MessageRead,
)
async def regenerate_assistant_message(
    conversation_id: int,
    message_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_owned_conversation(conversation_id, current_user, db)
    assistant_message = db.scalar(
        select(Message)
        .where(
            Message.id == message_id,
            Message.conversation_id == conversation_id,
        )
        .with_for_update()
    )

    if assistant_message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found.",
        )

    if assistant_message.role != "assistant":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only assistant messages can be regenerated.",
        )

    messages = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        ).all()
    )
    assistant_index = next(
        (
            index
            for index, item in enumerate(messages)
            if item.id == assistant_message.id
        ),
        None,
    )

    if assistant_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found.",
        )

    user_index = next(
        (
            index
            for index in range(assistant_index - 1, -1, -1)
            if messages[index].role == "user"
        ),
        None,
    )
    if user_index is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user prompt exists before this assistant message.",
        )

    original_history = message_history(messages)
    prompt_history = message_history(messages[: user_index + 1])
    graph = request.app.state.graph

    try:
        reply = await generate_reply(
            messages=prompt_history,
            graph=graph,
            conversation_id=conversation_id,
        )
    except Exception as exc:
        db.rollback()
        await restore_graph_history(original_history, graph, conversation_id)
        logger.exception(
            "Assistant response regeneration failed for conversation %s",
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not regenerate this assistant response. Please retry.",
        ) from exc

    updated_history = [message.copy() for message in original_history]
    updated_history[assistant_index]["content"] = reply
    assistant_message.content = reply

    try:
        await synchronize_graph_history(
            updated_history,
            graph,
            conversation_id,
        )
    except Exception as exc:
        db.rollback()
        await restore_graph_history(original_history, graph, conversation_id)
        logger.exception(
            "Could not synchronize LangGraph history for conversation %s",
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not synchronize the regenerated response. Please retry.",
        ) from exc

    try:
        db.commit()
        db.refresh(assistant_message)
    except SQLAlchemyError as exc:
        db.rollback()
        await restore_graph_history(original_history, graph, conversation_id)
        logger.exception(
            "Could not persist regenerated response for conversation %s",
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save the regenerated response. Please retry.",
        ) from exc

    return assistant_message
