from pydantic import BaseModel, Field

from app.schemas.message import MessageRead


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=12000)


class ChatResponse(BaseModel):
    user_message: MessageRead
    assistant_message: MessageRead