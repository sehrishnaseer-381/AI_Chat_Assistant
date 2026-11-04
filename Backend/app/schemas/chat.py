from pydantic import BaseModel, Field

from app.schemas.message import MessageRead
from app.schemas.model import ModelPreference


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=12000)
    model_preference: ModelPreference = "auto"


class ModelSelectionRequest(BaseModel):
    model_preference: ModelPreference = "auto"


class ChatResponse(BaseModel):
    user_message: MessageRead
    assistant_message: MessageRead