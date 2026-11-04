
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.model import ModelPreference


class MessageCreate(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class MessageUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=12000)
    model_preference: ModelPreference = "auto"


class ImageAttachment(BaseModel):
    filename: str
    content_type: str
    data_url: str


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str
    attachments: list[ImageAttachment] = Field(default_factory=list)
    created_at: datetime
