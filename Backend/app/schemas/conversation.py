
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    user_id: int
    title: str = Field(default="New chat", max_length=200)


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    title: str
    created_at: datetime



class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
