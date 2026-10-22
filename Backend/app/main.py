
from fastapi import FastAPI

from app.api.routes.users import router as users_router
from app.core.config import get_settings
from app.api.routes.conversations import router as conversations_router
from app.api.routes.messages import router as messages_router
from app.api.routes.chat import router as chat_router
from app.api.routes.auth import router as auth_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for a modular AI assistant.",
)

# Register API routers here
app.include_router(users_router, prefix=settings.api_prefix)
app.include_router(conversations_router, prefix=settings.api_prefix)
app.include_router(messages_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(auth_router, prefix=settings.api_prefix)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ai-chat-assistant",
    }
