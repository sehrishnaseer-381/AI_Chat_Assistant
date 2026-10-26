
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.messages import router as messages_router
from app.api.routes.users import router as users_router
from app.core.config import get_settings
from app.services.checkpoint_service import checkpoint_lifespan
from app.services.graph_service import build_graph
from app.services.mcp_service import load_mcp_tools

settings = get_settings()

import asyncio
import sys
import logging

logging.basicConfig(level=logging.INFO)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with checkpoint_lifespan(app):
        mcp_client, tools = await load_mcp_tools()

        try:
            app.state.mcp_client = mcp_client
            app.state.mcp_tools = tools
            app.state.graph = build_graph(
                checkpointer=app.state.checkpointer,
                tools=tools,
            )
            yield
        finally:
            # Release MCP client resources if the client provides cleanup.
            close = getattr(mcp_client, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result


app = FastAPI(lifespan=lifespan)

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
