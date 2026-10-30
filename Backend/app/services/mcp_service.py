
import asyncio
import logging
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
CALCULATOR_SERVER = (
    BACKEND_DIR / "app" / "mcp_servers" / "calculator_server.py"
)
MCP_DISCOVERY_TIMEOUT_SECONDS = 10


async def load_mcp_tools():
    client = None
    try:
        client = MultiServerMCPClient(
            {
                "calculator": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(CALCULATOR_SERVER)],
                }
            }
        )

        tools = await asyncio.wait_for(
            client.get_tools(),
            timeout=MCP_DISCOVERY_TIMEOUT_SECONDS,
        )
        if not tools:
            logger.warning("MCP calculator server discovered no tools.")
        return client, tools
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "MCP calculator server is unavailable; starting without tools."
        )
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
        return None, []
