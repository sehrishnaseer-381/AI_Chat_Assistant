
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


BACKEND_DIR = Path(__file__).resolve().parents[2]
CALCULATOR_SERVER = (
    BACKEND_DIR / "app" / "mcp_servers" / "calculator_server.py"
)


async def load_mcp_tools():
    client = MultiServerMCPClient(
        {
            "calculator": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(CALCULATOR_SERVER)],
            }
        }
    )

    tools = await client.get_tools()
    return client, tools
