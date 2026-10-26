
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url

from app.core.config import get_settings


@asynccontextmanager
async def checkpoint_lifespan(app):
    settings = get_settings()

    # Convert SQLAlchemy's URL into a psycopg-compatible URL.
    database_url = make_url(settings.database_url)
    conninfo = database_url.render_as_string(hide_password=False)

    if conninfo.startswith("postgresql+psycopg://"):
        conninfo = conninfo.replace(
            "postgresql+psycopg://",
            "postgresql://",
            1,
        )

    pool = AsyncConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True},
        open=False,
    )

    try:
        await pool.open(wait=True)

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        app.state.checkpoint_pool = pool
        app.state.checkpointer = checkpointer

        yield
    finally:
        await pool.close()
