
from sqlalchemy import text

from app.db.session import engine


def test_database_connection() -> None:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1")).scalar_one()

    assert result == 1
