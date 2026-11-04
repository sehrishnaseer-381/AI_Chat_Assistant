"""add image attachments to messages

Revision ID: 2d8d1e96c0af
Revises: 0e6b9b6012d7
Create Date: 2026-11-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "2d8d1e96c0af"
down_revision: Union[str, Sequence[str], None] = "0e6b9b6012d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "attachments",
            sa.JSON(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("messages", "attachments")
