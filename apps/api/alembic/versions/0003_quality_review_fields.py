"""add quality review fields

Revision ID: 0003_quality_review_fields
Revises: 0002_job_idempotency
Create Date: 2026-02-04 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "0003_quality_review_fields"
down_revision = "0002_job_idempotency"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("quality_scores", sa.Column("validator_notes", sa.String(), nullable=True))
    op.add_column("quality_scores", sa.Column("validator_id", sa.UUID(), nullable=True))
    op.add_column("quality_scores", sa.Column("validated_at", sa.DateTime(), nullable=True))


def downgrade():
    op.drop_column("quality_scores", "validated_at")
    op.drop_column("quality_scores", "validator_id")
    op.drop_column("quality_scores", "validator_notes")
