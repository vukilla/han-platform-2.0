"""add xmimic job timestamps and error

Revision ID: 0004_xmimic_job_timestamps_error
Revises: 0003_quality_review_fields
Create Date: 2026-02-09 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "0004_xmimic_job_timestamps_error"
down_revision = "0003_quality_review_fields"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("xmimic_jobs", sa.Column("started_at", sa.DateTime(), nullable=True))
    op.add_column("xmimic_jobs", sa.Column("finished_at", sa.DateTime(), nullable=True))
    op.add_column("xmimic_jobs", sa.Column("error", sa.String(), nullable=True))


def downgrade():
    op.drop_column("xmimic_jobs", "error")
    op.drop_column("xmimic_jobs", "finished_at")
    op.drop_column("xmimic_jobs", "started_at")

