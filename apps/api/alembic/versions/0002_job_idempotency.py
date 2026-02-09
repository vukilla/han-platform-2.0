"""add idempotency keys to job tables

Revision ID: 0002_job_idempotency
Revises: 0001_init
Create Date: 2026-02-04 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_job_idempotency"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("xgen_jobs", sa.Column("idempotency_key", sa.String(), nullable=True))
    op.add_column("xmimic_jobs", sa.Column("idempotency_key", sa.String(), nullable=True))
    op.create_index("ix_xgen_jobs_idempotency_key", "xgen_jobs", ["idempotency_key"], unique=False)
    op.create_index("ix_xmimic_jobs_idempotency_key", "xmimic_jobs", ["idempotency_key"], unique=False)


def downgrade():
    op.drop_index("ix_xmimic_jobs_idempotency_key", table_name="xmimic_jobs")
    op.drop_index("ix_xgen_jobs_idempotency_key", table_name="xgen_jobs")
    op.drop_column("xmimic_jobs", "idempotency_key")
    op.drop_column("xgen_jobs", "idempotency_key")
