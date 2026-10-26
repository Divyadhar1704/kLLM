-- Runs once when the local Postgres volume is first created.
-- Production (Cloud SQL) enables the same extensions via Alembic migration.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
