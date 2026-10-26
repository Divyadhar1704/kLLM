# syntax=docker/dockerfile:1.7
# Production image for KrishiMitra API (and worker, via a different command).

# ---------- builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --require-hashes -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app app ./app
# Migrations ship in the image so a Cloud Run Job can run `alembic upgrade head` before deploys.
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations

USER app
EXPOSE 8080

# Cloud Run injects $PORT. One worker per container; Cloud Run scales containers.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --no-server-header"]
