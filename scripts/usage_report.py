"""Usage and quality report from the database (all instances, persistent), per day.

For the farmer pilot: how many farmers asked, in which languages, how often with a photo or
video, how often answers were grounded in the documents, how fast, what it cost in tokens, and
what farmers rated badly (with their comments).

Usage:
    python -m scripts.usage_report               # last 7 days
    python -m scripts.usage_report --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db import Database
from app.models.conversation import Conversation, ConversationMessage, Feedback


async def load(db: Database, since: datetime) -> tuple[list[Any], list[Any]]:
    async with db.session() as session:
        answers = (
            await session.execute(
                select(
                    ConversationMessage.id,
                    ConversationMessage.created_at,
                    ConversationMessage.language,
                    ConversationMessage.sources,
                    ConversationMessage.diagnosis,
                    ConversationMessage.latency_ms,
                    ConversationMessage.usage_input_tokens,
                    ConversationMessage.usage_output_tokens,
                    Conversation.owner_id,
                )
                .join(Conversation, Conversation.id == ConversationMessage.conversation_id)
                .where(ConversationMessage.role == "assistant")
                .where(ConversationMessage.created_at >= since)
            )
        ).all()
        ratings = (
            await session.execute(
                select(Feedback.created_at, Feedback.rating, Feedback.comment).where(
                    Feedback.created_at >= since
                )
            )
        ).all()
    return list(answers), list(ratings)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1]


def report(answers: list[Any], ratings: list[Any], days: int) -> None:
    per_day: dict[date, list[Any]] = defaultdict(list)
    for a in answers:
        per_day[a.created_at.date()].append(a)
    print(
        f"{'day':10} {'users':>5} {'answers':>7} {'photo':>5} {'video':>5} {'grounded':>8} "
        f"{'p50 s':>6} {'p95 s':>6} {'tokens in/out':>15}  languages"
    )
    for day in sorted(per_day):
        rows = per_day[day]
        media = Counter(
            "video" if (r.diagnosis or {}).get("video") else "photo" for r in rows if r.diagnosis
        )
        latencies = [r.latency_ms / 1000 for r in rows if r.latency_ms]
        grounded = sum(1 for r in rows if r.sources)
        languages = Counter(r.language for r in rows)
        p50, p95 = percentile(latencies, 0.5), percentile(latencies, 0.95)
        print(
            f"{day:%Y-%m-%d} {len({r.owner_id for r in rows}):5} {len(rows):7} "
            f"{media['photo']:5} {media['video']:5} {100 * grounded / len(rows):7.0f}% "
            f"{p50 or 0:6.1f} {p95 or 0:6.1f} "
            f"{sum(r.usage_input_tokens or 0 for r in rows):>7}/"
            f"{sum(r.usage_output_tokens or 0 for r in rows):<7}  "
            + " ".join(f"{k}:{v}" for k, v in languages.most_common())
        )
    if not per_day:
        print(f"no answers in the last {days} days")

    statuses = Counter((a.diagnosis or {}).get("status") for a in answers if a.diagnosis)
    if statuses:
        print("\nphoto/video results: " + ", ".join(f"{k} {v}" for k, v in statuses.most_common()))

    up = sum(1 for r in ratings if r.rating == "up")
    down = sum(1 for r in ratings if r.rating == "down")
    share = f" ({100 * up / (up + down):.0f}% positive)" if up + down else ""
    print(f"\nfeedback: {up} up, {down} down{share}")
    comments = [r for r in ratings if r.rating == "down" and r.comment]
    for r in sorted(comments, key=lambda r: r.created_at, reverse=True)[:15]:
        print(f"  {r.created_at:%m-%d %H:%M}  {r.comment[:160]}")


async def main(days: int) -> int:
    settings = get_settings()
    db = Database.from_settings(settings)
    try:
        since = datetime.now(UTC) - timedelta(days=days)
        answers, ratings = await load(db, since)
    finally:
        await db.aclose()
    print(f"usage since {since:%Y-%m-%d %H:%M} UTC ({days} days)\n")
    report(answers, ratings, days)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=7)
    setup_logging(get_settings())
    raise SystemExit(asyncio.run(main(parser.parse_args().days)))
