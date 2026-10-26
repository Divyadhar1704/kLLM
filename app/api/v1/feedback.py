"""Answer feedback endpoint (thumbs up / down with an optional comment)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from app.api.deps import RequiredConversationsDep
from app.api.security import PrincipalDep
from app.schemas.chat import ConversationId
from app.schemas.errors import ErrorResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    message_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{32}$",
        description="The `message_id` of the answer being rated (from the chat response).",
    )
    conversation_id: ConversationId | None = Field(
        default=None, description="Rate a whole conversation when no message_id is given."
    )
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _target(self) -> FeedbackRequest:
        if self.message_id is None and self.conversation_id is None:
            raise ValueError("Give message_id (preferred) or conversation_id.")
        return self


class FeedbackResponse(BaseModel):
    id: str
    message_id: str | None
    conversation_id: str | None
    rating: Literal["up", "down"]


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=201,
    summary="Rate an answer",
    responses={
        404: {"model": ErrorResponse, "description": "Unknown message or conversation"},
        503: {"model": ErrorResponse, "description": "Conversation storage not configured"},
    },
)
async def save_feedback(
    body: FeedbackRequest, conversations: RequiredConversationsDep, principal: PrincipalDep
) -> FeedbackResponse:
    feedback_id = await conversations.save_feedback(
        owner=principal.owner,
        conversation_id=body.conversation_id,
        message_id=body.message_id,
        rating=body.rating,
        comment=body.comment.strip() if body.comment else None,
    )
    return FeedbackResponse(
        id=feedback_id,
        message_id=body.message_id,
        conversation_id=body.conversation_id,
        rating=body.rating,
    )
