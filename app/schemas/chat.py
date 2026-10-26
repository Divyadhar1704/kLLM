"""Text chat request/response schemas."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.schemas.vision import DiagnosisResponse
from app.services.language import LanguageCode

ConversationId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{8,64}$")]


class ChatRequest(BaseModel):
    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)
    ]
    language: LanguageCode | None = Field(
        default=None,
        description="Response language. Omit to auto-detect from the message script.",
    )
    conversation_id: ConversationId | None = Field(
        default=None, description="Continue an existing conversation. Omit to start a new one."
    )
    debug: bool = Field(
        default=False,
        description="Include retrieval/timing details. Honoured only in development.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "What are the symptoms of early blight in tomato?", "language": "en"},
                {"message": "టమాటాలో ఎర్లీ బ్లైట్ లక్షణాలు ఏమిటి?"},
            ]
        }
    }


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    thinking_tokens: int


class Source(BaseModel):
    index: int = Field(description="Citation number used in the answer, e.g. [1]")
    title: str
    page_start: int | None = None
    page_end: int | None = None
    doc_type: str
    source_org: str | None = None
    as_of: date | None = None


class DebugChunk(BaseModel):
    index: int
    chunk_id: str
    score: float
    vector_similarity: float | None
    vector_rank: int | None
    keyword_rank: int | None
    crops: list[str]
    text: str


class LanguageCheck(BaseModel):
    detection_method: str = Field(description="script | romanized | default | explicit")
    answer_script_share: float = Field(description="Share of the answer in the target script")
    foreign_script_chars: int = Field(description="Characters from unrelated Indic scripts")
    retried: bool = Field(description="True if regenerated because of the wrong language")


class ChatDebug(BaseModel):
    crops_detected: list[str]
    keyword_query: str | None
    retrieval_error: str | None
    timings_ms: dict[str, float]
    language_check: LanguageCheck | None = None
    chunks: list[DebugChunk]


class ChatResponse(BaseModel):
    response: str
    language: LanguageCode
    language_detected: bool = Field(description="True if the language was auto-detected")
    conversation_id: str
    message_id: str | None = Field(
        default=None,
        description="ID of this answer, for POST /feedback. Null if history could not be stored.",
    )
    model: str
    processing_time: float = Field(description="Server processing time in seconds")
    truncated: bool = Field(description="True if the answer hit the output token limit")
    grounded: bool = Field(description="True if the answer was generated from retrieved sources")
    cached: bool = Field(
        default=False, description="True if served from the answer cache (identical question)"
    )
    sources: list[Source] = Field(default_factory=list)
    safety_warnings: list[str] = Field(
        default_factory=list, description="Pesticide safety warnings appended to the answer"
    )
    usage: TokenUsage
    diagnosis: DiagnosisResponse | None = Field(
        default=None, description="Photo diagnosis (only for /chat/image)"
    )
    debug: ChatDebug | None = None
