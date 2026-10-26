"""Text chat orchestration (blocking and streaming).

    language -> retrieval (hybrid RAG) -> prompt with numbered sources -> LLM
             -> pesticide safety check -> response with citations

Both `answer` and `stream` share `_prepare` and `_finalize`, so the two endpoints cannot drift.
Retrieval failures never fail the request: the answer is generated without sources and marked
`grounded: false`.

Conversation history: with a `conversation_id`, the last turns are sent to the model as real
chat turns, and every answer (blocking, streamed, cached, photo, video) is stored best-effort
and returned with a `message_id` for feedback. The answer cache is used only for questions
without history: a follow-up's answer depends on the conversation, so it must never be reused.
"""

from __future__ import annotations

import dataclasses
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import Settings
from app.core.errors import BadRequestError
from app.core.logging import get_logger
from app.rag.crops import detect_query_crops
from app.rag.glossary import detect_terms
from app.rag.retriever import RetrievalResult, RetrievedChunk, Retriever
from app.schemas.chat import (
    ChatDebug,
    ChatRequest,
    ChatResponse,
    DebugChunk,
    LanguageCheck,
    Source,
    TokenUsage,
)
from app.services.answer_cache import AnswerCache
from app.services.citations import CitationStreamNormalizer, normalize_citations
from app.services.conversations import ConversationStore
from app.services.language import LanguageCode, detect_language_with_method, language_quality
from app.services.llm import ChatTurn, LLMProvider, LLMRequest, LLMResponse, StreamDelta, StreamEnd
from app.services.llm.errors import LLMError, LLMResponseError
from app.services.prompts import (
    PROMPT_VERSION,
    build_system_prompt,
    build_user_message,
    language_reminder,
)
from app.services.safety import PesticideSafety
from app.vision.taxonomy import CONDITIONS

logger = get_logger(__name__)

StreamEventName = Literal["meta", "delta", "done"]


def retrieval_query(message: str, history: tuple[ChatTurn, ...]) -> str:
    """Document search text. A follow-up that names no crop ("which medicine, what dose?")
    would otherwise retrieve other crops' doses: it is searched together with the farmer's
    latest earlier question that named a crop. Answers mention many crops in passing, so an
    answer is used only when no question named a crop and it names exactly one (photo chats,
    where the crop comes from the diagnosis)."""
    if not history or detect_query_crops(message):
        return message
    context = next(
        (t.text for t in reversed(history) if t.role == "user" and detect_query_crops(t.text)),
        None,
    )
    if context is None:
        context = next(
            (
                crops[0]
                for t in reversed(history)
                if t.role == "model" and len(crops := detect_query_crops(t.text)) == 1
            ),
            None,
        )
    if context is None:
        return message
    # The disease named in the last answer finds the control/dose pages; the question alone
    # finds symptom pages.
    last_answer = next((t.text for t in reversed(history) if t.role == "model"), "")
    diseases = conditions_named(last_answer, detect_query_crops(context))
    return "\n".join([context, message, *diseases])


def conditions_named(text: str, crops: list[str], limit: int = 2) -> list[str]:
    """English names of the given crops' conditions mentioned in `text` (answers usually give
    the English name in brackets: "ఎర్లీ బ్లైట్ (Early Blight)")."""
    lowered = text.lower()
    names = [
        condition.name
        for condition in CONDITIONS.values()
        if condition.crop in crops
        and condition.kind not in ("healthy", "unknown")
        and re.search(rf"\b{re.escape(condition.name.lower())}\b", lowered)
    ]
    return names[:limit]


@dataclass(slots=True)
class PreparedChat:
    request: ChatRequest
    language: LanguageCode
    language_detected: bool
    conversation_id: str
    retrieval: RetrievalResult
    used_chunks: list[RetrievedChunk]
    llm_request: LLMRequest
    start: float
    detection_method: str

    @property
    def sources(self) -> list[Source]:
        return [
            Source(
                index=i,
                title=c.hit.title,
                page_start=c.hit.page_start,
                page_end=c.hit.page_end,
                doc_type=c.hit.doc_type,
                source_org=c.hit.source_org,
                as_of=c.hit.as_of,
            )
            for i, c in enumerate(self.used_chunks, start=1)
        ]


class ChatService:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        settings: Settings,
        retriever: Retriever | None = None,
        safety: PesticideSafety | None = None,
        answer_cache: AnswerCache | None = None,
        conversations: ConversationStore | None = None,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._retriever = retriever if settings.rag_enabled else None
        self._safety = safety
        self._cache = answer_cache if answer_cache is not None and answer_cache.enabled else None
        self._conversations = conversations

    # ------------------------------------------------------------------ public

    async def answer(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        self._validate(request)
        history = await self._load_history(request)
        cache_key = None if history else self._cache_key(request)
        if (cached := self._from_cache(cache_key, request, start)) is not None:
            return await self.save(cached, question=request.message)
        prepared = await self._prepare(request, history)
        response = await self.generate_prepared(prepared)
        self._store(cache_key, prepared, response)
        return response

    @property
    def settings(self) -> Settings:
        return self._settings

    async def generate_prepared(
        self, prepared: PreparedChat, *, diagnosis: dict[str, Any] | None = None
    ) -> ChatResponse:
        """LLM call + answer-language check + safety check + response (shared with photo chat)."""
        llm_start = time.perf_counter()
        llm_response = await self._llm.generate(prepared.llm_request)
        llm_response, retried = await self._ensure_language(prepared, llm_response)
        llm_ms = round((time.perf_counter() - llm_start) * 1000, 1)
        response, _ = await self._finalize(
            prepared, llm_response, llm_ms, ttft_ms=None, language_retried=retried
        )
        return await self.save(response, question=prepared.request.message, diagnosis=diagnosis)

    async def save(
        self, response: ChatResponse, *, question: str, diagnosis: dict[str, Any] | None = None
    ) -> ChatResponse:
        """Store the question/answer pair (best-effort); the stored answer gets a message_id."""
        if self._conversations is None:
            return response
        message_id = await self._conversations.save_turn(
            conversation_id=response.conversation_id,
            language=response.language,
            question=question,
            answer=response.response,
            sources=[source.model_dump(mode="json") for source in response.sources],
            diagnosis=diagnosis,
            latency_ms=round(response.processing_time * 1000, 1),
            usage=(
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.usage.thinking_tokens,
            ),
        )
        if message_id is None:
            return response
        return response.model_copy(update={"message_id": message_id})

    async def _load_history(self, request: ChatRequest) -> tuple[ChatTurn, ...]:
        if self._conversations is None or request.conversation_id is None:
            return ()
        return await self._conversations.history(request.conversation_id)

    # ------------------------------------------------------------------ answer cache

    def _cache_key(self, request: ChatRequest) -> str | None:
        if self._cache is None:
            return None
        return AnswerCache.key(request.message, request.language, self._llm.model, PROMPT_VERSION)

    def _from_cache(
        self, key: str | None, request: ChatRequest, start: float
    ) -> ChatResponse | None:
        if key is None or self._cache is None or (stored := self._cache.get(key)) is None:
            return None
        response = stored.model_copy(
            update={
                "conversation_id": request.conversation_id or uuid.uuid4().hex,
                "processing_time": round(time.perf_counter() - start, 3),
                "cached": True,
                "debug": None,
            }
        )
        logger.info(
            "chat_answered",
            conversation_id=response.conversation_id,
            cached=True,
            language=response.language,
            grounded=response.grounded,
            processing_time=response.processing_time,
        )
        return response

    def _store(self, key: str | None, prepared: PreparedChat, response: ChatResponse) -> None:
        # Only clean answers: a degraded (no retrieval) or cut-off answer must not be repeated.
        if key is None or self._cache is None:
            return
        if prepared.retrieval.error is None and not response.truncated:
            self._cache.put(key, response.model_copy(update={"debug": None, "message_id": None}))

    async def _ensure_language(
        self, prepared: PreparedChat, llm_response: LLMResponse
    ) -> tuple[LLMResponse, bool]:
        """Regenerate once if the answer is not in the requested language (blocking only)."""
        quality = language_quality(llm_response.text, prepared.language)
        if quality.ok:
            return llm_response, False
        logger.warning(
            "language_mismatch",
            conversation_id=prepared.conversation_id,
            target=prepared.language,
            answer_script_share=quality.target_share,
        )
        *history, question = prepared.llm_request.messages
        retry_request = dataclasses.replace(
            prepared.llm_request,
            messages=(
                *history,
                ChatTurn(role="user", text=question.text + language_reminder(prepared.language)),
            ),
        )
        try:
            retry = await self._llm.generate(retry_request)
        except LLMError as exc:
            logger.warning("language_retry_failed", error_code=exc.code)
            return llm_response, True
        retry_quality = language_quality(retry.text, prepared.language)
        better = retry if retry_quality.target_share > quality.target_share else llm_response
        return better, True

    async def stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[tuple[StreamEventName, dict[str, Any]]]:
        """Yields ("meta", ...), ("delta", {"text"}) x N, ("done", ChatResponse)."""
        start = time.perf_counter()
        self._validate(request)
        history = await self._load_history(request)
        cache_key = None if history else self._cache_key(request)
        if (cached := self._from_cache(cache_key, request, start)) is not None:
            cached = await self.save(cached, question=request.message)
            yield (
                "meta",
                {
                    "conversation_id": cached.conversation_id,
                    "language": cached.language,
                    "language_detected": cached.language_detected,
                    "grounded": cached.grounded,
                    "sources": [s.model_dump(mode="json") for s in cached.sources],
                },
            )
            yield "delta", {"text": cached.response}
            yield "done", cached.model_dump(mode="json")
            return

        prepared = await self._prepare(request, history)
        yield (
            "meta",
            {
                "conversation_id": prepared.conversation_id,
                "language": prepared.language,
                "language_detected": prepared.language_detected,
                "grounded": bool(prepared.used_chunks),
                "sources": [s.model_dump(mode="json") for s in prepared.sources],
            },
        )

        llm_start = time.perf_counter()
        ttft_ms: float | None = None
        final: LLMResponse | None = None
        citations = CitationStreamNormalizer()
        async for event in self._llm.stream(prepared.llm_request):
            if isinstance(event, StreamDelta):
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - prepared.start) * 1000, 1)
                if text := citations.feed(event.text):
                    yield "delta", {"text": text}
            elif isinstance(event, StreamEnd):
                final = event.response
        if tail := citations.flush():
            yield "delta", {"text": tail}
        if final is None:
            raise LLMResponseError("The AI service ended the stream without a result.")
        llm_ms = round((time.perf_counter() - llm_start) * 1000, 1)

        response, warnings = await self._finalize(prepared, final, llm_ms, ttft_ms=ttft_ms)
        self._store(cache_key, prepared, response)
        response = await self.save(response, question=request.message)
        if warnings:
            # Clients that only render deltas must still show the safety warning.
            yield "delta", {"text": "\n\n" + "\n\n".join(warnings)}
        yield "done", response.model_dump(mode="json")

    # ------------------------------------------------------------------ steps

    def _validate(self, request: ChatRequest) -> None:
        if len(request.message) > self._settings.max_message_chars:
            raise BadRequestError(
                f"Message is too long (maximum {self._settings.max_message_chars} characters)."
            )

    async def _prepare(
        self, request: ChatRequest, history: tuple[ChatTurn, ...] = ()
    ) -> PreparedChat:
        start = time.perf_counter()
        self._validate(request)
        language_detected = request.language is None
        if request.language is not None:
            language, method = request.language, "explicit"
        else:
            language, method = detect_language_with_method(request.message)
        query = retrieval_query(request.message, history)
        retrieval = (
            await self._retriever.retrieve(query)
            if self._retriever is not None
            else RetrievalResult(crops=detect_query_crops(query))
        )
        user_message, used = build_user_message(
            request.message,
            retrieval.chunks,
            max_chars=self._settings.rag_max_context_chars,
            crops=retrieval.crops,
            terms=retrieval.terms or detect_terms(request.message),
        )
        return PreparedChat(
            request=request,
            language=language,
            language_detected=language_detected,
            conversation_id=request.conversation_id or uuid.uuid4().hex,
            retrieval=retrieval,
            used_chunks=retrieval.chunks[:used],
            llm_request=LLMRequest(
                messages=(*history, ChatTurn(role="user", text=user_message)),
                system_instruction=build_system_prompt(language),
            ),
            start=start,
            detection_method=method,
        )

    async def _finalize(
        self,
        prepared: PreparedChat,
        llm_response: LLMResponse,
        llm_ms: float,
        *,
        ttft_ms: float | None,
        language_retried: bool = False,
    ) -> tuple[ChatResponse, list[str]]:
        answer = normalize_citations(llm_response.text)
        quality = language_quality(answer, prepared.language)
        warnings: list[str] = []
        checker = await self._safety.ensure_loaded() if self._safety is not None else None
        if checker is not None:
            findings = checker.find(answer, prepared.retrieval.crops)
            if findings:
                warnings = checker.warnings(findings, prepared.language)
                # No rstrip: streamed deltas were already sent verbatim, and the joined deltas
                # must equal the final response exactly.
                answer = answer + "\n\n" + "\n\n".join(warnings)
                logger.warning(
                    "pesticide_safety_warning",
                    conversation_id=prepared.conversation_id,
                    findings=[(f.kind, f.name) for f in findings],
                )

        processing_time = round(time.perf_counter() - prepared.start, 3)
        retrieval = prepared.retrieval
        logger.info(
            "chat_answered",
            conversation_id=prepared.conversation_id,
            streamed=ttft_ms is not None,
            language=prepared.language,
            language_detected=prepared.language_detected,
            detection_method=prepared.detection_method,
            answer_script_share=quality.target_share,
            foreign_script_chars=quality.foreign_script_chars,
            language_retried=language_retried,
            history_turns=len(prepared.llm_request.messages) - 1,
            message_chars=len(prepared.request.message),
            grounded=bool(prepared.used_chunks),
            sources=len(prepared.used_chunks),
            retrieval_error=retrieval.error,
            retrieval_ms=retrieval.total_ms,
            llm_ms=llm_ms,
            ttft_ms=ttft_ms,
            model=llm_response.model,
            attempts=llm_response.attempts,
            processing_time=processing_time,
            truncated=llm_response.truncated,
            safety_warnings=len(warnings),
        )
        usage = llm_response.usage
        response = ChatResponse(
            response=answer,
            language=prepared.language,
            language_detected=prepared.language_detected,
            conversation_id=prepared.conversation_id,
            model=llm_response.model,
            processing_time=processing_time,
            truncated=llm_response.truncated,
            grounded=bool(prepared.used_chunks),
            sources=prepared.sources,
            safety_warnings=warnings,
            usage=TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                thinking_tokens=usage.thinking_tokens,
            ),
            debug=(
                self._debug(
                    retrieval,
                    llm_ms,
                    ttft_ms,
                    LanguageCheck(
                        detection_method=prepared.detection_method,
                        answer_script_share=quality.target_share,
                        foreign_script_chars=quality.foreign_script_chars,
                        retried=language_retried,
                    ),
                )
                if self._debug_allowed(prepared.request)
                else None
            ),
        )
        return response, warnings

    # ------------------------------------------------------------------ debug

    def _debug_allowed(self, request: ChatRequest) -> bool:
        return request.debug and self._settings.enable_dev_ui and not self._settings.is_production

    @staticmethod
    def _debug(
        retrieval: RetrievalResult,
        llm_ms: float,
        ttft_ms: float | None,
        language_check: LanguageCheck,
    ) -> ChatDebug:
        timings = {
            "embed": retrieval.embed_ms,
            "vector_search": retrieval.vector_ms,
            "keyword_search": retrieval.keyword_ms,
            "retrieval_total": retrieval.total_ms,
            "llm": llm_ms,
        }
        if ttft_ms is not None:
            timings["time_to_first_token"] = ttft_ms
        return ChatDebug(
            crops_detected=retrieval.crops,
            keyword_query=retrieval.keyword_query,
            retrieval_error=retrieval.error,
            timings_ms=timings,
            language_check=language_check,
            chunks=[
                DebugChunk(
                    index=i,
                    chunk_id=c.hit.chunk_id,
                    score=round(c.score, 5),
                    vector_similarity=(
                        round(c.vector_similarity, 4) if c.vector_similarity is not None else None
                    ),
                    vector_rank=c.vector_rank,
                    keyword_rank=c.keyword_rank,
                    crops=c.hit.crops,
                    text=c.hit.content,
                )
                for i, c in enumerate(retrieval.chunks, start=1)
            ],
        )
