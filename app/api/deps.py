"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.core.errors import ServiceUnavailableError
from app.rag.retriever import Retriever
from app.services.answer_cache import AnswerCache
from app.services.chat import ChatService
from app.services.conversations import ConversationStore
from app.services.health import HealthRegistry
from app.services.image_chat import ImageChatService
from app.services.llm import LLMProvider
from app.services.safety import PesticideSafety
from app.services.storage import StorageService
from app.vision.diagnosis import VisionDiagnoser


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_health_registry(request: Request) -> HealthRegistry:
    registry: HealthRegistry = request.app.state.health_registry
    return registry


def get_llm(request: Request) -> LLMProvider:
    llm: LLMProvider | None = getattr(request.app.state, "llm", None)
    if llm is None:
        raise ServiceUnavailableError("The AI service is not configured.")
    return llm


def get_retriever(request: Request) -> Retriever | None:
    retriever: Retriever | None = getattr(request.app.state, "retriever", None)
    return retriever


def get_safety(request: Request) -> PesticideSafety | None:
    safety: PesticideSafety | None = getattr(request.app.state, "safety", None)
    return safety


def get_diagnoser(request: Request) -> VisionDiagnoser:
    diagnoser: VisionDiagnoser | None = getattr(request.app.state, "diagnoser", None)
    if diagnoser is None:
        raise ServiceUnavailableError("Image diagnosis is not available.")
    return diagnoser


def get_answer_cache(request: Request) -> AnswerCache | None:
    cache: AnswerCache | None = getattr(request.app.state, "answer_cache", None)
    return cache


def get_conversations(request: Request) -> ConversationStore | None:
    store: ConversationStore | None = getattr(request.app.state, "conversations", None)
    return store


def require_conversations(request: Request) -> ConversationStore:
    store = get_conversations(request)
    if store is None:
        raise ServiceUnavailableError("Conversation storage is not configured.")
    return store


def get_storage(request: Request) -> StorageService:
    storage: StorageService | None = getattr(request.app.state, "storage", None)
    if storage is None:
        raise ServiceUnavailableError("Object storage is not configured.")
    return storage


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
HealthRegistryDep = Annotated[HealthRegistry, Depends(get_health_registry)]
LLMDep = Annotated[LLMProvider, Depends(get_llm)]
RetrieverDep = Annotated[Retriever | None, Depends(get_retriever)]
SafetyDep = Annotated[PesticideSafety | None, Depends(get_safety)]
AnswerCacheDep = Annotated[AnswerCache | None, Depends(get_answer_cache)]
StorageServiceDep = Annotated[StorageService, Depends(get_storage)]
DiagnoserDep = Annotated[VisionDiagnoser, Depends(get_diagnoser)]
ConversationsDep = Annotated[ConversationStore | None, Depends(get_conversations)]
RequiredConversationsDep = Annotated[ConversationStore, Depends(require_conversations)]


def get_chat_service(
    settings: SettingsDep,
    llm: LLMDep,
    retriever: RetrieverDep,
    safety: SafetyDep,
    answer_cache: AnswerCacheDep,
    conversations: ConversationsDep,
) -> ChatService:
    return ChatService(
        llm=llm,
        settings=settings,
        retriever=retriever,
        safety=safety,
        answer_cache=answer_cache,
        conversations=conversations,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


def get_image_chat_service(
    chat: ChatServiceDep, diagnoser: DiagnoserDep, retriever: RetrieverDep
) -> ImageChatService:
    return ImageChatService(chat=chat, diagnoser=diagnoser, retriever=retriever)


ImageChatServiceDep = Annotated[ImageChatService, Depends(get_image_chat_service)]
