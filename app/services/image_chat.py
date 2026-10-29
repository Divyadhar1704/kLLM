"""Photo and video questions: diagnosis + RAG + answer in the farmer's language.

photo or video (+ optional question, crop, language)
  -> crop: explicit field, else the single crop named in the question
  -> diagnosis (photo: quality gate; video: best frames, one model call; both calibrated)
  -> retake / not-a-plant: fixed message in the farmer's language (no LLM call)
  -> retrieval for the diagnosed condition (+ the farmer's own words)
  -> answer with the analysis and numbered sources; uncertain = "may be"
  -> pesticide safety check
  -> (background) cleaned photo / video frames kept with the answer (app.services.media_archive)
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal

from app.core import metrics
from app.core.errors import BadRequestError
from app.core.logging import get_logger
from app.rag.crops import detect_query_crops
from app.rag.retriever import RetrievalResult, Retriever
from app.schemas.chat import ChatRequest, ChatResponse, TokenUsage
from app.schemas.vision import DiagnosisResponse
from app.services.chat import ChatService, PreparedChat
from app.services.language import LanguageCode, detect_language_with_method
from app.services.llm import ChatTurn, LLMRequest
from app.services.media_archive import MediaArchive, MediaItem
from app.services.prompts import build_system_prompt, build_user_message
from app.vision.diagnosis import Diagnosis, VisionDiagnoser
from app.vision.taxonomy import CROPS, UNKNOWN, crop_name

logger = get_logger(__name__)

Media = Literal["photo", "video"]

DEFAULT_QUESTION: dict[Media, dict[LanguageCode, str]] = {
    "photo": {
        "en": "What is wrong with my plant in this photo, and what should I do?",
        "te": "ఈ ఫోటోలోని నా మొక్కకు ఏమైంది, నేను ఏమి చేయాలి?",
        "kn": "ಈ ಫೋಟೋದಲ್ಲಿರುವ ನನ್ನ ಗಿಡಕ್ಕೆ ಏನಾಗಿದೆ, ನಾನು ಏನು ಮಾಡಬೇಕು?",
        "hi": "इस फोटो में मेरे पौधे को क्या हुआ है, और मुझे क्या करना चाहिए?",
    },
    "video": {
        "en": "What is wrong with my plant in this video, and what should I do?",
        "te": "ఈ వీడియోలోని నా మొక్కకు ఏమైంది, నేను ఏమి చేయాలి?",
        "kn": "ಈ ವೀಡಿಯೊದಲ್ಲಿರುವ ನನ್ನ ಗಿಡಕ್ಕೆ ಏನಾಗಿದೆ, ನಾನು ಏನು ಮಾಡಬೇಕು?",
        "hi": "इस वीडियो में मेरे पौधे को क्या हुआ है, और मुझे क्या करना चाहिए?",
    },
}

RETAKE_MESSAGE: dict[Media, dict[LanguageCode, str]] = {
    "photo": {
        "en": "I could not check this photo properly. Please take a new photo in daylight, close "
        "to the affected leaf or fruit, and hold the phone steady. Telling me the crop name also "
        "helps.",
        "te": "ఈ ఫోటోను సరిగ్గా పరిశీలించలేకపోయాను. దయచేసి పగటి వెలుతురులో, సమస్య ఉన్న ఆకు లేదా "
        "కాయకు దగ్గరగా, ఫోన్‌ను కదలకుండా పట్టుకుని కొత్త ఫోటో తీయండి. పంట పేరు కూడా చెబితే ఇంకా "
        "బాగా సహాయపడుతుంది.",
        "kn": "ಈ ಫೋಟೋವನ್ನು ಸರಿಯಾಗಿ ಪರಿಶೀಲಿಸಲು ಆಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹಗಲಿನ ಬೆಳಕಿನಲ್ಲಿ, ಸಮಸ್ಯೆ ಇರುವ ಎಲೆ "
        "ಅಥವಾ ಕಾಯಿಯ ಹತ್ತಿರದಿಂದ, ಫೋನ್ ಅಲುಗಾಡದಂತೆ ಹಿಡಿದು ಹೊಸ ಫೋಟೋ ತೆಗೆಯಿರಿ. ಬೆಳೆಯ ಹೆಸರನ್ನೂ "
        "ತಿಳಿಸಿದರೆ ಇನ್ನೂ ಸಹಾಯವಾಗುತ್ತದೆ.",
        "hi": "मैं इस फोटो को ठीक से जाँच नहीं पाया। कृपया दिन की रोशनी में, प्रभावित पत्ती या फल के "
        "पास से, फोन को स्थिर रखकर नई फोटो लें। फसल का नाम भी बताएँ तो और मदद मिलेगी।",
    },
    "video": {
        "en": "I could not see the plant clearly in this video. Please record a new short video "
        "(10-20 seconds) in daylight, close to the affected leaves or fruit, moving the phone "
        "slowly. Telling me the crop name also helps.",
        "te": "ఈ వీడియోలో మొక్క స్పష్టంగా కనిపించలేదు. దయచేసి పగటి వెలుతురులో, సమస్య ఉన్న ఆకులు "
        "లేదా కాయలకు దగ్గరగా, ఫోన్‌ను నెమ్మదిగా కదిలిస్తూ 10-20 సెకన్ల కొత్త వీడియో తీయండి. పంట "
        "పేరు కూడా చెబితే ఇంకా బాగా సహాయపడుతుంది.",
        "kn": "ಈ ವೀಡಿಯೊದಲ್ಲಿ ಗಿಡ ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹಗಲಿನ ಬೆಳಕಿನಲ್ಲಿ, ಸಮಸ್ಯೆ ಇರುವ "
        "ಎಲೆಗಳು ಅಥವಾ ಕಾಯಿಗಳ ಹತ್ತಿರದಿಂದ, ಫೋನ್ ಅನ್ನು ನಿಧಾನವಾಗಿ ಸರಿಸುತ್ತಾ 10-20 ಸೆಕೆಂಡುಗಳ ಹೊಸ "
        "ವೀಡಿಯೊ ತೆಗೆಯಿರಿ. ಬೆಳೆಯ ಹೆಸರನ್ನೂ ತಿಳಿಸಿದರೆ ಇನ್ನೂ ಸಹಾಯವಾಗುತ್ತದೆ.",
        "hi": "इस वीडियो में पौधा साफ़ नहीं दिखा। कृपया दिन की रोशनी में, प्रभावित पत्तियों या फलों के "
        "पास से, फोन को धीरे-धीरे घुमाते हुए 10-20 सेकंड का नया वीडियो बनाएँ। फसल का नाम भी बताएँ "
        "तो और मदद मिलेगी।",
    },
}

NOT_A_PLANT_MESSAGE: dict[Media, dict[LanguageCode, str]] = {
    "photo": {
        "en": "I could not see a plant in this photo. Please send a close-up photo of the "
        "affected leaf, fruit or stem.",
        "te": "ఈ ఫోటోలో మొక్క కనిపించలేదు. దయచేసి సమస్య ఉన్న ఆకు, కాయ లేదా కాండం యొక్క దగ్గరి ఫోటో పంపండి.",
        "kn": "ಈ ಫೋಟೋದಲ್ಲಿ ಗಿಡ ಕಾಣಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಸಮಸ್ಯೆ ಇರುವ ಎಲೆ, ಕಾಯಿ ಅಥವಾ ಕಾಂಡದ ಹತ್ತಿರದ ಫೋಟೋ ಕಳುಹಿಸಿ.",
        "hi": "इस फोटो में मुझे पौधा नहीं दिखा। कृपया प्रभावित पत्ती, फल या तने की पास से ली गई फोटो भेजें।",
    },
    "video": {
        "en": "I could not see a plant in this video. Please record a short close-up video of the "
        "affected leaves, fruit or stem.",
        "te": "ఈ వీడియోలో మొక్క కనిపించలేదు. దయచేసి సమస్య ఉన్న ఆకులు, కాయలు లేదా కాండం యొక్క దగ్గరి చిన్న వీడియో పంపండి.",
        "kn": "ಈ ವೀಡಿಯೊದಲ್ಲಿ ಗಿಡ ಕಾಣಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಸಮಸ್ಯೆ ಇರುವ ಎಲೆಗಳು, ಕಾಯಿಗಳು ಅಥವಾ ಕಾಂಡದ ಹತ್ತಿರದ ಚಿಕ್ಕ ವೀಡಿಯೊ ಕಳುಹಿಸಿ.",
        "hi": "इस वीडियो में मुझे पौधा नहीं दिखा। कृपया प्रभावित पत्तियों, फलों या तने का पास से बना "
        "छोटा वीडियो भेजें।",
    },
}


def crop_from_text(text: str) -> str | None:
    """The crop named in the question, if exactly one supported crop is named."""
    crops = [c for c in detect_query_crops(text) if c in CROPS]
    return crops[0] if len(crops) == 1 else None


def retrieval_query(diagnosis: Diagnosis, question: str) -> str:
    """Document search for the diagnosed condition (and the runner-up when uncertain)."""
    conditions = [diagnosis.condition]
    if diagnosis.status == "uncertain" and diagnosis.alternatives:
        conditions.append(diagnosis.alternatives[0].condition)
    words = [crop_name(diagnosis.crop)]
    for condition in conditions:
        if condition.id != UNKNOWN:
            words.append(condition.name)
            words.extend(condition.search_terms)
    return " ".join(dict.fromkeys([*words, "management", question])).strip()


def photo_context(diagnosis: Diagnosis, crop_confirmed: bool) -> str:
    lines = ["PHOTO ANALYSIS (automatic, can be wrong):"]
    if diagnosis.video is not None:
        lines.append(
            f"- source: {len(diagnosis.video.frame_times)} frames from the farmer's "
            f"{diagnosis.video.duration_seconds:.0f}-second video"
        )
    lines += [
        f"- crop: {crop_name(diagnosis.crop)} "
        f"({'confirmed by the farmer' if crop_confirmed else 'identified from the photo'})",
        f"- result: {diagnosis.status}",
        f"- most likely: {diagnosis.condition.name} (confidence {diagnosis.confidence:.2f})",
    ]
    if diagnosis.alternatives:
        others = ", ".join(
            f"{a.condition.name} ({a.confidence:.2f})" for a in diagnosis.alternatives
        )
        lines.append(f"- other possibilities: {others}")
    if diagnosis.visible_symptoms:
        lines.append(f"- visible symptoms: {'; '.join(diagnosis.visible_symptoms)}")
    lines.append(f"- severity: {diagnosis.severity}")
    return "\n".join(lines)


# (note, crop) -> diagnosis + the cleaned images it was made from (for the media archive)
Diagnose = Callable[[str | None, str | None], Awaitable[tuple[Diagnosis, list[MediaItem]]]]


class ImageChatService:
    def __init__(
        self,
        *,
        chat: ChatService,
        diagnoser: VisionDiagnoser,
        retriever: Retriever | None,
        archive: MediaArchive | None = None,
    ) -> None:
        self._chat = chat
        self._diagnoser = diagnoser
        self._retriever = retriever
        self._archive = archive

    async def answer(
        self,
        *,
        image: bytes,
        message: str | None,
        language: LanguageCode | None,
        crop: str | None,
        conversation_id: str | None,
        debug: bool,
    ) -> ChatResponse:
        async def diagnose(note: str | None, crop: str | None) -> tuple[Diagnosis, list[MediaItem]]:
            prepared = await self._diagnoser.prepare(image)
            diagnosis = await self._diagnoser.diagnose_prepared(prepared, note=note, crop=crop)
            return diagnosis, [MediaItem("photo", prepared)]

        return await self._answer(
            "photo", diagnose, message, language, crop, conversation_id, debug
        )

    async def answer_video(
        self,
        *,
        video: bytes,
        message: str | None,
        language: LanguageCode | None,
        crop: str | None,
        conversation_id: str | None,
        debug: bool,
    ) -> ChatResponse:
        async def diagnose(note: str | None, crop: str | None) -> tuple[Diagnosis, list[MediaItem]]:
            prepared = await self._diagnoser.prepare_video(video)
            diagnosis = await self._diagnoser.diagnose_prepared_video(
                prepared, note=note, crop=crop
            )
            frames = [MediaItem("video_frame", f.image, f.time_seconds) for f in prepared.frames]
            return diagnosis, frames

        return await self._answer(
            "video", diagnose, message, language, crop, conversation_id, debug
        )

    async def _answer(
        self,
        media: Media,
        diagnose: Diagnose,
        message: str | None,
        language: LanguageCode | None,
        crop: str | None,
        conversation_id: str | None,
        debug: bool,
    ) -> ChatResponse:
        start = time.perf_counter()
        text = (message or "").strip()
        settings = self._chat.settings
        if len(text) > settings.max_message_chars:
            raise BadRequestError(
                f"Message is too long (maximum {settings.max_message_chars} characters)."
            )
        if crop is not None and crop not in CROPS:
            raise BadRequestError(f"Unknown crop '{crop[:40]}'. Supported: {', '.join(CROPS)}.")

        if language is not None:
            lang, method = language, "explicit"
        elif text:
            lang, method = detect_language_with_method(text)
        else:
            lang, method = "en", "default"
        crop = crop or crop_from_text(text)
        await self._chat.check_conversation(conversation_id)  # before any AI work
        conversation = conversation_id or uuid.uuid4().hex
        question = text or DEFAULT_QUESTION[media][lang]

        diagnosis, media_items = await diagnose(text or None, crop)
        diagnosis_out = DiagnosisResponse.of(diagnosis, 0.0)

        if diagnosis.status in ("retake_photo", "not_a_plant"):
            messages = RETAKE_MESSAGE if diagnosis.status == "retake_photo" else NOT_A_PLANT_MESSAGE
            processing_time = round(time.perf_counter() - start, 3)
            logger.info(
                "image_chat_answered",
                media=media,
                conversation_id=conversation,
                status=diagnosis.status,
                language=lang,
                processing_time=processing_time,
            )
            response = ChatResponse(
                response=messages[media][lang],
                language=lang,
                language_detected=language is None,
                conversation_id=conversation,
                model="none",
                processing_time=processing_time,
                truncated=False,
                grounded=False,
                usage=TokenUsage(input_tokens=0, output_tokens=0, thinking_tokens=0),
                diagnosis=diagnosis_out.model_copy(update={"processing_time": processing_time}),
            )
            metrics.answers.labels(media, lang, "false", "false").inc()  # fixed message, no AI
            return await self._chat.save(
                response, question=question, diagnosis=diagnosis_out.model_dump(mode="json")
            )

        retrieval = (
            await self._retriever.retrieve(retrieval_query(diagnosis, text))
            if self._retriever is not None and settings.rag_enabled
            else RetrievalResult(crops=[diagnosis.crop])
        )
        crops = sorted({diagnosis.crop, *retrieval.crops} - {UNKNOWN})
        retrieval.crops = crops  # the safety check uses the diagnosed crop too
        terms = [diagnosis.condition.name.lower()] if diagnosis.condition.id != UNKNOWN else []
        user_message, used = build_user_message(
            question,
            retrieval.chunks,
            max_chars=settings.rag_max_context_chars,
            crops=crops,
            terms=terms,
            extra_context=photo_context(diagnosis, crop_confirmed=crop in CROPS),
        )
        prepared = PreparedChat(
            request=ChatRequest(
                message=question, language=lang, conversation_id=conversation, debug=debug
            ),
            language=lang,
            language_detected=language is None,
            conversation_id=conversation,
            retrieval=retrieval,
            used_chunks=retrieval.chunks[:used],
            llm_request=LLMRequest(
                messages=(ChatTurn(role="user", text=user_message),),
                system_instruction=build_system_prompt(lang),
            ),
            start=start,
            detection_method=method,
            kind=media,
        )
        response = await self._chat.generate_prepared(
            prepared, diagnosis=diagnosis_out.model_dump(mode="json")
        )
        logger.info(
            "image_chat_answered",
            media=media,
            conversation_id=conversation,
            status=diagnosis.status,
            condition=diagnosis.condition.id,
            crop_confirmed=crop in CROPS,
            grounded=response.grounded,
            processing_time=response.processing_time,
        )
        if self._archive is not None and response.message_id is not None:
            # Background: the answer is returned now; only diagnosed media is ever kept.
            self._archive.archive_later(
                conversation_id=conversation,
                message_id=response.message_id,
                items=media_items,
                diagnosis=diagnosis,
            )
        return response.model_copy(
            update={
                "diagnosis": diagnosis_out.model_copy(
                    update={"processing_time": response.processing_time}
                )
            }
        )
