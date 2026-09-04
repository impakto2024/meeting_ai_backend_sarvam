import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sarvamai import SarvamAI

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

# English-only Meeting AI settings
FORCE_ENGLISH_ONLY = os.getenv("FORCE_ENGLISH_ONLY", "true").lower() == "true"

# Use translate mode so Hindi/Bengali/Indic speech becomes English text.
DEFAULT_STT_MODE = os.getenv("DEFAULT_STT_MODE", "translate")

# Use unknown because meeting audio may contain Hindi, Bengali, English, or mixed language.
DEFAULT_LANGUAGE_CODE = os.getenv("DEFAULT_LANGUAGE_CODE", "unknown")

DEFAULT_NUM_SPEAKERS = int(os.getenv("DEFAULT_NUM_SPEAKERS", "0"))

ENGLISH_ONLY_RULE = """
LANGUAGE RULE:
Output only English.

If the input contains Hindi, Bengali, Hinglish, Banglish, or any other language,
translate the meaning into clear natural English.

Do not output Hindi text.
Do not output Bengali text.
Do not output Devanagari script.
Do not output Bengali script.
Do not output mixed-language text.

Keep names, brand names, article codes, product codes, numbers, dates, prices,
amounts, and technical terms unchanged.
"""

SARVAM_CHAT_MAX_TOKENS = int(os.getenv("SARVAM_CHAT_MAX_TOKENS", "4096"))

if not SARVAM_API_KEY:
    raise RuntimeError("SARVAM_API_KEY is missing in environment variables")

app = FastAPI(title="Meeting AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs_lock = threading.Lock()
jobs: Dict[str, Dict[str, Any]] = {}


class AskQuestionRequest(BaseModel):
    transcriptText: str
    question: str


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Meeting AI backend is running",
    }


@app.post("/upload-debug")
async def upload_debug(audio: UploadFile = File(...)):
    started_at = time.time()
    temp_dir = tempfile.mkdtemp()
    file_name = os.path.basename(audio.filename or "debug_audio.m4a")
    audio_path = os.path.join(temp_dir, file_name)

    size_bytes = 0

    try:
        with open(audio_path, "wb") as buffer:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break

                size_bytes += len(chunk)
                buffer.write(chunk)

        elapsed_seconds = time.time() - started_at

        return {
            "status": "ok",
            "fileName": file_name,
            "sizeBytes": size_bytes,
            "sizeMb": round(size_bytes / 1024 / 1024, 2),
            "seconds": round(elapsed_seconds, 2),
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/process-meeting")
async def process_meeting(
    meeting_id: str = Form(...),
    audio: UploadFile = File(...),
    stt_mode: str = Form(DEFAULT_STT_MODE),
    language_code: str = Form(DEFAULT_LANGUAGE_CODE),
    num_speakers: int = Form(DEFAULT_NUM_SPEAKERS),
):
    temp_dir = tempfile.mkdtemp()
    file_name = os.path.basename(audio.filename or f"{meeting_id}.m4a")
    audio_path = os.path.join(temp_dir, file_name)

    try:
        with open(audio_path, "wb") as buffer:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        return start_processing_job(
            meeting_id=meeting_id,
            audio_path=audio_path,
            temp_dir=temp_dir,
            stt_mode=stt_mode,
            language_code=language_code,
            num_speakers=num_speakers,
        )

    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload-meeting-chunk")
async def upload_meeting_chunk(
    meeting_id: str = Form(...),
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file_name: str = Form(...),
    audio: UploadFile = File(...),
    stt_mode: str = Form(DEFAULT_STT_MODE),
    language_code: str = Form(DEFAULT_LANGUAGE_CODE),
    num_speakers: int = Form(DEFAULT_NUM_SPEAKERS),
):
    if chunk_index < 0:
        raise HTTPException(status_code=400, detail="Invalid chunk_index")

    if total_chunks <= 0:
        raise HTTPException(status_code=400, detail="Invalid total_chunks")

    if chunk_index >= total_chunks:
        raise HTTPException(
            status_code=400,
            detail="chunk_index must be less than total_chunks",
        )

    safe_upload_id = os.path.basename(str(upload_id).strip())

    if not safe_upload_id:
        raise HTTPException(status_code=400, detail="Invalid upload_id")

    base_upload_dir = os.path.join(tempfile.gettempdir(), "meeting_ai_chunk_uploads")
    upload_dir = os.path.join(base_upload_dir, safe_upload_id)
    os.makedirs(upload_dir, exist_ok=True)

    chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:06d}.part")
    temp_chunk_path = f"{chunk_path}.uploading"

    try:
        with open(temp_chunk_path, "wb") as buffer:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        os.replace(temp_chunk_path, chunk_path)

        uploaded_chunks = [
            name for name in os.listdir(upload_dir) if name.endswith(".part")
        ]

        if len(uploaded_chunks) < total_chunks:
            return {
                "status": "uploading",
                "uploadId": safe_upload_id,
                "chunkIndex": chunk_index,
                "uploadedChunks": len(uploaded_chunks),
                "totalChunks": total_chunks,
            }

        final_temp_dir = tempfile.mkdtemp()
        safe_file_name = os.path.basename(file_name or f"{meeting_id}.m4a")
        final_audio_path = os.path.join(final_temp_dir, safe_file_name)

        try:
            with open(final_audio_path, "wb") as final_file:
                for index in range(total_chunks):
                    part_path = os.path.join(upload_dir, f"chunk_{index:06d}.part")

                    if not os.path.exists(part_path):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Missing chunk {index}",
                        )

                    with open(part_path, "rb") as part_file:
                        shutil.copyfileobj(part_file, final_file)

            shutil.rmtree(upload_dir, ignore_errors=True)

            return start_processing_job(
                meeting_id=meeting_id,
                audio_path=final_audio_path,
                temp_dir=final_temp_dir,
                stt_mode=stt_mode,
                language_code=language_code,
                num_speakers=num_speakers,
            )

        except Exception:
            shutil.rmtree(final_temp_dir, ignore_errors=True)
            raise

    except HTTPException:
        raise

    except Exception as exc:
        if os.path.exists(temp_chunk_path):
            try:
                os.remove(temp_chunk_path)
            except Exception:
                pass

        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/meeting-status/{job_id}")
def meeting_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@app.post("/ask-question")
async def ask_question(request: AskQuestionRequest):
    try:
        answer = ask_from_transcript(
            transcript_text=request.transcriptText,
            question=request.question,
        )

        return {
            "answer": answer,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def start_processing_job(
    meeting_id: str,
    audio_path: str,
    temp_dir: str,
    stt_mode: str,
    language_code: str,
    num_speakers: int,
) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())

    if FORCE_ENGLISH_ONLY:
        validated_stt_mode = "translate"
        validated_language_code = "unknown"
    else:
        validated_stt_mode = validate_stt_mode(stt_mode)
        validated_language_code = validate_language_code(language_code)

    validated_num_speakers = validate_num_speakers(num_speakers)

    with jobs_lock:
        jobs[job_id] = {
            "jobId": job_id,
            "meetingId": meeting_id,
            "status": "queued",
            "message": "Meeting processing queued",
            "settings": {
                "sttMode": validated_stt_mode,
                "languageCode": validated_language_code,
                "numSpeakers": validated_num_speakers,
            },
            "result": None,
            "error": None,
        }

    worker_thread = threading.Thread(
        target=process_meeting_background,
        args=(
            job_id,
            meeting_id,
            audio_path,
            temp_dir,
            validated_stt_mode,
            validated_language_code,
            validated_num_speakers,
        ),
        daemon=True,
    )

    worker_thread.start()

    return {
        "jobId": job_id,
        "meetingId": meeting_id,
        "status": "queued",
        "message": "Meeting processing started",
        "settings": {
            "sttMode": validated_stt_mode,
            "languageCode": validated_language_code,
            "numSpeakers": validated_num_speakers,
        },
    }


def process_meeting_background(
    job_id: str,
    meeting_id: str,
    audio_path: str,
    temp_dir: str,
    stt_mode: str,
    language_code: str,
    num_speakers: int,
):
    try:
        update_job(
            job_id,
            status="transcribing",
            message="Transcribing meeting audio",
        )

        transcription_result = transcribe_with_sarvam(
            audio_path=audio_path,
            stt_mode=stt_mode,
            language_code=language_code,
            num_speakers=num_speakers,
        )

        transcript_text = transcription_result["transcriptText"]
        utterances = transcription_result["utterances"]

        update_job(
            job_id,
            status="analyzing",
            message="Creating premium meeting intelligence",
        )

        analysis_result = analyze_meeting(transcript_text)
        generated_at = current_utc_iso()

        meeting_title = str(analysis_result.get("meetingTitle", "") or "").strip()
        if not meeting_title:
            meeting_title = generate_title_from_text(
                analysis_result.get("overview", "")
                or analysis_result.get("summary", "")
                or transcript_text
            )

        result = {
            "meetingId": meeting_id,
            "audioCategory": analysis_result.get("audioCategory", "general_conversation"),
            "categoryLabel": analysis_result.get("categoryLabel", "General Conversation / Unknown"),
            "categoryConfidence": analysis_result.get("categoryConfidence", 0.0),
            "classificationReason": analysis_result.get("classificationReason", ""),
            "meetingTitle": meeting_title,
            "generatedAt": generated_at,
            "suggestedAudioFileName": build_suggested_audio_file_name(
                title=meeting_title,
                generated_at=generated_at,
                audio_path=audio_path,
            ),
            "overview": analysis_result.get("overview", ""),
            "summary": analysis_result.get("summary", ""),
            "keyPoints": analysis_result.get("keyPoints", []),
            "topics": analysis_result.get("topics", []),
            "discussionPoints": analysis_result.get("discussionPoints", []),
            "decisions": analysis_result.get("decisions", []),
            "actionItems": analysis_result.get("actionItems", []),
            "problemStatements": analysis_result.get("problemStatements", []),
            "solutions": analysis_result.get("solutions", []),
            "risks": analysis_result.get("risks", []),
            "followUps": analysis_result.get("followUps", []),
            "suggestions": analysis_result.get("suggestions", []),
            "approaches": analysis_result.get("approaches", []),
            "guide": analysis_result.get("guide", []),
            "workflowSteps": analysis_result.get("workflowSteps", []),
            "categorySpecificOutput": analysis_result.get("categorySpecificOutput", {}),
            "coverageCheck": analysis_result.get("coverageCheck", {}),
            "mindMap": analysis_result.get(
                "mindMap",
                {
                    "title": "Audio Mind Map",
                    "nodes": [],
                },
            ),
            "transcriptText": transcript_text,
            "utterances": utterances,
        }

        update_job(
            job_id,
            status="completed",
            message="Meeting processing completed",
            result=result,
        )

    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            message="Meeting processing failed",
            error=str(exc),
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def update_job(
    job_id: str,
    status: str,
    message: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
):
    with jobs_lock:
        if job_id not in jobs:
            return

        jobs[job_id]["status"] = status
        jobs[job_id]["message"] = message

        if result is not None:
            jobs[job_id]["result"] = result

        if error is not None:
            jobs[job_id]["error"] = error


def current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_suggested_audio_file_name(
    title: str,
    generated_at: str,
    audio_path: str,
) -> str:
    extension = os.path.splitext(str(audio_path or ""))[1].lower().strip()

    if extension not in [".m4a", ".mp4", ".aac", ".wav", ".mp3"]:
        extension = ".m4a"

    title_part = sanitize_file_name(title or "Meeting Notes")

    if not title_part:
        title_part = "Meeting Notes"

    try:
        date_part = datetime.fromisoformat(generated_at).strftime("%Y%m%d_%H%M")
    except Exception:
        date_part = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    return f"{date_part}_{title_part}{extension}"


def sanitize_file_name(value: str, max_length: int = 70) -> str:
    cleaned = " ".join(str(value or "").strip().split())

    if not cleaned:
        return ""

    allowed_chars = []

    for char in cleaned:
        if char.isalnum() or char in [" ", "-", "_"]:
            allowed_chars.append(char)
        else:
            allowed_chars.append(" ")

    cleaned = " ".join("".join(allowed_chars).split())
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.strip("_-")

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].strip("_-")

    return cleaned


def validate_stt_mode(value: str) -> str:
    allowed_modes = {
        "transcribe",
        "translate",
        "verbatim",
        "translit",
        "codemix",
    }

    mode = str(value or DEFAULT_STT_MODE).strip().lower()

    if mode not in allowed_modes:
        return DEFAULT_STT_MODE

    return mode


def validate_language_code(value: str) -> str:
    language_code = str(value or DEFAULT_LANGUAGE_CODE).strip()

    if not language_code:
        return DEFAULT_LANGUAGE_CODE

    return language_code


def validate_num_speakers(value: int) -> int:
    try:
        count = int(value)
    except Exception:
        count = DEFAULT_NUM_SPEAKERS

    if count < 0:
        return 0

    if count > 20:
        return 20

    return count


def get_sarvam_client() -> SarvamAI:
    return SarvamAI(api_subscription_key=SARVAM_API_KEY)


def transcribe_with_sarvam(
    audio_path: str,
    stt_mode: str,
    language_code: str,
    num_speakers: int,
) -> Dict[str, Any]:
    client = get_sarvam_client()

    create_job_args: Dict[str, Any] = {
        "model": SARVAM_STT_MODEL,
        "mode": stt_mode,
        "language_code": language_code,
        "with_diarization": True,
    }

    if num_speakers > 0:
        create_job_args["num_speakers"] = num_speakers

    job = client.speech_to_text_job.create_job(**create_job_args)

    job.upload_files(file_paths=[audio_path])
    job.start()
    job.wait_until_complete()

    file_results = job.get_file_results()
    failed_files = file_results.get("failed", [])

    if failed_files:
        error_message = failed_files[0].get(
            "error_message",
            "Sarvam transcription failed",
        )
        raise RuntimeError(error_message)

    successful_files = file_results.get("successful", [])

    if not successful_files:
        raise RuntimeError("Sarvam transcription did not return successful files")

    output_dir = tempfile.mkdtemp()

    try:
        job.download_outputs(output_dir=output_dir)

        result_json_path = find_first_json_file(output_dir)

        if not result_json_path:
            raise RuntimeError("Sarvam transcription output JSON not found")

        with open(result_json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return normalize_sarvam_transcription(data)

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def normalize_sarvam_transcription(data: Dict[str, Any]) -> Dict[str, Any]:
    transcript_text = str(data.get("transcript", "") or "")

    utterances: List[Dict[str, Any]] = []

    diarized_transcript = data.get("diarized_transcript") or {}
    diarized_entries = diarized_transcript.get("entries", [])

    if diarized_entries:
        for entry in diarized_entries:
            speaker_id = entry.get("speaker_id", "")
            speaker_label = (
                f"Speaker {speaker_id}"
                if speaker_id != ""
                else "Speaker"
            )

            text = str(entry.get("transcript", "") or "")

            if not text.strip():
                continue

            utterances.append(
                {
                    "speaker": speaker_label,
                    "startLabel": format_seconds(entry.get("start_time_seconds")),
                    "endLabel": format_seconds(entry.get("end_time_seconds")),
                    "text": text,
                }
            )

    if not transcript_text and utterances:
        transcript_text = "\n".join(
            f"{item['speaker']}: {item['text']}"
            for item in utterances
        )

    if not utterances and transcript_text:
        utterances.append(
            {
                "speaker": "Speaker",
                "startLabel": "00:00",
                "endLabel": "",
                "text": transcript_text,
            }
        )

    return {
        "transcriptText": transcript_text,
        "utterances": utterances,
    }



def category_label_for(category: str) -> str:
    labels = {
        "business_meeting": "Business Meeting",
        "sales_client_vendor_call": "Sales / Client / Vendor Call",
        "lecture_class_training": "Lecture / Class / Training",
        "interview_hr_discussion": "Interview / HR Discussion",
        "motivational_speech_seminar": "Motivational Speech / Seminar",
        "brainstorming_strategy": "Brainstorming / Strategy",
        "product_project_discussion": "Product / Project Discussion",
        "finance_legal_compliance": "Finance / Legal / Compliance",
        "customer_support_complaint": "Customer Support / Complaint",
        "personal_voice_note_idea": "Personal Voice Note / Idea Capture",
        "podcast_webinar_panel": "Podcast / Webinar / Panel",
        "general_conversation": "General Conversation / Unknown",
    }
    return labels.get(str(category or "").strip(), labels["general_conversation"])


VALID_AUDIO_CATEGORIES = {
    "business_meeting",
    "sales_client_vendor_call",
    "lecture_class_training",
    "interview_hr_discussion",
    "motivational_speech_seminar",
    "brainstorming_strategy",
    "product_project_discussion",
    "finance_legal_compliance",
    "customer_support_complaint",
    "personal_voice_note_idea",
    "podcast_webinar_panel",
    "general_conversation",
}


CATEGORY_SPECIFIC_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "business_meeting": {
        "meetingObjective": "",
        "keyAgenda": [],
        "managementSummary": "",
        "decisionsTaken": [],
        "ownersAndDeadlines": [],
        "blockers": [],
        "nextMeetingPreparation": [],
    },
    "sales_client_vendor_call": {
        "clientOrVendorName": "Not specified",
        "purposeOfCall": "",
        "clientNeeds": [],
        "objectionsOrConcerns": [],
        "priceOrCommercialPoints": [],
        "commitmentsMade": [],
        "followUpPlan": [],
        "suggestedResponse": [],
        "dealStatus": "open / closed / pending / unclear",
    },
    "lecture_class_training": {
        "topicTaught": "",
        "learningObjectives": [],
        "keyConcepts": [],
        "definitions": [],
        "examplesExplained": [],
        "importantFormulasOrFrameworks": [],
        "revisionNotes": [],
        "questionsToPractice": [],
        "studyGuide": [],
    },
    "interview_hr_discussion": {
        "interviewPurpose": "",
        "candidateProfile": "",
        "questionsAsked": [],
        "answerSummary": [],
        "strengths": [],
        "weaknessesOrConcerns": [],
        "skillsObserved": [],
        "cultureFitNotes": [],
        "finalRecommendation": "",
        "nextRoundQuestions": [],
    },
    "motivational_speech_seminar": {
        "coreMessage": "",
        "keyTakeaways": [],
        "memorableQuotes": [],
        "lifeLessons": [],
        "actionPrinciples": [],
        "practicalGuide": [],
        "audienceImpact": "",
        "suggestedDailyActions": [],
    },
    "brainstorming_strategy": {
        "brainstormingGoal": "",
        "ideasGenerated": [],
        "promisingIdeas": [],
        "rejectedOrWeakIdeas": [],
        "strategicDirection": [],
        "creativeApproaches": [],
        "executionPlan": [],
        "successMetrics": [],
    },
    "product_project_discussion": {
        "projectName": "",
        "requirements": [],
        "featuresDiscussed": [],
        "bugsOrIssues": [],
        "technicalDecisions": [],
        "dependencies": [],
        "roadmap": [],
        "testingChecklist": [],
        "releaseReadiness": "",
    },
    "finance_legal_compliance": {
        "financialSummary": "",
        "amountsMentioned": [],
        "documentsRequired": [],
        "approvalsNeeded": [],
        "complianceRisks": [],
        "legalConcerns": [],
        "paymentOrRefundStatus": "",
        "nextComplianceSteps": [],
    },
    "customer_support_complaint": {
        "customerIssue": "",
        "customerSentiment": "positive / neutral / negative / angry / unclear",
        "rootCause": "",
        "resolutionSuggested": [],
        "escalationNeeded": "",
        "refundOrReplacementMentioned": "",
        "supportFollowUp": [],
        "preventionSuggestion": [],
    },
    "personal_voice_note_idea": {
        "cleanedNote": "",
        "ideas": [],
        "tasks": [],
        "reminders": [],
        "priorityItems": [],
        "suggestedNextActions": [],
        "convertedToProfessionalNote": "",
    },
    "podcast_webinar_panel": {
        "mainTopic": "",
        "speakerViewpoints": [],
        "keyThemes": [],
        "importantArguments": [],
        "audienceTakeaways": [],
        "interestingQuotes": [],
        "summaryForSharing": "",
        "contentIdeasFromAudio": [],
    },
    "general_conversation": {
        "conversationSummary": "",
        "importantPoints": [],
        "possibleActions": [],
        "peopleMentioned": [],
        "datesOrNumbersMentioned": [],
        "unclearSections": [],
        "suggestedCategory": "",
    },
}


def default_classification() -> Dict[str, Any]:
    return {
        "audioCategory": "general_conversation",
        "categoryLabel": category_label_for("general_conversation"),
        "categoryConfidence": 0.35,
        "classificationReason": "The transcript did not strongly match a specific audio type.",
    }


def classify_audio_category(transcript_text: str) -> Dict[str, Any]:
    transcript_text = str(transcript_text or "").strip()
    if not transcript_text:
        return default_classification()

    options_text = "\n".join(
        f"- {category}: {category_label_for(category)}"
        for category in sorted(VALID_AUDIO_CATEGORIES)
    )

    prompt = f"""
{ENGLISH_ONLY_RULE}

Classify this audio transcript into exactly one category.

Allowed categories:
{options_text}

Return ONLY valid JSON in this format:
{{
  "audioCategory": "business_meeting",
  "categoryLabel": "Business Meeting",
  "categoryConfidence": 0.0,
  "classificationReason": "Short factual reason based on the transcript."
}}

Category rules:
- Choose lecture_class_training for teaching, class, training, concepts, examples, definitions, revision, or learning.
- Choose interview_hr_discussion for interviewer/candidate, HR, hiring, skills, experience, salary, joining, or evaluation.
- Choose motivational_speech_seminar for inspirational, life lesson, self-growth, leadership, success, or seminar tone.
- Choose sales_client_vendor_call for client/vendor/dealer/customer business call, price, quotation, order, margin, payment, negotiation, or commitment.
- Choose product_project_discussion for app, website, software, feature, bug, backend, API, design, testing, launch, product, or project planning.
- Choose finance_legal_compliance for audit, GST, invoice, PI, payment, tax, agreement, legal, compliance, approval, or documents.
- Choose customer_support_complaint for complaint, service issue, refund, replacement, escalation, customer dissatisfaction, or defect.
- Choose brainstorming_strategy for idea generation, campaign strategy, creative planning, roadmap, positioning, or brainstorming.
- Choose personal_voice_note_idea for one-person reminder, thought capture, idea note, task note, or personal planning.
- Choose podcast_webinar_panel for episode, webinar, public talk, panel, audience questions, or speaker viewpoints.
- Choose business_meeting for internal review, decision meeting, team planning, operations, or management discussion.
- Choose general_conversation only if no specific category fits.

Transcript sample:
{make_transcript_context(transcript_text, max_chars=9000)}
"""

    try:
        parsed = parse_json_object(call_sarvam_chat(prompt))
        return normalize_classification(parsed, transcript_text)
    except Exception:
        return heuristic_classification(transcript_text)


def normalize_classification(data: Any, transcript_text: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return heuristic_classification(transcript_text)

    category = str(
        data.get("audioCategory")
        or data.get("audio_category")
        or data.get("category")
        or ""
    ).strip()

    if category not in VALID_AUDIO_CATEGORIES:
        return heuristic_classification(transcript_text)

    try:
        confidence = float(data.get("categoryConfidence") or data.get("category_confidence") or 0.65)
    except Exception:
        confidence = 0.65

    confidence = max(0.0, min(confidence, 1.0))

    reason = str(
        data.get("classificationReason")
        or data.get("classification_reason")
        or ""
    ).strip()

    if not reason:
        reason = f"The transcript matched the {category_label_for(category)} pattern."

    return {
        "audioCategory": category,
        "categoryLabel": category_label_for(category),
        "categoryConfidence": confidence,
        "classificationReason": reason,
    }


def heuristic_classification(transcript_text: str) -> Dict[str, Any]:
    text = str(transcript_text or "").lower()

    keyword_groups = [
        ("lecture_class_training", ["class", "lecture", "student", "teacher", "chapter", "definition", "formula", "example", "exam", "training", "learning"]),
        ("interview_hr_discussion", ["interview", "candidate", "resume", "salary", "joining", "experience", "skills", "hr", "notice period", "tell me about yourself"]),
        ("motivational_speech_seminar", ["motivation", "success", "dream", "failure", "life", "inspire", "leadership", "mindset", "discipline", "seminar"]),
        ("sales_client_vendor_call", ["client", "vendor", "dealer", "customer", "quotation", "price", "margin", "order", "payment", "delivery", "follow up", "commitment"]),
        ("product_project_discussion", ["app", "api", "backend", "frontend", "feature", "bug", "testing", "release", "project", "product", "ui", "server", "github"]),
        ("finance_legal_compliance", ["gst", "invoice", "audit", "payment", "tax", "legal", "agreement", "compliance", "approval", "pi", "refund"]),
        ("customer_support_complaint", ["complaint", "issue", "problem", "defect", "replacement", "refund", "escalate", "support", "service"]),
        ("brainstorming_strategy", ["idea", "brainstorm", "strategy", "campaign", "creative", "concept", "positioning", "roadmap"]),
        ("podcast_webinar_panel", ["podcast", "webinar", "episode", "panel", "audience", "host", "speaker"]),
        ("personal_voice_note_idea", ["remind me", "note to self", "my idea", "i need to", "i should", "remember to"]),
        ("business_meeting", ["meeting", "review", "decision", "agenda", "team", "management", "approval", "timeline", "target"]),
    ]

    best_category = "general_conversation"
    best_score = 0

    for category, keywords in keyword_groups:
        score = sum(1 for keyword in keywords if keyword in text)
        if score > best_score:
            best_score = score
            best_category = category

    if best_score <= 0:
        confidence = 0.35
        reason = "No strong category-specific keywords were detected."
    else:
        confidence = min(0.55 + (best_score * 0.06), 0.86)
        reason = f"Matched category-specific terms for {category_label_for(best_category)}."

    return {
        "audioCategory": best_category,
        "categoryLabel": category_label_for(best_category),
        "categoryConfidence": confidence,
        "classificationReason": reason,
    }


def analyze_meeting(transcript_text: str) -> Dict[str, Any]:
    transcript_text = str(transcript_text or "").strip()

    if not transcript_text:
        return fallback_analysis_from_transcript("")

    classification = classify_audio_category(transcript_text)
    chunks = split_text_into_chunks(transcript_text, max_words=2200)

    if len(chunks) == 1:
        analysis = analyze_meeting_chunk(
            transcript_text=chunks[0],
            chunk_number=1,
            total_chunks=1,
            classification=classification,
        )
    else:
        chunk_results = []
        for index, chunk in enumerate(chunks):
            chunk_result = analyze_meeting_chunk(
                transcript_text=chunk,
                chunk_number=index + 1,
                total_chunks=len(chunks),
                classification=classification,
            )
            chunk_results.append(chunk_result)

        analysis = merge_meeting_analysis(
            chunk_results=chunk_results,
            transcript_text=transcript_text,
            classification=classification,
        )

    analysis = ensure_minimum_analysis(analysis, transcript_text)
    analysis.update(classification)

    title = generate_meaningful_title(
        transcript_text=transcript_text,
        analysis_result=analysis,
        classification=classification,
    )
    if title:
        analysis["meetingTitle"] = title

    return ensure_minimum_analysis(analysis, transcript_text)


def analyze_meeting_chunk(
    transcript_text: str,
    chunk_number: int,
    total_chunks: int,
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    category = classification.get("audioCategory", "general_conversation")
    category_label = classification.get("categoryLabel", category_label_for(category))
    category_schema = json.dumps(
        CATEGORY_SPECIFIC_SCHEMAS.get(category, CATEGORY_SPECIFIC_SCHEMAS["general_conversation"]),
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
{ENGLISH_ONLY_RULE}

You are a premium audio intelligence analyst.
The audio has been classified as: {category_label} ({category}).

Analyze the transcript chunk according to this category. The output must adapt to the category.
Do not force business-meeting sections when the audio is a lecture, interview, motivational speech, podcast, complaint, or voice note.

Return ONLY one valid JSON object.

Required JSON format:
{{
  "meetingTitle": "Temporary chunk title in 5 to 8 words",
  "audioCategory": "{category}",
  "categoryLabel": "{category_label}",
  "categoryConfidence": {classification.get('categoryConfidence', 0.65)},
  "classificationReason": "{escape_prompt_text(classification.get('classificationReason', ''))}",
  "overview": "Category-appropriate executive overview in 4 to 6 lines.",
  "summary": "Category-appropriate detailed summary in 6 to 10 lines.",
  "keyPoints": [{{"id":"key_1","title":"Key point title","description":"Important highlight from the transcript."}}],
  "topics": [{{"id":"topic_1","title":"Topic title","description":"What was discussed or taught under this topic."}}],
  "discussionPoints": [{{"id":"point_1","title":"Discussion title","description":"Only if the audio has discussion points.","status":"pending"}}],
  "decisions": [{{"id":"decision_1","title":"Decision title","description":"Only if a decision was actually made."}}],
  "actionItems": [{{"id":"action_1","title":"Action title","description":"Task details.","owner":"Owner if mentioned, otherwise Not specified","deadline":"Deadline if mentioned, otherwise Not specified","status":"pending"}}],
  "problemStatements": [{{"id":"problem_1","title":"Problem title","description":"Only if a real problem, blocker, concern, gap, or challenge was discussed."}}],
  "solutions": [{{"id":"solution_1","title":"Solution title","description":"Only if a solution, recommendation, approach, or answer was discussed.","relatedProblemId":"problem_1"}}],
  "risks": [{{"id":"risk_1","title":"Risk title","description":"Only if a risk, concern, uncertainty, objection, or dependency exists."}}],
  "followUps": [{{"id":"followup_1","title":"Follow-up title","description":"Only if follow-up or review is needed."}}],
  "suggestions": [{{"id":"suggestion_1","title":"Suggestion title","description":"Useful suggestion grounded in the transcript."}}],
  "approaches": [{{"id":"approach_1","title":"Approach title","description":"Practical way to handle or execute something from the transcript."}}],
  "guide": [{{"id":"guide_1","title":"Guide step title","description":"Step-by-step guidance relevant to this category."}}],
  "workflowSteps": [{{"step":1,"title":"Step title","description":"Only if a process or sequence is present."}}],
  "categorySpecificOutput": {category_schema},
  "coverageCheck": {{
    "importantItemsCovered": [],
    "possibleMissingItems": [],
    "unclearParts": [],
    "confidenceNotes": ""
  }},
  "mindMap": {{"title":"Audio Mind Map","nodes":[{{"id":"node_1","label":"Main topic","children":[]}}]}}
}}

Category-specific instructions:
- For Lecture/Class/Training: focus on topic taught, key concepts, definitions, examples, revision notes, questions to practice, and study guide.
- For Interview/HR: focus on candidate profile, questions, answer summaries, strengths, concerns, skills, fit, recommendation, and next questions.
- For Motivational Speech/Seminar: focus on core message, takeaways, memorable quotes, lessons, action principles, and practical guide.
- For Sales/Client/Vendor: focus on needs, objections, price/commercial points, commitments, deal status, follow-up plan, and suggested response.
- For Product/Project: focus on requirements, features, bugs, technical decisions, dependencies, roadmap, testing checklist, and release readiness.
- For Finance/Legal/Compliance: focus on amounts, documents, approvals, risks, legal/compliance concerns, and next compliance steps.
- For Customer Support/Complaint: focus on customer issue, sentiment, root cause, resolution, escalation, replacement/refund, and prevention.
- For Brainstorming/Strategy: focus on ideas, promising options, weak/rejected ideas, direction, creative approaches, execution plan, and success metrics.
- For Personal Voice Note: convert into cleaned note, ideas, tasks, reminders, priorities, and suggested next actions.
- For Podcast/Webinar/Panel: focus on speaker viewpoints, key themes, arguments, takeaways, quotes, shareable summary, and content ideas.
- For Business Meeting: focus on objective, agenda, decisions, action items, owners/deadlines, blockers, risks, and next meeting preparation.

Strict rules:
- Return only JSON. No markdown. No explanation.
- Output English only.
- Do not invent information.
- If a field has no real transcript support, return an empty string, empty array, or empty object for that field.
- Do not put placeholder values like "Not discussed", "No data", "None", or "N/A" inside arrays.
- Every list item must have meaningful title/description. Remove blank items.
- Preserve names, product codes, numbers, dates, prices, commitments, and deadlines.
- Suggestions, approaches, and guide must be practical but still grounded in the transcript.

Chunk {chunk_number} of {total_chunks}:
{transcript_text}
"""

    response_text = call_sarvam_chat(prompt)
    parsed = safe_json_parse(response_text)

    if not is_analysis_empty(parsed):
        parsed.update(classification)
        return ensure_minimum_analysis(parsed, transcript_text)

    return fallback_analysis_from_transcript(transcript_text, classification=classification)


def merge_meeting_analysis(
    chunk_results: List[Dict[str, Any]],
    transcript_text: str,
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    if not chunk_results:
        return fallback_analysis_from_transcript(transcript_text, classification=classification)

    category = classification.get("audioCategory", "general_conversation")
    category_label = classification.get("categoryLabel", category_label_for(category))
    category_schema = json.dumps(
        CATEGORY_SPECIFIC_SCHEMAS.get(category, CATEGORY_SPECIFIC_SCHEMAS["general_conversation"]),
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
{ENGLISH_ONLY_RULE}

You are a premium audio intelligence analyst.
Merge these chunk analyses into one final category-wise output.

Audio category: {category_label} ({category})

Return ONLY one valid JSON object.

Required JSON format:
{{
  "meetingTitle": "Meaningful title from the full conversation, not the first sentence, 5 to 8 words",
  "audioCategory": "{category}",
  "categoryLabel": "{category_label}",
  "categoryConfidence": {classification.get('categoryConfidence', 0.65)},
  "classificationReason": "{escape_prompt_text(classification.get('classificationReason', ''))}",
  "overview": "Final category-appropriate overview in 4 to 6 strong lines.",
  "summary": "Final complete summary in 6 to 10 lines.",
  "keyPoints": [{{"id":"key_1","title":"Key point title","description":"Important highlight."}}],
  "topics": [{{"id":"topic_1","title":"Topic title","description":"Topic details."}}],
  "discussionPoints": [{{"id":"point_1","title":"Discussion title","description":"Only if relevant.","status":"pending"}}],
  "decisions": [{{"id":"decision_1","title":"Decision title","description":"Only if a decision was made."}}],
  "actionItems": [{{"id":"action_1","title":"Action title","description":"Task details.","owner":"Owner if mentioned, otherwise Not specified","deadline":"Deadline if mentioned, otherwise Not specified","status":"pending"}}],
  "problemStatements": [{{"id":"problem_1","title":"Problem title","description":"Only if a real problem exists."}}],
  "solutions": [{{"id":"solution_1","title":"Solution title","description":"Only if a solution or approach exists.","relatedProblemId":"problem_1"}}],
  "risks": [{{"id":"risk_1","title":"Risk title","description":"Only if risk exists."}}],
  "followUps": [{{"id":"followup_1","title":"Follow-up title","description":"Only if follow-up exists."}}],
  "suggestions": [{{"id":"suggestion_1","title":"Suggestion title","description":"Useful suggestion grounded in the audio."}}],
  "approaches": [{{"id":"approach_1","title":"Approach title","description":"Practical approach grounded in the audio."}}],
  "guide": [{{"id":"guide_1","title":"Guide step title","description":"Practical guidance relevant to this category."}}],
  "workflowSteps": [{{"step":1,"title":"Step title","description":"Only if sequence exists."}}],
  "categorySpecificOutput": {category_schema},
  "coverageCheck": {{
    "importantItemsCovered": [],
    "possibleMissingItems": [],
    "unclearParts": [],
    "confidenceNotes": ""
  }},
  "mindMap": {{"title":"Audio Mind Map","nodes":[{{"id":"node_1","label":"Main topic","children":[]}}]}}
}}

Merge rules:
- The final output must match the category. Do not show generic meeting points for lecture/interview/motivation/podcast unless actually relevant.
- Remove duplicate and blank items.
- Preserve distinct decisions, tasks, risks, questions, numbers, dates, people, deadlines, prices, commitments, and names.
- Do not invent facts.
- Leave unsupported sections empty.
- Generate a meaningful meetingTitle after understanding all chunks.
- categorySpecificOutput must be filled according to {category_label}; do not use the same template for every category.
- coverageCheck must mention what was covered and any unclear/missing areas.

Chunk analyses:
{json.dumps(chunk_results, ensure_ascii=False)}
"""

    response_text = call_sarvam_chat(prompt)
    merged = safe_json_parse(response_text)

    if not is_analysis_empty(merged):
        merged.update(classification)
        return ensure_minimum_analysis(merged, transcript_text)

    return fallback_merge_analyses(chunk_results, transcript_text, classification)


def generate_meaningful_title(
    transcript_text: str,
    analysis_result: Dict[str, Any],
    classification: Dict[str, Any],
) -> str:
    category_label = classification.get("categoryLabel", "Audio Notes")
    existing_title = str(analysis_result.get("meetingTitle", "") or "").strip()

    prompt = f"""
{ENGLISH_ONLY_RULE}

Create one meaningful title for this audio after understanding the complete context.
Do NOT copy the first sentence of the transcript.
Do NOT include date, time, file extension, or speaker label.
Do NOT use generic titles like "Meeting Notes" or "Audio Summary" unless no topic is clear.

Audio category: {category_label}
Existing rough title: {existing_title}
Overview: {analysis_result.get('overview', '')}
Summary: {analysis_result.get('summary', '')}
Key topics: {json.dumps(analysis_result.get('topics', []), ensure_ascii=False)}

Transcript context:
{make_transcript_context(transcript_text, max_chars=9000)}

Return ONLY valid JSON:
{{"meetingTitle":"5 to 8 word meaningful title"}}
"""

    try:
        parsed = parse_json_object(call_sarvam_chat(prompt))
        title = str(parsed.get("meetingTitle") or parsed.get("title") or "").strip()
        title = clean_generated_title(title)
        if title:
            return title
    except Exception:
        pass

    return clean_generated_title(existing_title) or generate_title_from_analysis(analysis_result, transcript_text)


def clean_generated_title(value: str) -> str:
    title = " ".join(str(value or "").strip().split())
    if not title:
        return ""

    title = title.replace(".m4a", "").replace(".mp4", "").replace(".aac", "")
    title = title.replace("Speaker:", "").replace("Meeting Summary", "").strip(" -_:,.")

    generic = {"meeting notes", "audio notes", "audio summary", "meeting analysis", "discussion summary"}
    if title.lower() in generic:
        return ""

    words = title.split()
    if len(words) > 10:
        title = " ".join(words[:10])

    if len(title) > 80:
        title = title[:77].rstrip() + "..."

    return title


def generate_title_from_analysis(analysis_result: Dict[str, Any], transcript_text: str) -> str:
    for key in ["topics", "keyPoints", "decisions", "actionItems", "discussionPoints"]:
        items = analysis_result.get(key, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    title = clean_generated_title(str(item.get("title", "") or ""))
                    if title:
                        return title

    overview = clean_text_field(analysis_result.get("overview", ""))
    summary = clean_text_field(analysis_result.get("summary", ""))

    return generate_title_from_text(overview or summary or transcript_text)


def empty_meeting_analysis() -> Dict[str, Any]:
    return {
        "meetingTitle": "",
        "audioCategory": "general_conversation",
        "categoryLabel": category_label_for("general_conversation"),
        "categoryConfidence": 0.0,
        "classificationReason": "",
        "overview": "",
        "summary": "",
        "keyPoints": [],
        "topics": [],
        "discussionPoints": [],
        "decisions": [],
        "actionItems": [],
        "problemStatements": [],
        "solutions": [],
        "risks": [],
        "followUps": [],
        "suggestions": [],
        "approaches": [],
        "guide": [],
        "workflowSteps": [],
        "categorySpecificOutput": {},
        "coverageCheck": {},
        "mindMap": {"title": "Audio Mind Map", "nodes": []},
    }


def safe_json_parse(text: Any) -> Dict[str, Any]:
    parsed = parse_json_object(text)
    if not isinstance(parsed, dict):
        return empty_meeting_analysis()

    if isinstance(parsed.get("result"), dict):
        parsed = parsed["result"]
    if isinstance(parsed.get("analysis"), dict):
        parsed = parsed["analysis"]
    if isinstance(parsed.get("meetingAnalysis"), dict):
        parsed = parsed["meetingAnalysis"]

    return normalize_analysis_result(parsed)


def parse_json_object(text: Any) -> Dict[str, Any]:
    if text is None:
        return {}

    cleaned = str(text).strip()
    if not cleaned:
        return {}

    json_text = extract_json_object(cleaned)
    if not json_text:
        return {}

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            data = json.loads(repair_common_json_issues(json_text))
        except json.JSONDecodeError:
            return {}

    return data if isinstance(data, dict) else {}


def normalize_analysis_result(data: Dict[str, Any]) -> Dict[str, Any]:
    result = empty_meeting_analysis()

    category = str(data.get("audioCategory") or data.get("audio_category") or result["audioCategory"]).strip()
    if category not in VALID_AUDIO_CATEGORIES:
        category = "general_conversation"

    try:
        confidence = float(data.get("categoryConfidence") or data.get("category_confidence") or 0.0)
    except Exception:
        confidence = 0.0

    result["audioCategory"] = category
    result["categoryLabel"] = str(data.get("categoryLabel") or data.get("category_label") or category_label_for(category)).strip()
    result["categoryConfidence"] = max(0.0, min(confidence, 1.0))
    result["classificationReason"] = clean_text_field(data.get("classificationReason") or data.get("classification_reason") or "")
    result["meetingTitle"] = clean_generated_title(str(data.get("meetingTitle") or data.get("meeting_title") or data.get("title") or ""))
    result["overview"] = clean_text_field(data.get("overview", ""))
    result["summary"] = clean_text_field(data.get("summary", ""))

    if looks_like_json(result["overview"]):
        result["overview"] = ""
    if looks_like_json(result["summary"]):
        result["summary"] = ""

    result["keyPoints"] = normalize_text_items(data.get("keyPoints") or data.get("key_points") or data.get("highlights") or [], "key")
    result["topics"] = normalize_text_items(data.get("topics") or data.get("keyTopics") or data.get("key_topics") or [], "topic")
    result["discussionPoints"] = normalize_discussion_points(data.get("discussionPoints") or data.get("discussion_points") or [])
    result["decisions"] = normalize_text_items(data.get("decisions") or data.get("decisionsTaken") or data.get("decision_points") or [], "decision")
    result["actionItems"] = normalize_action_items(data.get("actionItems") or data.get("action_items") or data.get("tasks") or data.get("nextActions") or [])
    result["problemStatements"] = normalize_text_items(data.get("problemStatements") or data.get("problem_statements") or data.get("problems") or data.get("challenges") or [], "problem")
    result["solutions"] = normalize_solution_items(data.get("solutions") or data.get("solutionStatements") or data.get("solution_statements") or [])
    result["risks"] = normalize_text_items(data.get("risks") or data.get("concerns") or data.get("blockers") or [], "risk")
    result["followUps"] = normalize_text_items(data.get("followUps") or data.get("follow_ups") or data.get("followups") or [], "followup")
    result["suggestions"] = normalize_text_items(data.get("suggestions") or data.get("recommendations") or [], "suggestion")
    result["approaches"] = normalize_text_items(data.get("approaches") or data.get("approach") or [], "approach")
    result["guide"] = normalize_text_items(data.get("guide") or data.get("practicalGuide") or data.get("studyGuide") or [], "guide")
    result["workflowSteps"] = normalize_workflow_steps(data.get("workflowSteps") or data.get("workflow_steps") or data.get("steps") or [])
    result["categorySpecificOutput"] = normalize_dynamic_map(data.get("categorySpecificOutput") or data.get("category_specific_output") or {})
    result["coverageCheck"] = normalize_dynamic_map(data.get("coverageCheck") or data.get("coverage_check") or {})
    result["mindMap"] = normalize_mind_map(data.get("mindMap") or data.get("mind_map") or {})

    return result


def ensure_minimum_analysis(result: Dict[str, Any], transcript_text: str) -> Dict[str, Any]:
    if not isinstance(result, dict):
        result = empty_meeting_analysis()

    normalized = normalize_analysis_result(result)
    transcript_text = str(transcript_text or "").strip()

    if not normalized["overview"]:
        normalized["overview"] = generate_overview_fallback(transcript_text, normalized["summary"])
    if not normalized["summary"]:
        normalized["summary"] = generate_summary_fallback(transcript_text, normalized["overview"])
    if not normalized["meetingTitle"]:
        normalized["meetingTitle"] = generate_title_from_analysis(normalized, transcript_text)
    if not normalized["mindMap"].get("nodes"):
        normalized["mindMap"] = build_mind_map_from_analysis(normalized)

    return remove_empty_analysis_fields(normalized)


def fallback_analysis_from_transcript(
    transcript_text: str,
    classification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    classification = classification or default_classification()
    transcript_text = str(transcript_text or "").strip()

    if not transcript_text:
        result = empty_meeting_analysis()
        result.update(classification)
        result["meetingTitle"] = "Audio Analysis Unavailable"
        result["overview"] = "No clear speech transcript could be generated from this audio. Please check the recording quality and try again."
        result["summary"] = "No reliable summary could be generated because the transcript was empty or unclear."
        result["risks"] = [{"id": "risk_1", "title": "Recording Quality Issue", "description": "The uploaded audio may be empty, unclear, too short, or unsupported."}]
        result["mindMap"] = {"title": "Audio Mind Map", "nodes": [{"id": "node_1", "label": "Audio analysis unavailable", "children": []}]}
        return result

    result = empty_meeting_analysis()
    result.update(classification)
    excerpt = make_excerpt(transcript_text, max_chars=700)
    result["meetingTitle"] = generate_title_from_text(transcript_text)
    result["overview"] = f"The audio was transcribed successfully. This fallback overview is based directly on the transcript excerpt: {excerpt}"
    result["summary"] = "The transcript was captured, but the structured AI analysis response was not reliable enough to extract every section. Please review the transcript for exact details."
    result["keyPoints"] = [{"id": "key_1", "title": "Transcript Captured", "description": excerpt}]
    result["mindMap"] = {"title": "Audio Mind Map", "nodes": [{"id": "node_1", "label": result["meetingTitle"], "children": []}]}
    return result


def fallback_merge_analyses(
    chunk_results: List[Dict[str, Any]],
    transcript_text: str,
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    merged = empty_meeting_analysis()
    merged.update(classification)

    for item in chunk_results:
        normalized = ensure_minimum_analysis(item, "")
        for key in ["keyPoints", "topics", "discussionPoints", "decisions", "actionItems", "problemStatements", "solutions", "risks", "followUps", "suggestions", "approaches", "guide", "workflowSteps"]:
            existing = merged.get(key, [])
            if isinstance(existing, list):
                existing.extend(normalized.get(key, []))
                merged[key] = existing

        if normalized.get("overview"):
            merged["overview"] = (merged["overview"] + "\n" + normalized["overview"]).strip()
        if normalized.get("summary"):
            merged["summary"] = (merged["summary"] + "\n" + normalized["summary"]).strip()

        merged["categorySpecificOutput"] = merge_dynamic_maps(merged.get("categorySpecificOutput", {}), normalized.get("categorySpecificOutput", {}))
        merged["coverageCheck"] = merge_dynamic_maps(merged.get("coverageCheck", {}), normalized.get("coverageCheck", {}))

    return ensure_minimum_analysis(merged, transcript_text)


def remove_empty_analysis_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(result)

    for key in ["keyPoints", "topics", "discussionPoints", "decisions", "actionItems", "problemStatements", "solutions", "risks", "followUps", "suggestions", "approaches", "guide", "workflowSteps"]:
        value = cleaned.get(key, [])
        if isinstance(value, list):
            cleaned[key] = [item for item in value if has_displayable_value(item)]

    cleaned["categorySpecificOutput"] = normalize_dynamic_map(cleaned.get("categorySpecificOutput", {}))
    cleaned["coverageCheck"] = normalize_dynamic_map(cleaned.get("coverageCheck", {}))

    return cleaned


def merge_dynamic_maps(first: Any, second: Any) -> Dict[str, Any]:
    result = normalize_dynamic_map(first)
    other = normalize_dynamic_map(second)

    for key, value in other.items():
        if key not in result or not has_displayable_value(result[key]):
            result[key] = value
        elif isinstance(result[key], list) and isinstance(value, list):
            result[key].extend(value)
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dynamic_maps(result[key], value)

    return normalize_dynamic_map(result)


def normalize_dynamic_map(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    for key, item in value.items():
        cleaned_item = normalize_dynamic_value(item)
        if has_displayable_value(cleaned_item):
            cleaned[str(key)] = cleaned_item

    return cleaned


def normalize_dynamic_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        text = clean_text_field(value)
        if text.lower().strip() in {"n/a", "na", "none", "not applicable", "not discussed", "no data", "not mentioned"}:
            return ""
        return text
    if isinstance(value, list):
        cleaned_list = [normalize_dynamic_value(item) for item in value]
        return [item for item in cleaned_list if has_displayable_value(item)]
    if isinstance(value, dict):
        return normalize_dynamic_map(value)
    return value


def has_displayable_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(has_displayable_value(item) for item in value)
    if isinstance(value, dict):
        return any(has_displayable_value(item) for item in value.values())
    return bool(str(value).strip())


def normalize_discussion_points(value: Any) -> List[Dict[str, Any]]:
    items = normalize_list_of_dicts(value)
    normalized = []
    seen = set()

    for item in items:
        title = clean_text_field(item.get("title") or item.get("heading") or "")
        description = clean_text_field(item.get("description") or item.get("details") or item.get("text") or "")
        if not title and not description:
            continue

        status = str(item.get("status", "pending") or "pending").lower().strip()
        if status not in {"done", "pending", "not_done", "in_progress"}:
            status = "pending"

        key = f"{title.lower()}::{description.lower()}"
        if key in seen:
            continue
        seen.add(key)

        normalized.append({
            "id": f"point_{len(normalized) + 1}",
            "title": title or f"Point {len(normalized) + 1}",
            "description": description,
            "status": status,
        })

    return normalized


def normalize_text_items(value: Any, id_prefix: str) -> List[Dict[str, Any]]:
    items = normalize_list_of_dicts(value)
    normalized = []
    seen = set()

    for item in items:
        title = clean_text_field(item.get("title") or item.get("heading") or item.get("name") or "")
        description = clean_text_field(item.get("description") or item.get("details") or item.get("text") or item.get("value") or "")
        if not title and not description:
            continue

        key = f"{title.lower()}::{description.lower()}"
        if key in seen:
            continue
        seen.add(key)

        normalized.append({
            "id": f"{id_prefix}_{len(normalized) + 1}",
            "title": title or f"{id_prefix.title()} {len(normalized) + 1}",
            "description": description,
        })

    return normalized


def normalize_action_items(value: Any) -> List[Dict[str, Any]]:
    items = normalize_list_of_dicts(value)
    normalized = []
    seen = set()

    for item in items:
        title = clean_text_field(item.get("title") or item.get("task") or item.get("heading") or "")
        description = clean_text_field(item.get("description") or item.get("details") or item.get("text") or "")
        if not title and not description:
            continue

        owner = clean_text_field(item.get("owner") or item.get("assignedTo") or "Not specified") or "Not specified"
        deadline = clean_text_field(item.get("deadline") or item.get("dueDate") or "Not specified") or "Not specified"
        status = str(item.get("status", "pending") or "pending").lower().strip()
        if status not in {"done", "pending", "not_done", "in_progress"}:
            status = "pending"

        key = f"{title.lower()}::{description.lower()}::{owner.lower()}::{deadline.lower()}"
        if key in seen:
            continue
        seen.add(key)

        normalized.append({
            "id": f"action_{len(normalized) + 1}",
            "title": title or f"Action {len(normalized) + 1}",
            "description": description,
            "owner": owner,
            "deadline": deadline,
            "status": status,
        })

    return normalized


def normalize_solution_items(value: Any) -> List[Dict[str, Any]]:
    items = normalize_list_of_dicts(value)
    normalized = []
    seen = set()

    for item in items:
        title = clean_text_field(item.get("title") or item.get("heading") or "")
        description = clean_text_field(item.get("description") or item.get("details") or item.get("text") or "")
        if not title and not description:
            continue

        related = clean_text_field(item.get("relatedProblemId") or item.get("related_problem_id") or "")
        key = f"{title.lower()}::{description.lower()}::{related.lower()}"
        if key in seen:
            continue
        seen.add(key)

        normalized.append({
            "id": f"solution_{len(normalized) + 1}",
            "title": title or f"Solution {len(normalized) + 1}",
            "description": description,
            "relatedProblemId": related,
        })

    return normalized


def normalize_workflow_steps(value: Any) -> List[Dict[str, Any]]:
    items = normalize_list_of_dicts(value)
    normalized = []

    for item in items:
        title = clean_text_field(item.get("title") or item.get("heading") or "")
        description = clean_text_field(item.get("description") or item.get("details") or item.get("text") or "")
        if not title and not description:
            continue

        normalized.append({
            "step": len(normalized) + 1,
            "title": title or f"Step {len(normalized) + 1}",
            "description": description,
        })

    return normalized


def normalize_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    items = []
    for item in value:
        if isinstance(item, str):
            text = clean_text_field(item)
            if text:
                items.append({"description": text})
        elif isinstance(item, dict):
            items.append(dict(item))

    return items


def normalize_mind_map(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {"title": "Audio Mind Map", "nodes": []}

    title = clean_text_field(value.get("title") or "Audio Mind Map") or "Audio Mind Map"
    nodes = normalize_mind_map_nodes(value.get("nodes", []))
    return {"title": title, "nodes": nodes}


def normalize_mind_map_nodes(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    nodes = []
    for item in value:
        if isinstance(item, str):
            label = clean_text_field(item)
            children = []
        elif isinstance(item, dict):
            label = clean_text_field(item.get("label") or item.get("title") or "")
            children = normalize_mind_map_nodes(item.get("children", []))
        else:
            continue

        if not label:
            continue

        nodes.append({
            "id": f"node_{len(nodes) + 1}",
            "label": label,
            "children": children,
        })

    return nodes


def build_mind_map_from_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    groups = [
        ("Key Topics", result.get("topics", [])),
        ("Key Points", result.get("keyPoints", [])),
        ("Decisions", result.get("decisions", [])),
        ("Action Items", result.get("actionItems", [])),
        ("Problems", result.get("problemStatements", [])),
        ("Solutions", result.get("solutions", [])),
        ("Guide", result.get("guide", [])),
    ]

    nodes = []
    for group_label, items in groups:
        if not isinstance(items, list) or not items:
            continue
        children = []
        for item in items[:6]:
            if not isinstance(item, dict):
                continue
            label = clean_text_field(item.get("title") or item.get("description") or "")
            if label:
                children.append({"id": f"node_{len(nodes) + 1}_{len(children) + 1}", "label": label, "children": []})
        if children:
            nodes.append({"id": f"node_{len(nodes) + 1}", "label": group_label, "children": children})

    return {"title": "Audio Mind Map", "nodes": nodes}


def is_analysis_empty(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return True
    for key in ["overview", "summary", "meetingTitle", "keyPoints", "topics", "discussionPoints", "decisions", "actionItems", "problemStatements", "solutions", "risks", "followUps", "suggestions", "approaches", "guide", "workflowSteps", "categorySpecificOutput"]:
        if has_displayable_value(result.get(key)):
            return False
    return True


def extract_json_object(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    for start_index, char in enumerate(cleaned):
        if char != "{":
            continue
        candidate = extract_balanced_json_from_index(cleaned, start_index)
        if candidate:
            return candidate.strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace == -1 or last_brace <= first_brace:
        return ""
    return cleaned[first_brace:last_brace + 1].strip()


def extract_balanced_json_from_index(text: str, start_index: int) -> str:
    depth = 0
    in_string = False
    escape = False

    for index in range(start_index, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index:index + 1]
    return ""


def repair_common_json_issues(text: str) -> str:
    repaired = str(text or "").strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = repaired.replace(",\n}", "\n}").replace(",\n]", "\n]")
    repaired = repaired.replace(",}", "}").replace(",]", "]")
    return repaired


def clean_text_field(value: Any) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    if cleaned.lower().strip() in {"n/a", "na", "none", "not applicable", "not discussed", "no data", "not mentioned"}:
        return ""
    return cleaned


def looks_like_json(value: str) -> bool:
    cleaned = str(value or "").strip()
    if not cleaned:
        return False
    return cleaned.startswith("{") or cleaned.startswith("[") or '"overview"' in cleaned or '"discussionPoints"' in cleaned or '"audioCategory"' in cleaned


def generate_title_from_text(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return "Audio Notes"

    cleaned = cleaned.replace("Speaker:", "").strip()
    sentence_parts = [part.strip() for part in cleaned.replace("?", ".").replace("!", ".").split(".") if part.strip()]

    candidate = ""
    for part in sentence_parts[:5]:
        words = [word for word in part.split() if len(word) > 2]
        if len(words) >= 4:
            candidate = " ".join(words[:8])
            break

    if not candidate:
        words = [word for word in cleaned.split() if len(word) > 2]
        candidate = " ".join(words[:8])

    return clean_generated_title(candidate) or "Audio Notes"


def generate_overview_fallback(transcript_text: str, summary: str) -> str:
    if summary and not looks_like_json(summary):
        return summary
    transcript_text = str(transcript_text or "").strip()
    if transcript_text:
        return f"The audio transcript was captured successfully. Key excerpt: {make_excerpt(transcript_text, max_chars=600)}"
    return "No clear overview could be generated from this audio."


def generate_summary_fallback(transcript_text: str, overview: str) -> str:
    if overview and not looks_like_json(overview):
        return overview
    transcript_text = str(transcript_text or "").strip()
    if transcript_text:
        return f"The audio was transcribed, but a detailed structured summary could not be generated. Transcript excerpt: {make_excerpt(transcript_text, max_chars=600)}"
    return "No clear summary could be generated from this audio."


def make_excerpt(text: str, max_chars: int = 600) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def make_transcript_context(text: str, max_chars: int = 9000) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned

    part_len = max_chars // 3
    start = cleaned[:part_len]
    middle_start = max(0, (len(cleaned) // 2) - (part_len // 2))
    middle = cleaned[middle_start:middle_start + part_len]
    end = cleaned[-part_len:]

    return f"START:\n{start}\n\nMIDDLE:\n{middle}\n\nEND:\n{end}"


def escape_prompt_text(value: Any) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', "'").replace("\n", " ").strip()


def ask_from_transcript(transcript_text: str, question: str) -> str:
    transcript_text = str(transcript_text or "").strip()
    question = str(question or "").strip()

    if not transcript_text:
        return "I could not find any transcript text to answer from."

    if not question:
        return "Please ask a question about the meeting."

    relevant_chunks = find_relevant_transcript_chunks(
        transcript_text=transcript_text,
        question=question,
        max_chunks=5,
    )

    evidence_text = "\n\n".join(relevant_chunks).strip()

    if not evidence_text:
        evidence_text = transcript_text[:6000]

    prompt = f"""
{ENGLISH_ONLY_RULE}

Answer the user's question based only on the meeting transcript evidence below.

If the answer is not available in the evidence, say:
"I could not find this information in the meeting transcript."

Rules:
- Answer only in English.
- Be direct and useful.
- Do not invent information.
- If the question is in Hindi, Bengali, Hinglish, Banglish, or mixed language, understand it and answer in English.
- If the evidence contains Hindi, Bengali, Hinglish, Banglish, or mixed language, translate the meaning and answer in English.
- If there are decisions, numbers, deadlines, dates, names, or commitments, preserve them exactly.
- If useful, include a short "Evidence:" line using the relevant transcript wording.

Meeting transcript evidence:
{evidence_text}

User question:
{question}
"""

    answer = call_sarvam_chat(prompt)

    if answer is None:
        return ""

    return str(answer).strip()


def find_relevant_transcript_chunks(
    transcript_text: str,
    question: str,
    max_chunks: int = 5,
) -> List[str]:
    chunks = split_text_into_chunks(transcript_text, max_words=350)

    if not chunks:
        return []

    question_terms = {
        term.strip().lower()
        for term in question.replace("?", " ").replace(",", " ").replace(".", " ").split()
        if len(term.strip()) >= 3
    }

    scored_chunks = []

    for index, chunk in enumerate(chunks):
        lower_chunk = chunk.lower()
        score = 0

        for term in question_terms:
            if term in lower_chunk:
                score += 1

        if score > 0:
            scored_chunks.append((score, index, chunk))

    if not scored_chunks:
        return chunks[:max_chunks]

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    selected = scored_chunks[:max_chunks]
    selected.sort(key=lambda item: item[1])

    return [item[2] for item in selected]


def call_sarvam_chat(prompt: str) -> str:
    url = "https://api.sarvam.ai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_API_KEY,
    }

    final_prompt = f"{ENGLISH_ONLY_RULE}\n\n{prompt}" if FORCE_ENGLISH_ONLY else prompt

    payload = {
        "model": SARVAM_CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": final_prompt,
            }
        ],
        "temperature": 0.1,
        "max_tokens": SARVAM_CHAT_MAX_TOKENS,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=180,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Sarvam chat error {response.status_code}: {response.text}"
        )

    data = response.json()
    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(f"Sarvam chat returned no choices: {data}")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if content is None:
        content = ""

    return str(content)


def split_text_into_chunks(text: str, max_words: int = 2500) -> List[str]:
    words = str(text or "").split()

    if not words:
        return []

    chunks = []

    for start in range(0, len(words), max_words):
        chunk_words = words[start:start + max_words]
        chunks.append(" ".join(chunk_words))

    return chunks


def find_first_json_file(directory: str) -> Optional[str]:
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".json"):
                return os.path.join(root, file_name)

    return None


def format_seconds(value: Any) -> str:
    try:
        total_seconds = int(float(value or 0))
    except Exception:
        total_seconds = 0

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"
