import json
import os
import re
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


TITLE_STOPWORDS = {
    "about", "after", "again", "also", "audio", "because", "before", "being", "between", "could", "discussion",
    "everyone", "first", "from", "going", "hello", "here", "into", "just", "like", "meeting", "notes", "okay",
    "once", "only", "really", "should", "something", "speaker", "summary", "that", "their", "there", "these", "thing",
    "this", "today", "transcript", "very", "want", "what", "when", "where", "which", "with", "would", "your",
    "have", "will", "they", "them", "then", "than", "were", "been", "need", "time", "know", "make", "take",
    "give", "come", "done", "said", "tell", "talk", "talking", "discuss", "discussed", "point", "points",
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

    # Final premium pass: this is the important improvement.
    # It converts transcript-like notes into category-wise intelligence.
    refined = refine_analysis_premium(
        transcript_text=transcript_text,
        draft_analysis=analysis,
        classification=classification,
    )

    if not is_analysis_empty(refined):
        analysis = refined

    analysis.update(classification)
    analysis = ensure_minimum_analysis(analysis, transcript_text)
    analysis = ensure_category_specific_output(
        analysis_result=analysis,
        classification=classification,
        transcript_text=transcript_text,
    )
    analysis = adapt_analysis_for_category(analysis, classification)

    title = generate_meaningful_title(
        transcript_text=transcript_text,
        analysis_result=analysis,
        classification=classification,
    )

    if title:
        analysis["meetingTitle"] = title

    analysis = ensure_minimum_analysis(analysis, transcript_text)
    analysis = adapt_analysis_for_category(analysis, classification)
    return remove_empty_analysis_fields(analysis)


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

You are a premium audio intelligence analyst, not a normal transcript summarizer.
The audio category is: {category_label} ({category}).

Your job:
1. Understand the meaning of the audio.
2. Convert it into useful intelligence for the category.
3. Do NOT rewrite the transcript line by line.
4. Do NOT make every audio look like a business meeting.
5. Keep unsupported fields empty.

For this category, follow this output style:
{category_output_guidance(category)}

Return ONLY one valid JSON object.

Required JSON format:
{{
  "meetingTitle": "temporary analytical title, not copied from transcript opening",
  "audioCategory": "{category}",
  "categoryLabel": "{category_label}",
  "categoryConfidence": {classification.get('categoryConfidence', 0.65)},
  "classificationReason": "{escape_prompt_text(classification.get('classificationReason', ''))}",
  "overview": "Analytical overview. Explain purpose, context, and value. Do not copy transcript.",
  "summary": "Structured summary in natural professional language, not a transcript rewrite.",
  "keyPoints": [{{"id":"key_1","title":"Insight title","description":"Important insight, learning, conclusion, or takeaway."}}],
  "topics": [{{"id":"topic_1","title":"Topic title","description":"What this topic means and why it matters."}}],
  "discussionPoints": [{{"id":"point_1","title":"Only for real discussion/review point","description":"Use only when discussion format is relevant.","status":"pending"}}],
  "decisions": [{{"id":"decision_1","title":"Decision title","description":"Only if a decision/conclusion was made."}}],
  "actionItems": [{{"id":"action_1","title":"Action title","description":"Task details.","owner":"Owner if explicitly mentioned","deadline":"Deadline if explicitly mentioned","status":"pending"}}],
  "problemStatements": [{{"id":"problem_1","title":"Problem/challenge title","description":"Only if a real issue, blocker, doubt, gap, risk, or concern appears."}}],
  "solutions": [{{"id":"solution_1","title":"Solution/answer title","description":"Only if the audio gives a solution, answer, approach, or recommendation.","relatedProblemId":"problem_1"}}],
  "risks": [{{"id":"risk_1","title":"Risk/concern title","description":"Only if a risk, objection, dependency, uncertainty, or weakness exists."}}],
  "followUps": [{{"id":"followup_1","title":"Follow-up title","description":"Only if follow-up is required."}}],
  "suggestions": [{{"id":"suggestion_1","title":"Useful suggestion","description":"Useful, practical suggestion derived from the audio context."}}],
  "approaches": [{{"id":"approach_1","title":"Approach title","description":"Practical way to act on the audio."}}],
  "guide": [{{"id":"guide_1","title":"Guide step title","description":"Step-by-step guidance relevant to the category."}}],
  "workflowSteps": [{{"step":1,"title":"Process step","description":"Only if a real sequence/process exists."}}],
  "categorySpecificOutput": {category_schema},
  "coverageCheck": {{
    "importantItemsCovered": [],
    "possibleMissingItems": [],
    "unclearParts": [],
    "confidenceNotes": ""
  }},
  "mindMap": {{"title":"Audio Mind Map","nodes":[{{"id":"node_1","label":"Main theme","children":[]}}]}}
}}

Critical quality rules:
- Do not copy the first sentence as the title.
- Do not create bland transcript bullets like "Speaker discussed...".
- Every point must add structure, meaning, implication, decision, learning, action, risk, or next step.
- For lecture/training: keep discussionPoints empty unless there was a real discussion. Use topics, keyPoints, categorySpecificOutput, guide.
- For motivational speech: keep decisions/actionItems/discussionPoints empty unless truly present. Use takeaways, principles, guide, suggestions.
- For interview: put questions/answers/strengths/concerns inside categorySpecificOutput. Do not force meeting decisions.
- For personal voice note: convert into cleaned note, tasks, reminders, priorities, and next actions.
- For support/complaint: extract issue, root cause, sentiment, resolution, escalation, prevention.
- For sales/vendor: extract needs, objections, commercial points, commitments, follow-up, deal status.
- For product/project: extract requirements, bugs, dependencies, roadmap, testing, release readiness.
- For business meeting: extract decisions, owners, deadlines, blockers, risks, next meeting preparation.
- If an array has no supported item, return []. Never use placeholders.
- Preserve all names, product codes, article codes, numbers, dates, prices, commitments, and deadlines.
- Suggestions and guide can be practical, but they must be clearly based on the audio context.

Transcript chunk {chunk_number} of {total_chunks}:
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

You are creating a final premium audio intelligence report.
Merge the chunk analyses into one category-wise result.

Audio category: {category_label} ({category})
Category guidance:
{category_output_guidance(category)}

Return ONLY one valid JSON object.

Required JSON format:
{{
  "meetingTitle": "meaningful title based on full audio, not transcript opening",
  "audioCategory": "{category}",
  "categoryLabel": "{category_label}",
  "categoryConfidence": {classification.get('categoryConfidence', 0.65)},
  "classificationReason": "{escape_prompt_text(classification.get('classificationReason', ''))}",
  "overview": "Strong final overview in 4 to 6 lines.",
  "summary": "Complete synthesized summary in 6 to 10 lines. Do not rewrite transcript.",
  "keyPoints": [{{"id":"key_1","title":"Insight title","description":"Important final insight."}}],
  "topics": [{{"id":"topic_1","title":"Topic title","description":"Topic meaning and importance."}}],
  "discussionPoints": [{{"id":"point_1","title":"Discussion/review title","description":"Only if relevant to this category.","status":"pending"}}],
  "decisions": [{{"id":"decision_1","title":"Decision title","description":"Only if decision/conclusion exists."}}],
  "actionItems": [{{"id":"action_1","title":"Action title","description":"Task details.","owner":"Owner if explicitly mentioned","deadline":"Deadline if explicitly mentioned","status":"pending"}}],
  "problemStatements": [{{"id":"problem_1","title":"Problem title","description":"Real issue, blocker, doubt, gap, or challenge."}}],
  "solutions": [{{"id":"solution_1","title":"Solution title","description":"Solution, answer, approach, or recommendation.","relatedProblemId":"problem_1"}}],
  "risks": [{{"id":"risk_1","title":"Risk title","description":"Risk, concern, objection, dependency, or uncertainty."}}],
  "followUps": [{{"id":"followup_1","title":"Follow-up title","description":"Only if follow-up is needed."}}],
  "suggestions": [{{"id":"suggestion_1","title":"Suggestion title","description":"Practical suggestion derived from the audio."}}],
  "approaches": [{{"id":"approach_1","title":"Approach title","description":"Practical approach grounded in the audio."}}],
  "guide": [{{"id":"guide_1","title":"Guide step title","description":"Step-by-step guidance relevant to the category."}}],
  "workflowSteps": [{{"step":1,"title":"Step title","description":"Only if a process or sequence exists."}}],
  "categorySpecificOutput": {category_schema},
  "coverageCheck": {{
    "importantItemsCovered": [],
    "possibleMissingItems": [],
    "unclearParts": [],
    "confidenceNotes": ""
  }},
  "mindMap": {{"title":"Audio Mind Map","nodes":[{{"id":"node_1","label":"Main theme","children":[]}}]}}
}}

Merge and quality rules:
- Remove duplicates and transcript-like repetition.
- Do not force the same sections for every audio type.
- Leave unsupported sections empty.
- Put the strongest category-wise information into categorySpecificOutput.
- For lecture, motivation, personal voice note, and interview, avoid generic Discussion Points unless real discussion exists.
- Include practical Suggestions, Approaches, and Guide only when helpful and grounded.
- Use coverageCheck to identify covered areas, unclear parts, and potential missing details.
- Preserve names, numbers, dates, prices, product codes, deadlines, commitments, and speaker-specific facts.

Chunk analyses:
{json.dumps(chunk_results, ensure_ascii=False)}
"""

    response_text = call_sarvam_chat(prompt)
    merged = safe_json_parse(response_text)

    if not is_analysis_empty(merged):
        merged.update(classification)
        return ensure_minimum_analysis(merged, transcript_text)

    return fallback_merge_analyses(chunk_results, transcript_text, classification)


def refine_analysis_premium(
    transcript_text: str,
    draft_analysis: Dict[str, Any],
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

You are the final quality layer of a premium AI audio notes product.
The current draft may be too close to the transcript. Rewrite it into a better category-wise output.

Audio category: {category_label} ({category})
Category guidance:
{category_output_guidance(category)}

What to improve:
- Create insight, structure, and usefulness beyond plain transcript summary.
- Make the title meaningful from the complete audio, not the first sentence.
- Fill categorySpecificOutput according to the category schema.
- Keep only relevant top-level sections. Empty irrelevant sections.
- Add practical suggestions, approaches, and guide when useful.
- Preserve real facts only. Do not invent people, numbers, dates, amounts, promises, or decisions.

Return ONLY valid JSON with this exact shape:
{{
  "meetingTitle": "5 to 8 word context-aware title",
  "audioCategory": "{category}",
  "categoryLabel": "{category_label}",
  "categoryConfidence": {classification.get('categoryConfidence', 0.65)},
  "classificationReason": "{escape_prompt_text(classification.get('classificationReason', ''))}",
  "overview": "premium overview",
  "summary": "premium synthesized summary",
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
  "categorySpecificOutput": {category_schema},
  "coverageCheck": {{
    "importantItemsCovered": [],
    "possibleMissingItems": [],
    "unclearParts": [],
    "confidenceNotes": ""
  }},
  "mindMap": {{"title":"Audio Mind Map","nodes":[]}}
}}

Item format rules:
- For common lists use objects with id, title, description.
- For actionItems include id, title, description, owner, deadline, status.
- For discussionPoints include id, title, description, status.
- For workflowSteps include step, title, description.
- No placeholders like N/A, None, Not discussed, Not specified inside arrays. Use empty array instead.

Draft analysis:
{json.dumps(draft_analysis, ensure_ascii=False)}

Transcript context from start, middle, and end:
{make_transcript_context(transcript_text, max_chars=14000)}
"""

    try:
        parsed = safe_json_parse(call_sarvam_chat(prompt))
        if not is_analysis_empty(parsed):
            parsed.update(classification)
            return ensure_minimum_analysis(parsed, transcript_text)
    except Exception:
        pass

    return draft_analysis


def category_output_guidance(category: str) -> str:
    guidance = {
        "business_meeting": "Use executive notes: objective, agenda, decisions, action items, owners, deadlines, blockers, risks, next meeting preparation. Discussion points are relevant here.",
        "sales_client_vendor_call": "Use commercial notes: client/vendor need, objections, price/commercial terms, commitments, deal status, follow-up plan, suggested response. Discussion points are relevant only if they track negotiation topics.",
        "lecture_class_training": "Use learning notes: topic taught, learning objectives, key concepts, definitions, examples, formulas/frameworks, revision notes, practice questions, study guide. Avoid meeting-style decisions and discussion points unless students actually discussed something.",
        "interview_hr_discussion": "Use evaluation notes: interview purpose, candidate profile, questions asked, answer summaries, strengths, concerns, skills observed, culture fit, recommendation, next round questions. Avoid generic meeting discussion points.",
        "motivational_speech_seminar": "Use inspirational intelligence: core message, key takeaways, memorable quotes, life lessons, action principles, practical guide, audience impact, daily actions. Avoid decisions/action items unless explicit.",
        "brainstorming_strategy": "Use strategy notes: goal, ideas generated, promising ideas, weak ideas, strategic direction, creative approaches, execution plan, success metrics. Discussion points can be used if ideas were debated.",
        "product_project_discussion": "Use project intelligence: project name, requirements, features, bugs/issues, technical decisions, dependencies, roadmap, testing checklist, release readiness. Include action items when owners/tasks exist.",
        "finance_legal_compliance": "Use finance/legal notes: financial summary, amounts, documents, approvals, compliance risks, legal concerns, payment/refund status, next compliance steps. Avoid creative suggestions not grounded in compliance context.",
        "customer_support_complaint": "Use support notes: customer issue, sentiment, root cause, resolution, escalation, refund/replacement, support follow-up, prevention suggestions. Focus on solving the issue.",
        "personal_voice_note_idea": "Use personal productivity notes: cleaned note, ideas, tasks, reminders, priorities, next actions, converted professional note. Avoid meeting sections.",
        "podcast_webinar_panel": "Use content notes: main topic, speaker viewpoints, key themes, arguments, audience takeaways, quotes, shareable summary, content ideas. Avoid action items unless explicit.",
        "general_conversation": "Use universal notes: conversation summary, important points, possible actions, people/dates/numbers, unclear sections, suggested category.",
    }
    return guidance.get(category, guidance["general_conversation"])


def ensure_category_specific_output(
    analysis_result: Dict[str, Any],
    classification: Dict[str, Any],
    transcript_text: str,
) -> Dict[str, Any]:
    result = dict(analysis_result)
    category = classification.get("audioCategory", result.get("audioCategory", "general_conversation"))
    current = normalize_dynamic_map(result.get("categorySpecificOutput", {}))

    if current:
        result["categorySpecificOutput"] = current
        return result

    key_points = result.get("keyPoints", [])
    topics = result.get("topics", [])
    decisions = result.get("decisions", [])
    action_items = result.get("actionItems", [])
    problems = result.get("problemStatements", [])
    solutions = result.get("solutions", [])
    risks = result.get("risks", [])
    followups = result.get("followUps", [])
    suggestions = result.get("suggestions", [])
    guide = result.get("guide", [])
    summary = result.get("summary", "")
    overview = result.get("overview", "")
    title = result.get("meetingTitle", "") or generate_contextual_title_fallback(result, transcript_text)

    if category == "business_meeting":
        current = {
            "meetingObjective": overview,
            "keyAgenda": map_items_to_dynamic_list(topics or key_points),
            "decisionsTaken": map_items_to_dynamic_list(decisions),
            "actionItems": map_items_to_dynamic_list(action_items),
            "ownersAndDeadlines": map_items_to_dynamic_list(action_items),
            "blockers": map_items_to_dynamic_list(problems),
            "risks": map_items_to_dynamic_list(risks),
            "nextMeetingPreparation": map_items_to_dynamic_list(followups or guide),
            "managementSummary": summary,
        }
    elif category == "sales_client_vendor_call":
        current = {
            "purposeOfCall": overview,
            "clientNeeds": map_items_to_dynamic_list(topics or key_points),
            "objectionsOrConcerns": map_items_to_dynamic_list(risks or problems),
            "priceOrCommercialPoints": extract_commercial_mentions(transcript_text),
            "commitmentsMade": map_items_to_dynamic_list(decisions or action_items),
            "followUpPlan": map_items_to_dynamic_list(followups or action_items),
            "suggestedResponse": map_items_to_dynamic_list(suggestions),
            "dealStatus": infer_deal_status(result, transcript_text),
        }
    elif category == "lecture_class_training":
        current = {
            "topicTaught": title,
            "learningObjectives": map_items_to_dynamic_list(topics[:3] or key_points[:3]),
            "keyConcepts": map_items_to_dynamic_list(key_points or topics),
            "definitions": [],
            "examplesExplained": extract_example_sentences(transcript_text),
            "importantFormulasOrFrameworks": [],
            "revisionNotes": map_items_to_dynamic_list(topics or key_points),
            "questionsToPractice": build_practice_questions_from_topics(topics or key_points),
            "studyGuide": map_items_to_dynamic_list(guide or key_points),
        }
    elif category == "interview_hr_discussion":
        current = {
            "interviewPurpose": overview,
            "candidateProfile": extract_candidate_profile(transcript_text),
            "questionsAsked": extract_question_sentences(transcript_text),
            "answerSummary": map_items_to_dynamic_list(key_points or topics),
            "strengths": map_items_to_dynamic_list(solutions or key_points[:3]),
            "weaknessesOrConcerns": map_items_to_dynamic_list(risks or problems),
            "skillsObserved": extract_skill_mentions(transcript_text),
            "cultureFitNotes": summary,
            "finalRecommendation": infer_interview_recommendation(result),
            "nextRoundQuestions": build_practice_questions_from_topics(risks or topics),
        }
    elif category == "motivational_speech_seminar":
        current = {
            "coreMessage": overview or summary,
            "keyTakeaways": map_items_to_dynamic_list(key_points or topics),
            "memorableQuotes": extract_quote_like_lines(transcript_text),
            "lifeLessons": map_items_to_dynamic_list(topics or key_points),
            "actionPrinciples": map_items_to_dynamic_list(guide or suggestions),
            "practicalGuide": map_items_to_dynamic_list(guide),
            "audienceImpact": summary,
            "suggestedDailyActions": map_items_to_dynamic_list(suggestions or guide),
        }
    elif category == "brainstorming_strategy":
        current = {
            "brainstormingGoal": overview,
            "ideasGenerated": map_items_to_dynamic_list(topics or key_points),
            "promisingIdeas": map_items_to_dynamic_list(suggestions or solutions),
            "rejectedOrWeakIdeas": map_items_to_dynamic_list(risks or problems),
            "strategicDirection": map_items_to_dynamic_list(decisions or key_points),
            "creativeApproaches": map_items_to_dynamic_list(result.get("approaches", []) or suggestions),
            "executionPlan": map_items_to_dynamic_list(action_items or guide),
            "successMetrics": extract_metric_mentions(transcript_text),
        }
    elif category == "product_project_discussion":
        current = {
            "projectName": title,
            "requirements": map_items_to_dynamic_list(topics or key_points),
            "featuresDiscussed": map_items_to_dynamic_list(topics),
            "bugsOrIssues": map_items_to_dynamic_list(problems),
            "technicalDecisions": map_items_to_dynamic_list(decisions),
            "dependencies": map_items_to_dynamic_list(risks),
            "roadmap": map_items_to_dynamic_list(action_items or followups),
            "testingChecklist": map_items_to_dynamic_list(guide),
            "releaseReadiness": summary,
        }
    elif category == "finance_legal_compliance":
        current = {
            "financialSummary": overview or summary,
            "amountsMentioned": extract_commercial_mentions(transcript_text),
            "documentsRequired": extract_document_mentions(transcript_text),
            "approvalsNeeded": map_items_to_dynamic_list(action_items or followups),
            "complianceRisks": map_items_to_dynamic_list(risks or problems),
            "legalConcerns": map_items_to_dynamic_list(problems),
            "paymentOrRefundStatus": infer_payment_status(transcript_text),
            "nextComplianceSteps": map_items_to_dynamic_list(guide or action_items),
        }
    elif category == "customer_support_complaint":
        current = {
            "customerIssue": overview,
            "customerSentiment": infer_sentiment(transcript_text),
            "rootCause": first_item_description(problems),
            "resolutionSuggested": map_items_to_dynamic_list(solutions or suggestions),
            "escalationNeeded": bool(risks or followups),
            "refundOrReplacementMentioned": infer_refund_replacement(transcript_text),
            "supportFollowUp": map_items_to_dynamic_list(followups or action_items),
            "preventionSuggestion": map_items_to_dynamic_list(suggestions),
        }
    elif category == "personal_voice_note_idea":
        current = {
            "cleanedNote": summary or overview,
            "ideas": map_items_to_dynamic_list(topics or key_points),
            "tasks": map_items_to_dynamic_list(action_items),
            "reminders": map_items_to_dynamic_list(followups),
            "priorityItems": map_items_to_dynamic_list(key_points[:5]),
            "suggestedNextActions": map_items_to_dynamic_list(suggestions or guide),
            "convertedToProfessionalNote": summary or overview,
        }
    elif category == "podcast_webinar_panel":
        current = {
            "mainTopic": title,
            "speakerViewpoints": map_items_to_dynamic_list(topics or key_points),
            "keyThemes": map_items_to_dynamic_list(key_points or topics),
            "importantArguments": map_items_to_dynamic_list(result.get("approaches", []) or key_points),
            "audienceTakeaways": map_items_to_dynamic_list(suggestions or key_points),
            "interestingQuotes": extract_quote_like_lines(transcript_text),
            "summaryForSharing": summary,
            "contentIdeasFromAudio": map_items_to_dynamic_list(suggestions),
        }
    else:
        current = {
            "conversationSummary": summary or overview,
            "importantPoints": map_items_to_dynamic_list(key_points or topics),
            "possibleActions": map_items_to_dynamic_list(action_items or suggestions),
            "peopleMentioned": extract_people_like_terms(transcript_text),
            "datesOrNumbersMentioned": extract_metric_mentions(transcript_text),
            "unclearSections": [],
            "suggestedCategory": category_label_for(category),
        }

    result["categorySpecificOutput"] = normalize_dynamic_map(current)
    return result


def adapt_analysis_for_category(
    analysis_result: Dict[str, Any],
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    result = dict(analysis_result)
    category = classification.get("audioCategory", result.get("audioCategory", "general_conversation"))

    # These categories should not look like generic business meetings.
    non_discussion_categories = {
        "lecture_class_training",
        "motivational_speech_seminar",
        "interview_hr_discussion",
        "personal_voice_note_idea",
    }

    if category in non_discussion_categories:
        result["discussionPoints"] = []
        if category in {"lecture_class_training", "motivational_speech_seminar", "podcast_webinar_panel"}:
            result["decisions"] = []
            result["actionItems"] = []

    # Remove empty/generic transcript-like items from all sections.
    for key in ["keyPoints", "topics", "discussionPoints", "decisions", "actionItems", "problemStatements", "solutions", "risks", "followUps", "suggestions", "approaches", "guide", "workflowSteps"]:
        value = result.get(key, [])
        if isinstance(value, list):
            result[key] = [item for item in value if has_displayable_value(item) and not item_looks_like_transcript_filler(item)]

    return remove_empty_analysis_fields(result)


def map_items_to_dynamic_list(items: Any) -> List[Dict[str, str]]:
    if not isinstance(items, list):
        return []

    mapped = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            title = clean_text_field(item.get("title") or item.get("name") or f"Item {index + 1}")
            description = clean_text_field(item.get("description") or item.get("details") or item.get("text") or "")
            extra_parts = []
            for key in ["owner", "deadline", "status"]:
                value = clean_text_field(item.get(key, ""))
                if value:
                    extra_parts.append(f"{key.title()}: {value}")
            if extra_parts:
                description = (description + "\n" + " | ".join(extra_parts)).strip()
        else:
            title = f"Item {index + 1}"
            description = clean_text_field(item)

        if title or description:
            mapped.append({
                "title": title or f"Item {index + 1}",
                "description": description,
            })
    return mapped


def first_item_description(items: Any) -> str:
    mapped = map_items_to_dynamic_list(items)
    if not mapped:
        return ""
    return mapped[0].get("description") or mapped[0].get("title") or ""


def item_looks_like_transcript_filler(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    text = " ".join(str(item.get(key, "")) for key in ["title", "description"]).lower().strip()
    if not text:
        return True
    filler_patterns = [
        "speaker discussed",
        "the speaker discussed",
        "speaker mentioned",
        "the speaker mentioned",
        "audio was transcribed",
        "transcript captured",
        "main meeting discussion",
        "captured meeting discussion",
    ]
    return any(pattern in text for pattern in filler_patterns)


def extract_question_sentences(text: str) -> List[Dict[str, str]]:
    candidates = re.split(r"(?<=[?.!])\s+", str(text or ""))
    results = []
    for sentence in candidates:
        cleaned = clean_text_field(sentence)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        is_question = "?" in cleaned or lowered.startswith(("what ", "why ", "how ", "when ", "where ", "who ", "can you", "could you", "tell me", "explain"))
        if is_question:
            results.append({"title": f"Question {len(results) + 1}", "description": make_excerpt(cleaned, max_chars=240)})
        if len(results) >= 8:
            break
    return results


def extract_example_sentences(text: str) -> List[Dict[str, str]]:
    sentences = re.split(r"(?<=[?.!])\s+", str(text or ""))
    results = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(token in lowered for token in ["example", "for example", "such as", "suppose", "imagine"]):
            cleaned = clean_text_field(sentence)
            if cleaned:
                results.append({"title": f"Example {len(results) + 1}", "description": make_excerpt(cleaned, max_chars=260)})
        if len(results) >= 6:
            break
    return results


def extract_commercial_mentions(text: str) -> List[Dict[str, str]]:
    pattern = r"(?:₹|rs\.?|inr|usd|\$)?\s*\d+(?:[,.]\d+)*(?:\s*(?:k|lakh|lakhs|crore|crores|%|percent))?"
    matches = re.findall(pattern, str(text or ""), flags=re.IGNORECASE)
    cleaned = []
    for match in matches:
        value = clean_text_field(match)
        if value and value not in cleaned:
            cleaned.append(value)
    return [{"title": "Amount / Number Mentioned", "description": value} for value in cleaned[:10]]


def extract_metric_mentions(text: str) -> List[Dict[str, str]]:
    pattern = r"\b\d{1,4}(?:[,.]\d+)*(?:\s*(?:%|percent|days?|weeks?|months?|years?|hours?|minutes?|k|lakh|lakhs|crore|crores|pairs?|units?|orders?))?\b"
    matches = re.findall(pattern, str(text or ""), flags=re.IGNORECASE)
    unique = []
    for match in matches:
        value = clean_text_field(match)
        if value and value not in unique:
            unique.append(value)
    return [{"title": "Date / Number / Metric", "description": value} for value in unique[:12]]


def extract_document_mentions(text: str) -> List[Dict[str, str]]:
    terms = ["invoice", "agreement", "contract", "gst", "pi", "po", "purchase order", "receipt", "approval", "document", "certificate", "license"]
    lowered = str(text or "").lower()
    results = []
    for term in terms:
        if term in lowered:
            results.append({"title": term.title(), "description": f"{term.title()} was mentioned in the audio."})
    return results[:8]


def extract_skill_mentions(text: str) -> List[Dict[str, str]]:
    terms = ["excel", "python", "flutter", "marketing", "sales", "communication", "leadership", "management", "design", "backend", "frontend", "analytics", "finance", "accounting"]
    lowered = str(text or "").lower()
    return [{"title": term.title(), "description": f"{term.title()} was referenced."} for term in terms if term in lowered][:10]


def extract_people_like_terms(text: str) -> List[Dict[str, str]]:
    # Conservative fallback: do not guess too aggressively.
    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", str(text or ""))
    skip = {"Speaker", "Audio", "Meeting", "Today", "Tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    unique = []
    for match in matches:
        if match in skip:
            continue
        if match not in unique:
            unique.append(match)
    return [{"title": "Person / Name Mentioned", "description": value} for value in unique[:10]]


def extract_quote_like_lines(text: str) -> List[Dict[str, str]]:
    sentences = re.split(r"(?<=[?.!])\s+", str(text or ""))
    results = []
    for sentence in sentences:
        cleaned = clean_text_field(sentence)
        word_count = len(cleaned.split())
        if 6 <= word_count <= 28 and any(token in cleaned.lower() for token in ["must", "should", "never", "always", "life", "success", "believe", "remember"]):
            results.append({"title": f"Quote {len(results) + 1}", "description": cleaned})
        if len(results) >= 5:
            break
    return results


def build_practice_questions_from_topics(items: Any) -> List[Dict[str, str]]:
    mapped = map_items_to_dynamic_list(items)
    questions = []
    for item in mapped[:6]:
        title = item.get("title", "topic")
        questions.append({
            "title": f"Review {len(questions) + 1}",
            "description": f"Explain {title} in your own words and give one practical example.",
        })
    return questions


def infer_deal_status(result: Dict[str, Any], text: str) -> str:
    lowered = str(text or "").lower()
    if any(word in lowered for word in ["confirmed", "approved", "final", "closed", "done"]):
        return "likely confirmed"
    if any(word in lowered for word in ["pending", "follow up", "waiting", "discuss", "negotiate"]):
        return "pending / follow-up required"
    return "unclear"


def infer_payment_status(text: str) -> str:
    lowered = str(text or "").lower()
    if "paid" in lowered or "payment done" in lowered:
        return "payment appears completed"
    if "pending" in lowered and "payment" in lowered:
        return "payment appears pending"
    if "refund" in lowered:
        return "refund mentioned"
    return "unclear"


def infer_refund_replacement(text: str) -> str:
    lowered = str(text or "").lower()
    values = []
    if "refund" in lowered:
        values.append("refund mentioned")
    if "replacement" in lowered or "replace" in lowered:
        values.append("replacement mentioned")
    return ", ".join(values) if values else ""


def infer_sentiment(text: str) -> str:
    lowered = str(text or "").lower()
    if any(word in lowered for word in ["angry", "frustrated", "very bad", "not happy", "complaint", "issue"]):
        return "negative"
    if any(word in lowered for word in ["happy", "good", "satisfied", "thanks", "great"]):
        return "positive"
    return "neutral / unclear"


def infer_interview_recommendation(result: Dict[str, Any]) -> str:
    risks = result.get("risks") or result.get("problemStatements") or []
    strengths = result.get("solutions") or result.get("keyPoints") or []
    if strengths and not risks:
        return "Potentially positive, based on the captured discussion. Final decision needs human review."
    if strengths and risks:
        return "Mixed. There are positives, but concerns should be reviewed before final decision."
    return "No clear recommendation can be made from the captured audio."



def generate_meaningful_title(
    transcript_text: str,
    analysis_result: Dict[str, Any],
    classification: Dict[str, Any],
) -> str:
    category_label = classification.get("categoryLabel", "Audio Notes")
    existing_title = str(analysis_result.get("meetingTitle", "") or "").strip()

    prompt = f"""
{ENGLISH_ONLY_RULE}

Create ONE premium, meaningful title for this audio.
The title must be based on the complete context, not the first sentence.

Rules:
- 5 to 8 words.
- Do NOT copy the transcript opening.
- Do NOT start with filler words like hello, today, okay, so, basically, everyone.
- Do NOT include date, time, file extension, or speaker label.
- Do NOT use generic titles like Meeting Notes, Audio Summary, Discussion Summary.
- Make it sound like a professional saved note title.

Audio category: {category_label}
Overview: {analysis_result.get('overview', '')}
Summary: {analysis_result.get('summary', '')}
Topics: {json.dumps(analysis_result.get('topics', []), ensure_ascii=False)}
Key points: {json.dumps(analysis_result.get('keyPoints', []), ensure_ascii=False)}
Category-specific output: {json.dumps(analysis_result.get('categorySpecificOutput', {}), ensure_ascii=False)}

Transcript context from start, middle, and end:
{make_transcript_context(transcript_text, max_chars=12000)}

Return ONLY valid JSON:
{{"meetingTitle":"title here"}}
"""

    try:
        parsed = parse_json_object(call_sarvam_chat(prompt))
        title = clean_generated_title(str(parsed.get("meetingTitle") or parsed.get("title") or ""))
        if title and not is_bad_generated_title(title, transcript_text):
            return title
    except Exception:
        pass

    fallback = generate_contextual_title_fallback(analysis_result, transcript_text)
    if fallback and not is_bad_generated_title(fallback, transcript_text):
        return fallback

    if existing_title and not is_bad_generated_title(existing_title, transcript_text):
        return clean_generated_title(existing_title)

    return generate_title_from_text(transcript_text)


def is_bad_generated_title(title: str, transcript_text: str) -> bool:
    cleaned_title = clean_generated_title(title).lower()
    if not cleaned_title:
        return True

    generic_titles = {
        "meeting notes",
        "audio notes",
        "audio summary",
        "discussion summary",
        "meeting analysis",
        "general conversation",
        "captured meeting discussion",
    }
    if cleaned_title in generic_titles:
        return True

    bad_openers = {"hello", "hi", "today", "okay", "ok", "so", "basically", "everyone", "good", "morning", "afternoon", "evening"}
    first_word = cleaned_title.split()[0] if cleaned_title.split() else ""
    if first_word in bad_openers:
        return True

    opening = get_transcript_opening(transcript_text, max_words=35).lower()
    if not opening:
        return False

    title_words = meaningful_words(cleaned_title)
    opening_words = meaningful_words(opening)

    if not title_words:
        return True

    overlap = sum(1 for word in title_words if word in opening_words)
    overlap_ratio = overlap / max(len(title_words), 1)

    # If most title words are from only the first sentence, it is probably copied.
    return overlap_ratio >= 0.78 and len(title_words) >= 3


def get_transcript_opening(text: str, max_words: int = 35) -> str:
    words = str(text or "").split()
    return " ".join(words[:max_words])


def meaningful_words(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", str(text or "").lower())
    return [word for word in words if word not in TITLE_STOPWORDS and len(word) > 2]


def generate_contextual_title_fallback(analysis_result: Dict[str, Any], transcript_text: str) -> str:
    category = analysis_result.get("audioCategory", "general_conversation")
    category_label = category_label_for(category)

    for key in ["topics", "keyPoints", "decisions", "actionItems", "problemStatements", "solutions", "suggestions"]:
        items = analysis_result.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            title = clean_generated_title(str(item.get("title") or ""))
            if title and not is_bad_generated_title(title, transcript_text):
                return title

    category_specific = normalize_dynamic_map(analysis_result.get("categorySpecificOutput", {}))
    for _, value in category_specific.items():
        candidate = title_from_dynamic_value(value)
        candidate = clean_generated_title(candidate)
        if candidate and not is_bad_generated_title(candidate, transcript_text):
            return candidate

    keywords = extract_salient_keywords(transcript_text, max_terms=5)
    if keywords:
        topic = " ".join(word.title() for word in keywords[:4])
        if category == "lecture_class_training":
            return clean_generated_title(f"Learning Notes on {topic}")
        if category == "interview_hr_discussion":
            return clean_generated_title(f"Interview Discussion on {topic}")
        if category == "motivational_speech_seminar":
            return clean_generated_title(f"Motivational Talk on {topic}")
        if category == "sales_client_vendor_call":
            return clean_generated_title(f"Client Discussion on {topic}")
        if category == "product_project_discussion":
            return clean_generated_title(f"Project Discussion on {topic}")
        return clean_generated_title(f"{category_label.split('/')[0].strip()} on {topic}")

    return "Audio Intelligence Notes"


def title_from_dynamic_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            candidate = title_from_dynamic_value(item)
            if candidate:
                return candidate
    if isinstance(value, dict):
        for key in ["title", "topic", "mainTopic", "topicTaught", "projectName", "coreMessage", "purposeOfCall", "customerIssue"]:
            if value.get(key):
                return str(value.get(key))
        for item in value.values():
            candidate = title_from_dynamic_value(item)
            if candidate:
                return candidate
    return ""


def extract_salient_keywords(text: str, max_terms: int = 6) -> List[str]:
    words = meaningful_words(text)
    counts: Dict[str, int] = {}
    for word in words:
        if len(word) < 4:
            continue
        counts[word] = counts.get(word, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:max_terms]]


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
    return generate_contextual_title_fallback(analysis_result, transcript_text)

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
    result["meetingTitle"] = generate_contextual_title_fallback(result, transcript_text)
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
        return "Audio Intelligence Notes"

    keywords = extract_salient_keywords(cleaned, max_terms=5)
    if keywords:
        title = " ".join(word.title() for word in keywords[:5])
        return clean_generated_title(title) or "Audio Intelligence Notes"

    # Last-resort only. This should rarely be used.
    words = [word for word in cleaned.split() if len(word) > 3]
    return clean_generated_title(" ".join(words[:6])) or "Audio Intelligence Notes"

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
