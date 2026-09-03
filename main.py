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
            "meetingTitle": meeting_title,
            "generatedAt": generated_at,
            "suggestedAudioFileName": build_suggested_audio_file_name(
                title=meeting_title,
                generated_at=generated_at,
                audio_path=audio_path,
            ),
            "overview": analysis_result.get("overview", ""),
            "topics": analysis_result.get("topics", []),
            "discussionPoints": analysis_result.get("discussionPoints", []),
            "decisions": analysis_result.get("decisions", []),
            "actionItems": analysis_result.get("actionItems", []),
            "problemStatements": analysis_result.get("problemStatements", []),
            "solutions": analysis_result.get("solutions", []),
            "risks": analysis_result.get("risks", []),
            "followUps": analysis_result.get("followUps", []),
            "workflowSteps": analysis_result.get("workflowSteps", []),
            "mindMap": analysis_result.get(
                "mindMap",
                {
                    "title": "Meeting Mind Map",
                    "nodes": [],
                },
            ),
            "summary": analysis_result.get("summary", ""),
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


def analyze_meeting(transcript_text: str) -> Dict[str, Any]:
    transcript_text = str(transcript_text or "").strip()

    if not transcript_text:
        return fallback_analysis_from_transcript("")

    chunks = split_text_into_chunks(transcript_text, max_words=2500)

    if len(chunks) == 1:
        result = analyze_meeting_chunk(
            transcript_text=chunks[0],
            chunk_number=1,
            total_chunks=1,
        )
        return ensure_minimum_analysis(result, transcript_text)

    chunk_results = []

    for index, chunk in enumerate(chunks):
        chunk_result = analyze_meeting_chunk(
            transcript_text=chunk,
            chunk_number=index + 1,
            total_chunks=len(chunks),
        )
        chunk_results.append(chunk_result)

    merged = merge_meeting_analysis(chunk_results)
    return ensure_minimum_analysis(merged, transcript_text)


def analyze_meeting_chunk(
    transcript_text: str,
    chunk_number: int,
    total_chunks: int,
) -> Dict[str, Any]:
    prompt = f"""
{ENGLISH_ONLY_RULE}

You are a senior meeting intelligence analyst for a premium business productivity app.

Your job is to produce output that feels like a professional meeting analyst wrote it,
not like a basic AI summarizer.

Analyze this meeting transcript chunk and return exactly one valid JSON object.

Required JSON object:
{{
  "meetingTitle": "Short meaningful meeting title in 5 to 8 words. Do not include date or time.",
  "overview": "Executive overview of this meeting chunk in 4 to 6 strong business lines.",
  "topics": [
    {{
      "id": "topic_1",
      "title": "Main topic title",
      "description": "What was discussed under this topic."
    }}
  ],
  "discussionPoints": [
    {{
      "id": "point_1",
      "title": "Short discussion title",
      "description": "Detailed, useful business explanation of what was discussed.",
      "status": "done"
    }}
  ],
  "decisions": [
    {{
      "id": "decision_1",
      "title": "Decision title",
      "description": "Decision details, if a decision was made or confirmed."
    }}
  ],
  "actionItems": [
    {{
      "id": "action_1",
      "title": "Action item title",
      "description": "What needs to be done.",
      "owner": "Owner if mentioned, otherwise Not specified",
      "deadline": "Deadline if mentioned, otherwise Not specified",
      "status": "pending"
    }}
  ],
  "problemStatements": [
    {{
      "id": "problem_1",
      "title": "Short problem or challenge title",
      "description": "Clear explanation of the problem, blocker, concern, risk, or challenge."
    }}
  ],
  "solutions": [
    {{
      "id": "solution_1",
      "title": "Short solution title",
      "description": "Solution, recommendation, decision, or proposed approach.",
      "relatedProblemId": "problem_1"
    }}
  ],
  "risks": [
    {{
      "id": "risk_1",
      "title": "Risk or concern title",
      "description": "Risk, concern, dependency, objection, or uncertainty discussed."
    }}
  ],
  "followUps": [
    {{
      "id": "followup_1",
      "title": "Follow-up title",
      "description": "Follow-up, next discussion, review, approval, or confirmation needed."
    }}
  ],
  "workflowSteps": [
    {{
      "step": 1,
      "title": "Step title",
      "description": "Step-by-step action, process, workflow, or next step."
    }}
  ],
  "mindMap": {{
    "title": "Meeting Mind Map",
    "nodes": [
      {{
        "id": "node_1",
        "label": "Main topic",
        "children": [
          {{
            "id": "node_1_1",
            "label": "Sub topic",
            "children": []
          }}
        ]
      }}
    ]
  }},
  "summary": "Final summary of this chunk in 6 to 10 lines."
}}

Status rules:
- Use "done" if the point/action was completed, approved, fixed, finalized, resolved, or confirmed.
- Use "pending" if action is required but not completed.
- Use "not_done" if the meeting clearly says it was not completed.
- Use "in_progress" if someone is currently working on it.

Strict rules:
- Return only one valid JSON object.
- Do not add explanation before or after JSON.
- Do not use markdown code fence.
- Output all text in English only.
- If transcript contains Hindi, Hinglish, Bengali, Banglish, or mixed language, translate the meaning into English.
- Do not invent information not available in the transcript.
- Preserve names, brand names, article codes, product codes, numbers, dates, prices, commitments, and deadlines.
- Generate a useful meetingTitle from the actual topic. Do not use generic titles like "Meeting Summary" unless no topic is clear.
- Extract topic-wise intelligence, not only generic summary.
- Extract every important business discussion point.
- Extract decisions separately from discussion points.
- Extract action items separately with owner/deadline if available.
- Extract problems, blockers, objections, risks, and challenges separately.
- Extract solutions, decisions, approaches, and recommendations separately.
- Extract follow-ups separately.
- Extract step-by-step workflow only if the transcript contains process or action flow.
- If a section is not present in the transcript, return an empty array for that section.
- Always return mindMap with title and nodes array.
- Never put a JSON object or JSON string inside overview or summary.
- Keep titles short and useful.
- Keep descriptions business-friendly, factual, and clear.

Chunk:
{chunk_number} of {total_chunks}

Transcript chunk:
{transcript_text}
"""

    response_text = call_sarvam_chat(prompt)
    parsed = safe_json_parse(response_text)

    if not is_analysis_empty(parsed):
        return ensure_minimum_analysis(parsed, transcript_text)

    retry_prompt = prompt + """

Your previous response was empty, invalid, or not usable.
Return exactly one valid JSON object now.
Do not add any explanation.
Do not use markdown.
Make sure overview and summary are normal text, not JSON.
"""

    retry_response_text = call_sarvam_chat(retry_prompt)
    retry_parsed = safe_json_parse(retry_response_text)

    if not is_analysis_empty(retry_parsed):
        return ensure_minimum_analysis(retry_parsed, transcript_text)

    return fallback_analysis_from_transcript(transcript_text)


def merge_meeting_analysis(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunk_results:
        return fallback_analysis_from_transcript("")

    if len(chunk_results) == 1:
        return ensure_minimum_analysis(chunk_results[0], "")

    prompt = f"""
{ENGLISH_ONLY_RULE}

You are a senior meeting intelligence analyst.

Merge these meeting chunk analyses into one final premium meeting note.

Return exactly one valid JSON object.

Required JSON object:
{{
  "meetingTitle": "Short meaningful meeting title in 5 to 8 words. Do not include date or time.",
  "overview": "Executive overview of the complete meeting in 4 to 6 strong business lines.",
  "topics": [
    {{
      "id": "topic_1",
      "title": "Main topic title",
      "description": "What was discussed under this topic."
    }}
  ],
  "discussionPoints": [
    {{
      "id": "point_1",
      "title": "Short discussion title",
      "description": "Detailed useful business explanation.",
      "status": "pending"
    }}
  ],
  "decisions": [
    {{
      "id": "decision_1",
      "title": "Decision title",
      "description": "Decision details."
    }}
  ],
  "actionItems": [
    {{
      "id": "action_1",
      "title": "Action item title",
      "description": "What needs to be done.",
      "owner": "Owner if mentioned, otherwise Not specified",
      "deadline": "Deadline if mentioned, otherwise Not specified",
      "status": "pending"
    }}
  ],
  "problemStatements": [
    {{
      "id": "problem_1",
      "title": "Short problem or challenge title",
      "description": "Problem details."
    }}
  ],
  "solutions": [
    {{
      "id": "solution_1",
      "title": "Short solution title",
      "description": "Solution details.",
      "relatedProblemId": "problem_1"
    }}
  ],
  "risks": [
    {{
      "id": "risk_1",
      "title": "Risk or concern title",
      "description": "Risk details."
    }}
  ],
  "followUps": [
    {{
      "id": "followup_1",
      "title": "Follow-up title",
      "description": "Follow-up details."
    }}
  ],
  "workflowSteps": [
    {{
      "step": 1,
      "title": "Step title",
      "description": "Step details."
    }}
  ],
  "mindMap": {{
    "title": "Meeting Mind Map",
    "nodes": [
      {{
        "id": "node_1",
        "label": "Main topic",
        "children": []
      }}
    ]
  }},
  "summary": "Final meeting summary in 6 to 10 lines."
}}

Strict rules:
- Return only one valid JSON object.
- Do not add explanation before or after JSON.
- Do not use markdown code fence.
- Output English only.
- Generate a clean meetingTitle from the complete meeting topic.
- Do not use generic titles like "Meeting Summary" unless no topic is clear.
- Merge duplicate points.
- Do not remove distinct points.
- Preserve important decisions, pending tasks, blockers, confirmations, problems, solutions, risks, and follow-ups.
- Re-number topic ids as topic_1, topic_2, topic_3, etc.
- Re-number discussion point ids as point_1, point_2, point_3, etc.
- Re-number decision ids as decision_1, decision_2, decision_3, etc.
- Re-number action item ids as action_1, action_2, action_3, etc.
- Re-number problem ids as problem_1, problem_2, problem_3, etc.
- Re-number solution ids as solution_1, solution_2, solution_3, etc.
- Re-number workflow steps from 1.
- Use only these statuses: done, pending, not_done, in_progress.
- Create a clean mind map from the final merged meeting topics.
- Never put a JSON object or JSON string inside overview or summary.
- Do not invent information outside the given chunk analyses.
- Make the final output polished, useful, and boardroom-ready.

Chunk analyses:
{json.dumps(chunk_results, ensure_ascii=False)}
"""

    response_text = call_sarvam_chat(prompt)
    merged = safe_json_parse(response_text)

    if not is_analysis_empty(merged):
        return ensure_minimum_analysis(merged, json.dumps(chunk_results, ensure_ascii=False))

    retry_prompt = prompt + """

Your previous merge response was empty, invalid, or not usable.
Return exactly one valid JSON object now.
Do not add any explanation.
Do not use markdown.
Make sure overview and summary are normal text, not JSON.
"""

    retry_response_text = call_sarvam_chat(retry_prompt)
    retry_merged = safe_json_parse(retry_response_text)

    if not is_analysis_empty(retry_merged):
        return ensure_minimum_analysis(retry_merged, json.dumps(chunk_results, ensure_ascii=False))

    return fallback_merge_analyses(chunk_results)


def fallback_merge_analyses(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    overview_parts = []
    summary_parts = []

    topics = []
    discussion_points = []
    decisions = []
    action_items = []
    problem_statements = []
    solutions = []
    risks = []
    follow_ups = []
    workflow_steps = []
    mind_nodes = []

    for result in chunk_results:
        normalized = ensure_minimum_analysis(result, "")

        overview = str(normalized.get("overview", "") or "").strip()
        summary = str(normalized.get("summary", "") or "").strip()

        if overview and not looks_like_json(overview):
            overview_parts.append(overview)

        if summary and not looks_like_json(summary):
            summary_parts.append(summary)

        topics.extend(normalized.get("topics", []))
        discussion_points.extend(normalized.get("discussionPoints", []))
        decisions.extend(normalized.get("decisions", []))
        action_items.extend(normalized.get("actionItems", []))
        problem_statements.extend(normalized.get("problemStatements", []))
        solutions.extend(normalized.get("solutions", []))
        risks.extend(normalized.get("risks", []))
        follow_ups.extend(normalized.get("followUps", []))
        workflow_steps.extend(normalized.get("workflowSteps", []))

        mind_map = normalized.get("mindMap", {})

        if isinstance(mind_map, dict):
            nodes = mind_map.get("nodes", [])

            if isinstance(nodes, list):
                mind_nodes.extend(nodes)

    merged = {
        "meetingTitle": generate_title_from_text(" ".join(overview_parts + summary_parts)),
        "overview": "\n".join(overview_parts[:6]),
        "topics": normalize_text_items(topics, id_prefix="topic"),
        "discussionPoints": normalize_discussion_points(discussion_points),
        "decisions": normalize_text_items(decisions, id_prefix="decision"),
        "actionItems": normalize_action_items(action_items),
        "problemStatements": normalize_text_items(problem_statements, id_prefix="problem"),
        "solutions": normalize_solution_items(solutions),
        "risks": normalize_text_items(risks, id_prefix="risk"),
        "followUps": normalize_text_items(follow_ups, id_prefix="followup"),
        "workflowSteps": normalize_workflow_steps(workflow_steps),
        "mindMap": {
            "title": "Meeting Mind Map",
            "nodes": normalize_mind_map_nodes(mind_nodes),
        },
        "summary": "\n".join(summary_parts[:8]),
    }

    return ensure_minimum_analysis(merged, " ".join(overview_parts + summary_parts))


def empty_meeting_analysis() -> Dict[str, Any]:
    return {
        "meetingTitle": "",
        "overview": "",
        "topics": [],
        "discussionPoints": [],
        "decisions": [],
        "actionItems": [],
        "problemStatements": [],
        "solutions": [],
        "risks": [],
        "followUps": [],
        "workflowSteps": [],
        "mindMap": {
            "title": "Meeting Mind Map",
            "nodes": [],
        },
        "summary": "",
    }


def safe_json_parse(text: Any) -> Dict[str, Any]:
    empty_result = empty_meeting_analysis()

    if text is None:
        return empty_result

    cleaned = str(text).strip()

    if not cleaned:
        return empty_result

    json_text = extract_json_object(cleaned)

    if not json_text:
        return empty_result

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_common_json_issues(json_text)

        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return empty_result

    if not isinstance(data, dict):
        return empty_result

    if isinstance(data.get("result"), dict):
        data = data["result"]

    if isinstance(data.get("analysis"), dict):
        data = data["analysis"]

    if isinstance(data.get("meetingAnalysis"), dict):
        data = data["meetingAnalysis"]

    meeting_title = str(
        data.get("meetingTitle")
        or data.get("meeting_title")
        or data.get("title")
        or ""
    ).strip()

    overview = clean_text_field(data.get("overview", ""))
    summary = clean_text_field(data.get("summary", ""))

    if looks_like_json(overview):
        overview = ""

    if looks_like_json(summary):
        summary = ""

    if not overview and summary:
        overview = summary

    if not summary and overview:
        summary = overview

    topics = normalize_text_items(
        data.get("topics")
        or data.get("keyTopics")
        or data.get("key_topics")
        or [],
        id_prefix="topic",
    )

    discussion_points = normalize_discussion_points(
        data.get("discussionPoints")
        or data.get("discussion_points")
        or data.get("keyPoints")
        or data.get("key_points")
        or []
    )

    decisions = normalize_text_items(
        data.get("decisions")
        or data.get("decisionsTaken")
        or data.get("decision_points")
        or [],
        id_prefix="decision",
    )

    action_items = normalize_action_items(
        data.get("actionItems")
        or data.get("action_items")
        or data.get("tasks")
        or data.get("nextActions")
        or []
    )

    problem_statements = normalize_text_items(
        data.get("problemStatements")
        or data.get("problem_statements")
        or data.get("problems")
        or data.get("challenges")
        or [],
        id_prefix="problem",
    )

    solutions = normalize_solution_items(
        data.get("solutions")
        or data.get("solutionStatements")
        or data.get("solution_statements")
        or []
    )

    risks = normalize_text_items(
        data.get("risks")
        or data.get("concerns")
        or data.get("blockers")
        or [],
        id_prefix="risk",
    )

    follow_ups = normalize_text_items(
        data.get("followUps")
        or data.get("follow_ups")
        or data.get("followups")
        or data.get("next_steps")
        or [],
        id_prefix="followup",
    )

    workflow_steps = normalize_workflow_steps(
        data.get("workflowSteps")
        or data.get("workflow_steps")
        or data.get("steps")
        or []
    )

    mind_map = normalize_mind_map(
        data.get("mindMap")
        or data.get("mind_map")
        or {}
    )

    return {
        "meetingTitle": meeting_title,
        "overview": overview,
        "topics": topics,
        "discussionPoints": discussion_points,
        "decisions": decisions,
        "actionItems": action_items,
        "problemStatements": problem_statements,
        "solutions": solutions,
        "risks": risks,
        "followUps": follow_ups,
        "workflowSteps": workflow_steps,
        "mindMap": mind_map,
        "summary": summary,
    }


def extract_json_object(text: str) -> str:
    cleaned = str(text or "").strip()

    if not cleaned:
        return ""

    cleaned = cleaned.replace("```json", "")
    cleaned = cleaned.replace("```JSON", "")
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()

    for start_index, char in enumerate(cleaned):
        if char != "{":
            continue

        candidate = extract_balanced_json_from_index(cleaned, start_index)

        if not candidate:
            continue

        if (
            "overview" in candidate
            or "discussionPoints" in candidate
            or "discussion_points" in candidate
            or "summary" in candidate
            or "mindMap" in candidate
            or "meetingTitle" in candidate
            or "topics" in candidate
            or "actionItems" in candidate
            or "decisions" in candidate
        ):
            return candidate.strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
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

    repaired = repaired.replace("\u201c", '"')
    repaired = repaired.replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'")
    repaired = repaired.replace("\u2019", "'")

    repaired = repaired.replace(",\n}", "\n}")
    repaired = repaired.replace(",\n]", "\n]")
    repaired = repaired.replace(",}", "}")
    repaired = repaired.replace(",]", "]")

    return repaired


def clean_text_field(value: Any) -> str:
    cleaned = str(value or "").strip()

    if not cleaned:
        return ""

    cleaned = cleaned.replace("```json", "")
    cleaned = cleaned.replace("```JSON", "")
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()

    return cleaned


def looks_like_json(value: str) -> bool:
    cleaned = str(value or "").strip()

    if not cleaned:
        return False

    return (
        cleaned.startswith("{")
        or cleaned.startswith("[")
        or '"overview"' in cleaned
        or '"meetingTitle"' in cleaned
        or '"topics"' in cleaned
        or '"discussionPoints"' in cleaned
        or '"discussion_points"' in cleaned
        or '"decisions"' in cleaned
        or '"actionItems"' in cleaned
        or '"problemStatements"' in cleaned
        or '"workflowSteps"' in cleaned
        or '"mindMap"' in cleaned
        or '"summary"' in cleaned
    )


def is_analysis_empty(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return True

    mind_map = result.get("mindMap", {})

    mind_nodes = []
    if isinstance(mind_map, dict):
        nodes = mind_map.get("nodes", [])
        if isinstance(nodes, list):
            mind_nodes = nodes

    return not any(
        [
            str(result.get("meetingTitle", "") or "").strip(),
            str(result.get("overview", "") or "").strip(),
            str(result.get("summary", "") or "").strip(),
            result.get("topics", []),
            result.get("discussionPoints", []),
            result.get("decisions", []),
            result.get("actionItems", []),
            result.get("problemStatements", []),
            result.get("solutions", []),
            result.get("risks", []),
            result.get("followUps", []),
            result.get("workflowSteps", []),
            mind_nodes,
        ]
    )


def ensure_minimum_analysis(
    result: Dict[str, Any],
    transcript_text: str,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        result = empty_meeting_analysis()

    transcript_text = str(transcript_text or "").strip()

    meeting_title = str(result.get("meetingTitle", "") or "").strip()
    overview = clean_text_field(result.get("overview", ""))
    summary = clean_text_field(result.get("summary", ""))

    if looks_like_json(overview):
        overview = ""

    if looks_like_json(summary):
        summary = ""

    topics = normalize_text_items(
        result.get("topics", []),
        id_prefix="topic",
    )

    discussion_points = normalize_discussion_points(
        result.get("discussionPoints", [])
    )

    decisions = normalize_text_items(
        result.get("decisions", []),
        id_prefix="decision",
    )

    action_items = normalize_action_items(
        result.get("actionItems", [])
    )

    problem_statements = normalize_text_items(
        result.get("problemStatements", []),
        id_prefix="problem",
    )

    solutions = normalize_solution_items(
        result.get("solutions", [])
    )

    risks = normalize_text_items(
        result.get("risks", []),
        id_prefix="risk",
    )

    follow_ups = normalize_text_items(
        result.get("followUps", []),
        id_prefix="followup",
    )

    workflow_steps = normalize_workflow_steps(
        result.get("workflowSteps", [])
    )

    mind_map = normalize_mind_map(
        result.get("mindMap", {})
    )

    if not meeting_title:
        meeting_title = generate_title_from_text(
            overview or summary or transcript_text
        )

    if not overview:
        overview = generate_overview_fallback(transcript_text, summary)

    if not summary:
        summary = generate_summary_fallback(transcript_text, overview)

    if not discussion_points:
        discussion_points = [
            {
                "id": "point_1",
                "title": "Captured Meeting Discussion",
                "description": generate_discussion_fallback(transcript_text, overview, summary),
                "status": "pending",
            }
        ]

    if not topics:
        topics = normalize_text_items(
            [
                {
                    "title": meeting_title,
                    "description": overview,
                }
            ],
            id_prefix="topic",
        )

    if not has_mind_map_nodes(mind_map):
        mind_map = build_mind_map_from_analysis(
            topics=topics,
            decisions=decisions,
            action_items=action_items,
            risks=risks,
            follow_ups=follow_ups,
            discussion_points=discussion_points,
            problem_statements=problem_statements,
            solutions=solutions,
            workflow_steps=workflow_steps,
        )

    return {
        "meetingTitle": meeting_title,
        "overview": overview,
        "topics": topics,
        "discussionPoints": discussion_points,
        "decisions": decisions,
        "actionItems": action_items,
        "problemStatements": problem_statements,
        "solutions": solutions,
        "risks": risks,
        "followUps": follow_ups,
        "workflowSteps": workflow_steps,
        "mindMap": mind_map,
        "summary": summary,
    }


def fallback_analysis_from_transcript(transcript_text: str) -> Dict[str, Any]:
    transcript_text = str(transcript_text or "").strip()

    if not transcript_text:
        return {
            "meetingTitle": "Meeting Audio Analysis",
            "overview": "No clear speech transcript could be generated from this audio. Please check the recording quality and try again.",
            "topics": [],
            "discussionPoints": [
                {
                    "id": "point_1",
                    "title": "Audio Could Not Be Clearly Analyzed",
                    "description": "The system could not generate a reliable transcript from the uploaded meeting audio.",
                    "status": "pending",
                }
            ],
            "decisions": [],
            "actionItems": [],
            "problemStatements": [],
            "solutions": [],
            "risks": [
                {
                    "id": "risk_1",
                    "title": "Recording Quality Issue",
                    "description": "The uploaded audio may be empty, unclear, too short, or unsupported.",
                }
            ],
            "followUps": [],
            "workflowSteps": [],
            "mindMap": {
                "title": "Meeting Mind Map",
                "nodes": [
                    {
                        "id": "node_1",
                        "label": "Audio analysis unavailable",
                        "children": [],
                    }
                ],
            },
            "summary": "No reliable meeting summary could be generated because the transcript was empty or unclear.",
        }

    title = generate_title_from_text(transcript_text)
    excerpt = make_excerpt(transcript_text, max_chars=700)

    return {
        "meetingTitle": title,
        "overview": (
            "The meeting audio was transcribed successfully. "
            "A complete structured analysis could not be generated with high confidence, "
            "so this overview is based directly on the transcript excerpt: "
            f"{excerpt}"
        ),
        "topics": [
            {
                "id": "topic_1",
                "title": title,
                "description": excerpt,
            }
        ],
        "discussionPoints": [
            {
                "id": "point_1",
                "title": "Main Meeting Discussion",
                "description": excerpt,
                "status": "pending",
            }
        ],
        "decisions": [],
        "actionItems": [],
        "problemStatements": [],
        "solutions": [],
        "risks": [],
        "followUps": [],
        "workflowSteps": [],
        "mindMap": {
            "title": "Meeting Mind Map",
            "nodes": [
                {
                    "id": "node_1",
                    "label": "Main Meeting Discussion",
                    "children": [],
                }
            ],
        },
        "summary": (
            "The transcript was captured, but the AI analysis response was not reliable enough "
            "to extract every section. Please review the transcript for exact details."
        ),
    }


def generate_title_from_text(text: str) -> str:
    cleaned = str(text or "").strip()

    if not cleaned:
        return "Meeting Notes"

    cleaned = " ".join(cleaned.split())
    first_sentence = cleaned.split(".")[0].strip()

    if not first_sentence:
        first_sentence = cleaned

    first_sentence = first_sentence.replace("Speaker:", "").strip()

    words = first_sentence.split()

    if len(words) >= 4:
        title = " ".join(words[:9])
    else:
        title = " ".join(words)

    title = title.strip(" ,:-")

    if not title:
        return "Meeting Notes"

    if len(title) > 70:
        title = title[:67].strip() + "..."

    return title


def generate_overview_fallback(transcript_text: str, summary: str) -> str:
    if summary and not looks_like_json(summary):
        return summary

    transcript_text = str(transcript_text or "").strip()

    if transcript_text:
        return (
            "The meeting transcript was captured successfully. "
            f"Key transcript excerpt: {make_excerpt(transcript_text, max_chars=600)}"
        )

    return "No clear overview could be generated from this meeting audio."


def generate_summary_fallback(transcript_text: str, overview: str) -> str:
    if overview and not looks_like_json(overview):
        return overview

    transcript_text = str(transcript_text or "").strip()

    if transcript_text:
        return (
            "The meeting was transcribed, but a detailed structured summary could not be generated. "
            f"Transcript excerpt: {make_excerpt(transcript_text, max_chars=600)}"
        )

    return "No clear summary could be generated from this meeting audio."


def generate_discussion_fallback(
    transcript_text: str,
    overview: str,
    summary: str,
) -> str:
    for value in [overview, summary, transcript_text]:
        cleaned = str(value or "").strip()

        if cleaned and not looks_like_json(cleaned):
            return make_excerpt(cleaned, max_chars=600)

    return "The meeting was processed, but no clear discussion points were extracted."


def make_excerpt(text: str, max_chars: int = 600) -> str:
    cleaned = " ".join(str(text or "").split())

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def has_mind_map_nodes(mind_map: Dict[str, Any]) -> bool:
    if not isinstance(mind_map, dict):
        return False

    nodes = mind_map.get("nodes", [])

    return isinstance(nodes, list) and len(nodes) > 0


def build_mind_map_from_analysis(
    topics: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    action_items: List[Dict[str, Any]],
    risks: List[Dict[str, Any]],
    follow_ups: List[Dict[str, Any]],
    discussion_points: List[Dict[str, Any]],
    problem_statements: List[Dict[str, Any]],
    solutions: List[Dict[str, Any]],
    workflow_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    nodes = []

    if topics:
        nodes.append(
            {
                "id": "node_topics",
                "label": "Key Topics",
                "children": [
                    {
                        "id": f"node_topic_{index + 1}",
                        "label": str(item.get("title") or f"Topic {index + 1}"),
                        "children": [],
                    }
                    for index, item in enumerate(topics[:6])
                ],
            }
        )

    if decisions:
        nodes.append(
            {
                "id": "node_decisions",
                "label": "Decisions",
                "children": [
                    {
                        "id": f"node_decision_{index + 1}",
                        "label": str(item.get("title") or f"Decision {index + 1}"),
                        "children": [],
                    }
                    for index, item in enumerate(decisions[:6])
                ],
            }
        )

    if action_items:
        nodes.append(
            {
                "id": "node_actions",
                "label": "Action Items",
                "children": [
                    {
                        "id": f"node_action_{index + 1}",
                        "label": str(item.get("title") or f"Action {index + 1}"),
                        "children": [],
                    }
                    for index, item in enumerate(action_items[:6])
                ],
            }
        )

    if discussion_points:
        nodes.append(
            {
                "id": "node_discussions",
                "label": "Discussion Points",
                "children": [
                    {
                        "id": f"node_discussion_{index + 1}",
                        "label": str(item.get("title") or f"Point {index + 1}"),
                        "children": [],
                    }
                    for index, item in enumerate(discussion_points[:6])
                ],
            }
        )

    if problem_statements:
        nodes.append(
            {
                "id": "node_problems",
                "label": "Problems & Challenges",
                "children": [
                    {
                        "id": f"node_problem_{index + 1}",
                        "label": str(item.get("title") or f"Problem {index + 1}"),
                        "children": [],
                    }
                    for index, item in enumerate(problem_statements[:6])
                ],
            }
        )

    if solutions:
        nodes.append(
            {
                "id": "node_solutions",
                "label": "Solutions & Approach",
                "children": [
                    {
                        "id": f"node_solution_{index + 1}",
                        "label": str(item.get("title") or f"Solution {index + 1}"),
                        "children": [],
                    }
                    for index, item in enumerate(solutions[:6])
                ],
            }
        )

    if risks:
        nodes.append(
            {
                "id": "node_risks",
                "label": "Risks & Concerns",
                "children": [
                    {
                        "id": f"node_risk_{index + 1}",
                        "label": str(item.get("title") or f"Risk {index + 1}"),
                        "children": [],
                    }
                    for index, item in enumerate(risks[:6])
                ],
            }
        )

    if follow_ups:
        nodes.append(
            {
                "id": "node_followups",
                "label": "Follow-ups",
                "children": [
                    {
                        "id": f"node_followup_{index + 1}",
                        "label": str(item.get("title") or f"Follow-up {index + 1}"),
                        "children": [],
                    }
                    for index, item in enumerate(follow_ups[:6])
                ],
            }
        )

    if workflow_steps:
        nodes.append(
            {
                "id": "node_workflow",
                "label": "Workflow Steps",
                "children": [
                    {
                        "id": f"node_workflow_{index + 1}",
                        "label": str(item.get("title") or f"Step {index + 1}"),
                        "children": [],
                    }
                    for index, item in enumerate(workflow_steps[:6])
                ],
            }
        )

    if not nodes:
        nodes = [
            {
                "id": "node_1",
                "label": "Meeting Discussion",
                "children": [],
            }
        ]

    return {
        "title": "Meeting Mind Map",
        "nodes": nodes,
    }


def normalize_discussion_points(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized_points = []

    for index, point in enumerate(value):
        if isinstance(point, str):
            text = point.strip()

            if not text:
                continue

            normalized_points.append(
                {
                    "id": f"point_{index + 1}",
                    "title": f"Point {index + 1}",
                    "description": text,
                    "status": "pending",
                }
            )
            continue

        if not isinstance(point, dict):
            continue

        status = str(point.get("status", "pending")).lower().strip()

        if status not in ["done", "pending", "not_done", "in_progress"]:
            status = "pending"

        title = str(point.get("title") or f"Point {index + 1}").strip()
        description = str(point.get("description") or "").strip()

        if not title and not description:
            continue

        normalized_points.append(
            {
                "id": str(point.get("id") or f"point_{index + 1}"),
                "title": title or f"Point {index + 1}",
                "description": description,
                "status": status,
            }
        )

    return normalized_points


def normalize_text_items(value: Any, id_prefix: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized_items = []
    seen = set()

    for index, item in enumerate(value):
        if isinstance(item, str):
            title = f"{id_prefix.title()} {len(normalized_items) + 1}"
            description = item.strip()
        elif isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            description = str(item.get("description") or "").strip()
        else:
            continue

        if not title and not description:
            continue

        if not title:
            title = f"{id_prefix.title()} {len(normalized_items) + 1}"

        key = f"{title.lower()}::{description.lower()}"
        if key in seen:
            continue

        seen.add(key)

        normalized_items.append(
            {
                "id": f"{id_prefix}_{len(normalized_items) + 1}",
                "title": title,
                "description": description,
            }
        )

    return normalized_items


def normalize_action_items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized_items = []
    seen = set()

    for index, item in enumerate(value):
        if isinstance(item, str):
            title = f"Action {len(normalized_items) + 1}"
            description = item.strip()
            owner = "Not specified"
            deadline = "Not specified"
            status = "pending"
        elif isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            description = str(item.get("description") or "").strip()
            owner = str(item.get("owner") or "Not specified").strip()
            deadline = str(item.get("deadline") or "Not specified").strip()
            status = str(item.get("status") or "pending").lower().strip()
        else:
            continue

        if not title and not description:
            continue

        if status not in ["done", "pending", "not_done", "in_progress"]:
            status = "pending"

        if not title:
            title = f"Action {len(normalized_items) + 1}"

        key = f"{title.lower()}::{description.lower()}::{owner.lower()}::{deadline.lower()}::{status}"
        if key in seen:
            continue

        seen.add(key)

        normalized_items.append(
            {
                "id": f"action_{len(normalized_items) + 1}",
                "title": title,
                "description": description,
                "owner": owner or "Not specified",
                "deadline": deadline or "Not specified",
                "status": status,
            }
        )

    return normalized_items


def normalize_solution_items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized_items = []

    for index, item in enumerate(value):
        if isinstance(item, str):
            text = item.strip()

            if not text:
                continue

            normalized_items.append(
                {
                    "id": f"solution_{index + 1}",
                    "title": f"Solution {index + 1}",
                    "description": text,
                    "relatedProblemId": "",
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        related_problem_id = str(item.get("relatedProblemId") or "").strip()

        if not title and not description:
            continue

        normalized_items.append(
            {
                "id": str(item.get("id") or f"solution_{index + 1}"),
                "title": title or f"Solution {index + 1}",
                "description": description,
                "relatedProblemId": related_problem_id,
            }
        )

    return normalized_items


def normalize_workflow_steps(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized_steps = []

    for index, item in enumerate(value):
        if isinstance(item, str):
            text = item.strip()

            if not text:
                continue

            normalized_steps.append(
                {
                    "step": index + 1,
                    "title": f"Step {index + 1}",
                    "description": text,
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        try:
            step = int(item.get("step") or index + 1)
        except Exception:
            step = index + 1

        title = str(item.get("title") or f"Step {step}").strip()
        description = str(item.get("description") or "").strip()

        if not title and not description:
            continue

        normalized_steps.append(
            {
                "step": step,
                "title": title or f"Step {step}",
                "description": description,
            }
        )

    normalized_steps.sort(key=lambda item: item.get("step", 0))
    return normalized_steps


def normalize_mind_map(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "title": "Meeting Mind Map",
            "nodes": [],
        }

    title = str(value.get("title") or "Meeting Mind Map").strip()
    nodes = value.get("nodes", [])

    if not isinstance(nodes, list):
        nodes = []

    return {
        "title": title or "Meeting Mind Map",
        "nodes": normalize_mind_map_nodes(nodes),
    }


def normalize_mind_map_nodes(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized_nodes = []

    for index, node in enumerate(value):
        if isinstance(node, str):
            label = node.strip()

            if not label:
                continue

            normalized_nodes.append(
                {
                    "id": f"node_{index + 1}",
                    "label": label,
                    "children": [],
                }
            )
            continue

        if not isinstance(node, dict):
            continue

        label = str(node.get("label") or "").strip()

        if not label:
            continue

        children = node.get("children", [])

        normalized_nodes.append(
            {
                "id": str(node.get("id") or f"node_{index + 1}"),
                "label": label,
                "children": normalize_mind_map_nodes(children),
            }
        )

    return normalized_nodes


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
