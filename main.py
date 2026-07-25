import json
import os
import shutil
import tempfile
import threading
import time
import uuid
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
            message="Creating meeting analysis",
        )

        analysis_result = analyze_meeting(transcript_text)

        result = {
            "meetingId": meeting_id,
            "meetingTitle": analysis_result.get("meetingTitle", ""),
            "overview": analysis_result.get("overview", ""),
            "discussionPoints": analysis_result.get("discussionPoints", []),
            "problemStatements": analysis_result.get("problemStatements", []),
            "solutions": analysis_result.get("solutions", []),
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
You are a senior meeting analyst for a business productivity app.

Analyze this meeting transcript chunk and return exactly one valid JSON object.

Required JSON object:
{{
  "meetingTitle": "Short meaningful meeting title in 6 to 10 words.",
  "overview": "High-level overview of this meeting chunk in 4 to 6 lines.",
  "discussionPoints": [
    {{
      "id": "point_1",
      "title": "Short discussion title",
      "description": "What was discussed about this point.",
      "status": "done"
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
  "summary": "Final summary of this chunk in 5 to 8 lines."
}}

Status rules:
- Use "done" if the point was completed, approved, fixed, finalized, resolved, or confirmed.
- Use "pending" if action is required but not completed.
- Use "not_done" if the meeting clearly says it was not completed.
- Use "in_progress" if someone is currently working on it.

Strict rules:
- Return only one valid JSON object.
- Do not add explanation before or after JSON.
- Do not use markdown code fence.
- Output all text in English only.
- If transcript contains Hindi, Hinglish, Bengali, or mixed language, translate the meaning into English.
- Do not invent information not available in the transcript.
- Extract every important business discussion point.
- Extract problems, blockers, concerns, risks, and challenges separately.
- Extract solutions, decisions, approaches, and recommendations separately.
- Extract step-by-step workflow only if the transcript contains process or action flow.
- If no problem or challenge is discussed, return an empty problemStatements array.
- If no solution or approach is discussed, return an empty solutions array.
- If no workflow is discussed, return an empty workflowSteps array.
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
        return chunk_results[0]

    prompt = f"""
You are a senior meeting analyst.

Merge these meeting chunk analyses into one final meeting analysis.

Return exactly one valid JSON object.

Required JSON object:
{{
  "meetingTitle": "Short meaningful meeting title in 6 to 10 words.",
  "overview": "High-level overview of the complete meeting in 4 to 6 lines.",
  "discussionPoints": [
    {{
      "id": "point_1",
      "title": "Short discussion title",
      "description": "What was discussed about this point.",
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
        "children": []
      }}
    ]
  }},
  "summary": "Final meeting summary in 5 to 8 lines."
}}

Strict rules:
- Return only one valid JSON object.
- Do not add explanation before or after JSON.
- Do not use markdown code fence.
- Output English only.
- Merge duplicate points.
- Do not remove distinct points.
- Preserve important decisions, pending tasks, blockers, confirmations, problems, solutions, and follow-ups.
- Re-number discussion point ids as point_1, point_2, point_3, etc.
- Re-number problem ids as problem_1, problem_2, problem_3, etc.
- Re-number solution ids as solution_1, solution_2, solution_3, etc.
- Re-number workflow steps from 1.
- Use only these discussion statuses: done, pending, not_done, in_progress.
- Create a clean mind map from the final merged meeting topics.
- Never put a JSON object or JSON string inside overview or summary.
- Do not invent information outside the given chunk analyses.

Chunk analyses:
{json.dumps(chunk_results, ensure_ascii=False)}
"""

    response_text = call_sarvam_chat(prompt)
    merged = safe_json_parse(response_text)

    if not is_analysis_empty(merged):
        return merged

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
        return retry_merged

    return fallback_merge_analyses(chunk_results)


def fallback_merge_analyses(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    overview_parts = []
    summary_parts = []
    discussion_points = []
    problem_statements = []
    solutions = []
    workflow_steps = []
    mind_nodes = []

    seen_discussions = set()
    seen_problems = set()
    seen_solutions = set()
    seen_workflows = set()

    for result in chunk_results:
        overview = str(result.get("overview", "") or "").strip()
        summary = str(result.get("summary", "") or "").strip()

        if overview and not looks_like_json(overview):
            overview_parts.append(overview)

        if summary and not looks_like_json(summary):
            summary_parts.append(summary)

        for point in result.get("discussionPoints", []):
            if not isinstance(point, dict):
                continue

            title = str(point.get("title", "") or "").strip()
            description = str(point.get("description", "") or "").strip()

            if not title and not description:
                continue

            key = f"{title.lower()}::{description.lower()}"

            if key in seen_discussions:
                continue

            seen_discussions.add(key)

            status = str(point.get("status", "pending") or "pending").lower().strip()

            if status not in ["done", "pending", "not_done", "in_progress"]:
                status = "pending"

            discussion_points.append(
                {
                    "id": f"point_{len(discussion_points) + 1}",
                    "title": title or f"Point {len(discussion_points) + 1}",
                    "description": description,
                    "status": status,
                }
            )

        for problem in result.get("problemStatements", []):
            if not isinstance(problem, dict):
                continue

            title = str(problem.get("title", "") or "").strip()
            description = str(problem.get("description", "") or "").strip()

            if not title and not description:
                continue

            key = f"{title.lower()}::{description.lower()}"

            if key in seen_problems:
                continue

            seen_problems.add(key)

            problem_statements.append(
                {
                    "id": f"problem_{len(problem_statements) + 1}",
                    "title": title or f"Problem {len(problem_statements) + 1}",
                    "description": description,
                }
            )

        for solution in result.get("solutions", []):
            if not isinstance(solution, dict):
                continue

            title = str(solution.get("title", "") or "").strip()
            description = str(solution.get("description", "") or "").strip()
            related_problem_id = str(solution.get("relatedProblemId", "") or "").strip()

            if not title and not description:
                continue

            key = f"{title.lower()}::{description.lower()}"

            if key in seen_solutions:
                continue

            seen_solutions.add(key)

            solutions.append(
                {
                    "id": f"solution_{len(solutions) + 1}",
                    "title": title or f"Solution {len(solutions) + 1}",
                    "description": description,
                    "relatedProblemId": related_problem_id,
                }
            )

        for workflow in result.get("workflowSteps", []):
            if not isinstance(workflow, dict):
                continue

            title = str(workflow.get("title", "") or "").strip()
            description = str(workflow.get("description", "") or "").strip()

            if not title and not description:
                continue

            key = f"{title.lower()}::{description.lower()}"

            if key in seen_workflows:
                continue

            seen_workflows.add(key)

            workflow_steps.append(
                {
                    "step": len(workflow_steps) + 1,
                    "title": title or f"Step {len(workflow_steps) + 1}",
                    "description": description,
                }
            )

        mind_map = result.get("mindMap", {})

        if isinstance(mind_map, dict):
            nodes = mind_map.get("nodes", [])

            if isinstance(nodes, list):
                mind_nodes.extend(nodes)

    merged = {
        "meetingTitle": generate_title_from_text(" ".join(overview_parts + summary_parts)),
        "overview": "\n".join(overview_parts[:6]),
        "discussionPoints": discussion_points,
        "problemStatements": problem_statements,
        "solutions": solutions,
        "workflowSteps": workflow_steps,
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
        "discussionPoints": [],
        "problemStatements": [],
        "solutions": [],
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

    meeting_title = str(
        data.get("meetingTitle")
        or data.get("meeting_title")
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

    discussion_points = normalize_discussion_points(
        data.get("discussionPoints")
        or data.get("discussion_points")
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
        "discussionPoints": discussion_points,
        "problemStatements": problem_statements,
        "solutions": solutions,
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
        or '"discussionPoints"' in cleaned
        or '"discussion_points"' in cleaned
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
            str(result.get("overview", "") or "").strip(),
            str(result.get("summary", "") or "").strip(),
            result.get("discussionPoints", []),
            result.get("problemStatements", []),
            result.get("solutions", []),
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

    discussion_points = normalize_discussion_points(
        result.get("discussionPoints", [])
    )

    problem_statements = normalize_text_items(
        result.get("problemStatements", []),
        id_prefix="problem",
    )

    solutions = normalize_solution_items(
        result.get("solutions", [])
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

    if not has_mind_map_nodes(mind_map):
        mind_map = build_mind_map_from_analysis(
            discussion_points=discussion_points,
            problem_statements=problem_statements,
            solutions=solutions,
            workflow_steps=workflow_steps,
        )

    return {
        "meetingTitle": meeting_title,
        "overview": overview,
        "discussionPoints": discussion_points,
        "problemStatements": problem_statements,
        "solutions": solutions,
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
            "discussionPoints": [
                {
                    "id": "point_1",
                    "title": "Audio Could Not Be Clearly Analyzed",
                    "description": "The system could not generate a reliable transcript from the uploaded meeting audio.",
                    "status": "pending",
                }
            ],
            "problemStatements": [],
            "solutions": [],
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
        "discussionPoints": [
            {
                "id": "point_1",
                "title": "Main Meeting Discussion",
                "description": excerpt,
                "status": "pending",
            }
        ],
        "problemStatements": [],
        "solutions": [],
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
    discussion_points: List[Dict[str, Any]],
    problem_statements: List[Dict[str, Any]],
    solutions: List[Dict[str, Any]],
    workflow_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    nodes = []

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

    for index, item in enumerate(value):
        if isinstance(item, str):
            text = item.strip()

            if not text:
                continue

            normalized_items.append(
                {
                    "id": f"{id_prefix}_{index + 1}",
                    "title": f"{id_prefix.title()} {index + 1}",
                    "description": text,
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()

        if not title and not description:
            continue

        normalized_items.append(
            {
                "id": str(item.get("id") or f"{id_prefix}_{index + 1}"),
                "title": title or f"{id_prefix.title()} {index + 1}",
                "description": description,
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
    prompt = f"""
Answer the user's question based only on this meeting transcript.

If the answer is not available in the transcript, say:
"I could not find this information in the meeting transcript."

Rules:
- Answer only in English.
- If the question is in Hindi, Bengali, Hinglish, or mixed language, understand it and answer in English.
- If the transcript contains Hindi, Bengali, Hinglish, or mixed language, translate the meaning and answer in English.
- Do not use Hindi script.
- Do not use Bengali script.
- Do not use mixed-language text.
- Keep the answer direct and useful.

Transcript:
{transcript_text}

Question:
{question}
"""

    answer = call_sarvam_chat(prompt)

    if answer is None:
        return ""

    return str(answer).strip()


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
        "temperature": 0.2,
        "max_tokens": 4096,
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
