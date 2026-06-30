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

DEFAULT_STT_MODE = os.getenv("DEFAULT_STT_MODE", "transcribe")
DEFAULT_LANGUAGE_CODE = os.getenv("DEFAULT_LANGUAGE_CODE", "en-IN")
DEFAULT_NUM_SPEAKERS = int(os.getenv("DEFAULT_NUM_SPEAKERS", "0"))

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
            name
            for name in os.listdir(upload_dir)
            if name.endswith(".part")
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

    diarized_entries = (
        data.get("diarized_transcript", {})
        .get("entries", [])
    )

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
        return empty_meeting_analysis()

    chunks = split_text_into_chunks(transcript_text, max_words=2500)

    if len(chunks) == 1:
        return analyze_meeting_chunk(
            transcript_text=chunks[0],
            chunk_number=1,
            total_chunks=1,
        )

    chunk_results = []

    for index, chunk in enumerate(chunks):
        chunk_result = analyze_meeting_chunk(
            transcript_text=chunk,
            chunk_number=index + 1,
            total_chunks=len(chunks),
        )
        chunk_results.append(chunk_result)

    return merge_meeting_analysis(chunk_results)


def analyze_meeting_chunk(
    transcript_text: str,
    chunk_number: int,
    total_chunks: int,
) -> Dict[str, Any]:
    prompt = f"""
You are a meeting assistant for a production business app.

Analyze this meeting transcript chunk and return ONLY valid JSON.

Required JSON format:
{{
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

Discussion point status rules:
- Use "done" if the point was completed, approved, fixed, finalized, resolved, or confirmed.
- Use "pending" if action is required but not completed.
- Use "not_done" if the meeting clearly says it was not completed.
- Use "in_progress" if someone is currently working on it.

Important:
- Return JSON only.
- Do not return markdown.
- Do not use ```json.
- Output all text in English only.
- If transcript contains Hindi, Hinglish, Bengali, or mixed language, translate the meaning into English.
- Do not invent information not available in the transcript.
- Extract every important business discussion point.
- Extract problems, blockers, concerns, risks, and challenges separately.
- Extract solutions, decisions, approaches, and recommendations separately.
- Extract step-by-step workflow only if the transcript contains process/action flow.
- If no problem/challenge is discussed, return an empty problemStatements array.
- If no solution/approach is discussed, return an empty solutions array.
- If no workflow is discussed, return an empty workflowSteps array.
- Always return mindMap with title and nodes array.
- Keep titles short and useful.
- Keep descriptions business-friendly and clear.

Chunk:
{chunk_number} of {total_chunks}

Transcript chunk:
{transcript_text}
"""

    response_text = call_sarvam_chat(prompt)
    return safe_json_parse(response_text)


def merge_meeting_analysis(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunk_results:
        return empty_meeting_analysis()

    if len(chunk_results) == 1:
        return chunk_results[0]

    prompt = f"""
You are a meeting assistant.

Merge these meeting chunk analyses into one final meeting analysis.

Return ONLY valid JSON.

Required JSON format:
{{
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

Rules:
- Return JSON only.
- Do not return markdown.
- Do not use ```json.
- Output English only.
- Merge duplicate points.
- Do not remove distinct points.
- Preserve all important decisions, pending tasks, blockers, confirmations, problems, solutions, and follow-ups.
- Re-number discussion point ids as point_1, point_2, point_3, etc.
- Re-number problem ids as problem_1, problem_2, problem_3, etc.
- Re-number solution ids as solution_1, solution_2, solution_3, etc.
- Re-number workflow steps from 1.
- Use only these discussion statuses: done, pending, not_done, in_progress.
- Create a clean mind map from the final merged meeting topics.
- Do not invent information outside the given chunk analyses.

Chunk analyses:
{json.dumps(chunk_results, ensure_ascii=False)}
"""

    response_text = call_sarvam_chat(prompt)
    merged = safe_json_parse(response_text)

    has_content = (
        merged.get("overview")
        or merged.get("summary")
        or merged.get("discussionPoints")
        or merged.get("problemStatements")
        or merged.get("solutions")
        or merged.get("workflowSteps")
    )

    if has_content:
        return merged

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

        if overview:
            overview_parts.append(overview)

        if summary:
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

    return {
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


def empty_meeting_analysis() -> Dict[str, Any]:
    return {
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

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "")
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        fallback = dict(empty_result)
        fallback["overview"] = cleaned
        fallback["summary"] = cleaned
        return fallback

    if not isinstance(data, dict):
        return empty_result

    discussion_points = normalize_discussion_points(
        data.get("discussionPoints", [])
    )

    problem_statements = normalize_text_items(
        data.get("problemStatements", []),
        id_prefix="problem",
    )

    solutions = normalize_solution_items(
        data.get("solutions", [])
    )

    workflow_steps = normalize_workflow_steps(
        data.get("workflowSteps", [])
    )

    mind_map = normalize_mind_map(
        data.get("mindMap", {})
    )

    overview = str(data.get("overview", "") or "").strip()
    summary = str(data.get("summary", "") or "").strip()

    if not overview and summary:
        overview = summary

    if not summary and overview:
        summary = overview

    return {
        "overview": overview,
        "discussionPoints": discussion_points,
        "problemStatements": problem_statements,
        "solutions": solutions,
        "workflowSteps": workflow_steps,
        "mindMap": mind_map,
        "summary": summary,
    }


def normalize_discussion_points(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized_points = []

    for index, point in enumerate(value):
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
- Always answer in English only.
- Do not answer in Hindi unless the user explicitly asks for Hindi.
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

    payload = {
        "model": SARVAM_CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
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
