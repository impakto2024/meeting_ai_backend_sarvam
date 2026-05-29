# Meeting AI Backend

FastAPI backend for long meeting transcription, summary, discussion points, and Q&A.

## Endpoints

### GET /

Health check.

### POST /process-meeting

Form data:
- meeting_id
- audio

Returns:
- jobId
- status

### GET /meeting-status/{jobId}

Returns:
- queued
- transcribing
- analyzing
- completed
- failed

When completed, includes result:
- summary
- discussionPoints
- transcriptText
- utterances

### POST /ask-question

JSON:
{
  "transcriptText": "...",
  "question": "..."
}
