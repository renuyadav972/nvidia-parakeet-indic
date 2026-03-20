"""
Parakeet Indic ASR Demo Server
==============================
FastAPI server that:
  - POST /       — Plivo answer_url webhook (returns Stream XML)
  - WS   /ws     — Plivo bidirectional audio WebSocket
  - GET  /       — Dashboard UI
  - GET  /api/live — SSE endpoint for real-time transcripts
  - GET  /api/sessions — Session list
  - GET  /api/sessions/{id} — Session detail

Receives 8kHz mulaw audio from Plivo, converts to 16kHz PCM WAV,
sends to NVIDIA NIM Parakeet RNNT for transcription, and streams
results to a dashboard via SSE.
"""

import asyncio
import base64
import json
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pathlib import Path
from sse_starlette.sse import EventSourceResponse

import httpx

from audio_pipeline import AudioPipeline
from metrics_collector import MetricsCollector
from nim_client import NIMClient

load_dotenv(override=False)

app = FastAPI(title="Parakeet Indic ASR Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "sessions"
STATIC_DIR = BASE_DIR / "static"

DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN", os.getenv("PUBLIC_DOMAIN", "localhost:8000"))

# Target language for ASR (can be overridden via query param)
DEFAULT_LANGUAGE = os.getenv("ASR_LANGUAGE", "hi")

# How often to flush audio buffer and call NIM API (seconds)
TRANSCRIBE_INTERVAL = float(os.getenv("TRANSCRIBE_INTERVAL", "1.5"))

# SSE subscribers: list of asyncio.Queue
sse_queues: list[asyncio.Queue] = []

# NIM client singleton
nim_client = NIMClient()

# ---------------------------------------------------------------------------
# Plivo webhook
# ---------------------------------------------------------------------------

PLIVO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://{domain}/ws</Stream>
</Response>"""


@app.post("/")
async def plivo_webhook(request: Request):
    """Plivo answer_url - tells Plivo to open a bidirectional audio stream."""
    domain = request.headers.get("host") or DOMAIN
    xml = PLIVO_XML.format(domain=domain)
    logger.info(f"Plivo webhook hit - streaming to wss://{domain}/ws")
    return Response(content=xml, media_type="application/xml")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "dashboard.html"))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

async def sse_generator(queue: asyncio.Queue):
    """Yield SSE events from a per-client queue."""
    try:
        while True:
            data = await queue.get()
            yield {"event": "transcript", "data": json.dumps(data)}
    except asyncio.CancelledError:
        pass


@app.get("/api/live")
async def live_sse(request: Request):
    """SSE endpoint for real-time transcription events."""
    queue = asyncio.Queue()
    sse_queues.append(queue)

    async def on_disconnect():
        sse_queues.remove(queue)

    return EventSourceResponse(
        sse_generator(queue),
        ping=15,
    )


def broadcast_sse(event: dict):
    """Push an event to all connected SSE clients."""
    for q in sse_queues:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

def _read_session(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@app.get("/api/sessions")
async def list_sessions():
    if not DATA_DIR.exists():
        return []
    sessions = []
    for p in sorted(DATA_DIR.glob("*.json")):
        s = _read_session(p)
        if s:
            sessions.append({
                "session_id": s["session_id"],
                "started_at": s["started_at"],
                "ended_at": s.get("ended_at"),
                "summary": s.get("summary", {}),
            })
    return sessions


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    path = DATA_DIR / f"{session_id}.json"
    if not path.exists():
        return {"error": "Session not found"}
    return _read_session(path)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    path = DATA_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
        return {"message": "deleted"}
    return {"error": "not found"}


# ---------------------------------------------------------------------------
# Plivo call recording
# ---------------------------------------------------------------------------

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")


async def _start_recording(call_id: str):
    """Start Plivo call recording via REST API."""
    if not PLIVO_AUTH_ID or not PLIVO_AUTH_TOKEN:
        logger.debug("Plivo credentials not set, skipping recording")
        return
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"https://api.plivo.com/v1/Account/{PLIVO_AUTH_ID}/Call/{call_id}/Record/",
                auth=(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN),
                json={"time_limit": 300, "file_format": "mp3"},
            )
            logger.info(f"Plivo recording started: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to start recording: {e}")


async def _fetch_and_save_recording(call_id: str, session_id: str):
    """Poll Plivo for recording URL and save to session JSON."""
    if not PLIVO_AUTH_ID or not PLIVO_AUTH_TOKEN:
        return
    for attempt in range(12):
        await asyncio.sleep(5)
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"https://api.plivo.com/v1/Account/{PLIVO_AUTH_ID}/Recording/",
                    auth=(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN),
                    params={"call_uuid": call_id, "limit": 1},
                )
                if resp.status_code == 200:
                    objects = resp.json().get("objects", [])
                    if objects:
                        recording_url = objects[0].get("recording_url")
                        if recording_url:
                            logger.info(f"Recording ready: {recording_url}")
                            # Download and save locally
                            dl = await http.get(recording_url)
                            rec_path = DATA_DIR / f"{session_id}.mp3"
                            rec_path.write_bytes(dl.content)
                            logger.info(f"Recording saved: {rec_path}")
                            # Update session JSON
                            session_path = DATA_DIR / f"{session_id}.json"
                            if session_path.exists():
                                with open(session_path) as f:
                                    session = json.load(f)
                                session["recording_url"] = recording_url
                                session["recording_file"] = str(rec_path)
                                with open(session_path, "w") as f:
                                    json.dump(session, f, indent=2)
                            return
        except Exception as e:
            logger.warning(f"Recording fetch attempt {attempt+1}: {e}")


# ---------------------------------------------------------------------------
# Plivo WebSocket handler
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, language: str | None = None):
    """Handle Plivo bidirectional audio stream."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    session_id = str(uuid.uuid4())
    pipeline = AudioPipeline()
    metrics = MetricsCollector(session_id=session_id)
    language = language or DEFAULT_LANGUAGE
    logger.info(f"Session {session_id}: language={language}")

    # Shared state between receive and transcribe tasks
    audio_lock = asyncio.Lock()
    should_stop = asyncio.Event()

    async def receive_audio():
        """Receive Plivo WebSocket messages and feed audio to pipeline."""
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Plivo start handshake
                if "start" in msg:
                    start_data = msg["start"]
                    stream_id = start_data.get("streamId", "")
                    call_id = start_data.get("callId", "")
                    metrics.set_call_info(call_id, stream_id)
                    logger.info(f"Plivo stream started: call={call_id} stream={stream_id}")
                    # Start Plivo call recording
                    asyncio.create_task(_start_recording(call_id))
                    # Broadcast connection event
                    broadcast_sse({
                        "type": "connected",
                        "session_id": session_id,
                        "call_id": call_id,
                    })
                    continue

                # Plivo media message
                event = msg.get("event", "")
                if event == "media":
                    payload = msg.get("media", {}).get("payload", "")
                    if payload:
                        mulaw_bytes = base64.b64decode(payload)
                        async with audio_lock:
                            pipeline.process(mulaw_bytes)
                        metrics.on_audio_chunk()

                elif event == "stop":
                    logger.info("Plivo stream stopped")
                    break

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
        finally:
            should_stop.set()

    async def transcribe_loop():
        """Periodically flush audio buffer and call NIM API for transcription."""
        while not should_stop.is_set():
            await asyncio.sleep(TRANSCRIBE_INTERVAL)

            async with audio_lock:
                duration = pipeline.buffered_duration_s
                if duration < 0.3:
                    continue
                wav_bytes = pipeline.flush_as_wav()

            if not wav_bytes:
                continue

            result = await nim_client.transcribe(wav_bytes, language=language)
            text = result.get("text", "").strip()

            if not text:
                continue

            utterance = metrics.record_transcription(
                text=text,
                language=result.get("language", language),
                api_latency_ms=result.get("latency_ms", 0),
                audio_duration_s=duration,
                is_final=True,
            )

            event = metrics.to_sse_event(utterance)
            event["session_id"] = session_id
            broadcast_sse(event)

    # Run receive and transcribe concurrently
    receive_task = asyncio.create_task(receive_audio())
    transcribe_task = asyncio.create_task(transcribe_loop())

    try:
        await receive_task
    finally:
        should_stop.set()
        transcribe_task.cancel()
        try:
            await transcribe_task
        except asyncio.CancelledError:
            pass

        # Final transcription of remaining audio
        async with audio_lock:
            if pipeline.buffered_duration_s > 0.1:
                wav_bytes = pipeline.flush_as_wav()
                if wav_bytes:
                    result = await nim_client.transcribe(wav_bytes, language=language)
                    text = result.get("text", "").strip()
                    if text:
                        utterance = metrics.record_transcription(
                            text=text,
                            language=result.get("language", language),
                            api_latency_ms=result.get("latency_ms", 0),
                            audio_duration_s=pipeline.buffered_duration_s,
                            is_final=True,
                        )
                        event = metrics.to_sse_event(utterance)
                        event["session_id"] = session_id
                        broadcast_sse(event)

        metrics.finalize()
        broadcast_sse({
            "type": "disconnected",
            "session_id": session_id,
        })
        logger.info(f"Session {session_id} complete")

        # Fetch and save call recording in background
        if metrics._call_id:
            asyncio.create_task(
                _fetch_and_save_recording(metrics._call_id, session_id)
            )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Parakeet Indic ASR server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
