"""
WebSocket Replay Client
=======================
Sends pre-recorded WAV files to the server via Plivo WebSocket protocol
for testing ASR without a live phone call.

Usage:
    python scripts/replay_client.py test_utterances/hi/sample1.wav
    python scripts/replay_client.py test_utterances/hi/*.wav --language hi
    python scripts/replay_client.py --server localhost:8000 --language ta test_utterances/ta/*.wav
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
import wave

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

import websockets

CHUNK_MS = 20
ULAW_SAMPLE_RATE = 8000
ULAW_CHUNK_BYTES = int(ULAW_SAMPLE_RATE * CHUNK_MS / 1000)  # 160 bytes per 20ms


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def wav_to_ulaw_chunks(path: str, chunk_ms: int = CHUNK_MS) -> list[bytes]:
    """Read a WAV file and return a list of 8kHz mulaw chunks."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        orig_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if n_channels == 2:
        raw = audioop.tomono(raw, sampwidth, 0.5, 0.5)

    if sampwidth != 2:
        raw = audioop.lin2lin(raw, sampwidth, 2)
        sampwidth = 2

    if orig_rate != ULAW_SAMPLE_RATE:
        raw, _ = audioop.ratecv(raw, sampwidth, 1, orig_rate, ULAW_SAMPLE_RATE, None)

    ulaw_data = audioop.lin2ulaw(raw, 2)

    chunk_bytes = int(ULAW_SAMPLE_RATE * chunk_ms / 1000)
    chunks = []
    for i in range(0, len(ulaw_data), chunk_bytes):
        chunk = ulaw_data[i : i + chunk_bytes]
        if len(chunk) == chunk_bytes:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Plivo protocol helpers
# ---------------------------------------------------------------------------

def plivo_start_message(stream_id: str, call_id: str) -> str:
    return json.dumps({"start": {"streamId": stream_id, "callId": call_id}})


def plivo_media_message(ulaw_chunk: bytes) -> str:
    return json.dumps({
        "event": "media",
        "media": {
            "payload": base64.b64encode(ulaw_chunk).decode("ascii"),
            "contentType": "audio/x-mulaw",
            "sampleRate": ULAW_SAMPLE_RATE,
        },
    })


# ---------------------------------------------------------------------------
# Replay logic
# ---------------------------------------------------------------------------

async def replay_files(files: list[str], server: str, language: str):
    """Send WAV files to the server via Plivo WS protocol."""
    ws_url = f"ws://{server}/ws?language={language}"
    stream_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())

    print(f"Connecting to {ws_url}")
    print(f"Language: {language}")
    print(f"Files: {len(files)}")

    async def receive_loop(ws):
        """Print any messages from the server."""
        try:
            async for raw_msg in ws:
                try:
                    msg = json.loads(raw_msg)
                    print(f"  <- {msg}")
                except (json.JSONDecodeError, TypeError):
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass

    async with websockets.connect(ws_url) as ws:
        recv_task = asyncio.create_task(receive_loop(ws))

        # Send Plivo handshake
        await ws.send(plivo_start_message(stream_id, call_id))
        print("Handshake sent")

        for i, path in enumerate(files, 1):
            chunks = wav_to_ulaw_chunks(path)
            duration_s = len(chunks) * CHUNK_MS / 1000
            print(f"[{i}/{len(files)}] Sending {os.path.basename(path)} ({duration_s:.1f}s, {len(chunks)} chunks)")

            for chunk in chunks:
                await ws.send(plivo_media_message(chunk))
                await asyncio.sleep(CHUNK_MS / 1000)

            # Pause between files
            print(f"  Sent. Waiting for transcription...")
            await asyncio.sleep(2.0)

        # Send stop
        await ws.send(json.dumps({"event": "stop"}))
        print("Stop sent. Waiting for final results...")
        await asyncio.sleep(3.0)

        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Replay WAV files via Plivo WS protocol")
    parser.add_argument("files", nargs="+", help="WAV files to send")
    parser.add_argument("--server", default="localhost:8000", help="Server address")
    parser.add_argument("--language", "-l", default="hi", help="Language code (hi, ta, te, bn)")
    args = parser.parse_args()

    # Validate files exist
    for f in args.files:
        if not os.path.isfile(f):
            print(f"Error: File not found: {f}")
            sys.exit(1)

    asyncio.run(replay_files(args.files, args.server, args.language))


if __name__ == "__main__":
    main()
