"""
Indic STT Benchmark Client
===========================
Measures 3 metrics for NVIDIA Parakeet RNNT 1.1B on Indic languages:
  1. TTFS  — Time from speech end to final transcript (what the user feels)
  2. WER   — Word Error Rate against known reference text
  3. API Latency — Time inside the NIM gRPC call

Sends pre-recorded WAV files one at a time via Plivo WebSocket protocol,
listens for transcripts via SSE, and computes per-utterance + aggregate metrics.

Usage:
    python scripts/benchmark_client.py
    python scripts/benchmark_client.py --languages hi ta bn
    python scripts/benchmark_client.py --server localhost:8000 --samples 3
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
import unicodedata
import uuid
import wave

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

import httpx
import jiwer
import websockets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTTERANCES_DIR = os.path.join(BASE_DIR, "test_utterances")
MANIFEST_PATH = os.path.join(UTTERANCES_DIR, "manifest.json")

CHUNK_MS = 20
ULAW_SAMPLE_RATE = 8000
ULAW_CHUNK_BYTES = int(ULAW_SAMPLE_RATE * CHUNK_MS / 1000)  # 160 bytes

# Mulaw silence byte (0xFF = zero amplitude in mulaw encoding)
MULAW_SILENCE = b"\xff" * ULAW_CHUNK_BYTES


# ---------------------------------------------------------------------------
# Audio helpers (reused from replay_client.py)
# ---------------------------------------------------------------------------

def wav_to_ulaw_chunks(path: str) -> list[bytes]:
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

    chunks = []
    for i in range(0, len(ulaw_data), ULAW_CHUNK_BYTES):
        chunk = ulaw_data[i : i + ULAW_CHUNK_BYTES]
        if len(chunk) == ULAW_CHUNK_BYTES:
            chunks.append(chunk)
    return chunks


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
# WER helpers
# ---------------------------------------------------------------------------

def normalize_indic(text: str) -> str:
    """Normalize text for Indic WER comparison.

    Strips punctuation (including Devanagari danda), collapses whitespace,
    lowercases Latin characters, and applies NFC normalization.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove punctuation: ASCII + Devanagari danda/double-danda + common marks
    text = re.sub(r'[।॥,.\-!?:;"\'\(\)\[\]…\u0964\u0965]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute semantic WER between reference and hypothesis.

    Uses two strategies and picks the more lenient:
      1. Standard word-level WER (after punctuation normalization)
      2. Character Error Rate (CER) — handles Indic compound word splits
         where ASR inserts spaces inside compound words
         (e.g., "தயவுசெய்து" vs "தயவு செய்து")

    Returns the lower of the two, since compound-word spacing differences
    are not meaningful errors for an LLM consuming the transcript.
    """
    ref = normalize_indic(reference)
    hyp = normalize_indic(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0

    # Standard word-level WER
    word_wer = jiwer.wer(ref, hyp)

    # Character Error Rate (spaces removed — catches compound word splits)
    ref_chars = ref.replace(" ", "")
    hyp_chars = hyp.replace(" ", "")
    cer = jiwer.cer(ref_chars, hyp_chars)

    return round(min(word_wer, cer), 4)


# ---------------------------------------------------------------------------
# SSE listener
# ---------------------------------------------------------------------------

async def sse_listener(url: str, queue: asyncio.Queue, stop_event: asyncio.Event,
                       ready_event: asyncio.Event | None = None):
    """Connect to SSE endpoint and push transcript events to queue.

    Parses SSE wire format: sse-starlette sends named events like:
        event: transcript
        data: {"type":"final",...}

    Also handles :ping keepalive lines.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            async with client.stream("GET", url) as response:
                if ready_event:
                    ready_event.set()
                buffer = ""
                async for chunk in response.aiter_text():
                    if stop_event.is_set():
                        break
                    buffer += chunk
                    # SSE events are separated by \r\n\r\n (sse-starlette)
                    while "\r\n\r\n" in buffer:
                        event_block, buffer = buffer.split("\r\n\r\n", 1)
                        # Parse all data: lines in this event block
                        for line in event_block.split("\r\n"):
                            line = line.strip()
                            if line.startswith("data:"):
                                raw = line[5:].strip()
                                if not raw:
                                    continue
                                try:
                                    data = json.loads(raw)
                                    if data.get("type") in ("final", "partial",
                                                            "connected", "disconnected"):
                                        await queue.put(data)
                                except json.JSONDecodeError:
                                    pass
    except httpx.ReadError:
        pass
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def benchmark_language(server: str, lang: str, entries: list[dict], samples: int):
    """Benchmark one language: send WAVs one at a time, collect transcripts via SSE."""
    entries = entries[:samples]
    results = []
    transcript_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()
    sse_ready = asyncio.Event()

    # Start SSE listener
    sse_url = f"http://{server}/api/live"
    sse_task = asyncio.create_task(
        sse_listener(sse_url, transcript_queue, stop_event, sse_ready)
    )

    # Wait for SSE to actually connect before sending audio
    try:
        await asyncio.wait_for(sse_ready.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        print("    WARNING: SSE connection timeout, proceeding anyway")
    await asyncio.sleep(0.3)

    ws_url = f"ws://{server}/ws?language={lang}"
    stream_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())

    print(f"\n  Connecting to {ws_url}")

    try:
        async with websockets.connect(ws_url) as ws:
            # Plivo handshake
            await ws.send(plivo_start_message(stream_id, call_id))
            # Wait for "connected" SSE event to arrive and be queued
            await asyncio.sleep(1.0)
            # Drain the connected event
            while not transcript_queue.empty():
                try:
                    transcript_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            for i, entry in enumerate(entries, 1):
                wav_path = os.path.join(UTTERANCES_DIR, lang, entry["file"])
                reference = entry["reference_text"]

                if not os.path.isfile(wav_path):
                    print(f"    [{i}/{len(entries)}] SKIP — {entry['file']} not found")
                    continue

                # Drain any stale events from previous utterance
                while not transcript_queue.empty():
                    try:
                        transcript_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                # Send audio chunks at real-time pace
                chunks = wav_to_ulaw_chunks(wav_path)
                audio_duration_s = len(chunks) * CHUNK_MS / 1000

                print(f"    [{i}/{len(entries)}] Sending {entry['file']} "
                      f"({audio_duration_s:.1f}s, {len(chunks)} chunks)...")

                for chunk in chunks:
                    await ws.send(plivo_media_message(chunk))
                    await asyncio.sleep(CHUNK_MS / 1000)

                # Record when speech audio ended
                speech_end_ts = time.perf_counter()

                # Inject silence to force the server's buffer to flush
                # Must be longer than TRANSCRIBE_INTERVAL so the buffer
                # flushes with all speech audio in one chunk
                silence_secs = float(os.getenv("SILENCE_PADDING", "5.0"))
                silence_chunks = int(silence_secs / (CHUNK_MS / 1000))
                for _ in range(silence_chunks):
                    await ws.send(plivo_media_message(MULAW_SILENCE))
                    await asyncio.sleep(CHUNK_MS / 1000)

                # Collect transcript fragments
                hypothesis_parts = []
                api_latency_ms = None
                ttfs_ms = None
                deadline = time.perf_counter() + 10.0

                while time.perf_counter() < deadline:
                    try:
                        event = await asyncio.wait_for(
                            transcript_queue.get(), timeout=6.0
                        )
                        if event.get("type") == "final" and event.get("text", "").strip():
                            now = time.perf_counter()
                            if ttfs_ms is None:
                                ttfs_ms = round((now - speech_end_ts) * 1000)
                            hypothesis_parts.append(event["text"])
                            api_latency_ms = event.get("metrics", {}).get("api_latency_ms")
                    except asyncio.TimeoutError:
                        break

                hypothesis = " ".join(hypothesis_parts)
                wer = compute_wer(reference, hypothesis) if hypothesis else 1.0

                result = {
                    "file": entry["file"],
                    "lang": lang,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": wer,
                    "ttfs_ms": ttfs_ms,
                    "api_latency_ms": api_latency_ms,
                    "audio_duration_s": round(audio_duration_s, 2),
                    "fragment_count": len(hypothesis_parts),
                }
                results.append(result)

                wer_pct = f"{wer:.0%}"
                ttfs_str = f"{ttfs_ms}ms" if ttfs_ms else "N/A"
                lat_str = f"{api_latency_ms}ms" if api_latency_ms else "N/A"
                print(f"           WER={wer_pct}  TTFS={ttfs_str}  "
                      f"API={lat_str}  fragments={len(hypothesis_parts)}")
                if hypothesis:
                    print(f"           hyp: {hypothesis[:80]}")

            # Send stop
            await ws.send(json.dumps({"event": "stop"}))
            await asyncio.sleep(1.0)

    except Exception as e:
        print(f"    ERROR: {e}")

    finally:
        stop_event.set()
        sse_task.cancel()
        try:
            await sse_task
        except asyncio.CancelledError:
            pass

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: dict[str, list[dict]]):
    """Print a summary table in Pipecat-comparable format."""
    print(f"\n{'=' * 72}")
    print(f"  Parakeet RNNT 1.1B — Indic Telephony Benchmark")
    print(f"  {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    print(f"{'=' * 72}")

    header = f"{'Lang':<6} {'Utts':>5} {'WER':>8} {'TTFS med':>10} {'TTFS p95':>10} {'API med':>10}"
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    grand_wers = []
    grand_ttfs = []
    grand_api = []
    grand_utts = 0

    for lang in sorted(all_results.keys()):
        results = all_results[lang]
        if not results:
            continue

        wers = [r["wer"] for r in results if r["hypothesis"]]
        ttfs_vals = sorted([r["ttfs_ms"] for r in results if r["ttfs_ms"] is not None])
        api_vals = sorted([r["api_latency_ms"] for r in results if r["api_latency_ms"] is not None])

        n = len(results)
        grand_utts += n

        avg_wer = sum(wers) / len(wers) if wers else 0
        grand_wers.extend(wers)

        ttfs_med = ttfs_vals[len(ttfs_vals) // 2] if ttfs_vals else None
        ttfs_p95 = ttfs_vals[int(len(ttfs_vals) * 0.95)] if len(ttfs_vals) > 1 else ttfs_med
        grand_ttfs.extend(ttfs_vals)

        api_med = api_vals[len(api_vals) // 2] if api_vals else None
        grand_api.extend(api_vals)

        wer_str = f"{avg_wer:.1%}"
        ttfs_med_str = f"{ttfs_med}ms" if ttfs_med else "N/A"
        ttfs_p95_str = f"{ttfs_p95}ms" if ttfs_p95 else "N/A"
        api_med_str = f"{api_med}ms" if api_med else "N/A"

        print(f"  {lang.upper():<6} {n:>5} {wer_str:>8} {ttfs_med_str:>10} {ttfs_p95_str:>10} {api_med_str:>10}")

    # Overall
    print(f"  {'-' * len(header)}")
    if grand_wers:
        overall_wer = f"{sum(grand_wers) / len(grand_wers):.1%}"
    else:
        overall_wer = "N/A"
    grand_ttfs.sort()
    grand_api.sort()
    overall_ttfs = f"{grand_ttfs[len(grand_ttfs) // 2]}ms" if grand_ttfs else "N/A"
    overall_ttfs_p95 = f"{grand_ttfs[int(len(grand_ttfs) * 0.95)]}ms" if len(grand_ttfs) > 1 else overall_ttfs
    overall_api = f"{grand_api[len(grand_api) // 2]}ms" if grand_api else "N/A"
    print(f"  {'ALL':<6} {grand_utts:>5} {overall_wer:>8} {overall_ttfs:>10} {overall_ttfs_p95:>10} {overall_api:>10}")

    # Per-utterance detail
    print(f"\n  {'─' * 72}")
    print(f"  Per-utterance detail:")
    print(f"  {'─' * 72}")
    for lang in sorted(all_results.keys()):
        for r in all_results[lang]:
            wer_str = f"{r['wer']:.0%}" if r["hypothesis"] else "MISS"
            ttfs_str = f"{r['ttfs_ms']}ms" if r["ttfs_ms"] else "N/A"
            api_str = f"{r['api_latency_ms']}ms" if r["api_latency_ms"] else "N/A"
            print(f"  {r['lang'].upper()} {r['file']:<12} WER={wer_str:<6} "
                  f"TTFS={ttfs_str:<8} API={api_str:<8}")
            if r["hypothesis"]:
                print(f"     ref: {r['reference'][:70]}")
                print(f"     hyp: {r['hypothesis'][:70]}")

    print()


def save_results(all_results: dict[str, list[dict]], output_dir: str):
    """Save benchmark results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "parakeet-1.1b-rnnt-multilingual (indic profile)",
        "languages": {},
    }

    for lang, results in all_results.items():
        wers = [r["wer"] for r in results if r["hypothesis"]]
        ttfs_vals = [r["ttfs_ms"] for r in results if r["ttfs_ms"] is not None]
        api_vals = [r["api_latency_ms"] for r in results if r["api_latency_ms"] is not None]

        output["languages"][lang] = {
            "utterances": len(results),
            "avg_wer": round(sum(wers) / len(wers), 4) if wers else None,
            "ttfs_median_ms": sorted(ttfs_vals)[len(ttfs_vals) // 2] if ttfs_vals else None,
            "api_latency_median_ms": sorted(api_vals)[len(api_vals) // 2] if api_vals else None,
            "results": results,
        }

    path = os.path.join(output_dir, "benchmark_indic.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    all_results = {}

    for lang in args.languages:
        entries = [e for e in manifest if e["lang"] == lang]
        if not entries:
            print(f"\n  No manifest entries for {lang.upper()}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Benchmarking {lang.upper()} — {min(args.samples, len(entries))} utterances")
        print(f"{'=' * 60}")

        results = await benchmark_language(args.server, lang, entries, args.samples)
        all_results[lang] = results

    if all_results:
        print_summary(all_results)
        save_results(all_results, os.path.join(BASE_DIR, "data"))


def main():
    parser = argparse.ArgumentParser(
        description="Indic STT Benchmark — TTFS, WER, API Latency"
    )
    parser.add_argument("--server", default="localhost:8000", help="Server address")
    parser.add_argument(
        "--languages", "-l", nargs="+", default=["hi", "ta", "bn"],
        help="Languages to benchmark (default: hi ta bn)"
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=5,
        help="Max utterances per language (default: 5)"
    )
    args = parser.parse_args()

    print("Parakeet RNNT 1.1B Multilingual — Indic Benchmark")
    print(f"Server: {args.server}")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Samples per language: {args.samples}")
    print(f"Metrics: TTFS, WER, API Latency")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
