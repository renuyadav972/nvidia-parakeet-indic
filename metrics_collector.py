"""
Metrics Collector
=================
Per-utterance metrics for ASR transcription sessions.

Tracks TTFT, RTF, API latency, finalization latency, and WER.
Writes session JSON to data/sessions/ using atomic writes.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timezone

from loguru import logger


class MetricsCollector:
    """Collects per-utterance ASR metrics for a single call session."""

    def __init__(
        self,
        session_id: str,
        data_dir: str = "data/sessions",
    ):
        self.session_id = session_id
        self._data_dir = data_dir
        self._output_path = os.path.join(data_dir, f"{session_id}.json")

        self._started_at = datetime.now(timezone.utc).isoformat()
        self._ended_at: str | None = None
        self._call_id: str | None = None
        self._stream_id: str | None = None

        # Per-utterance records
        self._utterances: list[dict] = []
        self._utterance_num = 0

        # Timing state for current chunk
        self._first_audio_ts: float | None = None
        self._last_audio_ts: float | None = None
        self._audio_chunks_received = 0

        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"MetricsCollector: session={session_id} -> {self._output_path}")

    def set_call_info(self, call_id: str, stream_id: str):
        self._call_id = call_id
        self._stream_id = stream_id

    def on_audio_chunk(self):
        """Called when an audio chunk is received from Plivo."""
        now = time.perf_counter()
        if self._first_audio_ts is None:
            self._first_audio_ts = now
        self._last_audio_ts = now
        self._audio_chunks_received += 1

    def record_transcription(
        self,
        text: str,
        language: str,
        api_latency_ms: int,
        audio_duration_s: float,
        is_final: bool = True,
        reference_text: str | None = None,
    ):
        """Record a transcription result with metrics."""
        now = time.perf_counter()
        self._utterance_num += 1

        # TTFT: time from first audio byte to first transcript
        ttft_ms = None
        if self._first_audio_ts is not None:
            ttft_ms = round((now - self._first_audio_ts) * 1000)

        # RTF: processing time / audio duration
        rtf = None
        if audio_duration_s > 0:
            rtf = round(api_latency_ms / 1000 / audio_duration_s, 3)

        # Finalization latency: last audio byte to transcript
        finalization_ms = None
        if self._last_audio_ts is not None:
            finalization_ms = round((now - self._last_audio_ts) * 1000)

        # WER (computed later if reference provided)
        wer = None
        if reference_text and text:
            wer = self._compute_wer(reference_text, text)

        utterance = {
            "utterance_num": self._utterance_num,
            "text": text,
            "language": language,
            "is_final": is_final,
            "reference_text": reference_text,
            "ttft_ms": ttft_ms,
            "rtf": rtf,
            "api_latency_ms": api_latency_ms,
            "finalization_ms": finalization_ms,
            "audio_duration_s": round(audio_duration_s, 2),
            "wer": wer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._utterances.append(utterance)

        # Reset timing for next chunk
        self._first_audio_ts = None
        self._last_audio_ts = None

        logger.info(
            f"MetricsCollector: utt#{self._utterance_num} "
            f"lang={language} api={api_latency_ms}ms rtf={rtf} "
            f"text='{text[:60]}'"
        )

        self.flush()
        return utterance

    def _compute_wer(self, reference: str, hypothesis: str) -> float | None:
        try:
            import unicodedata

            import jiwer

            ref = unicodedata.normalize("NFC", reference.strip())
            hyp = unicodedata.normalize("NFC", hypothesis.strip())
            return round(jiwer.wer(ref, hyp), 4)
        except Exception as e:
            logger.warning(f"WER computation failed: {e}")
            return None

    def finalize(self):
        self._ended_at = datetime.now(timezone.utc).isoformat()
        self.flush()
        logger.info(
            f"MetricsCollector: session {self.session_id} finalized — "
            f"{len(self._utterances)} utterances"
        )

    def _build_summary(self) -> dict:
        api_latencies = [u["api_latency_ms"] for u in self._utterances]
        rtfs = [u["rtf"] for u in self._utterances if u["rtf"] is not None]
        ttfts = [u["ttft_ms"] for u in self._utterances if u["ttft_ms"] is not None]
        wers = [u["wer"] for u in self._utterances if u["wer"] is not None]

        return {
            "total_utterances": len(self._utterances),
            "avg_api_latency_ms": round(sum(api_latencies) / len(api_latencies)) if api_latencies else 0,
            "min_api_latency_ms": min(api_latencies) if api_latencies else 0,
            "max_api_latency_ms": max(api_latencies) if api_latencies else 0,
            "avg_rtf": round(sum(rtfs) / len(rtfs), 3) if rtfs else None,
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts)) if ttfts else None,
            "avg_wer": round(sum(wers) / len(wers), 4) if wers else None,
            "languages": list({u["language"] for u in self._utterances}),
        }

    def flush(self):
        """Atomic write session JSON to disk."""
        session = {
            "session_id": self.session_id,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "call_id": self._call_id,
            "stream_id": self._stream_id,
            "utterances": self._utterances,
            "summary": self._build_summary(),
        }

        fd, tmp_path = tempfile.mkstemp(
            dir=self._data_dir, suffix=".tmp", prefix=".session_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(session, f, indent=2)
            os.rename(tmp_path, self._output_path)
        except Exception:
            logger.exception("MetricsCollector: failed to write session JSON")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def to_sse_event(self, utterance: dict) -> dict:
        """Format an utterance as an SSE event payload."""
        return {
            "type": "final" if utterance["is_final"] else "partial",
            "text": utterance["text"],
            "lang": utterance["language"],
            "utterance_num": utterance["utterance_num"],
            "metrics": {
                "ttft_ms": utterance["ttft_ms"],
                "rtf": utterance["rtf"],
                "api_latency_ms": utterance["api_latency_ms"],
                "finalization_ms": utterance["finalization_ms"],
                "wer": utterance["wer"],
            },
        }
