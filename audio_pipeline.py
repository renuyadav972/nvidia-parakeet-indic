"""
Audio Pipeline
==============
Converts Plivo's 8kHz mulaw audio stream to 16kHz PCM16 WAV for the NIM ASR API.

Stateful: maintains audioop ratecv state for seamless resampling across chunks.
"""

import io
import struct
import wave

try:
    import audioop
except ImportError:
    import audioop_lts as audioop


class AudioPipeline:
    """Accumulates mulaw 8kHz audio and converts to 16kHz PCM16 WAV."""

    def __init__(self):
        self._ratecv_state = None
        self._pcm16_16k_buffer = bytearray()
        self._total_bytes_in = 0

    def process(self, mulaw_8k: bytes) -> bytes:
        """Convert mulaw 8kHz bytes to PCM16 16kHz bytes.

        Returns the converted PCM16 16kHz chunk. Also accumulates internally
        for flush_as_wav().
        """
        self._total_bytes_in += len(mulaw_8k)

        # mulaw -> PCM16 at 8kHz
        pcm16_8k = audioop.ulaw2lin(mulaw_8k, 2)

        # Resample 8kHz -> 16kHz
        pcm16_16k, self._ratecv_state = audioop.ratecv(
            pcm16_8k, 2, 1, 8000, 16000, self._ratecv_state
        )

        self._pcm16_16k_buffer.extend(pcm16_16k)
        return pcm16_16k

    def flush_as_wav(self) -> bytes:
        """Return accumulated audio as an in-memory WAV file and clear the buffer."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(bytes(self._pcm16_16k_buffer))
        self._pcm16_16k_buffer.clear()
        return buf.getvalue()

    def peek_as_wav(self) -> bytes:
        """Return accumulated audio as WAV without clearing the buffer."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(bytes(self._pcm16_16k_buffer))
        return buf.getvalue()

    @property
    def buffered_duration_s(self) -> float:
        """Duration of audio currently in the buffer (seconds)."""
        return len(self._pcm16_16k_buffer) / (16000 * 2)

    @property
    def total_duration_s(self) -> float:
        """Total duration of all audio processed (seconds)."""
        return self._total_bytes_in / 8000  # mulaw is 1 byte per sample at 8kHz

    def clear(self):
        """Clear the buffer without returning data."""
        self._pcm16_16k_buffer.clear()
