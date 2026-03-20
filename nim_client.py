"""
NVIDIA NIM ASR Client
=====================
Client for the NVIDIA NIM Parakeet RNNT 1.1B Multilingual model.

Uses the Riva gRPC API for batch transcription of WAV audio buffers.
Supports both cloud (grpc.nvcf.nvidia.com) and local (localhost:50051) endpoints.
"""

import io
import os
import time
import wave

import riva.client
from loguru import logger

# Language code mapping: short code -> Riva locale
LANG_MAP = {
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "bn": "bn-IN",
    "en": "en-US",
}

# Cloud NIM function ID for parakeet-1.1b-rnnt-multilingual-asr
FUNCTION_ID = "71203149-d3b7-4460-8231-1be2543a1fca"


class NIMClient:
    """Client for NVIDIA NIM ASR (Parakeet RNNT 1.1B Multilingual)."""

    def __init__(
        self,
        api_key: str | None = None,
        server: str | None = None,
    ):
        self._api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self._server = server or os.getenv(
            "NIM_ASR_SERVER", "grpc.nvcf.nvidia.com:443"
        )
        self._is_cloud = "nvcf.nvidia.com" in self._server

        # Build metadata for authentication
        metadata = []
        if self._is_cloud:
            metadata.append(["function-id", FUNCTION_ID])
            metadata.append(["authorization", f"Bearer {self._api_key}"])

        self._auth = riva.client.Auth(
            uri=self._server,
            use_ssl=self._is_cloud,
            metadata_args=metadata if metadata else None,
        )
        self._asr = riva.client.ASRService(self._auth)
        logger.info(f"NIMClient: server={self._server} cloud={self._is_cloud}")

    async def transcribe(self, wav_bytes: bytes, language: str = "hi") -> dict:
        """Transcribe a WAV audio buffer.

        Args:
            wav_bytes: In-memory WAV file bytes (16kHz mono PCM16).
            language: Language code (hi, ta, te, bn, en, etc.).

        Returns:
            dict with keys: text, latency_ms, language
        """
        start = time.perf_counter()

        # Map short language code to Riva locale
        lang_code = LANG_MAP.get(language, f"{language}-IN")

        try:
            # Extract raw audio from WAV
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                raw_audio = wf.readframes(wf.getnframes())

            config = riva.client.RecognitionConfig(
                language_code=lang_code,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                audio_channel_count=1,
            )
            config.encoding = riva.client.AudioEncoding.LINEAR_PCM
            config.sample_rate_hertz = 16000

            response = self._asr.offline_recognize(raw_audio, config)
            elapsed_ms = round((time.perf_counter() - start) * 1000)

            # Extract transcript
            text = ""
            for result in response.results:
                if result.alternatives:
                    text += result.alternatives[0].transcript

            text = text.strip()

            logger.info(
                f"NIM ASR: lang={lang_code} latency={elapsed_ms}ms text='{text[:80]}'"
            )
            return {
                "text": text,
                "latency_ms": elapsed_ms,
                "language": language,
            }

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            logger.error(f"NIM API error: {e}")
            return {
                "text": "",
                "latency_ms": elapsed_ms,
                "language": language,
                "error": str(e),
            }

    async def close(self):
        pass
