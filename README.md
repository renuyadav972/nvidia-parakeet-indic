# NVIDIA Parakeet Indic Demo

A live Hindi voice agent built with **NVIDIA Parakeet RNNT 1.1B Multilingual** for speech recognition, **Google Gemini** for conversation, **ElevenLabs** for text-to-speech, and **Plivo** for telephony. The agent answers a phone call, transcribes the caller's Hindi speech, and responds in real time.

This is a working test bed for evaluating Parakeet's Hindi ASR over real telephony audio. It includes a transcription benchmark harness and a session metrics collector.

Findings from real test calls (recordings + transcripts): https://nvidia-parakeet-indic.vercel.app/

## How It Works

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌──────────┐     ┌────────────┐
│  Phone  │────▶│  Plivo  │────▶│ FastAPI │────▶│ Parakeet │────▶│   Gemini   │
│  Call   │◀────│   WS    │◀────│ Server  │◀────│   STT    │     │    LLM     │
└─────────┘     └─────────┘     └─────────┘     └──────────┘     └────────────┘
                                      │                                  │
                                      ▼                                  ▼
                                ┌──────────┐                      ┌────────────┐
                                │ Metrics  │                      │ ElevenLabs │
                                │Collector │                      │    TTS     │
                                └──────────┘                      └────────────┘
```

1. A phone call is placed (or received) via Plivo
2. Plivo opens a bidirectional WebSocket to the FastAPI server
3. Inbound audio is streamed in chunks to NVIDIA NIM (Parakeet RNNT 1.1B)
4. Transcribed text goes to Gemini for the conversation reply
5. The reply is synthesized by ElevenLabs and streamed back over the same WebSocket
6. Per-turn latencies are written to `data/sessions/`

## Project Layout

```
nim_client.py          # gRPC client for NVIDIA NIM (cloud or self-hosted)
audio_pipeline.py      # μ-law / PCM conversion for telephony audio
server.py              # FastAPI app, Plivo webhook + WebSocket handler
metrics_collector.py   # Per-session waterfall + transcript capture
scripts/               # Call placement + benchmark utilities
  make_call.py         # Place an outbound test call via Plivo
test_utterances/       # Reference Hindi utterances for benchmarking
data/sessions/         # Session JSON output (gitignored)
static/                # Static assets served by FastAPI
```

## Prerequisites

- Python 3.11+
- A [Plivo](https://www.plivo.com/) account with a phone number
- An [NVIDIA NIM](https://build.nvidia.com/) API key (cloud) or a self-hosted Parakeet NIM container
- [ngrok](https://ngrok.com/) for exposing the local server to Plivo
- A Gemini API key and an ElevenLabs API key (loaded via your environment)

## Quick Start

1. **Clone and install**
   ```bash
   git clone https://github.com/renuyadav972/nvidia-parakeet-indic.git
   cd nvidia-parakeet-indic
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Fill in:
   ```
   PLIVO_AUTH_ID=
   PLIVO_AUTH_TOKEN=
   PLIVO_PHONE_NUMBER=
   NVIDIA_API_KEY=
   NIM_ASR_SERVER=grpc.nvcf.nvidia.com:443   # or localhost:50051 for self-hosted
   PORT=8000
   PUBLIC_DOMAIN=localhost:8000
   ```

3. **Start ngrok** (in a separate terminal)
   ```bash
   ngrok http 8000
   ```
   Copy the HTTPS forwarding URL into `PUBLIC_DOMAIN` (without the `https://` prefix).

4. **Run the server**
   ```bash
   SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())") \
     python server.py
   ```

5. **Place a test call**
   ```bash
   python scripts/make_call.py --to +1XXXXXXXXXX
   ```
   The Plivo number will dial out and connect the answering side to your agent.

## Cloud vs Self-Hosted NIM

The default `NIM_ASR_SERVER` points to `grpc.nvcf.nvidia.com:443`, NVIDIA's hosted endpoint. The hosted endpoint exposes the **Default** profile of Parakeet RNNT 1.1B Multilingual, which supports Hindi (along with the other languages in that profile).

For Tamil, Bengali, or other languages that live in the **Indic** profile, you currently need to run a self-hosted NIM container and point `NIM_ASR_SERVER` at it (e.g. `localhost:50051`). The cloud function ID does not appear to expose the Indic profile at this time.

## Notes on Hindi ASR Quality

A few things from real call testing:

- **Formal Hindi**: accurate. Numbers, domain terminology (we tested with cricket vocabulary), and gendered verb forms all transcribe correctly.
- **Hinglish / code-switching**: unreliable. English words embedded in Hindi sentences are often mis-transcribed.
- **Output is formalized**: casual speech and shortened forms get cleaned up into textbook Hindi. If you need transcriptions that reflect how someone actually spoke (for QA audits, sentiment analysis, training data), this may matter.

Full writeup with audio: https://nvidia-parakeet-indic.vercel.app/

## License

MIT
