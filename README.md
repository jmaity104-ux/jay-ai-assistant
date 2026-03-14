# JARVIS — Live AI Assistant

A real-time voice + text AI assistant. Speak into your mic or type — get intelligent responses instantly, spoken back aloud.

Built with **FastAPI**, **Whisper** (local STT), **Groq + LLaMA 3.3** (LLM), and vanilla JS frontend.

---

## Demo

> Hold mic → speak → release → JARVIS responds in text + voice

---

## Features

- Voice input via browser microphone (WebM recording)
- Local speech-to-text using OpenAI Whisper (no STT API cost)
- LLM responses via Groq API running LLaMA 3.3 70B (free tier)
- Streaming text responses using Server-Sent Events
- Browser-native text-to-speech for voice output
- Full conversation memory within a session
- Works on desktop and mobile

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Speech-to-Text | OpenAI Whisper (runs locally) |
| LLM | Groq API — LLaMA 3.3 70B |
| Streaming | Server-Sent Events (SSE) |
| Frontend | HTML, CSS, Vanilla JS |
| TTS | Web Speech API (browser built-in) |

---

## Project Structure

```
jarvis/
├── main.py              # FastAPI server — all routes & pipelines
├── requirements.txt
├── .env.example
└── templates/
    └── index.html       # Full frontend — single file, no framework
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- ffmpeg installed (`brew install ffmpeg` / `sudo apt install ffmpeg`)
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Setup

```bash
# Clone
git clone https://github.com/yourusername/jarvis.git
cd jarvis

# Virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install
pip install -r requirements.txt

# Environment
cp .env.example .env
# Add your GROQ_API_KEY inside .env
```

### Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend UI |
| POST | `/api/voice` | Full pipeline: audio → STT → LLM → reply |
| POST | `/api/chat` | Streaming text chat |
| POST | `/api/transcribe` | Audio → transcript only |
| GET | `/api/health` | Health check |

---

## Voice Pipeline

```
Browser mic (WebM blob)
        │
        ▼
  POST /api/voice
        │
        ├─── Whisper transcribes locally
        │
        ├─── Transcript → Groq LLaMA 3.3
        │
        └─── { transcript, reply } → Browser
                        │
                        ▼
              SpeechSynthesisUtterance (spoken)
```

## Text Pipeline

```
User types → POST /api/chat → Groq streams tokens
                                      │
                        Server-Sent Events (SSE)
                                      │
                        Words appear in real time
```

---

## Whisper Model Sizes

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| tiny | 39 MB | Fastest | Basic |
| base | 74 MB | Fast | Good ← default |
| small | 244 MB | Medium | Better |
| medium | 769 MB | Slow | Best (local) |

Change via `WHISPER_MODEL=small` in `.env`

---

## Roadmap

- [ ] ElevenLabs TTS for more natural voice output
- [ ] Deepgram STT for lower latency (< 300ms)
- [ ] Persistent chat history with SQLite
- [ ] Wake word detection ("Hey JARVIS")
- [ ] Docker support

---

## License

MIT
