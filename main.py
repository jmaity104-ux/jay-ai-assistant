import os
os.environ["PATH"] += os.pathsep + r"C:\Users\jmait\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"

from dotenv import load_dotenv
load_dotenv()

import json
import tempfile
from typing import List

import whisper
from groq import Groq

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# Config


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

LLM_MODEL = "llama-3.3-70b-versatile"


SYSTEM_PROMPT = """
You are JAY AI, a smart and helpful AI assistant.

Identity rule:
If anyone asks who made you, who created you, who built you, or who developed you,
always answer:

"I was created by Jaydeb Maity as a personal AI assistant powered by LLaMA and Whisper."

Rules:
- When asked to write code or a program, always show clean formatted code only.
- Never explain code in paragraph form — show the code block directly.
- For voice responses: keep it to 1-2 short sentences.
- For text responses: be clear and structured.
- Never use ** or ## markdown symbols in plain text responses.
- Always sound natural, never robotic.
- Always sound natural and friendly.
- Never say you were created by OpenAI, Groq, or any company.
"""



# App Init


app = FastAPI(title="JAY AI - Live AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")



# Load Models at Startup


print(f"[JAY AI] Loading Whisper ({WHISPER_MODEL_SIZE})...")
stt_model = whisper.load_model(WHISPER_MODEL_SIZE)
print("[JAY AI] Whisper ready.")

llm_client = Groq(api_key=GROQ_API_KEY)
print("[JAY AI] Groq client ready.")



# Schemas


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = True



# Routes


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Convert speech audio to text using Whisper."""

    contents = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1] if audio.filename else "webm"

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = stt_model.transcribe(tmp_path, fp16=False)
        transcript = result["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)

    return {"transcript": transcript}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Send messages to Groq LLaMA and stream the reply back."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:

        def generate():
            stream = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=1024,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'delta': delta})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # fallback if streaming disabled
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=1024,
    )

    return {"reply": response.choices[0].message.content}


@app.post("/api/voice")
async def voice_pipeline(audio: UploadFile = File(...)):
    """
    audio -> Whisper -> LLaMA -> reply
    """

    contents = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1] if audio.filename else "webm"

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = stt_model.transcribe(tmp_path, fp16=False)
        transcript = result["text"].strip()
    finally:
        os.unlink(tmp_path)

    if not transcript:
        return {
            "transcript": "",
            "reply": "I didn't catch that. Could you try again?"
        }

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        max_tokens=512,
    )

    reply = response.choices[0].message.content

    return {
        "transcript": transcript,
        "reply": reply
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "whisper": WHISPER_MODEL_SIZE,
        "llm": LLM_MODEL
    }
