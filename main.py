import logging
import os
import secrets
import shutil
import sqlite3
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import AsyncOpenAI

load_dotenv()


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value or ""


BOT_TOKEN = env("TELEGRAM_BOT_TOKEN", required=True)
OPENAI_API_KEY = env("OPENAI_API_KEY", required=True)
PUBLIC_BASE_URL = env("PUBLIC_BASE_URL", "").rstrip("/")
WEBHOOK_SECRET = env("WEBHOOK_SECRET", required=True)
ALLOWED_TELEGRAM_USER_IDS = {
    int(x.strip())
    for x in env("ALLOWED_TELEGRAM_USER_IDS", required=True).split(",")
    if x.strip()
}
TRANSCRIBE_MODEL = env("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
SUMMARY_MODEL = env("OPENAI_SUMMARY_MODEL", "gpt-4.1-mini")
TRANSCRIPT_TTL_DAYS = int(env("TRANSCRIPT_TTL_DAYS", "7"))
DB_PATH = Path(env("DB_PATH", "data/tgtldr.sqlite3"))
LOG_LEVEL = env("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("tgtldr")

TELEGRAM_DOWNLOAD_LIMIT_BYTES = 20 * 1024 * 1024
SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".ogg", ".oga", ".webm", ".flac", ".aac", ".mpga", ".mpeg"
}
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm"
}

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcripts (
                token TEXT PRIMARY KEY,
                transcript TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def purge_expired_transcripts() -> None:
    now = int(time.time())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM transcripts WHERE expires_at < ?", (now,))
        conn.commit()


def save_transcript(transcript: str) -> str:
    token = secrets.token_urlsafe(18)
    now = int(time.time())
    expires_at = now + TRANSCRIPT_TTL_DAYS * 24 * 60 * 60
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO transcripts (token, transcript, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, transcript, now, expires_at),
        )
        conn.commit()
    return token


def load_transcript(token: str) -> str | None:
    now = int(time.time())
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT transcript, expires_at FROM transcripts WHERE token = ?",
            (token,),
        ).fetchone()

    if not row:
        return None

    transcript, expires_at = row
    if expires_at < now:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM transcripts WHERE token = ?", (token,))
            conn.commit()
        return None

    return transcript


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    purge_expired_transcripts()
    yield


app = FastAPI(title="tgtldr", lifespan=lifespan)


async def telegram_api(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error in {method}: {data}")

    return data["result"]


async def send_message(chat_id: int, text: str, reply_to_message_id: int | None = None) -> None:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id

    await telegram_api("sendMessage", payload)


async def get_telegram_file(file_id: str, target_path: Path) -> None:
    result = await telegram_api("getFile", {"file_id": file_id})
    file_path = result["file_path"]
    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(file_url)
        response.raise_for_status()
        target_path.write_bytes(response.content)


def pick_media(message: dict[str, Any]) -> dict[str, Any] | None:
    if voice := message.get("voice"):
        return {
            "kind": "voice",
            "file_id": voice["file_id"],
            "file_size": voice.get("file_size", 0),
            "suffix": ".ogg",
            "needs_audio_extract": False,
        }

    if audio := message.get("audio"):
        suffix = Path(audio.get("file_name", "audio.mp3")).suffix.lower() or ".mp3"
        return {
            "kind": "audio",
            "file_id": audio["file_id"],
            "file_size": audio.get("file_size", 0),
            "suffix": suffix,
            "needs_audio_extract": False,
        }

    if video_note := message.get("video_note"):
        return {
            "kind": "video_note",
            "file_id": video_note["file_id"],
            "file_size": video_note.get("file_size", 0),
            "suffix": ".mp4",
            "needs_audio_extract": True,
        }

    if video := message.get("video"):
        suffix = Path(video.get("file_name", "video.mp4")).suffix.lower() or ".mp4"
        return {
            "kind": "video",
            "file_id": video["file_id"],
            "file_size": video.get("file_size", 0),
            "suffix": suffix,
            "needs_audio_extract": True,
        }

    if document := message.get("document"):
        suffix = Path(document.get("file_name", "document.bin")).suffix.lower()
        mime = (document.get("mime_type") or "").lower()

        if mime.startswith("audio/") or suffix in SUPPORTED_AUDIO_EXTENSIONS:
            return {
                "kind": "document_audio",
                "file_id": document["file_id"],
                "file_size": document.get("file_size", 0),
                "suffix": suffix or ".bin",
                "needs_audio_extract": False,
            }

        if mime.startswith("video/") or suffix in SUPPORTED_VIDEO_EXTENSIONS:
            return {
                "kind": "document_video",
                "file_id": document["file_id"],
                "file_size": document.get("file_size", 0),
                "suffix": suffix or ".bin",
                "needs_audio_extract": True,
            }

    return None


def extract_audio(input_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for video/video_note support but was not found on PATH.")

    output_path = input_path.with_suffix(".mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")

    return output_path


async def transcribe_audio(audio_path: Path) -> str:
    with audio_path.open("rb") as media_file:
        result = await openai_client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=media_file,
            response_format="text",
        )
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", "")
    return text.strip()


async def summarize_transcript(transcript: str) -> str:
    prompt = """
You summarize transcripts from Telegram voice notes and short videos.

Rules:
- Detect the transcript language yourself.
- If the transcript is in Russian or English, write the summary in that same language.
- If the transcript is in any other language, write the summary in English.
- Keep it concise and practical.
- Preserve names, dates, places, requests, decisions, promises, and next actions.
- Use short paragraphs or bullet points.
- Do not start with phrases like "Here is the summary".
- If the transcript is too short, too noisy, or unclear, say so briefly.
""".strip()

    response = await openai_client.responses.create(
        model=SUMMARY_MODEL,
        input=f"{prompt}\n\nTranscript:\n{transcript}",
    )
    return response.output_text.strip()


async def process_message(message: dict[str, Any]) -> None:
    media = pick_media(message)
    if not media:
        return

    sender = message.get("from") or {}
    sender_id = sender.get("id")
    chat_id = message["chat"]["id"]
    message_id = message["message_id"]

    if sender_id not in ALLOWED_TELEGRAM_USER_IDS:
        log.info("Ignoring message from sender_id=%s", sender_id)
        return

    if media["file_size"] and media["file_size"] > TELEGRAM_DOWNLOAD_LIMIT_BYTES:
        await send_message(
            chat_id,
            "Too large for V1. Please keep it under 20 MB.",
            reply_to_message_id=message_id,
        )
        return

    await send_message(chat_id, "Got it. Transcribing...", reply_to_message_id=message_id)

    try:
        with tempfile.TemporaryDirectory(prefix="tgtldr-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / f"input{media['suffix']}"
            await get_telegram_file(media["file_id"], source_path)

            audio_path = source_path
            if media["needs_audio_extract"]:
                audio_path = extract_audio(source_path)

            transcript = await transcribe_audio(audio_path)
            if not transcript:
                await send_message(
                    chat_id,
                    "I couldn't get a usable transcript from that.",
                    reply_to_message_id=message_id,
                )
                return

            summary = await summarize_transcript(transcript)
            token = save_transcript(transcript)

            lines = [summary]
            if PUBLIC_BASE_URL:
                lines.append("")
                lines.append(f"Transcript: {PUBLIC_BASE_URL}/t/{token}")
            else:
                lines.append("")
                lines.append("Transcript link disabled: set PUBLIC_BASE_URL.")

            await send_message(chat_id, "\n".join(lines), reply_to_message_id=message_id)

    except Exception as exc:
        log.exception("Processing failed")
        await send_message(
            chat_id,
            f"Sorry, processing failed: {exc}",
            reply_to_message_id=message_id,
        )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "tgtldr"})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/t/{token}")
async def transcript(token: str) -> PlainTextResponse:
    transcript_text = load_transcript(token)
    if transcript_text is None:
        raise HTTPException(status_code=404, detail="Transcript not found or expired")
    return PlainTextResponse(transcript_text)


@app.post("/telegram/webhook/{webhook_secret}")
async def telegram_webhook(
    webhook_secret: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    if webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=404, detail="Not found")

    update = await request.json()
    message = update.get("message")
    if message:
        background_tasks.add_task(process_message, message)

    return JSONResponse({"ok": True})
