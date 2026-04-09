# tgtldr

Minimal private Telegram bot for Arseni + Sasha.

Supported inputs:
- voice notes
- audio files
- Telegram video notes ("circles")
- short videos (audio only is extracted)

Flow:
1. Receive Telegram message
2. Reply with quick ack
3. Download media from Telegram
4. Extract audio for video/video_note with ffmpeg
5. Transcribe with OpenAI
6. Summarize
7. Store transcript locally in SQLite
8. Return a random public transcript URL valid for 7 days

Notes:
- You need ffmpeg on PATH for video notes / videos.
- Transcript URLs are public but unguessable.
- Only Telegram user IDs listed in ALLOWED_TELEGRAM_USER_IDS are allowed.
