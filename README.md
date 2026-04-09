# tgtldr

Minimal private Telegram bot for transcription + summaries.

## What it does

- Accepts Telegram voice notes
- Accepts audio files
- Accepts video notes and short videos
- Transcribes with OpenAI
- Summarizes the transcript
- Stores the full transcript
- Returns a short summary plus a transcript link

## Storage

- Local development defaults to SQLite via `DB_PATH`
- Upsun can use PostgreSQL automatically via `DATABASE_URL` or `POSTGRESQL_*`

## URL handling

- `PUBLIC_BASE_URL` is optional
- On Upsun, the app can derive its public base URL from `PLATFORM_ROUTES`
- Locally, it falls back to `http://localhost:${PORT:-8000}`

## Webhook secret

- `WEBHOOK_SECRET` is optional
- If omitted, the app derives a stable secret from your bot token and Upsun project entropy when available
- You can still set `WEBHOOK_SECRET` explicitly if you want a fixed value

## Recommended models

- `OPENAI_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe`
- `OPENAI_SUMMARY_MODEL=gpt-5.4-mini`

Use `gpt-4o-transcribe` instead if you want slightly better transcript quality and are fine paying more.

## Local run

```bash
uv sync
uv run uvicorn main:app --reload
```

## Upsun notes

- `.upsun/config.yaml` starts the app with Uvicorn on the Upsun socket
- `.environment` maps Upsun PostgreSQL service variables to `DATABASE_URL`
- Transcript URLs are public but unguessable
- Only Telegram user IDs listed in `ALLOWED_TELEGRAM_USER_IDS` are allowed
