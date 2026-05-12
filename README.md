# personal-security

Watches a Reolink FTP folder for new clips, asks Claude to describe what's in
each one, and sends an alert to Telegram.

## How it works

1. Polls `WATCH_DIR` for new `.jpg` thumbnails the cameras drop via FTP.
2. Sends the thumbnail to Claude for a fast classification pass
   (`interesting?`, `alert_level`, subjects, one-line summary).
3. If the thumbnail looks interesting, samples ~6 frames from the matching
   `.mp4` with `ffmpeg` and asks Claude for a richer description across the
   clip.
4. Posts the thumbnail + description to a Telegram chat.
5. Tracks every processed file in a small SQLite DB so restarts don't re-spam.

The script assumes the Reolink convention of a thumbnail `.jpg` and a video
`.mp4` sharing the same basename in the same directory. Subdirectory layout
under `WATCH_DIR` is free-form — the first path component below `WATCH_DIR` is
used as the camera label in the alert.

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `$PATH` (`apt install ffmpeg`)
- Anthropic API key
- Telegram bot token + chat ID (see `.env.example` for how to get them)

## Setup

```sh
pip install -r requirements.txt
cp .env.example .env
$EDITOR .env          # fill in the secrets and WATCH_DIR
python reolink_analyzer.py
```

Useful flags:

- `--once` — run a single scan and exit (handy for cron or smoke-testing).
- `--no-seed` — process the existing backlog instead of marking it as seen on
  first startup.

## Notes

- Defaults to Claude Opus 4.7 for both passes. For lower cost and latency, set
  `THUMB_MODEL=claude-haiku-4-5` and/or `VIDEO_MODEL=claude-sonnet-4-6` in
  `.env`.
- Telegram captions are capped at 1024 characters; longer descriptions get a
  follow-up text message.
- The script writes its state DB next to itself by default
  (`STATE_DB=./reolink_analyzer.db`). Override if you want it elsewhere.
- If the matching video hasn't finished uploading by the time the thumbnail
  is processed, the script waits up to `VIDEO_WAIT_SECONDS` for it; otherwise
  it sends the thumbnail-only summary.
