#!/usr/bin/env python3
"""Watch a Reolink FTP directory, describe new clips with Claude, alert via Telegram."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"missing required env var: {name}")
    return val or ""


ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", required=True)
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN", required=True)
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID", required=True)
WATCH_DIR = Path(_env("WATCH_DIR", required=True)).resolve()
STATE_DB = Path(_env("STATE_DB", "./reolink_analyzer.db"))
POLL_SECONDS = int(_env("POLL_SECONDS", "10"))
MIN_FILE_AGE_SECONDS = int(_env("MIN_FILE_AGE_SECONDS", "5"))
SKIP_OLDER_THAN_HOURS = float(_env("SKIP_OLDER_THAN_HOURS", "1"))
VIDEO_WAIT_SECONDS = int(_env("VIDEO_WAIT_SECONDS", "60"))
VIDEO_FRAMES = int(_env("VIDEO_FRAMES", "6"))
THUMB_MODEL = _env("THUMB_MODEL", "claude-opus-4-7")
VIDEO_MODEL = _env("VIDEO_MODEL", "claude-opus-4-7")
ALERT_ON = _env("ALERT_ON", "interesting").lower()

# Pre-filter: cheap first pass to skip clearly empty frames before calling Opus.
# PREFILTER_BACKEND: "none" (disabled), "ollama" (local SLM), "claude" (cloud model)
# PREFILTER_MODEL:   model name — e.g. "moondream" for ollama, "claude-haiku-4-5" for claude
# PREFILTER_OLLAMA_URL: base URL of the Ollama server
PREFILTER_BACKEND = _env("PREFILTER_BACKEND", "none").lower()
PREFILTER_MODEL = _env("PREFILTER_MODEL", "moondream:v2")
PREFILTER_OLLAMA_URL = _env("PREFILTER_OLLAMA_URL", "http://localhost:11434")
PREFILTER_TIMEOUT = int(_env("PREFILTER_TIMEOUT", "120"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("reolink")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# (input $/MTok, output $/MTok, cache_write $/MTok, cache_read $/MTok)
_MODEL_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-haiku-4-5":           (0.80,  4.00,  1.00,  0.08),
    "claude-haiku-4-5-20251001":  (0.80,  4.00,  1.00,  0.08),
    "claude-sonnet-4-6":          (3.00, 15.00,  3.75,  0.30),
    "claude-opus-4-7":           (15.00, 75.00, 18.75,  1.50),
}


def _calc_cost(model: str, usage: object) -> float:
    prices = _MODEL_PRICING.get(model, (15.00, 75.00, 18.75, 1.50))
    ip, op, cwp, crp = prices
    return (
        getattr(usage, "input_tokens", 0)                  / 1_000_000 * ip
        + getattr(usage, "output_tokens", 0)               / 1_000_000 * op
        + getattr(usage, "cache_creation_input_tokens", 0) / 1_000_000 * cwp
        + getattr(usage, "cache_read_input_tokens", 0)     / 1_000_000 * crp
    )


THUMB_SYSTEM = """You analyze still thumbnails from a home security camera system.

The household has three known family members:
- Adult male: wears glasses and has a beard
- Adult female: wears glasses and long black hair
- Young boy: approximately 5 years old

The property has three cameras with different security contexts:
- Garage: front of the property / driveway. Watch for dogs defecating on the lawn, deliveries \
(couriers, packages), and any unfamiliar person near the entrance.
- Backyard: enclosed private area within the property. Any unfamiliar person here is a \
serious concern regardless of time of day.
- Right_walkway: side access within the property perimeter. Any unfamiliar person here is a \
serious concern regardless of time of day.

Return a JSON object with these fields:

- `interesting` = true if the scene shows any of:
  - An unfamiliar person, or someone whose identity cannot be confirmed as a family member
  - A delivery (courier, postal worker, package left at door)
  - A dog on the property (especially defecating on the front lawn)
  - Suspicious behavior: loitering, peering through windows, testing doors/gates/fences
  - Any person in Backyard or Right_walkway not confirmed as a family member
  False for: confirmed family members doing routine activity, empty scenes, wind-blown vegetation, \
lighting changes, distant street traffic with no interaction with the property.

- `alert_level`:
  - "high": unfamiliar person in Backyard or Right_walkway at any time; anyone testing entry \
points (doors, gates, windows); loitering near the property at night; unrecognized person at \
Garage after dark
  - "medium": delivery or package drop-off; unfamiliar person at Garage during the day; dog on \
the lawn; unrecognized vehicle parked near the property
  - "low": confirmed family members going about normal activity; empty scene; passing street traffic

- `summary` = one sentence (≤ 200 chars), present tense, no preamble. If a person is visible, \
note whether they appear to be a family member or an unfamiliar individual.

- `subjects` = list of nouns present: person, vehicle, animal, dog, package, delivery. \
Empty list if none."""


THUMB_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "alert_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "interesting": {"type": "boolean"},
    },
    "required": ["summary", "subjects", "alert_level", "interesting"],
    "additionalProperties": False,
}


VIDEO_SYSTEM = """You analyze frames sampled evenly across a short home security camera clip.

The household has three known family members:
- Adult male: wears glasses and has a beard
- Adult female: wears glasses and long black hair
- Young boy: approximately 5 years old

Describe what happens across the clip in 2-4 sentences:
- If a person is visible: note whether they appear to be a family member or an unfamiliar \
individual. For unfamiliar individuals, describe their appearance, movement pattern (purposeful, \
hesitant, loitering), and any interaction with the property (doors, windows, gates, fences, \
packages).
- If an animal is visible: describe species, behavior, and exact location on the property.
- If a delivery is visible: note the courier/service if identifiable, how the package is handled, \
and where it is left.
Be concise and factual. No preamble, no JSON — plain prose only."""


@dataclass
class ThumbResult:
    summary: str
    subjects: list[str]
    alert_level: str
    interesting: bool


# ---------- state ----------

def init_db() -> sqlite3.Connection:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS processed ("
        "  path TEXT PRIMARY KEY,"
        "  processed_at INTEGER NOT NULL"
        ")"
    )
    conn.commit()
    return conn


def is_processed(conn: sqlite3.Connection, path: Path) -> bool:
    cur = conn.execute("SELECT 1 FROM processed WHERE path = ?", (str(path),))
    return cur.fetchone() is not None


def mark_processed(conn: sqlite3.Connection, path: Path) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO processed (path, processed_at) VALUES (?, ?)",
        (str(path), int(time.time())),
    )
    conn.commit()


def seed_backlog(conn: sqlite3.Connection) -> int:
    """On startup, mark anything older than SKIP_OLDER_THAN_HOURS as already processed."""
    cutoff = time.time() - SKIP_OLDER_THAN_HOURS * 3600
    count = 0
    for path in WATCH_DIR.rglob("*.jpg"):
        try:
            if path.stat().st_mtime < cutoff and not is_processed(conn, path):
                mark_processed(conn, path)
                count += 1
        except OSError:
            continue
    return count


# ---------- file discovery ----------

def find_new_thumbnails(conn: sqlite3.Connection) -> list[Path]:
    now = time.time()
    out: list[Path] = []
    for path in WATCH_DIR.rglob("*.jpg"):
        if is_processed(conn, path):
            continue
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age < MIN_FILE_AGE_SECONDS:
            continue
        out.append(path)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def is_stable(path: Path, interval: float = 2.0) -> bool:
    try:
        size1 = path.stat().st_size
    except OSError:
        return False
    if size1 == 0:
        return False
    time.sleep(interval)
    try:
        size2 = path.stat().st_size
    except OSError:
        return False
    return size1 == size2


def matching_video(thumb: Path) -> Path | None:
    candidate = thumb.with_suffix(".mp4")
    return candidate if candidate.exists() else None


def wait_for_video(thumb: Path, timeout: int) -> Path | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        video = matching_video(thumb)
        if video and is_stable(video):
            return video
        time.sleep(2)
    return matching_video(thumb)


# ---------- Claude ----------

def _image_block(path: Path) -> dict:
    suffix = path.suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _camera_name(thumb: Path) -> str:
    """Extract camera name from Reolink filename (e.g. Garage_00_20260513... → Garage)."""
    m = re.match(r"^(.+?)_\d{2}_\d{14}", thumb.stem)
    if m:
        return m.group(1)
    try:
        return thumb.relative_to(WATCH_DIR).parts[0]
    except (ValueError, IndexError):
        return thumb.parent.name


def analyze_thumbnail(thumb: Path, camera: str) -> tuple[ThumbResult, float]:
    response = client.messages.create(
        model=THUMB_MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": THUMB_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": THUMB_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block(thumb),
                    {"type": "text", "text": f"Camera: {camera}. Analyze this thumbnail."},
                ],
            }
        ],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    data = json.loads(text)
    result = ThumbResult(
        summary=data["summary"].strip(),
        subjects=list(data["subjects"]),
        alert_level=data["alert_level"],
        interesting=bool(data["interesting"]),
    )
    return result, _calc_cost(THUMB_MODEL, response.usage)


def extract_frames(video: Path, out_dir: Path, num_frames: int) -> list[Path]:
    duration_raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        text=True,
    ).strip()
    duration = float(duration_raw)
    if duration <= 0:
        raise ValueError(f"non-positive duration for {video}: {duration}")

    frames: list[Path] = []
    for i in range(num_frames):
        ts = duration * (i + 0.5) / num_frames
        out_path = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{ts:.2f}",
                "-i", str(video),
                "-frames:v", "1",
                "-vf", "scale=1024:-1",
                "-q:v", "3",
                str(out_path),
            ],
            check=True,
        )
        frames.append(out_path)
    return frames


def analyze_video(video: Path) -> tuple[str, float]:
    with tempfile.TemporaryDirectory(prefix="reolink_frames_") as tmp:
        try:
            frames = extract_frames(video, Path(tmp), VIDEO_FRAMES)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
            log.warning("frame extraction failed for %s: %s", video, e)
            return "", 0.0

        content: list[dict] = [_image_block(f) for f in frames]
        content.append({"type": "text", "text": "Describe what happens across these frames."})
        response = client.messages.create(
            model=VIDEO_MODEL,
            max_tokens=1024,
            system=VIDEO_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return text, _calc_cost(VIDEO_MODEL, response.usage)


# ---------- Telegram ----------

def send_telegram(thumb: Path, caption: str) -> None:
    photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    short = caption if len(caption) <= 1024 else caption[:1020] + "..."
    with thumb.open("rb") as f:
        resp = requests.post(
            photo_url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": short},
            files={"photo": f},
            timeout=30,
        )
    resp.raise_for_status()
    if len(caption) > 1024:
        msg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        rest = caption[1020:]
        resp = requests.post(
            msg_url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": rest},
            timeout=30,
        )
        resp.raise_for_status()


def format_caption(thumb: Path, thumb_result: ThumbResult, video_desc: str, cost_usd: float) -> str:
    camera = _camera_name(thumb)

    icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(thumb_result.alert_level, "📷")
    lines = [
        f"{icon} {camera} — {thumb_result.alert_level}",
        thumb_result.summary,
    ]
    if thumb_result.subjects:
        lines.append("Subjects: " + ", ".join(thumb_result.subjects))
    if video_desc:
        lines.append("")
        lines.append(video_desc)
    lines.append("")
    lines.append(f"File: {thumb.name}")
    lines.append(f"Cost: ${cost_usd:.4f}")
    return "\n".join(lines)


# ---------- pre-filter ----------

_PREFILTER_PROMPT = "Describe what you see in this security camera image in one sentence."

_PREFILTER_KEYWORDS = {
    "person", "people", "man", "woman", "child", "boy", "girl",
    "individual", "figure", "human", "someone", "somebody",
    "dog",
}


def _resize_for_prefilter(thumb: Path, max_width: int = 1024) -> bytes:
    """Downscale image to max_width for the prefilter model; returns original bytes on failure."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(thumb),
             "-vf", f"scale='min({max_width},iw)':-2", "-q:v", "3", str(tmp_path)],
            check=True,
        )
        return tmp_path.read_bytes()
    except Exception as e:
        log.warning("prefilter resize failed for %s: %s — using original", thumb.name, e)
        return thumb.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def _prefilter_ollama(thumb: Path) -> bool:
    data = base64.standard_b64encode(_resize_for_prefilter(thumb)).decode("ascii")
    try:
        resp = requests.post(
            f"{PREFILTER_OLLAMA_URL}/api/chat",
            json={"model": PREFILTER_MODEL, "stream": False,
                  "messages": [{"role": "user", "content": _PREFILTER_PROMPT,
                                 "images": [data]}]},
            timeout=PREFILTER_TIMEOUT,
        )
        resp.raise_for_status()
        answer = resp.json().get("message", {}).get("content", "").strip()
        if not answer:
            log.warning("prefilter ollama: empty response for %s — passing through", thumb.name)
            return True
        words = set(re.findall(r"[a-z]+", answer.lower()))
        result = bool(words & _PREFILTER_KEYWORDS)
        log.info("prefilter ollama: %s → %s | SLM said: %r", thumb.name, "pass" if result else "skip", answer)
        return result
    except Exception as e:
        log.warning("prefilter_ollama failed for %s: %s — passing through", thumb.name, e)
        return True


def _prefilter_claude(thumb: Path) -> bool:
    try:
        response = client.messages.create(
            model=PREFILTER_MODEL,
            max_tokens=60,
            messages=[{"role": "user", "content": [
                _image_block(thumb),
                {"type": "text", "text": _PREFILTER_PROMPT},
            ]}],
        )
        answer = next((b.text for b in response.content if b.type == "text"), "").strip()
        words = set(re.findall(r"[a-z]+", answer.lower()))
        result = bool(words & _PREFILTER_KEYWORDS)
        log.info("prefilter claude: %s → %s | SLM said: %r", thumb.name, "pass" if result else "skip", answer)
        return result
    except Exception as e:
        log.warning("prefilter_claude failed for %s: %s — passing through", thumb.name, e)
        return True


def prefilter_thumbnail(thumb: Path) -> bool:
    """Return False if the image is clearly empty; errors always pass through to avoid missed alerts."""
    if PREFILTER_BACKEND == "none":
        return True
    if PREFILTER_BACKEND == "ollama":
        return _prefilter_ollama(thumb)
    if PREFILTER_BACKEND == "claude":
        return _prefilter_claude(thumb)
    log.warning("unknown PREFILTER_BACKEND %r — passing through", PREFILTER_BACKEND)
    return True


# ---------- main loop ----------

def process_thumbnail(conn: sqlite3.Connection, thumb: Path) -> None:
    log.info("─" * 60)
    log.info("processing %s", thumb)
    if not is_stable(thumb):
        log.info("not stable, will retry next cycle: %s", thumb)
        return

    camera = _camera_name(thumb)

    if not prefilter_thumbnail(thumb):
        mark_processed(conn, thumb)
        return

    try:
        thumb_result, thumb_cost = analyze_thumbnail(thumb, camera)
    except anthropic.OverloadedError:
        log.warning("API overloaded for %s, will retry after backoff", thumb)
        raise
    except anthropic.APIError as e:
        log.exception("thumbnail API error for %s: %s", thumb, e)
        return
    except (json.JSONDecodeError, KeyError) as e:
        log.exception("thumbnail parse error for %s: %s", thumb, e)
        mark_processed(conn, thumb)  # don't retry on a malformed response forever
        return
    log.info(
        "thumb: interesting=%s level=%s subjects=%s cost=$%.4f",
        thumb_result.interesting,
        thumb_result.alert_level,
        thumb_result.subjects,
        thumb_cost,
    )

    video_desc = ""
    video_cost = 0.0
    if thumb_result.interesting:
        video = wait_for_video(thumb, VIDEO_WAIT_SECONDS)
        if video:
            log.info("analyzing video frames from %s", video)
            try:
                video_desc, video_cost = analyze_video(video)
            except anthropic.OverloadedError:
                log.warning("API overloaded for video %s, sending thumbnail-only alert", video)
            except anthropic.APIError as e:
                log.exception("video API error for %s: %s", video, e)
        else:
            log.info("no matching video found within timeout for %s", thumb)

    should_notify = ALERT_ON == "always" or (
        ALERT_ON == "interesting" and thumb_result.interesting
    )
    if should_notify:
        caption = format_caption(thumb, thumb_result, video_desc, thumb_cost + video_cost)
        try:
            send_telegram(thumb, caption)
            log.info("alert sent for %s", thumb)
        except requests.RequestException as e:
            log.exception("telegram send failed for %s: %s", thumb, e)
            return  # leave unmarked so we retry next cycle

    mark_processed(conn, thumb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan and exit (useful for cron or testing).",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Don't pre-mark existing old files as processed on startup.",
    )
    args = parser.parse_args(argv)

    if not WATCH_DIR.is_dir():
        log.error("WATCH_DIR does not exist or is not a directory: %s", WATCH_DIR)
        return 1
    if ALERT_ON not in ("always", "interesting"):
        log.error("ALERT_ON must be 'always' or 'interesting', got %r", ALERT_ON)
        return 1

    conn = init_db()
    if not args.no_seed:
        seeded = seed_backlog(conn)
        if seeded:
            log.info("seeded %d backlog file(s) as already processed", seeded)

    prefilter_info = (
        f"ollama/{PREFILTER_MODEL} @ {PREFILTER_OLLAMA_URL}"
        if PREFILTER_BACKEND == "ollama"
        else f"claude/{PREFILTER_MODEL}"
        if PREFILTER_BACKEND == "claude"
        else "none"
    )
    log.info(
        "watching %s every %ds (prefilter=%s, thumb=%s, video=%s, alert_on=%s)",
        WATCH_DIR, POLL_SECONDS, prefilter_info, THUMB_MODEL, VIDEO_MODEL, ALERT_ON,
    )

    while True:
        try:
            for thumb in find_new_thumbnails(conn):
                process_thumbnail(conn, thumb)
        except KeyboardInterrupt:
            log.info("stopping")
            return 0
        except anthropic.OverloadedError:
            log.warning("API overloaded; pausing 60s before retrying")
            time.sleep(60)
        except Exception:  # noqa: BLE001
            log.exception("scan loop error")
        if args.once:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
