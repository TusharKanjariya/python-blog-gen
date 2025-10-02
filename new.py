# yt2md.py
import os
import re
import json
import requests
import datetime
import tempfile
import shutil
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)

# --------- Load env ---------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


# --------- Helpers ---------
def video_id_from_url(url: str) -> str:
    u = urlparse(url)
    if u.netloc in ("youtu.be", "www.youtu.be"):
        return u.path.lstrip("/")
    if "watch" in u.path:
        return parse_qs(u.query).get("v", [""])[0]
    m = re.search(r"/(embed|shorts)/([A-Za-z0-9_-]{11})", u.path)
    if m:
        return m.group(2)
    raise ValueError("Could not extract video id.")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------- Caption path (free) ---------
def fetch_transcript_text_robust(video_id: str):
    """
    Returns (text, source_label).
    Strategy:
      1) English captions (en, en-US, en-GB)
      2) Prefer manually created -> English -> anything
         2a) try original language
         2b) if translatable -> translate to EN
      else -> ("", "missing")
    """

    def _join(tr):
        return " ".join(_clean(x.get("text", "")) for x in tr if x.get("text", "").strip())

    # 1) English first
    try:
        tr = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
        text = _join(tr)
        if text:
            return text, "captions:en"
    except Exception:
        pass

    # 2) Probe everything (manual > English > anything)
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        ordered = []
        for t in transcripts:
            try:
                manual = (hasattr(t, "is_generated") and not t.is_generated)
                lang = getattr(t, "language_code", "") or ""
                score = ((0 if manual else 1), (0 if lang.startswith("en") else 1))
                ordered.append((score, t))
            except Exception:
                continue

        ordered.sort(key=lambda x: x[0])

        for _, t in ordered:
            # Try original
            try:
                tr = t.fetch()
                text = _join(tr)
                if text:
                    lang = getattr(t, "language_code", "unknown")
                    return text, f"captions:{lang}"
            except Exception:
                pass

            # Try translate → EN
            try:
                if getattr(t, "is_translatable", False):
                    tr_en = t.translate("en").fetch()
                    text = _join(tr_en)
                    if text:
                        return text, "captions-translated:en"
            except Exception:
                pass

    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        pass

    return "", "missing"


# --------- Local ASR fallback (free, offline) ---------
def local_whisper_transcribe(
    youtube_url: str,
    model_size: str = "small",
    device: str = "auto",
    ffmpeg_location: str = None
):
    """
    Download audio with yt-dlp and transcribe via faster-whisper.
    - Forces a YouTube player client that avoids SABR/web issues.
    - Falls back once with a different client if needed.

    model_size: tiny, base, small, medium, large-v3
    device: "auto" -> cuda if available else cpu; or "cpu" / "cuda"
    Returns (text, "asr:faster-whisper:<model>:<device>")
    """
    # Lazy import to avoid heavy deps when not needed
    from yt_dlp import YoutubeDL
    from faster_whisper import WhisperModel

    # temp dir for audio
    tmpdir = tempfile.mkdtemp(prefix="yt2md_")
    try:
        # Base options (Android client fights SABR + nsig problems)
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "noprogress": True,
            "outtmpl": os.path.join(tmpdir, "%(title)s.%(ext)s"),
            "socket_timeout": 30,
            "extractor_args": {"youtube": {"player_client": ["android"]}},
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "192"
            }],
        }
        if ffmpeg_location:
            ydl_opts["ffmpeg_location"] = ffmpeg_location

        # Attempt 1
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
        except Exception:
            # Attempt 2: try another client as a fallback
            ydl_opts_retry = dict(ydl_opts)
            ydl_opts_retry["extractor_args"] = {"youtube": {"player_client": ["tv_embedded"]}}
            with YoutubeDL(ydl_opts_retry) as ydl:
                info = ydl.extract_info(youtube_url, download=True)

        # The postprocessor converts to .m4a; rebuild path accordingly
        # prepare_filename reflects the pre-pp name; then we swap to .m4a
        base = YoutubeDL(ydl_opts).prepare_filename(info)
        audio_path = os.path.splitext(base)[0] + ".m4a"

        # Auto device selection
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8_float16"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        segments, _ = model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            vad_filter=True
        )
        text = " ".join(_clean(getattr(seg, "text", "")) for seg in segments if getattr(seg, "text", "").strip())
        return text, f"asr:faster-whisper:{model_size}:{device}"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------- OpenRouter ---------
def openrouter_chat(messages, model=MODEL, temperature=0.3, max_tokens=1600):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://local-script",
        "X-Title": "yt2blog-local-asr",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    if r.status_code != 200:
        # Expose body so user can see exact model/key problem
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


# --------- Prompting ---------
def build_messages(video_title, channel, video_url, transcript_text, transcript_source):
    today = datetime.date.today().isoformat()
    system = (
        "You are a skilled writer creating engaging Medium articles.\n"
        "- Blog title must be curiosity-driven, scroll-stopping, and irresistible.\n"
        "- Write in a human, authoritative voice with short, easy-to-read paragraphs.\n"
        "- Always use very simple words, but keep the tone professional.\n"
        "- Add real-world examples to illustrate points.\n"
        "- If topic relates to programming, include clear code blocks or step-by-step tutorials.\n"
        "- Structure: Start with a TL;DR summary → main content with subheadings → examples/tutorials → concise conclusion.\n"
        "- At the end, suggest **Top 5 Medium tags** that will improve reach.\n"
        "- Return content in Markdown format suitable for Medium (no YAML front matter).\n"
        "- Give Curios Title, Subtitle, And Kicker.\n"
        "- Add Conclusion or any relevant section at the very end of the blog.\n"
        "- Minimum length: 1500 characters.\n"
        "- Don't generate fake quotes, sources, or references.\n"
        "- don't generate any section like TL;DR like all robotic style blogs only human written tone like section is applicable .\n"
        "- I am writing on Medium so please follow the Medium writing style.\n"
    )
    user = f"""
VIDEO TITLE: {video_title}
CHANNEL: {channel}
VIDEO URL: {video_url}
DATE: {today}
TRANSCRIPT SOURCE: {transcript_source}

TRANSCRIPT:
{transcript_text[:200000]}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------- Main flow ---------
def youtube_to_md(url: str, outdir="output", asr_model="small", device="auto", ffmpeg_location=None):
    vid = video_id_from_url(url)

    # Lightweight metadata via oEmbed
    try:
        meta_resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=15,
            headers={"User-Agent": "yt2md/1.0"}
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        title = meta.get("title", f"YouTube {vid}")
        author = meta.get("author_name", "Unknown Channel")
    except Exception:
        title, author = f"YouTube {vid}", "Unknown Channel"

    # 1) Try captions (free)
    transcript, source = fetch_transcript_text_robust(vid)

    # 2) Local ASR fallback if missing
    if not transcript:
        print("No captions found. Running local transcription (faster-whisper)…")
        try:
            transcript, source = local_whisper_transcribe(
                url,
                model_size=asr_model,
                device=device,
                ffmpeg_location=ffmpeg_location
            )
        except Exception as e:
            raise RuntimeError(f"Local transcription failed: {e}")

    # 3) Generate Markdown grounded in transcript
    messages = build_messages(title, author, url, transcript, source)
    blog_md = openrouter_chat(messages, model=MODEL)

    # 4) Save
    os.makedirs(outdir, exist_ok=True)
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-') or f"video-{vid}"
    path = os.path.join(outdir, f"{slug}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(blog_md)
    return path, source


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Markdown blog from YouTube with captions or local ASR.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--asr-model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"], help="faster-whisper model size")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument("--ffmpeg-location", default=None, help="Path to ffmpeg/ffprobe bin dir")
    args = parser.parse_args()

    md_file, source = youtube_to_md(
        args.url,
        outdir=args.outdir,
        asr_model=args.asr_model,
        device=args.device,
        ffmpeg_location=args.ffmpeg_location
    )
    print(f"Transcript source: {source}")
    print(f"Markdown saved to: {md_file}")

# Example:
# python -m yt2md "https://www.youtube.com/watch?v=TlIOk8VuEBU" --asr-model small --ffmpeg-location "C:\ffmpeg\bin"
