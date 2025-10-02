# yt2md.py
import os, re, json, requests, datetime, tempfile, shutil
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
      1) English captions
      2) Any captions
      3) If non-EN and translatable -> translate to EN
      else -> ("", "missing")
    """
    # 1) English
    try:
        tr = YouTubeTranscriptApi.get_transcript(video_id, languages=["en","en-US","en-GB"])
        text = " ".join(_clean(x["text"]) for x in tr if x["text"].strip())
        if text:
            return text, "captions:en"
    except Exception:
        pass

    # 2) Any + 3) translate
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer manually created
        preferred = None
        for t in transcripts:
            try:
                if hasattr(t, "is_generated") and not t.is_generated:
                    preferred = t
                    break
            except Exception:
                continue
        if preferred is None:
            preferred = next(iter(transcripts), None)

        if preferred:
            # try original
            try:
                tr = preferred.fetch()
                text = " ".join(_clean(x["text"]) for x in tr if x["text"].strip())
                lang = getattr(preferred, "language_code", "unknown")
                if text:
                    return text, f"captions:{lang}"
            except Exception:
                pass
            # try translate → EN
            try:
                if preferred.is_translatable:
                    tr_en = preferred.translate("en").fetch()
                    text = " ".join(_clean(x["text"]) for x in tr_en if x["text"].strip())
                    if text:
                        return text, "captions-translated:en"
            except Exception:
                pass
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        pass

    return "", "missing"

# --------- Local ASR fallback (free, offline) ---------
def local_whisper_transcribe(youtube_url: str, model_size: str = "small", device: str = "auto", ffmpeg_location: str = None):
    """
    Download audio with yt-dlp and transcribe via faster-whisper.
    model_size options: tiny, base, small, medium, large-v3
    device: "auto" -> cuda if available else cpu; "cpu"; "cuda"
    Returns (text, "faster-whisper:<model>")
    """
    # Lazy import to avoid heavy deps when not needed
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import PostProcessingError, DownloadError
    from faster_whisper import WhisperModel

    # temp dir for audio
    tmpdir = tempfile.mkdtemp(prefix="yt2md_")

    # Build a base ydl options dict (we’ll vary player_client)
    def base_opts(player_client: str):
        opts = {
            "format": "bestaudio[acodec!=none]/best[acodec!=none]",
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "quiet": True,
            "noprogress": True,
            "noplaylist": True,
            "retries": 3,
            "fragment_retries": 3,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "192",
            }],
            "extractor_args": {"youtube": {"player_client": [player_client]}},
            # "prefer_ipv4": True,  # uncomment if your network has IPv6 issues
        }
        if ffmpeg_location:
            opts["ffmpeg_location"] = ffmpeg_location
        return opts

    # Try multiple player clients to dodge SABR / PO-token issues
    tried_errors = []
    audio_path = None
    for client in ("ios", "android", "tvhtml5"):
        try:
            with YoutubeDL(base_opts(client)) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                audio_path = os.path.join(tmpdir, f"{info['id']}.m4a")
                break  # success
        except PostProcessingError as e:
            # Usually: ffmpeg/ffprobe not found or wrong --ffmpeg-location
            raise RuntimeError(
                "Post-processing failed. Ensure ffmpeg and ffprobe exist in the folder you passed via --ffmpeg-location "
                "and that the path points to the directory that contains ffmpeg.exe and ffprobe.exe on Windows. "
                f"Details: {e}"
            )
        except DownloadError as e:
            tried_errors.append((client, str(e)))
        except Exception as e:
            tried_errors.append((client, repr(e)))

    if audio_path is None:
        raise RuntimeError(
            "yt-dlp could not fetch an audio stream with any player client "
            f"(tried ios, android, tvhtml5). Errors: {tried_errors}"
        )

    try:
        # Auto device selection
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # Transcribe (segment wise)
        segments, _info = model.transcribe(audio_path, language="en", task="transcribe", vad_filter=True)
        parts = []
        for seg in segments:
            txt = _clean(seg.text)
            if txt:
                parts.append(txt)
        text = " ".join(parts)
        return text, f"asr:faster-whisper:{model_size}:{device}"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# --------- OpenRouter ---------
def openrouter_chat(messages, model=MODEL, temperature=0.3, max_tokens=1600):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://local-script",
        "X-Title": "yt2blog-local-asr",
        "Content-Type": "application/json",
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
        data=json.dumps(payload),
        timeout=300,
    )
    if r.status_code != 200:
        # show the body so you know exactly what's wrong (e.g., model id)
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
    return [{"role":"system","content":system},{"role":"user","content":user}]

# --------- Main flow ---------
def youtube_to_md(url: str, outdir="output", asr_model="small", device="auto", ffmpeg_location=None):
    vid = video_id_from_url(url)
    # Lightweight metadata via oEmbed
    try:
        meta = requests.get("https://www.youtube.com/oembed", params={"url":url,"format":"json"}, timeout=15).json()
        title, author = meta.get("title", f"YouTube {vid}"), meta.get("author_name", "Unknown Channel")
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
    slug = re.sub(r'[^a-z0-9]+','-', title.lower()).strip('-') or f"video-{vid}"
    path = os.path.join(outdir, f"{slug}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(blog_md)
    return path, source

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Markdown blog from YouTube with captions or local ASR.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--asr-model", default="small", choices=["tiny","base","small","medium","large-v3"], help="faster-whisper model size")
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Computation device")
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
