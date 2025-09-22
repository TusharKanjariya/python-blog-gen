# yt2md.py
import os, re, json, requests, datetime, tempfile, shutil, math
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
    if m: return m.group(2)
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
        if text: return text, "captions:en"
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
                    preferred = t; break
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
                if text: return text, f"captions:{lang}"
            except Exception:
                pass
            # try translate → EN
            try:
                if preferred.is_translatable:
                    tr_en = preferred.translate("en").fetch()
                    text = " ".join(_clean(x["text"]) for x in tr_en if x["text"].strip())
                    if text: return text, "captions-translated:en"
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
    from faster_whisper import WhisperModel

    # temp dir for audio
    tmpdir = tempfile.mkdtemp(prefix="yt2md_")
    audio_path = os.path.join(tmpdir, "audio.m4a")

    try:
        # 1) Download best audio-only
        ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "%(title)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "192"
            }],
        }
        if ffmpeg_location:
            ydl_opts["ffmpeg_location"] = ffmpeg_location
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            # Find produced m4a file
            base = ydl.prepare_filename(info)
            base_noext = os.path.splitext(base)[0]
            audio_path = base_noext + ".m4a"

        # 2) Build model
        # Auto device selection
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # 3) Transcribe (segment wise)
        segments, info = model.transcribe(audio_path, language="en", task="transcribe", vad_filter=True)
        # Combine into plain text
        parts = []
        for seg in segments:
            txt = _clean(seg.text)
            if txt: parts.append(txt)
        text = " ".join(parts)
        return text, f"asr:faster-whisper:{model_size}:{device}"

    finally:
        # Clean temp files
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
        "- Write in a human, authoritative voice with short, easy-to-read paragraphs.\n"
        "- Use simple words, professional tone.\n"
        "- Add real-world examples where possible.\n"
        "- If programming-related, include code blocks or step-by-step tutorials.\n"
        "- Structure with clear subheadings; finish with a concise Conclusion.\n"
        "- END WITH a **Key Takeaways** section of 3–5 bullets.\n"
        "- Suggest **Top 5 Medium tags** at the very end.\n"
        "- Return Markdown (no YAML). Include a curiosity Title, short Subtitle, and a Kicker.\n"
        "- Minimum length: ~1500 characters.\n"
        "- Do not invent sources or quotes.\n"
    )
    transcript_snippet = transcript_text[:120000]
    user = f"""
VIDEO TITLE: {video_title}
CHANNEL: {channel}
VIDEO URL: {video_url}
DATE: {today}
TRANSCRIPT SOURCE: {transcript_source}

TRANSCRIPT:
{transcript_snippet}
"""
    return [{"role":"system","content":system},{"role":"user","content":user}]


def split_into_chunks(text: str, target_chars: int = 12000, overlap: int = 500):
    """
    Greedy, whitespace-aware splitter.
    - target_chars: ~size of each chunk (characters, not tokens).
    - overlap: small overlap so context flows across chunks.
    Returns list[str].
    """
    text = text.strip()
    if len(text) <= target_chars:
        return [text]

    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + target_chars, n)
        # try to break at whitespace near the end
        if end < n:
            j = text.rfind(" ", i + int(0.8*target_chars), end)
            if j != -1:
                end = j
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = max(end - overlap, i + 1)
    return chunks


def summarize_chunk_with_llm(chunk_text: str, model=MODEL):
    """
    Summarize a single chunk into a concise, structured outline paragraph.
    Keep names, numbers, and key steps.
    """
    messages = [
        {"role": "system", "content": (
            "You summarize transcripts into concise notes. "
            "Keep only the essentials: key points, steps, numbers, names, and definitions. "
            "Prefer bullet-y prose (short lines). Avoid fluff."
        )},
        {"role": "user", "content": f"Summarize this transcript chunk:\n\n{chunk_text}"}
    ]
    return openrouter_chat(messages, model=model, temperature=0.2, max_tokens=600)


def compose_article_from_summaries(video_title: str, channel: str, video_url: str,
                                   chunk_summaries: list[str], model=MODEL, max_tokens=1800):
    """
    Turn chunk-level summaries into one cohesive Medium-style article.
    Enforce a 'Key Takeaways' section at the end.
    """
    joined = "\n\n---\n\n".join(chunk_summaries)
    today = datetime.date.today().isoformat()

    system = (
        "You are a skilled writer creating engaging Medium articles.\n"
        "- Write in a human, authoritative voice with short paragraphs.\n"
        "- Use simple words, professional tone; add real examples where possible.\n"
        "- Organize with clear subheadings; keep flow logical.\n"
        "- If relevant to programming, include code blocks or step-by-step guidance.\n"
        "- Return Markdown (no YAML). Include Title, Subtitle, and a short Kicker at top.\n"
        "- END WITH a **Key Takeaways** section of 3–5 bullets—no exceptions.\n"
        "- Add a concise Conclusion before Key Takeaways.\n"
        "- Suggest **Top 5 Medium tags** at the very end.\n"
        "- Minimum length: ~1500 characters.\n"
        "- Do not invent sources or quotes."
    )

    user = (
        f"VIDEO TITLE: {video_title}\n"
        f"CHANNEL: {channel}\n"
        f"VIDEO URL: {video_url}\n"
        f"DATE: {today}\n\n"
        "Below are structured summaries of the full transcript, in order. "
        "Write ONE cohesive article covering all the important points. "
        "Preserve technical accuracy and any numbers/steps mentioned.\n\n"
        f"{joined}"
    )

    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    return openrouter_chat(messages, model=model, temperature=0.3, max_tokens=max_tokens)

# --------- Main flow ---------
def youtube_to_md(url: str, outdir="output", asr_model="small", device="auto"):
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
            transcript, source = local_whisper_transcribe(url, model_size=asr_model, device=device, ffmpeg_location=args.ffmpeg_location)
        except Exception as e:
            raise RuntimeError(f"Local transcription failed: {e}")
        
        # After obtaining `transcript` and before generating Markdown:
    # Decide: single-pass vs chunked 2-pass
    CHUNK_THRESHOLD = 20000   # characters; adjust for your model/costs
    USE_CHUNKING = len(transcript) > CHUNK_THRESHOLD

    if USE_CHUNKING:
        print(f"Transcript is long ({len(transcript):,} chars). Using 2-pass chunking...")
        chunks = split_into_chunks(transcript, target_chars=12000, overlap=500)

        # 1) Summarize each chunk
        chunk_summaries = []
        for idx, ch in enumerate(chunks, 1):
            print(f"Summarizing chunk {idx}/{len(chunks)} (~{len(ch):,} chars)")
            summary = summarize_chunk_with_llm(ch, model=MODEL)
            chunk_summaries.append(f"Chunk {idx} summary:\n{summary}")

        # 2) Compose final article from all summaries
        blog_md = compose_article_from_summaries(
            video_title=title,
            channel=author,
            video_url=url,
            chunk_summaries=chunk_summaries,
            model=MODEL,
            max_tokens=1800  # you can raise if you want longer output
        )
    else:
        # Short enough → single pass (still ends with Key Takeaways via build_messages)
        messages = build_messages(title, author, url, transcript, source)
        blog_md = openrouter_chat(messages, model=MODEL, max_tokens=1600)

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

    md_file, source = youtube_to_md(args.url, outdir=args.outdir, asr_model=args.asr_model, device=args.device)
    print(f"Transcript source: {source}")
    print(f"Markdown saved to: {md_file}")


# python -m yt2md "https://www.youtube.com/watch?v=gDVxBOGL99Q" --asr-model small