# yt2md.py
import os, re, json, requests, datetime
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# ------------------- YT helpers -------------------

def video_id_from_url(url: str) -> str:
    u = urlparse(url)
    if u.netloc in ("youtu.be", "www.youtu.be"):
        return u.path.lstrip("/")
    if "watch" in u.path:
        return parse_qs(u.query).get("v", [""])[0]
    m = re.search(r"/(embed|shorts)/([A-Za-z0-9_-]{11})", u.path)
    if m: return m.group(2)
    raise ValueError("Could not extract video id.")

def fetch_transcript_text_robust(video_id: str):
    """
    Returns (text, source_label). Tries:
    1) English (en, en-US, en-GB)
    2) Any available transcript
    3) If non-English but translatable -> translate to en
    """
    # 1) Try English directly
    try:
        tr = YouTubeTranscriptApi.get_transcript(video_id, languages=["en","en-US","en-GB"])
        text = " ".join(_clean_chunk(t["text"]) for t in tr if t["text"].strip())
        if text.strip():
            return text, "captions:en"
    except Exception:
        pass

    # 2) Try listing anything, prefer manually created, else auto-generated
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer a manually created transcript if available
        preferred = None
        for t in transcripts:
            try:
                if hasattr(t, "is_generated") and not t.is_generated:
                    preferred = t
                    break
            except Exception:
                continue
        if preferred is None:
            # pick the first available
            preferred = next(iter(transcripts), None)

        if preferred:
            # fetch as-is
            try:
                tr = preferred.fetch()
                text = " ".join(_clean_chunk(x["text"]) for x in tr if x["text"].strip())
                if text.strip():
                    return text, f"captions:{getattr(preferred, 'language_code', 'unknown')}"
            except Exception:
                pass

            # 3) Try translating to English if supported
            try:
                if preferred.is_translatable:
                    tr_en = preferred.translate("en").fetch()
                    text = " ".join(_clean_chunk(x["text"]) for x in tr_en if x["text"].strip())
                    if text.strip():
                        return text, f"captions-translated:en"
            except Exception:
                pass
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        pass

    return "", "missing"

def _clean_chunk(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# ------------------- OpenRouter -------------------

def openrouter_chat(messages, model=MODEL, temperature=0.4, max_tokens=1600):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://local-script",
        "X-Title": "yt2blog-demo",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ------------------- Prompting -------------------

def build_messages(video_title, channel, video_url, transcript_text, transcript_source, allow_no_transcript=False):
    today = datetime.date.today().isoformat()

    if transcript_text.strip():
        system = (
            "You are a senior editor. Write a clear, accurate, SEO-friendly blog post grounded ONLY in the provided transcript. "
            "Include: TL;DR, 5–8 Key Takeaways, 3–5 Pull Quotes, Conclusion. "
            "Use H2/H3 headings. Grade 8–10 reading level. "
            "Return Markdown with YAML front matter (title, slug, description, date, tags, video_url). "
            "Meta title <60 chars, meta description <155. Avoid inventing facts beyond the transcript."
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
    else:
        # No transcript — explicitly instruct the model to still produce a helpful draft
        if not allow_no_transcript:
            # default: still create a helpful scaffolded draft with disclaimers
            allow_no_transcript = True

        system = (
            "You are a senior editor. No transcript is available. "
            "Create a SAFE, useful blog draft using only the video title and channel. "
            "Do NOT assert specific facts from the video. "
            "Structure: TL;DR, outline with H2/H3, likely topics (clearly labeled as hypotheses), "
            "Key Takeaways (general, non-specific), Pull Quotes as placeholders, and a Conclusion. "
            "Add a disclaimer box at the top noting that captions were unavailable and that details are placeholders. "
            "Return Markdown with YAML front matter (title, slug, description, date, tags, video_url). "
            "Meta title <60 chars, meta description <155."
        )
        user = f"""
NO TRANSCRIPT AVAILABLE.

VIDEO TITLE: {video_title}
CHANNEL: {channel}
VIDEO URL: {video_url}
DATE: {today}

Please generate a general-purpose draft with placeholders and clearly labeled assumptions.
"""

    return [{"role":"system","content":system},{"role":"user","content":user}]

# ------------------- Main -------------------

def youtube_to_md(url: str, outdir="output", allow_no_transcript=False):
    vid = video_id_from_url(url)
    # Lightweight metadata via oEmbed
    try:
        meta = requests.get("https://www.youtube.com/oembed", params={"url":url,"format":"json"}, timeout=15).json()
        title, author = meta.get("title", f"YouTube {vid}"), meta.get("author_name", "Unknown Channel")
    except Exception:
        title, author = f"YouTube {vid}", "Unknown Channel"

    transcript, src = fetch_transcript_text_robust(vid)

    messages = build_messages(title, author, url, transcript, src, allow_no_transcript=allow_no_transcript)
    blog_md = openrouter_chat(messages, model=MODEL)

    os.makedirs(outdir, exist_ok=True)
    slug = re.sub(r'[^a-z0-9]+','-', title.lower()).strip('-') or f"video-{vid}"
    path = os.path.join(outdir, f"{slug}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(blog_md)
    return path, src

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Markdown blog from YouTube video.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--allow-no-transcript", action="store_true",
                        help="If captions are missing, still create a draft with placeholders and a disclaimer.")
    args = parser.parse_args()

    md_file, source = youtube_to_md(args.url, outdir=args.outdir, allow_no_transcript=args.allow_no_transcript)
    print(f"Transcript source: {source}")
    print(f"Markdown saved to: {md_file}")
