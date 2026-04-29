import os
import pathlib
import re
import shutil
import json

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


def _has_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None

# Cookie handling: inside Docker use a mounted cookies file,
# on the host use Chrome cookies directly.
_COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "/app/cookies.txt")

def _yt_dlp_opts(**extra):
    """Base yt-dlp options. Cookies are optional — yt-dlp works without them
    by using alternative YouTube clients (Android VR) that bypass n-challenge."""
    opts = {"quiet": True, "no_warnings": True}
    _cookies_path = pathlib.Path(_COOKIES_FILE)
    if _cookies_path.is_file() and _cookies_path.stat().st_size > 0:
        opts["cookiefile"] = _COOKIES_FILE
    opts.update(extra)
    return opts


def create_folder(folder_name):
    """creates folder (and parents) if it does not exist -- relative path"""
    print(f"creating path: {folder_name}")
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    return True

def delete_folder(folder_name, ignore_error=True):
    """deletes <folder_name> and all its content"""
    print(f"removing path: {folder_name}")
    shutil.rmtree(folder_name, ignore_errors=ignore_error)
    return True

def _extract_video_id(url):
    """Extract the 11-char video ID from a YouTube URL."""
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    if not m:
        raise ValueError(f"Cannot extract video ID from URL: {url}")
    return m.group(1)

def get_video_info(url):
    """returns video_id, video_title"""
    with yt_dlp.YoutubeDL(_yt_dlp_opts(skip_download=True)) as ydl:
        info = ydl.extract_info(url, download=False, process=False)
    return info["id"], info["title"]

def download_video(url, destination_folder, filename=None):
    """downloads YouTube Video (mp4) from URL, skipping if file already exists.
    If *filename* is given it is used as the stem; otherwise the YouTube title
    is used with colons and pipes stripped."""
    vid_id, title = get_video_info(url)
    safe_title = filename or re.sub(r'[:|]', '', title).strip()
    save_path = pathlib.Path(destination_folder) / (safe_title + ".mp4")
    if save_path.exists():
        print(f"Skipping (already exists): {title}")
        return str(save_path)
    print(f"Downloading: {title}...", end=" ", flush=True)
    if _has_ffmpeg():
        # Best quality: separate video+audio streams merged by ffmpeg
        fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"
        extra = {"merge_output_format": "mp4"}
    else:
        # ffmpeg not available: use a pre-muxed single-file mp4 stream
        fmt = "best[ext=mp4]/best"
        extra = {}

    ydl_opts = _yt_dlp_opts(
        format=fmt,
        outtmpl=str(pathlib.Path(destination_folder) / (safe_title + ".%(ext)s")),
        **extra,
    )

    # Dev affordance: when FW_DOWNLOAD_DURATION_SECONDS is a positive integer,
    # truncate the download to the first N seconds. Lets the slow CPU TTS
    # stage finish in minutes instead of hours during pipeline iteration.
    # Leave unset (or 0) for normal full-length downloads.
    duration_s = int(os.getenv("FW_DOWNLOAD_DURATION_SECONDS", "0") or 0)
    if duration_s > 0:
        ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(0, duration_s)]
        )
        ydl_opts["force_keyframes_at_cuts"] = True
        print(f"[FW_DOWNLOAD_DURATION_SECONDS={duration_s}s — truncating clip]", end=" ", flush=True)
        _purge_stale_artifacts(pathlib.Path(destination_folder).parent, safe_title)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Success!")
    return str(save_path)


def _purge_stale_artifacts(api_dir: pathlib.Path, safe_title: str) -> None:
    """Delete cached downstream artifacts for *safe_title* under api_dir.

    Called when FW_DOWNLOAD_DURATION_SECONDS forces a fresh truncated download —
    transcribe/translate/TTS/stitch all cache by filename, so stale full-length
    artifacts would otherwise short-circuit the re-run.
    """
    leaf_globs = [
        api_dir / "youtube_captions" / f"{safe_title}.*",
        api_dir / "transcriptions" / "whisper" / f"{safe_title}.*",
        api_dir / "translations" / "argos" / f"{safe_title}.*",
        api_dir / "dubbed_videos" / f"{safe_title}.*",
        api_dir / "dubbed_captions" / f"{safe_title}.*",
    ]
    for pattern in leaf_globs:
        for stale in pattern.parent.glob(pattern.name):
            stale.unlink(missing_ok=True)
            print(f"[dev] purged stale: {stale.relative_to(api_dir)}")
    # TTS audio is namespaced by config id (c-XXXXXXX/{safe_title}.*)
    tts_root = api_dir / "tts_audio" / "chatterbox"
    if tts_root.is_dir():
        for cfg_dir in tts_root.iterdir():
            if not cfg_dir.is_dir():
                continue
            for stale in cfg_dir.glob(f"{safe_title}.*"):
                stale.unlink(missing_ok=True)
                print(f"[dev] purged stale: {stale.relative_to(api_dir)}")

def download_caption(url, destination_folder, filename=None):
    """download english captions to <filename.txt> in destination_folder, skipping if file already exists.
    If *filename* is given it is used as the stem; otherwise the YouTube title
    is used with colons and pipes stripped."""
    video_id, title = get_video_info(url)
    safe_title = filename or re.sub(r'[:|]', '', title).strip()
    save_path = pathlib.Path(destination_folder) / (safe_title + ".txt")

    # When truncating downloads, the cached full-length captions are stale —
    # purge them in download_video, and here we always re-fetch + clip.
    duration_s = int(os.getenv("FW_DOWNLOAD_DURATION_SECONDS", "0") or 0)

    if save_path.exists() and duration_s == 0:
        print(f"Skipping captions (already exists): {title}")
        return str(save_path)

    print(f"Downloading captions for {title}... ", end=" ", flush=True)
    api = YouTubeTranscriptApi()
    caption = api.fetch(video_id).to_raw_data()
    if duration_s > 0:
        before = len(caption)
        caption = [s for s in caption if s.get("start", 0) < duration_s]
        print(f"[truncated {before}→{len(caption)} segments to first {duration_s}s]", end=" ", flush=True)
    with open(save_path, 'w') as outfile:
        for segment in caption:
            outfile.write(json.dumps(segment) + "\n")
    print("Success!")
    return str(save_path)

if __name__ == '__main__':
    vid_urls = ["https://www.youtube.com/watch?v=G3Eup4mfJdA&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=1",
                "https://www.youtube.com/watch?v=480OGItLZNo&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=2",
                "https://www.youtube.com/watch?v=OA2Tj75T3fI&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=4",
                "https://www.youtube.com/watch?v=qrvK_KuIeJk&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=5",
                "https://www.youtube.com/watch?v=oFVuQ0RP_As&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=6",
                "https://www.youtube.com/watch?v=4aPp8KX6EiU&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=7",
                "https://www.youtube.com/watch?v=h8PSWeRLGXs&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=8",
                "https://www.youtube.com/watch?v=Z8qC2tVkGeU&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=9",
                "https://www.youtube.com/watch?v=Y9nM_9oBj2k&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=10",
                "https://www.youtube.com/watch?v=ervLwxz7xPo&list=PLI1yx5Z0Lrv77D_g1tvF9u3FVqnrNbCRL&index=11"]

    # make a directory and download 10 videos into it
    video_folder = "./raw_videos"
    captions_folder = "./raw_captions"

    delete_folder(video_folder)
    delete_folder(captions_folder)
    create_folder(video_folder)
    create_folder(captions_folder)

    for url in vid_urls:
        download_video(url, video_folder)
        download_caption(url, captions_folder)
