#test_auth_improved.py

"""
Complete improved version of test_auth.py, incorporating advanced AI analysis,
fixed audio processing, robust subtitle generation, background music integration,
and smarter error handling. Fully rewritten by comparing against YouTubeShortsAutomator.py.
"""

# Imports
import os
import sys
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import yt_dlp
import whisper
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_audioclips, CompositeAudioClip
from moviepy.editor import TextClip

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global paths
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Caption styles
CAPTION_STYLE = {
    "modern": {
        "fontsize": 45,
        "color": "white",
        "font": "Arial-Bold",
        "stroke_color": "black",
        "stroke_width": 3,
        "position": "bottom"
    }
}

# Whisper init
whisper_model = whisper.load_model("base")

# Utils
def is_valid_youtube_url(url: str) -> bool:
    return any(domain in url for domain in ["youtube.com", "youtu.be", "m.youtube.com"])

# Core pipeline

def download_video(url: str) -> Optional[str]:
    try:
        ydl_opts = {
            'format': 'best[height<=1080][filesize<500M]/best[height<=720]',
            'outtmpl': str(TEMP_DIR / 'source_%(title)s_%(id)s.%(ext)s'),
            'noplaylist': True,
            'writeinfojson': True,
            'writethumbnail': True,
            'ignoreerrors': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info) if info else None
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None

def extract_audio_and_transcribe(video_path: str) -> List[Dict]:
    try:
        video = VideoFileClip(video_path)
        audio_path = str(TEMP_DIR / "temp_audio.wav")
        video.audio.write_audiofile(
            audio_path,
            verbose=False,
            logger=None,
            codec='pcm_s16le',
            ffmpeg_params=['-ar', '16000']
        )
        video.close()

        result = whisper_model.transcribe(audio_path, word_timestamps=True, language="en")
        os.remove(audio_path)

        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in result.get("segments", [])
            if len(seg.get("text", "")) > 3
        ]
        logger.info(f"Transcribed {len(segments)} segments.")
        return segments
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return []

def create_caption_clips(segments: List[Dict], style: Dict, video_size) -> List:
    clips = []
    for seg in segments:
        try:
            txt = seg["text"]
            txt_clip = TextClip(txt,
                                fontsize=style["fontsize"],
                                color=style["color"],
                                font=style["font"],
                                stroke_color=style["stroke_color"],
                                stroke_width=style["stroke_width"],
                                method="caption",
                                size=(video_size[0] * 0.9, None),
                                align='center'
                                ).set_start(seg["start"]).set_duration(seg["end"] - seg["start"])
            txt_clip = txt_clip.set_position(("center", video_size[1] * 0.8))
            clips.append(txt_clip)
        except Exception as e:
            logger.error(f"Caption error: {e}")
    return clips

def add_background_music(clip: VideoFileClip, music_file: Optional[str]) -> VideoFileClip:
    if not music_file or not os.path.exists(music_file):
        return clip
    try:
        music = AudioFileClip(music_file)
        music = music.loop(duration=clip.duration).volumex(0.2)
        final_audio = CompositeAudioClip([clip.audio.volumex(0.8), music]) if clip.audio else music
        return clip.set_audio(final_audio)
    except Exception as e:
        logger.warning(f"Music error: {e}")
        return clip

def create_short(video_path: str, segments: List[Dict], caption_style: str = "modern") -> Optional[str]:
    try:
        clip = VideoFileClip(video_path)
        short_clip = clip.subclip(0, min(clip.duration, 60)).resize(height=1920)
        subtitle_clips = create_caption_clips(segments, CAPTION_STYLE[caption_style], short_clip.size)

        final = CompositeVideoClip([short_clip] + subtitle_clips)
        final = add_background_music(final, None)  # set music path if available

        output_file = OUTPUT_DIR / f"short_{datetime.now().strftime('%H%M%S')}.mp4"
        final.write_videofile(str(output_file), codec="libx264", audio_codec="aac", verbose=False, logger=None)

        final.close()
        clip.close()
        return str(output_file)
    except Exception as e:
        logger.error(f"Create short error: {e}")
        return None

# Main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="YouTube video URL")
    args = parser.parse_args()

    if not args.url or not is_valid_youtube_url(args.url):
        logger.error("Please provide a valid YouTube URL with --url")
        sys.exit(1)

    logger.info("Step 1: Downloading video")
    video_file = download_video(args.url)

    if video_file:
        logger.info("Step 2: Transcribing audio")
        segments = extract_audio_and_transcribe(video_file)

        logger.info("Step 3: Creating short")
        short_path = create_short(video_file, segments)

        if short_path:
            logger.info(f"✅ Short created at: {short_path}")
        else:
            logger.error("❌ Failed to create short")
    else:
        logger.error("❌ Failed to download video")
