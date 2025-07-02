import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import random
import time
import traceback
import re
from collections import defaultdict
import torch
import colorsys
import matplotlib.colors
from matplotlib.colors import to_rgb
# Fix potential whisper import conflict
try:
    import whisper
except ImportError as e:
    print(f"Whisper import failed: {e}")
    print("Please install openai-whisper: pip install openai-whisper")
    sys.exit(1)

try:
    import yt_dlp
    from moviepy.editor import *
    from moviepy.video.tools.subtitles import SubtitlesClip
    import schedule
    import requests
    import numpy as np
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"Required package missing: {e}")
    print("Please install required packages:")
    print("pip install yt-dlp moviepy schedule requests scikit-learn numpy")
    sys.exit(1)

YOUTUBE_API_AVAILABLE = False
try:
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    print("YouTube API libraries not found. Upload functionality will be disabled.")
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_shorts_automator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YouTubeShortsAutomator:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the YouTube Shorts Automator with enhanced features"""
        self.config = self.load_config(config_path)
        self.youtube_service = None
        self.whisper_model = None
        self.output_dir = Path("output")
        self.temp_dir = Path("temp")
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize Whisper model
        self._initialize_whisper()
        
        # Enhanced caption styles configuration with more options
        self.caption_styles = {
            "modern": {
                "fontsize": 48,
                "color": "white",
                "font": "Arial-Bold",
                "stroke_color": "black",
                "stroke_width": 4,
                "bg_color": None,
                "position": "bottom",
                "alignment": "center",
                "interline": 5
            },
            "neon": {
                "fontsize": 52,
                "color": "#00ffff",
                "font": "Impact",
                "stroke_color": "#0000ff",
                "stroke_width": 5,
                "bg_color": None,
                "position": "center",
                "alignment": "center",
                "interline": 8
            },
            "minimal": {
                "fontsize": 44,
                "color": "white",
                "font": "Helvetica",
                "stroke_color": "#333333",
                "stroke_width": 2,
                "bg_color": None,
                "position": "bottom",
                "alignment": "center",
                "interline": 3
            },
            "bold": {
                "fontsize": 56,
                "color": "#ffff00",
                "font": "Arial-Black",
                "stroke_color": "black",
                "stroke_width": 6,
                "bg_color": None,
                "position": "top",
                "alignment": "center",
                "interline": 6
            },
            "bubble": {
                "fontsize": 46,
                "color": "black",
                "font": "Comic-Sans-MS-Bold",
                "stroke_color": "white",
                "stroke_width": 2,
                "bg_color": "white",
                "position": "center",
                "alignment": "center",
                "interline": 5,
                "radius": 0.5,
                "margin": 5
            },
            "highlight": {
                "fontsize": 50,
                "color": "white",
                "font": "Arial-Bold",
                "stroke_color": "#ff5500",
                "stroke_width": 4,
                "bg_color": "#333333",
                "position": "center",
                "alignment": "center",
                "interline": 6,
                "opacity": 0.8
            }
        }
        
    def _initialize_whisper(self):
        """Initialize Whisper model with error handling and GPU optimization"""
        try:
            model_size = self.config.get("whisper_model", "base")
            logger.info(f"Loading Whisper model: {model_size}")
            
            # Use GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model(model_size, device=device)
            
            # Optimize model for faster inference
            if device == "cuda":
                self.whisper_model = self.whisper_model.half()
            
            logger.info(f"Whisper model loaded successfully on {device.upper()}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.info("Whisper will be disabled. Captions will not be generated.")
            self.whisper_model = None
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file with enhanced defaults"""
        default_config = {
            "youtube_api_credentials": "credentials.json",
            "youtube_api_token": "token.json",
            "whisper_model": "base",
            "output_quality": "1080p",  # Default to higher quality
            "shorts_duration": {"min": 30, "max": 60},
            "upload_schedule": {
                "times": ["09:00", "12:00", "15:00", "18:00", "21:00"],
                "timezone": "UTC"
            },
            "background_music_dir": "music",
            "video_formats": ["mp4", "webm", "mkv", "avi"],
            "audio_formats": ["mp3", "wav", "flac", "aac"],
            "max_file_size_mb": 500,
            "concurrent_processing": 2,
            "scopes": ["https://www.googleapis.com/auth/youtube.upload"],
            "auto_cleanup": True,  # Automatically delete source files after processing
            "caption_style": "modern",
            "default_tags": ["shorts", "viral", "trending", "youtube", "shortvideo"],
            "min_engagement_score": 0.7,  # Minimum score for segment to be considered
            "max_short_count": 5,  # Maximum number of shorts to create per video
            "engagement_analysis": True,  # Analyze watch time data
            "dynamic_captions": True,  # Adjust captions based on content
            "auto_description": True  # Automatically generate descriptions
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                merged_config = {**default_config, **config}
            else:
                merged_config = default_config
                logger.info(f"Config file not found, creating default config at {config_path}")
                
            # Save the merged config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(merged_config, f, indent=2)
                
            return merged_config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def download_video(self, youtube_url: str) -> Optional[Dict]:
        """Download video from YouTube with enhanced metadata collection"""
        try:
            # if not self._is_valid_youtube_url(youtube_url):
            #     raise ValueError("Invalid YouTube URL")
            
            ydl_opts = {
                'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]',
                'outtmpl': str(self.temp_dir / 'source_%(title)s_%(id)s.%(ext)s'),
                'noplaylist': True,
                'extractaudio': False,
                'writeinfojson': True,
                'writethumbnail': True,
                'writesubtitles': True,
                'subtitlesformat': 'vtt',
                'subtitleslangs': ['en'],
                'ignoreerrors': True,
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False,
                'getcomments': False,
                'get_heatmap': True  # Try to get engagement data
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading video from: {youtube_url}")
                info = ydl.extract_info(youtube_url, download=True)
                
                if not info:
                    raise ValueError("Failed to extract video information")
                
                filename = ydl.prepare_filename(info)
                
                # Check if file exists and has reasonable size
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"Downloaded file not found: {filename}")
                
                file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
                if file_size > self.config["max_file_size_mb"]:
                    logger.warning(f"File size ({file_size:.1f}MB) exceeds limit")
                
                # Get metadata file path
                json_filename = os.path.splitext(filename)[0] + '.info.json'
                
                # Return both video path and metadata
                result = {
                    'video_path': filename,
                    'metadata_path': json_filename if os.path.exists(json_filename) else None,
                    'thumbnail_path': os.path.splitext(filename)[0] + '.webp',
                    'subtitles_path': os.path.splitext(filename)[0] + '.en.vtt',
                    'original_url': youtube_url
                }
                
                logger.info(f"Downloaded video: {filename} ({file_size:.1f}MB)")
                return result
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None
    
    def _is_valid_youtube_url(self, url: str) -> bool:
        """Validate YouTube URL"""
        youtube_domains = ["youtube.com", "youtu.be", "m.youtube.com"]
        return any(domain in url for domain in youtube_domains)
    
    def analyze_engagement(self, metadata_path: str) -> Optional[List[float]]:
        """Analyze video engagement data to find most watched segments"""
        if not metadata_path or not os.path.exists(metadata_path):
            return None
            
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Try to get heatmap data (if available)
            heatmap = metadata.get('heatmap')
            if not heatmap:
                logger.info("No engagement heatmap data available")
                return None
            
            # Convert heatmap to normalized values
            engagement = [float(x) for x in heatmap]
            max_val = max(engagement) if engagement else 1.0
            normalized = [x / max_val for x in engagement]
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error analyzing engagement data: {e}")
            return None
    
    def extract_audio_and_transcribe(self, video_path: str) -> List[Dict]:
        """Enhanced audio extraction and transcription with better segment handling"""
        if not self.whisper_model:
            logger.warning("Whisper model not available, skipping transcription")
            return []
            
        try:
            logger.info("Extracting audio and transcribing...")
            
            # Load video and extract audio with optimized settings
            video = VideoFileClip(video_path, audio_fps=16000)  # Lower sample rate for faster processing
            if not video.audio:
                logger.warning("No audio track found in video")
                video.close()
                return []
            
            audio_path = str(self.temp_dir / "temp_audio.wav")
            video.audio.write_audiofile(
                audio_path, 
                fps=16000,  # Match whisper's preferred sample rate
                nbytes=2,   # 16-bit audio
                codec='pcm_s16le',
                verbose=False, 
                logger=None,
                temp_audiofile=str(self.temp_dir / "temp_audio_temp.wav")
            )
            video.close()
            
            # Transcribe with Whisper using optimized parameters
            logger.info("Transcribing audio with enhanced settings...")
            result = self.whisper_model.transcribe(
                audio_path, 
                word_timestamps=True,
                language="en",
                fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
                temperature=0.2,  # More deterministic output
                best_of=3,       # Better quality
                beam_size=5       # Better quality
            )
            
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Enhanced segment processing
            segments = []
            for segment in result["segments"]:
                text = segment["text"].strip()
                if len(text) > 3:  # Filter out very short segments
                    # Clean up text
                    text = self._clean_text(text)
                    
                    # Calculate speech rate (words per second)
                    word_count = len(text.split())
                    duration = segment["end"] - segment["start"]
                    speech_rate = word_count / duration if duration > 0 else 0
                    
                    # Calculate confidence score
                    avg_prob = np.mean([word['probability'] for word in segment.get('words', [])]) if 'words' in segment else 0.9
                    
                    segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": text,
                        "speech_rate": speech_rate,
                        "confidence": avg_prob,
                        "word_count": word_count
                    })
            
            logger.info(f"Transcribed {len(segments)} segments with enhanced metadata")
            return segments
            
        except Exception as e:
            logger.error(f"Error in enhanced transcription: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean up transcribed text for better display"""
        # Remove unwanted characters
        text = re.sub(r'\[.*?\]', '', text)  # Remove anything in brackets
        text = re.sub(r'\(.*?\)', '', text)  # Remove anything in parentheses
        text = re.sub(r'[^\w\s\'\",.?!-]', '', text)  # Keep only common punctuation
        
        # Fix common transcription errors
        replacements = {
            " you re ": " you're ",
            " we re ": " we're ",
            " they re ": " they're ",
            " i m ": " I'm ",
            " dont ": " don't ",
            " cant ": " can't ",
            " wont ": " won't ",
            " its ": " it's ",
            " whats ": " what's "
        }
        
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text.strip()
    
    def find_best_segments(self, segments: List[Dict], engagement_data: Optional[List[float]] = None, 
                          total_duration: float = 0) -> List[Dict]:
        """Find the best segments for shorts using multiple factors"""
        if not segments:
            # If no segments, create random time intervals
            num_segments = min(3, self.config["max_short_count"])
            segment_duration = 45
            segments_info = []
            
            for i in range(num_segments):
                start_time = (total_duration / num_segments) * i
                end_time = min(start_time + segment_duration, total_duration)
                if end_time - start_time >= 30:  # Minimum 30 seconds
                    segments_info.append({
                        "start": start_time,
                        "end": end_time,
                        "text": f"Highlight {i+1}",
                        "score": 0.7  # Default score
                    })
            return segments_info
        
        # Score segments based on multiple factors
        scored_segments = []
        for segment in segments:
            duration = segment["end"] - segment["start"]
            
            # Base score components (normalized)
            duration_score = min(1.0, duration / 60)  # Prefer segments around 60s
            speech_score = min(2.0, segment["speech_rate"] / 3)  # Ideal ~3 words/sec
            confidence_score = segment["confidence"]
            word_score = min(1.5, segment["word_count"] / 20)  # Ideal ~20 words
            
            # Engagement score (if available)
            if engagement_data:
                start_idx = int((segment["start"] / total_duration) * len(engagement_data))
                end_idx = int((segment["end"] / total_duration) * len(engagement_data))
                engagement_score = np.mean(engagement_data[start_idx:end_idx]) if end_idx > start_idx else 0
            else:
                engagement_score = 0.5  # Neutral if no data
            
            # Calculate composite score with weights
            score = (
                0.3 * duration_score +
                0.2 * speech_score +
                0.15 * confidence_score +
                0.15 * word_score +
                0.2 * engagement_score
            )
            
            # Only consider segments with minimum quality
            if score >= self.config["min_engagement_score"] and duration >= 5:
                scored_segments.append({
                    **segment,
                    "score": score,
                    "engagement_score": engagement_score
                })
        
        # Cluster similar segments to avoid redundancy
        if len(scored_segments) > 5:
            try:
                # Create feature vectors for clustering
                features = []
                for seg in scored_segments:
                    features.append([
                        seg["start"] / total_duration,  # Normalized position
                        seg["end"] / total_duration,
                        seg["speech_rate"],
                        seg["word_count"],
                        seg["score"]
                    ])
                
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=min(5, len(scored_segments)//2), random_state=42)
                clusters = kmeans.fit_predict(features)
                
                # Select best segment from each cluster
                best_segments = []
                for cluster_id in set(clusters):
                    cluster_segments = [seg for seg, c in zip(scored_segments, clusters) if c == cluster_id]
                    best_in_cluster = max(cluster_segments, key=lambda x: x["score"])
                    best_segments.append(best_in_cluster)
                
                scored_segments = best_segments
            except Exception as e:
                logger.error(f"Clustering failed, using all segments: {e}")
        
        # Sort by score and return top segments
        scored_segments.sort(key=lambda x: x["score"], reverse=True)
        return scored_segments[:self.config["max_short_count"]]
    
    def create_subtitle_clips(self, segments: List[Dict], style: str, video_size: Tuple[int, int]) -> List:
        """Create enhanced subtitle clips with dynamic styling"""
        if not segments:
            return []
            
        style_config = self.caption_styles.get(style, self.caption_styles["modern"])
        subtitle_clips = []
        
        try:
            for segment in segments:
                text = segment["text"]
                
                # Dynamic font size adjustment based on text length
                max_chars = max(len(line) for line in text.split('\n')) if '\n' in text else len(text)
                dynamic_fontsize = max(
                    style_config["fontsize"] * 0.8,
                    min(
                        style_config["fontsize"] * 1.5,
                        style_config["fontsize"] * (1 + (50 - max_chars) / 100)
                    )
                )
                
                # Dynamic positioning based on speech rate
                position = style_config["position"]
                if style_config.get("dynamic_position", True):
                    if segment.get("speech_rate", 0) > 3.5:  # Fast speech
                        position = "center"
                    elif segment.get("speech_rate", 0) < 2.0:  # Slow speech
                        position = "bottom"
                
                # Create text clip with enhanced styling
                txt_clip = TextClip(
                    text,
                    fontsize=int(dynamic_fontsize),
                    color=style_config["color"],
                    font=style_config["font"],
                    stroke_color=style_config["stroke_color"],
                    stroke_width=style_config["stroke_width"],
                    method="caption",
                    size=(video_size[0] * 0.9, None),
                    align=style_config["alignment"],
                    interline=style_config.get("interline", 0)
                ).set_start(segment["start"]).set_duration(segment["end"] - segment["start"])
                
                # Add background if specified
                if style_config.get("bg_color"):
                    bg_opacity = style_config.get("opacity", 0.7)
                    bg_clip = ColorClip(
                        size=(int(video_size[0] * 0.95), int(txt_clip.size[1] * 1.2)),
                        color= [int(c * 255) for c in to_rgb(style_config["bg_color"])],
                        duration=segment["end"] - segment["start"]
                    ).set_opacity(bg_opacity)
                    
                    if style_config.get("radius", 0) > 0:
                        bg_clip = bg_clip.fx(vfx.mask_color, color=[0, 0, 0]).set_pos("center")
                    
                    # Composite text with background
                    txt_clip = CompositeVideoClip([bg_clip, txt_clip.set_position("center")])
                
                # Position subtitle
                if position == "bottom":
                    y_pos = video_size[1] * 0.85 - txt_clip.size[1]
                    txt_clip = txt_clip.set_position(("center", y_pos))
                elif position == "center":
                    txt_clip = txt_clip.set_position("center")
                else:  # top
                    txt_clip = txt_clip.set_position(("center", video_size[1] * 0.15))
                
                # Add fade in/out effects
                fade_duration = min(0.5, (segment["end"] - segment["start"]) / 3)
                txt_clip = txt_clip.crossfadein(fade_duration).crossfadeout(fade_duration)
                
                subtitle_clips.append(txt_clip)
            
            logger.info(f"Created {len(subtitle_clips)} enhanced subtitle clips")
            return subtitle_clips
            
        except Exception as e:
            logger.error(f"Error creating enhanced subtitle clips: {e}")
            return []
    
    def _add_animated_elements(self, clip: VideoFileClip) -> VideoFileClip:
        """Add animated elements to make shorts more engaging"""
        try:
            # Add subtle zoom effect
            zoom_factor = 1.03  # 3% zoom
            zoom_duration = clip.duration / 2
            
            def zoom_effect(t):
                progress = min(1.0, t / zoom_duration)
                return 1 + (zoom_factor - 1) * progress
            
            zoomed_clip = clip.fx(vfx.resize, zoom_effect)
            
            # Add subtle color enhancement
            color_enhanced = zoomed_clip.fx(vfx.colorx, 1.1)
            
            return color_enhanced
        except Exception as e:
            logger.error(f"Error adding animated elements: {e}")
            return clip
    
    def _add_intro_outro(self, clip: VideoFileClip, video_info: Dict) -> VideoFileClip:
        """Add branded intro/outro to the short"""
        try:
            # Create simple text intro
            intro_duration = 1.5
            intro_text = "Trending Clip"  # Could be customized from config
            
            intro = TextClip(
                intro_text,
                fontsize=70,
                color='white',
                font="Impact",
                stroke_color='black',
                stroke_width=4,
                size=clip.size
            ).set_duration(intro_duration).set_position("center")
            
            # Fade in/out effects
            intro = intro.crossfadein(0.5).crossfadeout(0.5)
            
            # Create outro with call to action
            outro_duration = 2.0
            outro_text = "Watch Full Video!\nLink in Description"
            
            outro = TextClip(
                outro_text,
                fontsize=60,
                color='yellow',
                font="Arial-Bold",
                stroke_color='black',
                stroke_width=3,
                size=(clip.w * 0.8, None),
                method="caption",
                align="center"
            ).set_duration(outro_duration).set_position("center")
            
            outro = outro.crossfadein(0.5).crossfadeout(0.5)
            
            # Composite the clips
            main_clip = clip.set_start(intro_duration)
            final_clip = CompositeVideoClip([
                intro,
                main_clip,
                outro.set_start(intro_duration + clip.duration - outro_duration)
            ]).set_duration(intro_duration + clip.duration)
            
            return final_clip
        except Exception as e:
            logger.error(f"Error adding intro/outro: {e}")
            return clip
    
    def create_shorts_from_video(self, video_info: Dict, segments: List[Dict], 
                                caption_style: str) -> List[str]:
        """Create multiple short videos with enhanced processing"""
        shorts_paths = []
        video_path = video_info['video_path']
        
        try:
            video = VideoFileClip(video_path)
            total_duration = video.duration
            
            if total_duration < 30:
                logger.warning("Video too short to create meaningful shorts")
                video.close()
                return []
            
            # Analyze engagement data if available
            engagement_data = None
            if self.config["engagement_analysis"]:
                engagement_data = self.analyze_engagement(video_info.get('metadata_path'))
            
            # Find the best segments for shorts
            best_segments = self.find_best_segments(segments, engagement_data, total_duration)
            logger.info(f"Found {len(best_segments)} high-quality segments for shorts")
            
            for i, segment_info in enumerate(best_segments):
                try:
                    short_path = self._create_single_short(
                        video, segment_info, segments, caption_style, i + 1, video_info
                    )
                    if short_path:
                        shorts_paths.append(short_path)
                        
                except Exception as e:
                    logger.error(f"Error creating short {i+1}: {e}")
                    continue
            
            video.close()
            
            # Auto cleanup if configured
            if self.config["auto_cleanup"]:
                self._cleanup_source_files(video_info)
            
            logger.info(f"Successfully created {len(shorts_paths)} enhanced shorts")
            return shorts_paths
            
        except Exception as e:
            logger.error(f"Error in create_shorts_from_video: {e}")
            return []
    
    def _create_single_short(self, video: VideoFileClip, segment_info: Dict, 
                           all_segments: List[Dict], caption_style: str, 
                           short_number: int, video_info: Dict) -> Optional[str]:
        """Create a single enhanced short video"""
        try:
            # Calculate timing with buffer
            short_duration = random.randint(
                self.config["shorts_duration"]["min"], 
                self.config["shorts_duration"]["max"]
            )
            
            # Adjust start time to include context
            start_buffer = min(2, segment_info["start"])  # Up to 2s before
            end_buffer = min(2, video.duration - segment_info["end"])  # Up to 2s after
            
            start_time = max(0, segment_info["start"] - start_buffer)
            end_time = min(video.duration, segment_info["end"] + end_buffer)
            
            # Ensure minimum duration
            if (end_time - start_time) < self.config["shorts_duration"]["min"]:
                end_time = min(video.duration, start_time + self.config["shorts_duration"]["min"])
            
            # Extract clip
            short_clip = video.subclip(start_time, end_time)
            
            # Enhanced processing
            short_clip = self._resize_to_shorts_format(short_clip)
            short_clip = self._add_animated_elements(short_clip)
            
            # Get relevant subtitles
            relevant_segments = self._get_relevant_segments(all_segments, start_time, end_time)
            
            # Create subtitle clips with dynamic styling
            subtitle_clips = []
            if relevant_segments:
                subtitle_clips = self.create_subtitle_clips(
                    relevant_segments, caption_style, (1080, 1920)
                )
            
            # Add background music
            short_clip = self._add_background_music(short_clip)
            
            # Composite final video
            if subtitle_clips:
                final_clip = CompositeVideoClip([short_clip] + subtitle_clips)
            else:
                final_clip = short_clip
            
            # Add intro/outro
            final_clip = self._add_intro_outro(final_clip, video_info)
            
            # Export with optimized settings
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = str(self.output_dir / f"short_{short_number}_{timestamp}.mp4")
            
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate="8000k",  # Higher quality
                threads=4,        # Use multiple threads
                preset='fast',   # Balance between speed and quality
                temp_audiofile=str(self.temp_dir / f'temp-audio-{short_number}.m4a'),
                remove_temp=True,
                verbose=False,
                logger=None,
                fps=30
            )
            
            final_clip.close()
            
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"Created enhanced short video: {output_path} ({file_size:.1f}MB)")
                return output_path
            else:
                logger.error(f"Failed to create valid short video: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating enhanced short: {e}")
            return None
    
    def _cleanup_source_files(self, video_info: Dict):
        """Clean up source files after processing"""
        try:
            files_to_remove = [
                video_info['video_path'],
                video_info.get('metadata_path'),
                video_info.get('thumbnail_path'),
                video_info.get('subtitles_path')
            ]
            
            for file_path in files_to_remove:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up source file: {file_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up source files: {e}")
    
    def generate_video_description(self, original_url: str, segment_text: str = "") -> str:
        """Generate an optimized video description"""
        description = ""
        
        # Add call to action
        description += "🔥 Check out this viral clip! 🔥\n\n"
        
        # Add segment text if available
        if segment_text:
            description += f"\"{segment_text}\"\n\n"
        
        # Add original video link
        description += f"Watch the full video here: {original_url}\n\n"
        
        # Add hashtags and tags
        description += "#shorts #viral #trending #youtube #shortvideo\n"
        
        # Add channel promotion if configured
        if self.config.get("channel_name"):
            description += f"\nSubscribe to {self.config['channel_name']} for more amazing content!"
        
        return description
    
    def generate_video_title(self, base_title: str, segment_text: str = "") -> str:
        """Generate an optimized video title"""
        # Clean up base title
        title = base_title.replace("|", "-").replace(":", "-")[:50]
        
        # Add emoji if not present
        if not any(c in title for c in ["🔥", "💯", "🎥", "👀"]):
            title = "🔥 " + title
        
        # Add segment highlight if space allows
        if segment_text and len(title) < 40:
            short_text = segment_text[:30].split('.')[0]
            if len(title + " - " + short_text) < 80:
                title += " - " + short_text
        
        return title
    
    def upload_to_youtube(self, video_path: str, video_info: Dict, 
                         segment_text: str = "", scheduled_time: Optional[datetime] = None) -> Optional[str]:
        """Enhanced YouTube upload with better metadata"""
        if not YOUTUBE_API_AVAILABLE:
            logger.error("YouTube API not available for upload")
            return None
            
        try:
            if not self.youtube_service and not self.authenticate_youtube():
                return None
            
            # Validate video file
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None
            
            # Generate optimized metadata
            title = self.generate_video_title(
                os.path.basename(video_path).replace('_', ' ').replace('.mp4', ''),
                segment_text
            )
            
            description = self.generate_video_description(
                video_info['original_url'],
                segment_text
            )
            
            tags = self.config.get("default_tags", [])
            
            body = {
                'snippet': {
                    'title': title[:100],
                    'description': description[:5000],
                    'tags': tags[:15],
                    'categoryId': '22',  # People & Blogs
                    'defaultLanguage': 'en',
                    'defaultAudioLanguage': 'en'
                },
                'status': {
                    'privacyStatus': 'private' if scheduled_time else 'public',
                    'selfDeclaredMadeForKids': False,
                    'embeddable': True
                }
            }
            
            if scheduled_time:
                body['status']['publishAt'] = scheduled_time.isoformat() + 'Z'
            
            # Add thumbnail if available
            if video_info.get('thumbnail_path') and os.path.exists(video_info['thumbnail_path']):
                try:
                    thumbnail_response = self.youtube_service.thumbnails().set(
                        videoId=video_id,
                        media_body=MediaFileUpload(video_info['thumbnail_path'])
                    ).execute()
                    logger.info(f"Thumbnail uploaded: {thumbnail_response}")
                except Exception as e:
                    logger.error(f"Error uploading thumbnail: {e}")
            
            media = MediaFileUpload(
                video_path, 
                chunksize=-1, 
                resumable=True, 
                mimetype='video/mp4'
            )
            
            request = self.youtube_service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = request.execute()
            video_id = response['id']
            
            logger.info(f"Successfully uploaded video with ID: {video_id}")
            return video_id
            
        except Exception as e:
            logger.error(f"Error uploading to YouTube: {e}")
            return None
    
    async def process_video_url(self, youtube_url: str, caption_style: str = None, 
                               num_shorts: int = None, auto_upload: bool = False) -> Dict:
        """Enhanced processing function with better error handling and reporting"""
        result = {
            "success": False,
            "original_video": None,
            "shorts_created": 0,
            "shorts_paths": [],
            "segments": 0,
            "uploads_scheduled": False,
            "error": None,
            "engagement_data": None,
            "best_segments": []
        }
        
        try:
            logger.info(f"Starting enhanced video processing: {youtube_url}")
            
            # Step 1: Download video with metadata
            video_info = self.download_video(youtube_url)
            if not video_info or not video_info.get('video_path'):
                result["error"] = "Failed to download video"
                return result
                
            result["original_video"] = video_info['video_path']
            
            # Step 2: Extract audio and transcribe
            segments = self.extract_audio_and_transcribe(video_info['video_path'])
            result["segments"] = len(segments)
            
            # Step 3: Analyze engagement if available
            if self.config["engagement_analysis"]:
                result["engagement_data"] = self.analyze_engagement(video_info.get('metadata_path'))
            
            # Step 4: Create shorts with enhanced processing
            caption_style = caption_style or self.config.get("caption_style", "modern")
            shorts_paths = self.create_shorts_from_video(video_info, segments, caption_style)
            
            result["shorts_created"] = len(shorts_paths)
            result["shorts_paths"] = shorts_paths
            
            if not shorts_paths:
                result["error"] = "Failed to create any shorts"
                return result
            
            # Step 5: Schedule uploads if requested
            if auto_upload and shorts_paths and YOUTUBE_API_AVAILABLE:
                try:
                    # Extract video info for upload metadata
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(youtube_url, download=False)
                        video_title = info.get('title', 'Viral Clip')[:50]
                    
                    base_title = f"Shorts: {video_title}"
                    tags = self.config.get("default_tags", [])
                    
                    self.schedule_uploads(shorts_paths, video_info, base_title, tags)
                    result["uploads_scheduled"] = True
                    
                except Exception as e:
                    logger.error(f"Error scheduling uploads: {e}")
                    result["error"] = f"Shorts created but upload scheduling failed: {e}"
            
            result["success"] = True
            logger.info(f"Enhanced processing completed: {result['shorts_created']} shorts created")
            
        except Exception as e:
            logger.error(f"Error in enhanced process_video_url: {e}")
            logger.error(traceback.format_exc())
            result["error"] = str(e)
        
        return result
    
    def schedule_uploads(self, video_paths: List[str], video_info: Dict, 
                       base_title: str, tags: List[str]):
        """Schedule multiple videos for upload with enhanced logic"""
        if not video_paths:
            return
            
        upload_times = self.config["upload_schedule"]["times"]
        
        for i, video_path in enumerate(video_paths):
            try:
                # Calculate upload time with spreading
                days_ahead = i // len(upload_times)
                time_index = i % len(upload_times)
                
                upload_time = datetime.now() + timedelta(days=days_ahead)
                hour, minute = map(int, upload_times[time_index].split(':'))
                upload_time = upload_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # Ensure future time
                if upload_time <= datetime.now():
                    upload_time += timedelta(days=1)
                
                # Get segment text for metadata (if available)
                segment_text = ""
                if i < len(video_info.get('segment_texts', [])):
                    segment_text = video_info['segment_texts'][i]
                
                title = f"{base_title} - Part {i+1}"
                
                # Schedule the upload
                schedule.every().day.at(upload_times[time_index]).do(
                    self._scheduled_upload_wrapper,
                    video_path=video_path,
                    video_info=video_info,
                    title=title,
                    tags=tags,
                    segment_text=segment_text,
                    scheduled_time=upload_time
                )
                
                logger.info(f"Scheduled enhanced upload for {os.path.basename(video_path)} at {upload_time}")
                
            except Exception as e:
                logger.error(f"Error scheduling upload for video {i+1}: {e}")
    
    def _scheduled_upload_wrapper(self, video_path: str, video_info: Dict, title: str, 
                                tags: List[str], segment_text: str, scheduled_time: datetime):
        """Wrapper for scheduled uploads with enhanced error handling"""
        try:
            video_id = self.upload_to_youtube(
                video_path, 
                video_info,
                segment_text,
                scheduled_time
            )
            if video_id:
                logger.info(f"Successfully uploaded scheduled video: {video_id}")
                # Clean up after successful upload if configured
                if self.config["auto_cleanup"] and os.path.exists(video_path):
                    os.remove(video_path)
            else:
                logger.error(f"Failed to upload scheduled video: {video_path}")
        except Exception as e:
            logger.error(f"Error in scheduled upload: {e}")
def main():
    """Enhanced main function with better error handling and examples"""
    print("YouTube Shorts Automator v2.0")
    print("=" * 50)
    
    try:
        automator = YouTubeShortsAutomator()
        
        # Interactive mode
        if len(sys.argv) == 1:
            print("\nInteractive Mode")
            print("Enter a YouTube URL to process, or 'quit' to exit:")
            
            while True:
                try:
                    user_input = input("\nYouTube URL: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not user_input:
                        continue
                    
                    # Get processing options
                    print("\nProcessing Options:")
                    caption_style = input("Caption style (modern/neon/minimal/bold/bubble) [modern]: ").strip() or "modern"
                    
                    try:
                        num_shorts = int(input("Number of shorts to create [3]: ") or "3")
                    except ValueError:
                        num_shorts = 3
                    
                    auto_upload = input("Auto-upload to YouTube? (y/n) [n]: ").strip().lower() == 'y'
                    
                    print(f"\nProcessing video...")
                    print(f"URL: {user_input}")
                    print(f"Caption Style: {caption_style}")
                    print(f"Number of Shorts: {num_shorts}")
                    print(f"Auto Upload: {auto_upload}")
                    print("-" * 30)
                    
                    # Process the video
                    result = asyncio.run(automator.process_video_url(
                        youtube_url=user_input,
                        caption_style=caption_style,
                        num_shorts=num_shorts,
                        auto_upload=auto_upload
                    ))
                    
                    # Display results
                    print("\nProcessing Results:")
                    print(f"Success: {result['success']}")
                    print(f"Shorts Created: {result['shorts_created']}")
                    print(f"Segments Transcribed: {result['segments']}")
                    
                    if result['shorts_paths']:
                        print("Created Files:")
                        for path in result['shorts_paths']:
                            print(f"  - {os.path.basename(path)}")
                    
                    if result['error']:
                        print(f"Error: {result['error']}")
                    
                    if result['uploads_scheduled']:
                        print("Uploads have been scheduled!")
                    
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user")
                    break
                except Exception as e:
                    print(f"Error processing video: {e}")
                    logger.error(f"Interactive mode error: {e}")
        
        # Command line mode
        elif len(sys.argv) >= 2:
            youtube_url = sys.argv[1]
            caption_style = sys.argv[2] if len(sys.argv) > 2 else "modern"
            num_shorts = int(sys.argv[3]) if len(sys.argv) > 3 else 3
            auto_upload = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False
            
            print(f"Processing URL: {youtube_url}")
            
            result = asyncio.run(automator.process_video_url(
                youtube_url=youtube_url,
                caption_style=caption_style,
                num_shorts=num_shorts,
                auto_upload=auto_upload
            ))
            
            print("Processing Result:")
            print(json.dumps(result, indent=2, default=str))
        
        # Scheduler mode
        print("\nStarting scheduler...")
        print("Scheduled upload times:", automator.config["upload_schedule"]["times"])
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\nScheduler stopped by user")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Fatal error: {e}")
        sys.exit(1)

class ShortsAutomatorAPI:
    """Enhanced Web API wrapper for the automator"""
    
    def __init__(self, config_path: str = "config.json"):
        self.automator = YouTubeShortsAutomator(config_path)
    
    async def create_shorts(self, request_data: Dict) -> Dict:
        """API endpoint for creating shorts with comprehensive validation"""
        try:
            # Validate required fields
            youtube_url = request_data.get("youtube_url")
            if not youtube_url:
                return {"success": False, "error": "YouTube URL is required"}
            
            # Validate optional fields with defaults
            caption_style = request_data.get("caption_style", "modern")
            if caption_style not in self.automator.caption_styles:
                return {"success": False, "error": f"Invalid caption style. Options: {list(self.automator.caption_styles.keys())}"}
            
            num_shorts = request_data.get("num_shorts", 3)
            if not isinstance(num_shorts, int) or num_shorts < 1 or num_shorts > 10:
                return {"success": False, "error": "num_shorts must be between 1 and 10"}
            
            auto_upload = request_data.get("auto_upload", False)
            if not isinstance(auto_upload, bool):
                return {"success": False, "error": "auto_upload must be boolean"}
            
            # Process the video
            result = await self.automator.process_video_url(
                youtube_url, caption_style, num_shorts, auto_upload
            )
            
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_status(self) -> Dict:
        """Get system status"""
        try:
            return {
                "success": True,
                "data": {
                    "whisper_available": self.automator.whisper_model is not None,
                    "youtube_api_available": YOUTUBE_API_AVAILABLE,
                    "caption_styles": list(self.automator.caption_styles.keys()),
                    "output_directory": str(self.automator.output_dir),
                    "temp_directory": str(self.automator.temp_dir),
                    "config": {
                        "shorts_duration": self.automator.config["shorts_duration"],
                        "upload_schedule": self.automator.config["upload_schedule"],
                        "whisper_model": self.automator.config["whisper_model"]
                    }
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# Example usage and testing functions
def test_automator():
    """Test function for development"""
    automator = YouTubeShortsAutomator()
    
    # Test configuration
    print("Configuration loaded:", automator.config)
    
    # Test caption styles
    print("Available caption styles:", list(automator.caption_styles.keys()))
    
    # Test directories
    print(f"Output directory: {automator.output_dir}")
    print(f"Temp directory: {automator.temp_dir}")
    
    # Test Whisper model
    print(f"Whisper model loaded: {automator.whisper_model is not None}")
    
    # Test YouTube API
    print(f"YouTube API available: {YOUTUBE_API_AVAILABLE}")

def create_sample_config():
    """Create a sample configuration file with detailed comments"""
    sample_config = {
        "youtube_api_credentials": "credentials.json",
        "youtube_api_token": "token.json",
        "whisper_model": "base",
        "output_quality": "720p",
        "shorts_duration": {
            "min": 30,
            "max": 60
        },
        "upload_schedule": {
            "times": ["09:00", "15:00", "21:00"],
            "timezone": "UTC"
        },
        "background_music_dir": "music",
        "video_formats": ["mp4", "webm", "mkv", "avi"],
        "audio_formats": ["mp3", "wav", "flac", "aac"],
        "max_file_size_mb": 500,
        "concurrent_processing": 2,
        "scopes": ["https://www.googleapis.com/auth/youtube.upload"]
    }
    
    with open("sample_config.json", "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=2)
    
    print("Sample configuration created: sample_config.json")

# Main function and other supporting code remains similar with minor adjustments
# to accommodate the enhanced features...

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_automator()
            sys.exit(0)
        elif sys.argv[1] == "config":
            create_sample_config()
            sys.exit(0)
        elif sys.argv[1] == "help":
            print("YouTube Shorts Automator - Usage:")
            print("python YouTubeShortsAutomator.py [URL] [style] [count] [upload]")
            print("python YouTubeShortsAutomator.py test          # Test configuration")
            print("python YouTubeShortsAutomator.py config        # Create sample config")
            print("python YouTubeShortsAutomator.py help          # Show this help")
            print("\nInteractive mode: python YouTubeShortsAutomator.py")
            print("\nCaption styles: modern, neon, minimal, bold, bubble")
            sys.exit(0)
    main()