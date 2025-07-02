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
except ImportError as e:
    print(f"Required package missing: {e}")
    print("Please install required packages:")
    print("pip install yt-dlp moviepy schedule requests")
    sys.exit(1)

# Optional YouTube API imports (will work without these)
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
    print("To enable uploads, install: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

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
        """Initialize the YouTube Shorts Automator"""
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
        
        # Enhanced caption styles configuration
        self.caption_styles = {
            "modern": {
                "fontsize": 45,
                "color": "white",
                "font": "Arial-Bold",
                "stroke_color": "black",
                "stroke_width": 3,
                "bg_color": None,
                "position": "bottom"
            },
            "neon": {
                "fontsize": 48,
                "color": "cyan",
                "font": "Arial-Bold",
                "stroke_color": "blue",
                "stroke_width": 4,
                "bg_color": None,
                "position": "bottom"
            },
            "minimal": {
                "fontsize": 40,
                "color": "white",
                "font": "Arial",
                "stroke_color": "gray",
                "stroke_width": 2,
                "bg_color": None,
                "position": "bottom"
            },
            "bold": {
                "fontsize": 52,
                "color": "yellow",
                "font": "Arial-Bold",
                "stroke_color": "black",
                "stroke_width": 5,
                "bg_color": None,
                "position": "bottom"
            },
            "bubble": {
                "fontsize": 42,
                "color": "black",
                "font": "Arial-Bold",
                "stroke_color": "white",
                "stroke_width": 2,
                "bg_color": "white",
                "position": "bottom"
            }
        }
        
    def _initialize_whisper(self):
        """Initialize Whisper model with error handling"""
        try:
            model_size = self.config.get("whisper_model", "base")
            logger.info(f"Loading Whisper model: {model_size}")
            self.whisper_model = whisper.load_model(model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.info("Whisper will be disabled. Captions will not be generated.")
            self.whisper_model = None
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file with improved defaults"""
        default_config = {
            "youtube_api_credentials": "credentials.json",
            "youtube_api_token": "token.json",
            "whisper_model": "base",  # Options: tiny, base, small, medium, large
            "output_quality": "720p",
            "shorts_duration": {"min": 30, "max": 60},
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
    
    def authenticate_youtube(self) -> bool:
        """Authenticate with YouTube API with improved error handling"""
        if not YOUTUBE_API_AVAILABLE:
            logger.error("YouTube API libraries not available")
            return False
            
        try:
            creds = None
            token_path = self.config["youtube_api_token"]
            
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, self.config["scopes"])
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.config["youtube_api_credentials"]):
                        logger.error(f"Credentials file not found: {self.config['youtube_api_credentials']}")
                        return False
                        
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.config["youtube_api_credentials"], self.config["scopes"])
                    creds = flow.run_local_server(port=0)
                
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            self.youtube_service = build('youtube', 'v3', credentials=creds)
            logger.info("YouTube API authenticated successfully")
            return True
            
        except Exception as e:
            logger.error(f"YouTube authentication failed: {e}")
            return False
    
    def download_video(self, youtube_url: str) -> Optional[str]:
        """Download video from YouTube with improved error handling"""
        try:
            # Validate URL
            if not self._is_valid_youtube_url(youtube_url):
                raise ValueError("Invalid YouTube URL")
            
            ydl_opts = {
                'format': 'best[height<=1080][filesize<500M]/best[height<=720]',
                'outtmpl': str(self.temp_dir / 'source_%(title)s_%(id)s.%(ext)s'),
                'noplaylist': True,
                'extractaudio': False,
                'writeinfojson': True,
                'writethumbnail': True,
                'ignoreerrors': True,
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
                
                logger.info(f"Downloaded video: {filename} ({file_size:.1f}MB)")
                return filename
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None
    
    def _is_valid_youtube_url(self, url: str) -> bool:
        """Validate YouTube URL"""
        youtube_domains = ["youtube.com", "youtu.be", "m.youtube.com"]
        return any(domain in url for domain in youtube_domains)
    
    def extract_audio_and_transcribe(self, video_path: str) -> List[Dict]:
        """Extract audio and generate transcription with timestamps"""
        if not self.whisper_model:
            logger.warning("Whisper model not available, skipping transcription")
            return []
            
        try:
            logger.info("Extracting audio and transcribing...")
            
            # Load video and extract audio
            video = VideoFileClip(video_path)
            if not video.audio:
                logger.warning("No audio track found in video")
                video.close()
                return []
            
            audio_path = str(self.temp_dir / "temp_audio.wav")
            video.audio.write_audiofile(
                audio_path, 
                verbose=False, 
                logger=None,
                temp_audiofile=str(self.temp_dir / "temp_audio_temp.wav")
            )
            video.close()
            
            # Transcribe with Whisper
            logger.info("Transcribing audio...")
            result = self.whisper_model.transcribe(
                audio_path, 
                word_timestamps=True,
                language="en"  # Specify language for better accuracy
            )
            
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Format segments for subtitle creation
            segments = []
            for segment in result["segments"]:
                text = segment["text"].strip()
                if len(text) > 3:  # Filter out very short segments
                    segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": text
                    })
            
            logger.info(f"Transcribed {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def create_subtitle_clips(self, segments: List[Dict], style: str, video_size: Tuple[int, int]) -> List:
        """Create subtitle clips with improved styling"""
        if not segments:
            return []
            
        subtitle_clips = []
        style_config = self.caption_styles.get(style, self.caption_styles["modern"])
        
        try:
            for segment in segments:
                # Break long text into multiple lines
                text = self._format_text_for_display(segment["text"], max_chars_per_line=40)
                
                # Create text clip with background if specified
                txt_clip = TextClip(
                    text,
                    fontsize=style_config["fontsize"],
                    color=style_config["color"],
                    font=style_config["font"],
                    stroke_color=style_config["stroke_color"],
                    stroke_width=style_config["stroke_width"],
                    method="caption",
                    size=(video_size[0] * 0.9, None),
                    align='center'
                ).set_start(segment["start"]).set_duration(segment["end"] - segment["start"])
                
                # Position subtitle
                if style_config["position"] == "bottom":
                    txt_clip = txt_clip.set_position(("center", video_size[1] * 0.8))
                elif style_config["position"] == "center":
                    txt_clip = txt_clip.set_position("center")
                else:
                    txt_clip = txt_clip.set_position(("center", video_size[1] * 0.2))
                
                subtitle_clips.append(txt_clip)
            
            logger.info(f"Created {len(subtitle_clips)} subtitle clips")
            return subtitle_clips
            
        except Exception as e:
            logger.error(f"Error creating subtitle clips: {e}")
            return []
    
    def _format_text_for_display(self, text: str, max_chars_per_line: int = 40) -> str:
        """Format text for better display on video"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars_per_line:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)
    
    def get_background_music(self) -> Optional[str]:
        """Get random background music file with improved selection"""
        music_dir = Path(self.config["background_music_dir"])
        if not music_dir.exists():
            logger.info("Background music directory not found, creating it...")
            music_dir.mkdir(exist_ok=True)
            return None
        
        # Look for various audio formats
        music_files = []
        for ext in self.config["audio_formats"]:
            music_files.extend(list(music_dir.glob(f"*.{ext}")))
        
        if not music_files:
            logger.info("No music files found in background music directory")
            return None
        
        selected_music = str(random.choice(music_files))
        logger.info(f"Selected background music: {os.path.basename(selected_music)}")
        return selected_music
    
    def create_shorts_from_video(self, video_path: str, segments: List[Dict], 
                                caption_style: str, num_shorts: int = 3) -> List[str]:
        """Create multiple short videos with improved processing"""
        shorts_paths = []
        
        try:
            video = VideoFileClip(video_path)
            total_duration = video.duration
            
            if total_duration < 30:
                logger.warning("Video too short to create meaningful shorts")
                video.close()
                return []
            
            logger.info(f"Creating {num_shorts} shorts from video (duration: {total_duration:.1f}s)")
            
            # Find good segments for shorts
            interesting_segments = self._find_interesting_segments(segments, total_duration)
            
            for i in range(min(num_shorts, len(interesting_segments))):
                try:
                    short_path = self._create_single_short(
                        video, interesting_segments[i], segments, caption_style, i + 1
                    )
                    if short_path:
                        shorts_paths.append(short_path)
                        
                except Exception as e:
                    logger.error(f"Error creating short {i+1}: {e}")
                    continue
            
            video.close()
            logger.info(f"Successfully created {len(shorts_paths)} shorts")
            return shorts_paths
            
        except Exception as e:
            logger.error(f"Error in create_shorts_from_video: {e}")
            return []
    
    def _find_interesting_segments(self, segments: List[Dict], total_duration: float) -> List[Dict]:
        """Find interesting segments for creating shorts"""
        if not segments:
            # If no segments, create random time intervals
            num_segments = 3
            segment_duration = 45
            segments_info = []
            
            for i in range(num_segments):
                start_time = (total_duration / num_segments) * i
                end_time = min(start_time + segment_duration, total_duration)
                if end_time - start_time >= 30:  # Minimum 30 seconds
                    segments_info.append({
                        "start": start_time,
                        "end": end_time,
                        "text": f"Segment {i+1}"
                    })
            return segments_info
        
        # Filter segments by length and content quality
        interesting = []
        for segment in segments:
            text_length = len(segment["text"])
            duration = segment["end"] - segment["start"]
            
            # Score based on text length and duration
            score = text_length * 0.1 + duration * 0.5
            
            if score > 5 and duration > 3:  # Minimum thresholds
                interesting.append({**segment, "score": score})
        
        # Sort by score and return top segments
        interesting.sort(key=lambda x: x["score"], reverse=True)
        return interesting[:10]  # Return top 10 segments
    
    def _create_single_short(self, video: VideoFileClip, segment_info: Dict, 
                           all_segments: List[Dict], caption_style: str, short_number: int) -> Optional[str]:
        """Create a single short video"""
        try:
            # Calculate timing
            short_duration = random.randint(
                self.config["shorts_duration"]["min"], 
                self.config["shorts_duration"]["max"]
            )
            
            start_time = max(0, segment_info["start"] - 2)
            end_time = min(video.duration, start_time + short_duration)
            
            # Extract clip
            short_clip = video.subclip(start_time, end_time)
            
            # Resize to 9:16 aspect ratio (1080x1920)
            short_clip = self._resize_to_shorts_format(short_clip)
            
            # Get relevant subtitles
            relevant_segments = self._get_relevant_segments(all_segments, start_time, end_time)
            
            # Create subtitle clips
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
            
            # Export
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = str(self.output_dir / f"short_{short_number}_{timestamp}.mp4")
            
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / f'temp-audio-{short_number}.m4a'),
                remove_temp=True,
                verbose=False,
                logger=None,
                fps=30
            )
            
            final_clip.close()
            
            # Verify file was created and has reasonable size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                logger.info(f"Created short video: {output_path}")
                return output_path
            else:
                logger.error(f"Failed to create valid short video: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating single short: {e}")
            return None
    
    def _resize_to_shorts_format(self, clip: VideoFileClip) -> VideoFileClip:
        """Resize video to 9:16 aspect ratio for YouTube Shorts"""
        target_width, target_height = 1080, 1920
        
        # Calculate current aspect ratio
        current_ratio = clip.w / clip.h
        target_ratio = target_width / target_height
        
        if current_ratio > target_ratio:
            # Video is wider, crop width
            new_width = int(clip.h * target_ratio)
            clip = clip.crop(x_center=clip.w/2, width=new_width)
        else:
            # Video is taller, crop height
            new_height = int(clip.w / target_ratio)
            clip = clip.crop(y_center=clip.h/2, height=new_height)
        
        # Resize to target dimensions
        clip = clip.resize((target_width, target_height))
        
        return clip
    
    def _get_relevant_segments(self, segments: List[Dict], start_time: float, end_time: float) -> List[Dict]:
        """Get segments relevant to the time range and adjust timing"""
        relevant = []
        
        for segment in segments:
            seg_start, seg_end = segment["start"], segment["end"]
            
            # Check if segment overlaps with our time range
            if seg_end > start_time and seg_start < end_time:
                # Adjust timing relative to clip start
                adjusted_start = max(0, seg_start - start_time)
                adjusted_end = min(end_time - start_time, seg_end - start_time)
                
                if adjusted_end > adjusted_start:
                    relevant.append({
                        "start": adjusted_start,
                        "end": adjusted_end,
                        "text": segment["text"]
                    })
        
        return relevant
    
    def _add_background_music(self, clip: VideoFileClip) -> VideoFileClip:
        """Add background music to video clip"""
        music_path = self.get_background_music()
        if not music_path:
            return clip
            
        try:
            bg_music = AudioFileClip(music_path)
            
            # Loop music if needed
            if bg_music.duration < clip.duration:
                loops_needed = int(clip.duration / bg_music.duration) + 1
                bg_music = concatenate_audioclips([bg_music] * loops_needed)
            
            # Trim to clip duration
            bg_music = bg_music.subclip(0, clip.duration)
            bg_music = bg_music.volumex(0.2)  # Lower volume for background
            
            # Mix with original audio
            if clip.audio:
                final_audio = CompositeAudioClip([clip.audio.volumex(0.8), bg_music])
            else:
                final_audio = bg_music
            
            clip = clip.set_audio(final_audio)
            
        except Exception as e:
            logger.error(f"Error adding background music: {e}")
        
        return clip
    
    def upload_to_youtube(self, video_path: str, title: str, description: str, 
                         tags: List[str], scheduled_time: Optional[datetime] = None) -> Optional[str]:
        """Upload video to YouTube with improved error handling"""
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
            
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            logger.info(f"Uploading video: {os.path.basename(video_path)} ({file_size:.1f}MB)")
            
            body = {
                'snippet': {
                    'title': title[:100],  # YouTube title limit
                    'description': description[:5000],  # YouTube description limit
                    'tags': tags[:15],  # YouTube tags limit
                    'categoryId': '22',  # People & Blogs
                    'defaultLanguage': 'en',
                    'defaultAudioLanguage': 'en'
                },
                'status': {
                    'privacyStatus': 'private' if scheduled_time else 'public',
                    'selfDeclaredMadeForKids': False,
                }
            }
            
            if scheduled_time:
                body['status']['publishAt'] = scheduled_time.isoformat() + 'Z'
            
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
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    async def process_video_url(self, youtube_url: str, caption_style: str = "modern", 
                               num_shorts: int = 3, auto_upload: bool = False) -> Dict:
        """Main processing function with comprehensive error handling"""
        result = {
            "success": False,
            "original_video": None,
            "shorts_created": 0,
            "shorts_paths": [],
            "segments": 0,
            "uploads_scheduled": False,
            "error": None
        }
        
        try:
            logger.info(f"Starting video processing: {youtube_url}")
            
            # Step 1: Download video
            video_path = self.download_video(youtube_url)
            if not video_path:
                result["error"] = "Failed to download video"
                return result
                
            result["original_video"] = video_path
            
            # Step 2: Extract audio and transcribe
            segments = self.extract_audio_and_transcribe(video_path)
            result["segments"] = len(segments)
            
            # Step 3: Create shorts
            shorts_paths = self.create_shorts_from_video(
                video_path, segments, caption_style, num_shorts
            )
            
            result["shorts_created"] = len(shorts_paths)
            result["shorts_paths"] = shorts_paths
            
            if not shorts_paths:
                result["error"] = "Failed to create any shorts"
                return result
            
            # Step 4: Schedule uploads if requested
            if auto_upload and shorts_paths and YOUTUBE_API_AVAILABLE:
                try:
                    # Extract video info for upload metadata
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(youtube_url, download=False)
                        video_title = info.get('title', 'Untitled')[:50]  # Truncate title
                    
                    base_title = f"Shorts: {video_title}"
                    description = f"Automatically generated shorts from: {youtube_url}\n\nCreated with YouTube Shorts Automator"
                    tags = ["shorts", "youtube", "automated", "viral", "content"]
                    
                    self.schedule_uploads(shorts_paths, base_title, description, tags)
                    result["uploads_scheduled"] = True
                    
                except Exception as e:
                    logger.error(f"Error scheduling uploads: {e}")
                    result["error"] = f"Shorts created but upload scheduling failed: {e}"
            
            result["success"] = True
            logger.info(f"Processing completed successfully: {result['shorts_created']} shorts created")
            
        except Exception as e:
            logger.error(f"Error in process_video_url: {e}")
            logger.error(traceback.format_exc())
            result["error"] = str(e)
        
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()
        
        return result
    
    def schedule_uploads(self, video_paths: List[str], base_title: str, 
                        base_description: str, tags: List[str]):
        """Schedule multiple videos for upload with improved logic"""
        if not video_paths:
            return
            
        upload_times = self.config["upload_schedule"]["times"]
        
        for i, video_path in enumerate(video_paths):
            try:
                # Calculate upload time
                days_ahead = i // len(upload_times)
                time_index = i % len(upload_times)
                
                upload_time = datetime.now() + timedelta(days=days_ahead)
                hour, minute = map(int, upload_times[time_index].split(':'))
                upload_time = upload_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # Ensure future time
                if upload_time <= datetime.now():
                    upload_time += timedelta(days=1)
                
                title = f"{base_title} - Part {i+1}"
                
                # Schedule the upload
                schedule.every().day.at(upload_times[time_index]).do(
                    self._scheduled_upload_wrapper,
                    video_path=video_path,
                    title=title,
                    description=base_description,
                    tags=tags,
                    scheduled_time=upload_time
                )
                
                logger.info(f"Scheduled upload for {os.path.basename(video_path)} at {upload_time}")
                
            except Exception as e:
                logger.error(f"Error scheduling upload for video {i+1}: {e}")
    
    def _scheduled_upload_wrapper(self, video_path: str, title: str, description: str, 
                                 tags: List[str], scheduled_time: datetime):
        """Wrapper for scheduled uploads with error handling"""
        try:
            video_id = self.upload_to_youtube(video_path, title, description, tags, scheduled_time)
            if video_id:
                logger.info(f"Successfully uploaded scheduled video: {video_id}")
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

if __name__ == "__main__":
    # Check for special commands
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