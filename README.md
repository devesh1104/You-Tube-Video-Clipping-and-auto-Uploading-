# YouTube Shorts Automator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)]()

An intelligent Python tool that automatically converts long-form YouTube videos into engaging short-form content (YouTube Shorts) with AI-powered transcription, custom captions, and automated scheduling.

## 🌟 Features

### Core Functionality
- **🎥 Video Processing**: Download and process YouTube videos automatically
- **🤖 AI Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **✂️ Smart Segmentation**: Intelligently identifies and extracts engaging segments
- **📱 Shorts Optimization**: Automatically resizes to 9:16 aspect ratio (1080x1920)
- **🎨 Custom Captions**: 5 built-in caption styles with customizable appearance
- **🎵 Background Music**: Automatic background music integration
- **📅 Automated Scheduling**: Schedule uploads across multiple time slots
- **🔄 Batch Processing**: Create multiple shorts from a single video

### Caption Styles
- **Modern**: Clean white text with black stroke
- **Neon**: Vibrant cyan with blue glow effect
- **Minimal**: Simple white text with gray stroke
- **Bold**: Eye-catching yellow with heavy black outline
- **Bubble**: Text with white background bubble

### Advanced Features
- **Multi-format Support**: MP4, WebM, MKV, AVI video formats
- **Audio Format Support**: MP3, WAV, FLAC, AAC for background music
- **Concurrent Processing**: Process multiple videos simultaneously
- **Error Recovery**: Robust error handling and logging
- **Interactive & CLI Modes**: Flexible usage options
- **API Wrapper**: Easy integration with other applications

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- FFmpeg installed and in PATH
- YouTube API credentials (optional, for uploads)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/youtube-shorts-automator.git
cd youtube-shorts-automator
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Create required directories**
```bash
mkdir output temp music
```

4. **Run the setup**
```bash
python YouTubeShortsAutomator.py config
```

## 📋 Requirements

Create a `requirements.txt` file with these dependencies:

```
openai-whisper>=20231117
yt-dlp>=2023.12.30
moviepy>=1.0.3
schedule>=1.2.0
requests>=2.31.0
google-auth>=2.23.0
google-auth-oauthlib>=1.1.0
google-auth-httplib2>=0.2.0
google-api-python-client>=2.108.0
```

## ⚙️ Configuration

### Basic Configuration

The tool automatically creates a `config.json` file with default settings:

```json
{
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
  "concurrent_processing": 2
}
```

### YouTube API Setup (Optional)

For automatic uploads, you'll need YouTube API credentials:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable YouTube Data API v3
4. Create credentials (OAuth 2.0 Client ID)
5. Download the credentials as `credentials.json`
6. Place the file in the project root directory

### Whisper Model Options

Choose the appropriate Whisper model based on your needs:
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy
- `medium`: High accuracy
- `large`: Best accuracy, slowest

## 🎯 Usage

### Interactive Mode (Recommended for beginners)

```bash
python YouTubeShortsAutomator.py
```

Follow the prompts to:
1. Enter YouTube URL
2. Select caption style
3. Choose number of shorts to create
4. Enable/disable auto-upload

### Command Line Mode

```bash
# Basic usage
python YouTubeShortsAutomator.py "https://youtube.com/watch?v=VIDEO_ID"

# With custom options
python YouTubeShortsAutomator.py "https://youtube.com/watch?v=VIDEO_ID" "neon" 5 true
```

**Parameters:**
- `URL`: YouTube video URL
- `style`: Caption style (modern/neon/minimal/bold/bubble)
- `count`: Number of shorts to create (1-10)
- `upload`: Auto-upload to YouTube (true/false)

### API Usage

```python
from YouTubeShortsAutomator import ShortsAutomatorAPI
import asyncio

api = ShortsAutomatorAPI()

# Create shorts
result = await api.create_shorts({
    "youtube_url": "https://youtube.com/watch?v=VIDEO_ID",
    "caption_style": "modern",
    "num_shorts": 3,
    "auto_upload": False
})

print(result)
```

### Helper Commands

```bash
# Test configuration
python YouTubeShortsAutomator.py test

# Create sample config
python YouTubeShortsAutomator.py config

# Show help
python YouTubeShortsAutomator.py help
```

## 📁 Project Structure

```
youtube-shorts-automator/
├── YouTubeShortsAutomator.py    # Main application
├── config.json                  # Configuration file
├── credentials.json             # YouTube API credentials (you create)
├── token.json                   # YouTube API token (auto-generated)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── output/                      # Generated shorts videos
├── temp/                        # Temporary processing files
├── music/                       # Background music files
└── logs/                        # Application logs
```

## 🎵 Adding Background Music

1. Create a `music` directory in the project root
2. Add your music files (MP3, WAV, FLAC, AAC)
3. The tool will randomly select music for each short
4. Ensure you have rights to use the music

## 📊 Output Information

Each processing session generates:
- **Short videos**: Saved in `output/` directory
- **Processing logs**: Detailed logs in `youtube_shorts_automator.log`
- **Metadata**: Information about segments and processing results

## 🔧 Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Whisper installation issues:**
```bash
pip install --upgrade openai-whisper
# or
pip install git+https://github.com/openai/whisper.git
```

**YouTube API errors:**
- Verify credentials.json is in the correct location
- Check API quotas in Google Cloud Console
- Ensure YouTube Data API v3 is enabled

**Video download failures:**
- Check if the video is publicly accessible
- Verify the URL format
- Some videos may be geo-restricted

### Performance Optimization

- Use smaller Whisper models (`tiny` or `base`) for faster processing
- Reduce `concurrent_processing` if experiencing memory issues
- Keep videos under 500MB for optimal processing
- Use SSD storage for temp directory if possible

## 📈 Advanced Usage

### Batch Processing Script

```python
import asyncio
from YouTubeShortsAutomator import YouTubeShortsAutomator

async def batch_process():
    automator = YouTubeShortsAutomator()
    
    urls = [
        "https://youtube.com/watch?v=VIDEO_ID_1",
        "https://youtube.com/watch?v=VIDEO_ID_2",
        "https://youtube.com/watch?v=VIDEO_ID_3"
    ]
    
    for url in urls:
        result = await automator.process_video_url(url, "modern", 2, False)
        print(f"Processed {url}: {result['shorts_created']} shorts created")

asyncio.run(batch_process())
```

### Custom Caption Styling

Modify the `caption_styles` dictionary in the code to create custom styles:

```python
self.caption_styles["custom"] = {
    "fontsize": 50,
    "color": "red",
    "font": "Arial-Bold",
    "stroke_color": "white",
    "stroke_width": 3,
    "bg_color": None,
    "position": "center"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/youtube-shorts-automator.git
cd youtube-shorts-automator
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If you create this for dev dependencies
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

- **Copyright**: Ensure you have rights to use the source videos and music
- **Fair Use**: This tool is for educational and fair use purposes
- **Rate Limits**: Respect YouTube's API rate limits and terms of service
- **Content Policy**: Generated content must comply with YouTube's community guidelines

## 🔗 Related Projects

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube video downloader
- [Whisper](https://github.com/openai/whisper) - OpenAI's speech recognition
- [MoviePy](https://github.com/Zulko/moviepy) - Video editing library

## 📞 Support

- **Issues**: Report bugs and request features through [GitHub Issues](https://github.com/yourusername/youtube-shorts-automator/issues)
- **Discussions**: Join community discussions in [GitHub Discussions](https://github.com/yourusername/youtube-shorts-automator/discussions)
- **Wiki**: Check the [project wiki](https://github.com/yourusername/youtube-shorts-automator/wiki) for detailed guides

## 🎉 Acknowledgments

- OpenAI team for the Whisper speech recognition model
- MoviePy developers for the excellent video processing library
- yt-dlp community for the robust YouTube downloader
- All contributors who help improve this project

---

**Made with ❤️ for content creators who want to maximize their reach with engaging short-form content.**
