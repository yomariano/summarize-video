import os
import yt_dlp
from typing import Dict, Any, Optional
from pathlib import Path

def download_youtube_video(url: str, output_dir: str = "downloads") -> Optional[str]:
    """
    Download a YouTube video and return the path to the downloaded file.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save the downloaded video
        
    Returns:
        Path to the downloaded video file or None if download failed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output filename template
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    # yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'restrictfilenames': True,
        'noplaylist': True,
        'nocheckcertificate': True,
        'ignoreerrors': False,
        'logtostderr': False,
        'quiet': False,
        'no_warnings': False,
        'default_search': 'auto',
        'source_address': '0.0.0.0',
    }
    
    try:
        # Extract info first to get the filename
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return None
                
            # Download the video
            ydl.download([url])
            
            # Get the downloaded file path
            video_title = info.get('title', 'video')
            video_ext = info.get('ext', 'mp4')
            # Clean filename (yt-dlp does this internally with restrictfilenames=True)
            video_title = ydl.sanitize_info(info)['title']
            
            # Construct the expected file path
            video_path = os.path.join(output_dir, f"{video_title}.{video_ext}")
            
            # Double-check if file exists
            if os.path.isfile(video_path):
                return video_path
            
            # If exact file not found, try to find a matching file
            for file in os.listdir(output_dir):
                if file.startswith(video_title) and file.endswith(f".{video_ext}"):
                    return os.path.join(output_dir, file)
            
            # Check if any file was recently created in the output directory
            files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
            if files:
                # Sort by creation time, newest first
                files.sort(key=os.path.getctime, reverse=True)
                return files[0]
                
            return None
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return None

def get_video_info(url: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a YouTube video.
    
    Args:
        url: YouTube URL
        
    Returns:
        Dictionary with video information or None if failed
    """
    ydl_opts = {
        'format': 'best',
        'noplaylist': True,
        'nocheckcertificate': True,
        'ignoreerrors': False,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return None
                
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', None),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'upload_date': info.get('upload_date', None),
            }
    except Exception as e:
        print(f"Error getting YouTube video info: {e}")
        return None 