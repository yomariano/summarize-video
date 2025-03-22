import os
import subprocess
from typing import List, Dict, Any, Optional
import shutil


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are installed.
    
    Returns:
        Dict with dependency status
    """
    dependencies = {
        "ffmpeg": False,
        "whisper": False,
        "transformers": False,
        "torch": False,
    }
    
    # Check FFmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        dependencies["ffmpeg"] = True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check Python packages
    try:
        import whisper
        dependencies["whisper"] = True
    except ImportError:
        pass
    
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        pass
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    return dependencies


def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a video file using FFmpeg.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information or None if failed
    """
    if not os.path.isfile(video_path):
        return None
    
    try:
        # Get video information using FFprobe
        result = subprocess.run(
            [
                "ffprobe", 
                "-v", "error", 
                "-select_streams", "v:0", 
                "-show_entries", "stream=width,height,duration,r_frame_rate", 
                "-of", "csv=p=0", 
                video_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output
        values = result.stdout.strip().split(',')
        if len(values) >= 4:
            # Parse frame rate (can be in format "num/den")
            frame_rate = values[3]
            if '/' in frame_rate:
                num, den = map(float, frame_rate.split('/'))
                frame_rate = num / den if den != 0 else 0
            else:
                frame_rate = float(frame_rate)
            
            return {
                "width": int(values[0]),
                "height": int(values[1]),
                "duration": float(values[2]) if values[2] != "N/A" else None,
                "frame_rate": frame_rate
            }
    except (subprocess.SubprocessError, ValueError):
        pass
    
    return None


def cleanup_temp_files(frame_dir: str) -> None:
    """
    Remove temporary files and directories.
    
    Args:
        frame_dir: Directory containing frame images
    """
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
    
    if os.path.exists(frame_dir) and os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)


def get_whisper_model_sizes() -> List[str]:
    """
    Get available Whisper model sizes.
    
    Returns:
        List of available model sizes
    """
    return ["tiny", "base", "small", "medium", "large"] 