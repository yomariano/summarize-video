#!/usr/bin/env python3
"""
Simple example of using the video summarizer.
"""

import os
import sys

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.summarize_video import (
    extract_audio_and_transcribe,
    extract_key_frames,
    summarize_with_gemma
)
from src.utils import cleanup_temp_files, get_video_info


def simple_summary(video_path, output_dir="frames", model="google/gemma-27b-multimodal"):
    """
    Create a simple summary of a video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        model: Gemma model name/path
    """
    # Print video info
    video_info = get_video_info(video_path)
    if video_info:
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        if video_info['duration']:
            minutes, seconds = divmod(video_info['duration'], 60)
            print(f"Duration: {int(minutes)}m {int(seconds)}s")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract audio and transcribe
    print("Transcribing audio...")
    transcript = extract_audio_and_transcribe(video_path, whisper_model_size="base")
    
    # Extract frames
    print("Extracting frames...")
    frame_paths = extract_key_frames(video_path, output_dir, frame_rate=0.5)
    
    # Generate summary
    print("Generating summary...")
    summary = summarize_with_gemma(transcript, frame_paths, model, max_frames=5)
    
    # Print and save summary
    print("\nSummary:")
    print(summary)
    
    # Save summary to file
    summary_file = f"{os.path.splitext(os.path.basename(video_path))[0]}_summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to {summary_file}")
    
    # Clean up
    cleanup_temp_files(output_dir)
    print("Temporary files cleaned up")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_summary.py <video_path>")
        sys.exit(1)
    
    simple_summary(sys.argv[1]) 