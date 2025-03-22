import os
import subprocess
import argparse
from pathlib import Path
import sys
import torch
from PIL import Image

# Add local imports
from utils import (
    check_dependencies,
    get_video_info,
    cleanup_temp_files,
    get_whisper_model_sizes,
)

try:
    import whisper
except ImportError:
    print("Whisper not installed. Run: pip install openai-whisper")
    exit(1)

try:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
except ImportError:
    print("Transformers not installed or version is too old.")
    print("Run: pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3")
    exit(1)


def extract_audio_and_transcribe(video_path, whisper_model_size="base"):
    """
    Extract audio from video and transcribe it using Whisper.
    
    Args:
        video_path: Path to the video file
        whisper_model_size: Size of Whisper model to use
        
    Returns:
        transcript: Transcribed text
    """
    audio_path = "temp_audio.wav"
    
    # Extract audio using FFmpeg
    print(f"Extracting audio from {video_path}...")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    
    # Load Whisper model for transcription
    print(f"Transcribing audio with Whisper ({whisper_model_size})...")
    whisper_model = whisper.load_model(whisper_model_size)
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"]
    
    # Clean up temporary audio file
    os.remove(audio_path)
    return transcript


def extract_key_frames(video_path, output_dir, frame_rate=1):
    """
    Extract key frames from video using FFmpeg.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_rate: Number of frames to extract per second
        
    Returns:
        frame_paths: List of paths to extracted frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames using FFmpeg
    print(f"Extracting frames at {frame_rate} fps...")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"fps={frame_rate}", f"{output_dir}/frame_%04d.jpg", "-y"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    
    # Get list of extracted frames
    frames = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
    return sorted(frames)


def summarize_with_gemma(transcript, frame_paths, model_name, max_frames=5, max_tokens=150):
    """
    Summarize video content using Gemma 3 multimodal model.
    
    Args:
        transcript: Transcribed text from video
        frame_paths: List of paths to key frames
        model_name: Name/path of the Gemma model to use
        max_frames: Maximum number of frames to include
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        summary: Generated summary text
    """
    print(f"Loading Gemma model: {model_name}...")
    
    # Load model, processor, and move to device
    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Limit number of frames for efficiency
    selected_frames = frame_paths[:max_frames]
    print(f"Processing {len(selected_frames)} frames...")
    
    # Load images
    images = []
    for frame_path in selected_frames:
        image = Image.open(frame_path).convert("RGB")
        images.append(image)
    
    # Create input prompt text with correct image tokens
    # Gemma 3 expects <start_of_image> token for each image
    prompt_text = f"Summarize this video based on the transcript and key frames.\n\nTranscript: {transcript}\n\nKey frames:"
    for _ in range(len(images)):
        prompt_text += f" {processor.image_token}"
    
    # Process inputs with processor
    inputs = processor(
        text=prompt_text, 
        images=images,
        return_tensors="pt"
    ).to(device)
    
    # Generate summary
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Get the length of input tokens to extract only the generated text
    input_length = inputs.input_ids.shape[1]
    
    # Decode only the newly generated tokens
    summary = processor.decode(output_ids[0][input_length:], skip_special_tokens=True)
    
    return summary.strip()


def main():
    parser = argparse.ArgumentParser(description="Summarize video using Gemma 3 multimodal model")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output-dir", default="frames", help="Directory to save extracted frames")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Gemma model name/path")
    parser.add_argument("--frame-rate", type=float, default=1, help="Frame extraction rate (fps)")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum frames to process")
    parser.add_argument("--whisper-model", default="base", 
                        choices=get_whisper_model_sizes(),
                        help="Whisper model size")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument("--cleanup", action="store_true", help="Remove extracted frames after processing")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        deps = check_dependencies()
        print("Dependency Status:")
        for dep, installed in deps.items():
            print(f"  {dep}: {'✓' if installed else '✗'}")
        
        missing = [dep for dep, installed in deps.items() if not installed]
        if missing:
            print("\nMissing dependencies:")
            for dep in missing:
                if dep == "ffmpeg":
                    print("  - FFmpeg: Install with 'brew install ffmpeg' (macOS) or 'apt install ffmpeg' (Linux)")
                elif dep == "whisper":
                    print("  - Whisper: Install with 'pip install openai-whisper'")
                elif dep == "transformers":
                    print("  - Transformers: Install with 'pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3'")
                elif dep == "torch":
                    print("  - PyTorch: Install with 'pip install torch'")
        sys.exit(0)
    
    video_path = args.video_path
    output_dir = args.output_dir
    
    # Verify video exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Get video info
    video_info = get_video_info(video_path)
    if video_info:
        print(f"Video: {Path(video_path).name}")
        print(f"  Resolution: {video_info['width']}x{video_info['height']}")
        if video_info['duration']:
            minutes, seconds = divmod(video_info['duration'], 60)
            print(f"  Duration: {int(minutes)}m {int(seconds)}s")
        print(f"  Frame rate: {video_info['frame_rate']:.2f} fps")
    
    # Extract audio and transcribe
    transcript = extract_audio_and_transcribe(video_path, args.whisper_model)
    print(f"Transcript generated ({len(transcript)} characters)")
    
    # Extract key frames
    frame_paths = extract_key_frames(video_path, output_dir, args.frame_rate)
    print(f"Extracted {len(frame_paths)} frames")
    
    # Generate summary
    summary = summarize_with_gemma(
        transcript, 
        frame_paths, 
        args.model, 
        args.max_frames,
        args.max_tokens
    )
    
    # Print summary
    print("\n=== VIDEO SUMMARY ===")
    print(summary)
    print("====================\n")
    
    # Save summary to file
    summary_file = f"{Path(video_path).stem}_summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary)
    print(f"Summary saved to {summary_file}")
    
    # Clean up frames if requested
    if args.cleanup:
        print("Cleaning up extracted frames...")
        cleanup_temp_files(output_dir)


if __name__ == "__main__":
    main() 