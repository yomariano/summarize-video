# Video Summarizer with Gemma 3

This project creates summaries of videos using Google's Gemma 3 multimodal model. It extracts key frames and transcribes audio from the video, then uses Gemma 3 to generate a concise summary based on this information.

## How It Works

1. Audio Extraction & Transcription: Uses Whisper to convert video audio to text
2. Key Frame Extraction: Uses FFmpeg to extract significant images from the video
3. Processing with Gemma 3: Feeds the transcript and images into Gemma 3 for summarization
4. Summary Output: Generates and saves a textual summary

## Requirements

- Python 3.8 or higher
- FFmpeg installed on your system
- A GPU is recommended for optimal performance with Gemma 3
- Access to Gemma 3 multimodal model

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/summarize-video.git
   cd summarize-video
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

4. Check if all dependencies are installed:
   ```bash
   python src/summarize_video.py --check-deps
   ```

## Usage

Basic usage:

```bash
python src/summarize_video.py path/to/your/video.mp4
```

Advanced options:

```bash
python src/summarize_video.py path/to/your/video.mp4 \
  --model "google/gemma-3-12b-it" \
  --frame-rate 0.5 \
  --max-frames 5 \
  --whisper-model medium \
  --max-tokens 300 \
  --cleanup
```

### Options

- `--output-dir`: Directory to save extracted frames (default: "frames")
- `--model`: Gemma model name/path (default: "google/gemma-3-4b-it")
- `--frame-rate`: Frame extraction rate in fps (default: 1)
- `--max-frames`: Maximum frames to process (default: 5)
- `--whisper-model`: Whisper model size (tiny, base, small, medium, large) (default: "base")
- `--max-tokens`: Maximum tokens to generate for summary (default: 500)
- `--cleanup`: Remove extracted frames after processing
- `--check-deps`: Check dependencies and exit

## Notes on Gemma 3

Gemma 3 is Google's multimodal model that can process both text and images. This script uses the instruction-tuned version of Gemma 3 by default (google/gemma-3-4b-it), but you can also use other variants like:

- google/gemma-3-4b-pt (pre-trained, 4B parameters)
- google/gemma-3-12b-it (instruction-tuned, 12B parameters)
- google/gemma-3-27b-it (instruction-tuned, 27B parameters)

The instruction-tuned models tend to produce better summaries.

## License

[MIT License](LICENSE)

## Acknowledgments

- [Whisper](https://github.com/openai/whisper) for audio transcription
- [FFmpeg](https://ffmpeg.org/) for video frame extraction
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for model integration
- [Google Gemma 3](https://ai.google.dev/gemma) for the multimodal model 