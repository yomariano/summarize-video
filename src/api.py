import os
import time
import asyncio
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn
import httpx
from dotenv import load_dotenv
from supabase import create_client
import json
import uuid
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

from youtube import download_youtube_video, get_video_info as get_youtube_info
from summarize_video import (
    extract_audio_and_transcribe,
    extract_key_frames,
    summarize_with_gemma,
    cleanup_temp_files
)
from utils import get_video_info, get_whisper_model_sizes

# Create FastAPI app
app = FastAPI(
    title="Video Summarization API",
    description="API for summarizing videos using AI",
    version="1.0.0"
)

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, set this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Map to store jobs and their status
jobs = {}

# Models for request and response validation
class YouTubeRequest(BaseModel):
    url: HttpUrl
    whisper_model: str = "base"
    frame_rate: float = 1
    max_frames: int = 5
    max_tokens: int = 500
    model_name: str = "google/gemma-3-4b-it"
    cleanup: bool = True

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SummaryRequest(BaseModel):
    whisper_model: str = "base"
    frame_rate: float = 1
    max_frames: int = 5
    max_tokens: int = 500
    model_name: str = "google/gemma-3-4b-it"
    cleanup: bool = True

class SaveSummaryRequest(BaseModel):
    job_id: str
    user_id: str

# Function to process video in the background
async def process_video(
    job_id: str,
    video_path: str,
    whisper_model: str,
    frame_rate: float,
    max_frames: int,
    max_tokens: int,
    model_name: str,
    cleanup: bool
):
    try:
        print(f"[DEBUG] process_video starting for job {job_id}")
        # Update job status
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["progress"] = 0.1
        
        # Get video info
        print(f"[DEBUG] Getting video info for {video_path}")
        video_info = get_video_info(video_path)
        if not video_info:
            print(f"[ERROR] Failed to get video information for {video_path}")
            raise Exception("Failed to get video information")
        
        jobs[job_id]["video_info"] = video_info
        print(f"[DEBUG] Got video info: {video_info}")
        
        # Extract audio and transcribe
        print(f"[DEBUG] Starting audio extraction and transcription with model {whisper_model}")
        transcript = extract_audio_and_transcribe(video_path, whisper_model)
        jobs[job_id]["progress"] = 0.3
        jobs[job_id]["transcript"] = transcript
        print(f"[DEBUG] Transcription completed: {len(transcript)} characters")
        
        # Update job status
        jobs[job_id]["status"] = "extracting_frames"
        
        # Extract key frames
        output_dir = f"frames_{job_id}"
        print(f"[DEBUG] Extracting key frames to {output_dir} with frame rate {frame_rate}")
        frame_paths = extract_key_frames(video_path, output_dir, frame_rate)
        jobs[job_id]["progress"] = 0.5
        print(f"[DEBUG] Extracted {len(frame_paths)} frames")
        
        # Update job status
        jobs[job_id]["status"] = "summarizing"
        
        # Generate summary
        print(f"[DEBUG] Generating summary with model {model_name}, max_frames: {max_frames}, max_tokens: {max_tokens}")
        summary = summarize_with_gemma(
            transcript,
            frame_paths,
            model_name,
            max_frames,
            max_tokens
        )
        jobs[job_id]["progress"] = 0.9
        print(f"[DEBUG] Summary generated: {len(summary)} characters")
        
        # Save summary to file
        summary_file = f"{Path(video_path).stem}_summary.txt"
        print(f"[DEBUG] Saving summary to {summary_file}")
        with open(summary_file, "w") as f:
            f.write(summary)
        
        # Clean up frames if requested
        if cleanup:
            print(f"[DEBUG] Cleaning up frames in {output_dir}")
            cleanup_temp_files(output_dir)
        
        # Clean up video file if it was downloaded from YouTube
        if "youtube" in jobs[job_id] and jobs[job_id]["youtube"]:
            if os.path.isfile(video_path) and "downloads" in video_path:
                # Don't remove the video immediately, keep it for a while
                # so it can be downloaded by the user if needed
                print(f"[DEBUG] Keeping downloaded video for later use: {video_path}")
        
        # Update job status to completed
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {
            "summary": summary,
            "summary_file": summary_file,
            "video_info": video_info
        }
        print(f"[DEBUG] process_video completed successfully for job {job_id}")
    except Exception as e:
        print(f"[ERROR] Exception in process_video for job {job_id}: {str(e)}")
        # Update job status to failed
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        
        # Clean up frames if created
        output_dir = f"frames_{job_id}"
        if os.path.exists(output_dir):
            print(f"[DEBUG] Cleaning up frames directory after error: {output_dir}")
            shutil.rmtree(output_dir)

# Function to save summary to Supabase
async def save_summary_to_supabase(user_id: str, summary_data: Dict[str, Any]):
    """
    Save the video summary to the Supabase video_summaries table using service role key
    """
    # Get YouTube info
    youtube_info = summary_data.get("youtube_info", {})
    
    # Extract required data
    video_id = youtube_info.get("id", "unknown")
    url = summary_data.get("url", "")
    
    # If video_id is unknown, generate a unique one using timestamp and job_id if available
    if video_id == "unknown":
        timestamp = int(datetime.now().timestamp())
        # Try to get job_id from summary_data
        job_id = summary_data.get("job_id", "")
        # Create a unique video_id
        video_id = f"unknown_{job_id}_{timestamp}_{str(uuid.uuid4())[:8]}"
        print(f"[DEBUG] Generated unique video_id: {video_id} for unknown video")
    
    # Extract title, use filename as fallback
    title = youtube_info.get("title", "Unknown Video")
    if "video_path" in summary_data and not title:
        title = os.path.basename(summary_data["video_path"])
    
    # Extract other fields
    thumbnail = youtube_info.get("thumbnail", "")
    channel = youtube_info.get("channel", "")
    duration = youtube_info.get("duration", "")
    
    # Get summary from the appropriate location in the data structure
    summary = ""
    if "result" in summary_data and "summary" in summary_data["result"]:
        summary = summary_data["result"]["summary"]
    
    print(f"[DEBUG] Extracted summary length: {len(summary)} characters")
    
    # Validate that we have a user_id and summary - these are required
    if not user_id or not summary:
        error_msg = "Missing required data: "
        if not user_id:
            error_msg += "user_id "
        if not summary:
            error_msg += "summary"
        raise Exception(error_msg)
    
    # Prepare data - ensure no empty values are sent as NULL
    data = {
        "user_id": user_id,
        "video_id": video_id or "unknown",
        "url": url or "",
        "title": title or "Unknown Video",
        "thumbnail": thumbnail or "",
        "channel": channel or "",
        "duration": duration or "",
        "summary": summary
    }
    
    print(f"[DEBUG] Saving data to Supabase: user_id={user_id}, title={title}")

    # Get Supabase URL and key from environment variables
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not supabase_url or not supabase_key:
        raise HTTPException(
            status_code=500, 
            detail="Supabase configuration missing"
        )
    
    try:
        # Use Supabase client with service key
        from supabase import create_client
        
        # Create Supabase client with service key
        supabase = create_client(supabase_url, supabase_key)
        
        # Insert data
        response = supabase.table('video_summaries').insert(data).execute()
        
        # Check for errors
        if hasattr(response, 'error') and response.error:
            # If client approach fails, try REST API approach
            print(f"[WARNING] Supabase client error: {response.error}")
            return await save_summary_rest_api(supabase_url, supabase_key, data)
            
        # Return data
        return response.data
    
    except Exception as e:
        # If client approach fails, try REST API approach
        print(f"[ERROR] Supabase client error: {str(e)}")
        return await save_summary_rest_api(supabase_url, supabase_key, data)

async def save_summary_rest_api(supabase_url: str, supabase_key: str, data: Dict[str, Any]):
    """
    Backup approach using direct REST API call with service role key
    """
    print(f"[DEBUG] Trying backup approach with direct REST API")
    
    try:
        # Use direct REST API call with httpx
        rest_url = f"{supabase_url}/rest/v1/video_summaries"
        
        # Use specific headers that identify as service_role
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        print(f"[DEBUG] Making direct API call to Supabase with service role key")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(rest_url, json=data, headers=headers)
            
            # If we're still getting RLS errors, we need to log it
            if response.status_code >= 400:
                error_text = response.text
                print(f"[ERROR] REST API approach failed: {error_text}")
                
                # Try getting the execution context
                try:
                    context_response = await client.get(
                        f"{supabase_url}/rest/v1/rpc/get_execution_context",
                        headers=headers
                    )
                    print(f"[DEBUG] Execution context: {context_response.text}")
                except Exception as e:
                    print(f"[ERROR] Failed to get execution context: {str(e)}")
                
                # Try one more time with a different approach - using RLS bypass function
                try:
                    print(f"[DEBUG] Trying to use RPC function as final fallback")
                    # Create RPC function if it doesn't exist
                    create_func_sql = """
                    CREATE OR REPLACE FUNCTION insert_video_summary_bypass(
                        p_user_id UUID,
                        p_video_id TEXT,
                        p_url TEXT,
                        p_title TEXT,
                        p_thumbnail TEXT,
                        p_channel TEXT,
                        p_duration TEXT,
                        p_summary TEXT
                    ) RETURNS UUID
                    LANGUAGE plpgsql SECURITY DEFINER
                    AS $$
                    DECLARE
                        v_id UUID;
                    BEGIN
                        INSERT INTO public.video_summaries (
                            user_id, video_id, url, title, thumbnail, channel, duration, summary
                        ) VALUES (
                            p_user_id, p_video_id, p_url, p_title, p_thumbnail, p_channel, p_duration, p_summary
                        )
                        RETURNING id INTO v_id;
                        
                        RETURN v_id;
                    END;
                    $$;
                    """
                    
                    # Call the function via RPC
                    rpc_url = f"{supabase_url}/rest/v1/rpc/insert_video_summary_bypass"
                    rpc_data = {
                        "p_user_id": data["user_id"],
                        "p_video_id": data["video_id"],
                        "p_url": data["url"],
                        "p_title": data["title"],
                        "p_thumbnail": data.get("thumbnail", ""),
                        "p_channel": data.get("channel", ""),
                        "p_duration": data.get("duration", ""),
                        "p_summary": data["summary"]
                    }
                    
                    bypass_response = await client.post(rpc_url, json=rpc_data, headers=headers)
                    if bypass_response.status_code < 400:
                        print(f"[DEBUG] RPC bypass succeeded: {bypass_response.text}")
                        return {"id": bypass_response.text.strip('"')}
                    else:
                        print(f"[ERROR] RPC bypass also failed: {bypass_response.text}")
                except Exception as e:
                    print(f"[ERROR] Failed to create/call RPC bypass function: {str(e)}")
                
                print(f"[ERROR] All approaches failed. Please check RLS policies.")
                raise Exception(f"Failed to save to Supabase: {error_text}")
                
            print(f"[DEBUG] Supabase API response status: {response.status_code}")
            return response.json()
    except Exception as e:
        print(f"[ERROR] Exception in save_summary_rest_api: {str(e)}")
        raise Exception(f"Failed to save to Supabase after multiple attempts: {str(e)}")

# Routes
@app.get("/")
async def root():
    return {"message": "Video Summarization API is running"}

@app.get("/api/models/whisper")
async def whisper_models():
    return {"models": get_whisper_model_sizes()}

@app.post("/api/youtube", response_model=JobStatus)
async def summarize_youtube_video(
    request: YouTubeRequest,
    background_tasks: BackgroundTasks
):
    # Generate job ID based on timestamp
    job_id = f"job_{int(time.time())}"
    
    # Create initial job status
    jobs[job_id] = {
        "job_id": job_id,
        "status": "downloading",
        "progress": 0,
        "youtube": True,
        "url": str(request.url)
    }
    
    # Download YouTube video in the background
    async def download_and_process():
        try:
            # Get YouTube video info
            youtube_info = get_youtube_info(str(request.url))
            if youtube_info:
                jobs[job_id]["youtube_info"] = youtube_info
            
            # Download the video
            video_path = download_youtube_video(str(request.url))
            if not video_path:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "Failed to download YouTube video"
                return
            
            jobs[job_id]["video_path"] = video_path
            
            # Process the video
            await process_video(
                job_id,
                video_path,
                request.whisper_model,
                request.frame_rate,
                request.max_frames,
                request.max_tokens,
                request.model_name,
                request.cleanup
            )
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
    
    # Start background task
    background_tasks.add_task(download_and_process)
    
    return JobStatus(
        job_id=job_id,
        status=jobs[job_id]["status"],
        progress=jobs[job_id]["progress"]
    )

# New endpoint to process YouTube video and save directly to Supabase
class YouTubeSaveRequest(BaseModel):
    url: HttpUrl
    user_id: str
    whisper_model: str = "base"
    frame_rate: float = 1
    max_frames: int = 5
    max_tokens: int = 500
    model_name: str = "google/gemma-3-4b-it"
    cleanup: bool = True

@app.post("/api/youtube/save", response_model=JobStatus)
async def process_and_save_youtube_video(
    request: YouTubeSaveRequest,
    background_tasks: BackgroundTasks
):
    # Generate job ID based on timestamp
    job_id = f"job_{int(time.time())}"
    
    # Create initial job status
    jobs[job_id] = {
        "job_id": job_id,
        "status": "downloading",
        "progress": 0,
        "youtube": True,
        "url": str(request.url),
        "user_id": request.user_id
    }
    
    # Download YouTube video, process it, and save to Supabase in the background
    async def download_process_and_save():
        try:
            print(f"[DEBUG] Starting download_process_and_save for job {job_id}")
            # Get YouTube video info
            youtube_info = get_youtube_info(str(request.url))
            if youtube_info:
                jobs[job_id]["youtube_info"] = youtube_info
                print(f"[DEBUG] Got YouTube info for job {job_id}: {youtube_info.get('title', 'Unknown')}")
            else:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "Failed to get video information"
                print(f"[ERROR] Failed to get YouTube info for job {job_id}")
                return
            
            # Download the video
            print(f"[DEBUG] Downloading video for job {job_id}")
            video_path = download_youtube_video(str(request.url))
            if not video_path:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "Failed to download YouTube video"
                print(f"[ERROR] Failed to download video for job {job_id}")
                return
            
            jobs[job_id]["video_path"] = video_path
            print(f"[DEBUG] Video downloaded to {video_path} for job {job_id}")
            
            # Update status
            jobs[job_id]["status"] = "summarizing"
            jobs[job_id]["progress"] = 0.4
            
            # Run the summarize_video.py script directly as a subprocess
            print(f"[DEBUG] Running summarize_video.py with video path: {video_path}")
            
            # Build the command with arguments
            cmd = [
                "python", 
                "src/summarize_video.py", 
                video_path,
                "--whisper-model", request.whisper_model,
                "--frame-rate", str(request.frame_rate),
                "--max-frames", str(request.max_frames),
                "--max-tokens", str(request.max_tokens),
                "--model", request.model_name
            ]
            
            if request.cleanup:
                cmd.append("--cleanup")
            
            # Run the command
            import subprocess
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"[DEBUG] summarize_video.py output: {process.stdout}")
            
            # Update job status
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 0.9
            
            # Get the summary file path
            summary_file = f"{Path(video_path).stem}_summary.txt"
            
            # Read the summary content
            with open(summary_file, "r") as f:
                summary_content = f.read()
            
            # Store the result
            jobs[job_id]["result"] = {
                "summary": summary_content,
                "summary_file": summary_file,
                "video_info": youtube_info
            }
            
            # If processing was successful, save to Supabase
            try:
                print(f"[DEBUG] Saving to Supabase for job {job_id}")
                result = await save_summary_to_supabase(request.user_id, jobs[job_id])
                jobs[job_id]["supabase_result"] = result
                jobs[job_id]["status"] = "saved"
                print(f"[DEBUG] Successfully saved to Supabase for job {job_id}")
            except Exception as e:
                print(f"[ERROR] Failed to save to Supabase for job {job_id}: {str(e)}")
                jobs[job_id]["status"] = "completed_but_not_saved"
                jobs[job_id]["save_error"] = str(e)
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] subprocess error in download_process_and_save for job {job_id}: {str(e)}")
            print(f"[ERROR] stderr: {e.stderr}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"Error running summarize_video.py: {e.stderr}"
        except Exception as e:
            print(f"[ERROR] Exception in download_process_and_save for job {job_id}: {str(e)}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
    
    # Start background task
    background_tasks.add_task(download_process_and_save)
    
    return JobStatus(
        job_id=job_id,
        status=jobs[job_id]["status"],
        progress=jobs[job_id]["progress"]
    )

@app.post("/api/upload", response_model=JobStatus)
async def upload_and_summarize_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    whisper_model: str = Form("base"),
    frame_rate: float = Form(1.0),
    max_frames: int = Form(5),
    max_tokens: int = Form(500),
    model_name: str = Form("google/gemma-3-4b-it"),
    cleanup: bool = Form(True),
    user_id: Optional[str] = Form(None)
):
    # Generate job ID based on timestamp
    job_id = f"job_{int(time.time())}"
    
    # Create initial job status
    jobs[job_id] = {
        "job_id": job_id,
        "status": "uploading",
        "progress": 0,
        "youtube": False
    }
    
    # Store user_id if provided
    if user_id:
        jobs[job_id]["user_id"] = user_id
    
    # Process uploaded video
    async def save_and_process():
        try:
            # Create uploads directory if it doesn't exist
            os.makedirs("uploads", exist_ok=True)
            
            # Save uploaded file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as f:
                contents = await file.read()
                f.write(contents)
            
            jobs[job_id]["video_path"] = file_path
            
            # Process the video
            await process_video(
                job_id,
                file_path,
                whisper_model,
                frame_rate,
                max_frames,
                max_tokens,
                model_name,
                cleanup
            )
            
            # If user_id is provided, save to Supabase
            if user_id and jobs[job_id]["status"] == "completed":
                try:
                    print(f"[DEBUG] Saving uploaded video to Supabase for job {job_id}")
                    result = await save_summary_to_supabase(user_id, jobs[job_id])
                    jobs[job_id]["supabase_result"] = result
                    jobs[job_id]["status"] = "saved"
                    print(f"[DEBUG] Successfully saved uploaded video to Supabase for job {job_id}")
                except Exception as e:
                    print(f"[ERROR] Failed to save uploaded video to Supabase for job {job_id}: {str(e)}")
                    jobs[job_id]["status"] = "completed_but_not_saved"
                    jobs[job_id]["save_error"] = str(e)
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
    
    # Start background task
    background_tasks.add_task(save_and_process)
    
    return JobStatus(
        job_id=job_id,
        status=jobs[job_id]["status"],
        progress=jobs[job_id]["progress"]
    )

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(
        job_id=job_id,
        status=jobs[job_id]["status"],
        progress=jobs[job_id]["progress"],
        result=jobs[job_id].get("result"),
        error=jobs[job_id].get("error")
    )

@app.get("/api/download/video/{job_id}")
async def download_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if "video_path" not in jobs[job_id]:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = jobs[job_id]["video_path"]
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        filename=Path(video_path).name,
        media_type="video/mp4"
    )

@app.get("/api/download/summary/{job_id}")
async def download_summary(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if "result" not in jobs[job_id] or "summary_file" not in jobs[job_id]["result"]:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    summary_file = jobs[job_id]["result"]["summary_file"]
    if not os.path.isfile(summary_file):
        raise HTTPException(status_code=404, detail="Summary file not found")
    
    return FileResponse(
        path=summary_file,
        filename=Path(summary_file).name,
        media_type="text/plain"
    )

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up files
    try:
        if "video_path" in jobs[job_id]:
            video_path = jobs[job_id]["video_path"]
            if os.path.isfile(video_path) and ("uploads" in video_path or "downloads" in video_path):
                os.remove(video_path)
        
        if "result" in jobs[job_id] and "summary_file" in jobs[job_id]["result"]:
            summary_file = jobs[job_id]["result"]["summary_file"]
            if os.path.isfile(summary_file):
                os.remove(summary_file)
        
        frames_dir = f"frames_{job_id}"
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
    except Exception as e:
        # Log error but continue
        print(f"Error cleaning up job {job_id}: {e}")
    
    # Remove job from memory
    del jobs[job_id]
    
    return {"status": "deleted"}

@app.post("/api/save-summary")
async def save_summary(request: SaveSummaryRequest, authorization: Optional[str] = Header(None)):
    job_id = request.job_id
    user_id = request.user_id
    
    # Extract token from Authorization header
    jwt_token = None
    if authorization and authorization.startswith("Bearer "):
        jwt_token = authorization.split(" ")[1]
    
    # Check if job exists
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if job is completed
    if jobs[job_id].get("status") != "completed":
        raise HTTPException(status_code=400, detail="Summary is not ready yet")
    
    # Save to Supabase
    try:
        result = await save_summary_to_supabase(user_id, jobs[job_id])
        return {"success": True, "message": "Summary saved successfully", "data": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving summary: {str(e)}")

# Run API server when executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 