#!/usr/bin/env python3
"""
Start the Video Summarization API server.

Usage:
    python start_api.py
"""
import os
import argparse
import uvicorn
from utils import check_dependencies

def main():
    parser = argparse.ArgumentParser(description="Start the Video Summarization API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        deps = check_dependencies()
        missing = [dep for dep, installed in deps.items() if not installed]
        
        print("Dependency Status:")
        for dep, installed in deps.items():
            print(f"  {dep}: {'✓' if installed else '✗'}")
        
        # Check additional dependencies
        try:
            import fastapi
            print("  fastapi: ✓")
        except ImportError:
            print("  fastapi: ✗")
            missing.append("fastapi")
        
        try:
            import yt_dlp
            print("  yt_dlp: ✓")
        except ImportError:
            print("  yt_dlp: ✗")
            missing.append("yt_dlp")
        
        if missing:
            print("\nMissing dependencies, install with:")
            print("pip install -r requirements.txt")
            return
    
    # Create directories if they don't exist
    for directory in ["uploads", "downloads", "frames"]:
        os.makedirs(directory, exist_ok=True)
    
    # Start API server
    print(f"Starting API server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 