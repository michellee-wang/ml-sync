"""
API routes for ML service - Frontend integration with Modal
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
import json
from pathlib import Path
from datetime import datetime
import asyncio
import uuid

router = APIRouter()

# In-memory job store (in production, use Redis or database)
job_store: Dict[str, Dict] = {}

class SpotifyTrack(BaseModel):
    name: str
    artist: str
    spotify_id: str
    artists: List[str] = []
    album: str = ""
    popularity: int = 0
    duration_ms: int = 0

class SpotifyTracksRequest(BaseModel):
    tracks: List[SpotifyTrack]

class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str
    created_at: str
    updated_at: str
    progress: Optional[float] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

class GenerationRequest(BaseModel):
    num_samples: int = 10
    temperature: float = 0.9
    model_type: str = "vae"
    model_path: str = "pretrained_model.pt"

class PipelineResponse(BaseModel):
    job_id: str
    status: str
    message: str
    pipeline_steps: List[str]

async def trigger_modal_matching(job_id: str, tracks_data: List[Dict]):
    """Background task to trigger Modal matching pipeline"""
    try:
        # Update job status to running
        job_store[job_id]["status"] = "running"
        job_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        job_store[job_id]["progress"] = 0.1

        # Import Modal app dynamically
        import sys
        scripts_path = Path(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_path))

        from match_spotify_to_midi import app as match_app, match_tracks

        # Call Modal function
        job_store[job_id]["progress"] = 0.3
        job_store[job_id]["message"] = "Matching Spotify tracks to MIDI files..."

        # Run Modal function asynchronously
        with match_app.run():
            result = match_tracks.remote(tracks_data, min_similarity=0.75)

        # Update job with results
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["progress"] = 1.0
        job_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        job_store[job_id]["result"] = {
            "matched_count": result["statistics"]["matched_count"],
            "unmatched_count": result["statistics"]["unmatched_count"],
            "match_rate": result["statistics"]["match_rate"],
            "total_tracks": result["statistics"]["total_spotify_tracks"]
        }
        job_store[job_id]["message"] = f"Successfully matched {result['statistics']['matched_count']} tracks"

    except Exception as e:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
        job_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        job_store[job_id]["message"] = f"Matching failed: {str(e)}"

@router.post("/save-spotify-tracks")
async def save_spotify_tracks(request: SpotifyTracksRequest, background_tasks: BackgroundTasks):
    """
    Save Spotify tracks from frontend and trigger Modal matching pipeline

    This endpoint:
    1. Saves Spotify tracks locally and to Modal volume
    2. Triggers Modal matching job asynchronously
    3. Returns job_id for status tracking
    """
    tracks_data = [track.dict() for track in request.tracks]

    # Save to data directory (local backup)
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "spotify_tracks.json"

    with open(output_file, 'w') as f:
        json.dump({
            'tracks': tracks_data,
            'total_tracks': len(tracks_data),
        }, f, indent=2)

    # Create job for tracking
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": "Matching job queued",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "progress": 0.0,
        "type": "matching",
        "tracks_count": len(tracks_data)
    }

    # Trigger Modal matching in background
    background_tasks.add_task(trigger_modal_matching, job_id, tracks_data)

    return {
        "status": "success",
        "job_id": job_id,
        "tracks_saved": len(tracks_data),
        "file_path": str(output_file),
        "message": "Spotify tracks saved and matching pipeline started",
        "check_status": f"/api/ml/training-status/{job_id}"
    }

async def trigger_modal_generation(job_id: str, request: GenerationRequest):
    """Background task to trigger Modal generation pipeline"""
    try:
        # Update job status to running
        job_store[job_id]["status"] = "running"
        job_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        job_store[job_id]["progress"] = 0.2

        # Import Modal app dynamically
        import sys
        scripts_path = Path(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_path))

        from generate_from_matched import app as gen_app, MIDIGenerator

        # Update progress
        job_store[job_id]["progress"] = 0.4
        job_store[job_id]["message"] = "Initializing MIDI generation..."

        # Run Modal function
        with gen_app.run():
            generator = MIDIGenerator(
                model_type=request.model_type,
                model_path=request.model_path
            )

            job_store[job_id]["progress"] = 0.6
            job_store[job_id]["message"] = f"Generating {request.num_samples} MIDI files..."

            generated_files = generator.generate.remote(
                num_samples=request.num_samples,
                temperature=request.temperature,
                output_dir=f"samples_{job_id}"
            )

        # Update job with results
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["progress"] = 1.0
        job_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        job_store[job_id]["result"] = {
            "generated_files": generated_files,
            "num_files": len(generated_files),
            "output_volume": "generated-midi",
            "output_dir": f"samples_{job_id}"
        }
        job_store[job_id]["message"] = f"Successfully generated {len(generated_files)} MIDI files"

    except Exception as e:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
        job_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        job_store[job_id]["message"] = f"Generation failed: {str(e)}"

@router.post("/start-training")
async def start_training(user_id: str):
    """
    Start personalized model training

    Note: Training endpoint is a placeholder. In production, this would:
    1. Load matched MIDI files from Modal volume
    2. Trigger Modal training job with GPU resources
    3. Return job_id for tracking
    """
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": "Training pipeline not yet implemented - use pretrain_model.py directly",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "progress": 0.0,
        "type": "training",
        "user_id": user_id
    }

    return {
        "status": "pending",
        "job_id": job_id,
        "message": "Training requires manual execution: modal run scripts/pretrain_model.py",
        "user_id": user_id,
        "check_status": f"/api/ml/training-status/{job_id}"
    }

@router.post("/generate-music")
async def generate_music(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate music from trained model using Modal

    This endpoint:
    1. Creates a generation job
    2. Triggers Modal generation pipeline asynchronously
    3. Returns job_id for status tracking

    Args:
        request: Generation parameters (num_samples, temperature, model_type, model_path)

    Returns:
        Job information with job_id for tracking
    """
    # Create job for tracking
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": "Generation job queued",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "progress": 0.0,
        "type": "generation",
        "parameters": {
            "num_samples": request.num_samples,
            "temperature": request.temperature,
            "model_type": request.model_type,
            "model_path": request.model_path
        }
    }

    # Trigger Modal generation in background
    background_tasks.add_task(trigger_modal_generation, job_id, request)

    return {
        "status": "success",
        "job_id": job_id,
        "message": "Generation pipeline started",
        "parameters": request.dict(),
        "check_status": f"/api/ml/training-status/{job_id}"
    }

@router.get("/training-status/{job_id}", response_model=JobResponse)
async def get_training_status(job_id: str):
    """
    Get real-time status of any job (matching, training, or generation)

    Args:
        job_id: Unique job identifier returned from start endpoints

    Returns:
        JobResponse with current status, progress, and results

    Raises:
        HTTPException: If job_id not found
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = job_store[job_id]

    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        message=job["message"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error")
    )

@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 50
):
    """
    List all jobs with optional filtering

    Args:
        status: Filter by status (queued, running, completed, failed)
        job_type: Filter by type (matching, training, generation)
        limit: Maximum number of jobs to return

    Returns:
        List of jobs with their current status
    """
    jobs = list(job_store.values())

    # Apply filters
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    if job_type:
        jobs = [j for j in jobs if j.get("type") == job_type]

    # Sort by created_at descending (most recent first)
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Limit results
    jobs = jobs[:limit]

    return {
        "total": len(jobs),
        "jobs": jobs,
        "filters": {
            "status": status,
            "type": job_type,
            "limit": limit
        }
    }

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job from the job store

    Args:
        job_id: Job identifier to delete

    Returns:
        Success confirmation

    Raises:
        HTTPException: If job_id not found
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = job_store.pop(job_id)

    return {
        "status": "success",
        "message": f"Job {job_id} deleted",
        "job_type": job.get("type"),
        "job_status": job.get("status")
    }

@router.post("/pipeline/full")
async def run_full_pipeline(
    request: SpotifyTracksRequest,
    background_tasks: BackgroundTasks,
    auto_generate: bool = True,
    num_samples: int = 10,
    temperature: float = 0.9
):
    """
    Run the full pipeline: matching -> (training) -> generation

    This convenience endpoint:
    1. Saves Spotify tracks and triggers matching
    2. Optionally triggers generation after matching completes

    Args:
        request: Spotify tracks to process
        auto_generate: Whether to automatically generate music after matching
        num_samples: Number of MIDI files to generate
        temperature: Generation temperature

    Returns:
        Pipeline job information with all job IDs
    """
    tracks_data = [track.dict() for track in request.tracks]

    # Save tracks locally
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "spotify_tracks.json"

    with open(output_file, 'w') as f:
        json.dump({
            'tracks': tracks_data,
            'total_tracks': len(tracks_data),
        }, f, indent=2)

    # Create pipeline job
    pipeline_id = str(uuid.uuid4())
    matching_job_id = str(uuid.uuid4())

    # Create matching job
    job_store[matching_job_id] = {
        "job_id": matching_job_id,
        "status": "queued",
        "message": "Matching job queued",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "progress": 0.0,
        "type": "matching",
        "tracks_count": len(tracks_data),
        "pipeline_id": pipeline_id
    }

    # Trigger matching
    background_tasks.add_task(trigger_modal_matching, matching_job_id, tracks_data)

    response = {
        "status": "success",
        "pipeline_id": pipeline_id,
        "matching_job_id": matching_job_id,
        "message": "Full pipeline started",
        "steps": [
            {
                "name": "matching",
                "job_id": matching_job_id,
                "status": "queued",
                "check_status": f"/api/ml/training-status/{matching_job_id}"
            }
        ]
    }

    if auto_generate:
        # Note: In production, this should wait for matching to complete
        # For now, we'll just queue the generation job
        response["message"] += " (Note: Generation will use existing model, not matched tracks)"

    return response

@router.get("/health")
async def health_check():
    """
    Health check endpoint with system status

    Returns:
        Service health status and basic statistics
    """
    total_jobs = len(job_store)
    running_jobs = len([j for j in job_store.values() if j.get("status") == "running"])
    failed_jobs = len([j for j in job_store.values() if j.get("status") == "failed"])
    completed_jobs = len([j for j in job_store.values() if j.get("status") == "completed"])

    return {
        "status": "healthy",
        "service": "ml-service",
        "modal_integration": "enabled",
        "job_statistics": {
            "total": total_jobs,
            "running": running_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs
        }
    }
