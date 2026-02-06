"""API endpoints for batch game running and data export."""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()


# In-memory tracking of running batches
_running_batches: Dict[str, dict] = {}


class BatchRunRequest(BaseModel):
    """Request to start a batch run."""
    num_games: int = 100
    player_count: int = 5
    models: List[dict]  # [{"name": "qwen-plus", "provider": "qwen"}, ...]
    rotate_models: bool = True
    parallel: int = 1  # Number of games to run in parallel
    tag: Optional[str] = None


class BatchStatusResponse(BaseModel):
    """Status of a batch run."""
    batch_id: str
    status: str  # "running", "completed", "failed", "stopped"
    total_games: int
    completed_games: int
    failed_games: int
    good_wins: int
    evil_wins: int
    started_at: str
    finished_at: Optional[str] = None
    errors: List[str] = []


class ExportRequest(BaseModel):
    """Request to export training data."""
    batch_id: Optional[str] = None
    tag: Optional[str] = None  # Experiment tag (e.g., "experiment_v1")
    game_ids: Optional[List[str]] = None
    output_filename: Optional[str] = None
    include_web: bool = False  # Default: only batch rollout data


class ExportResponse(BaseModel):
    """Response with export results."""
    filename: str
    total_games: int
    total_decisions: int
    download_url: str


@router.post("/run")
async def start_batch_run(
    request: BatchRunRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Start a batch run in the background."""
    from server.batch.runner import BatchGameRunner, BatchConfig
    
    batch_id = str(uuid.uuid4())[:8]
    
    # Convert models format
    models = [(m["name"], m["provider"]) for m in request.models]
    
    config = BatchConfig(
        num_games=request.num_games,
        player_count=request.player_count,
        models=models,
        rotate_models=request.rotate_models,
        parallel=request.parallel,
        batch_tag=request.tag,
        progress_callback=lambda done, total, gid: _update_progress(batch_id, done, total, gid),
    )
    
    # Initialize tracking
    _running_batches[batch_id] = {
        "status": "running",
        "total_games": request.num_games,
        "completed_games": 0,
        "failed_games": 0,
        "good_wins": 0,
        "evil_wins": 0,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "errors": [],
        "runner": None,
    }
    
    # Run in background
    runner = BatchGameRunner(config)
    _running_batches[batch_id]["runner"] = runner
    
    background_tasks.add_task(_run_batch, batch_id, runner)
    
    return {"batch_id": batch_id, "message": f"Started batch run with {request.num_games} games"}


def _update_progress(batch_id: str, done: int, total: int, game_id: Optional[str]):
    """Update progress for a running batch."""
    if batch_id in _running_batches:
        _running_batches[batch_id]["completed_games"] = done


async def _run_batch(batch_id: str, runner):
    """Run a batch and update status when done."""
    try:
        result = await runner.run()
        
        if batch_id in _running_batches:
            _running_batches[batch_id].update({
                "status": "completed",
                "completed_games": result.completed_games,
                "failed_games": result.failed_games,
                "good_wins": result.good_wins,
                "evil_wins": result.evil_wins,
                "finished_at": result.finished_at,
                "errors": result.errors,
            })
    except Exception as e:
        if batch_id in _running_batches:
            _running_batches[batch_id].update({
                "status": "failed",
                "finished_at": datetime.now().isoformat(),
                "errors": [str(e)],
            })


@router.get("/status/{batch_id}")
async def get_batch_status(batch_id: str) -> BatchStatusResponse:
    """Get status of a batch run."""
    if batch_id not in _running_batches:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch = _running_batches[batch_id]
    return BatchStatusResponse(
        batch_id=batch_id,
        status=batch["status"],
        total_games=batch["total_games"],
        completed_games=batch["completed_games"],
        failed_games=batch["failed_games"],
        good_wins=batch["good_wins"],
        evil_wins=batch["evil_wins"],
        started_at=batch["started_at"],
        finished_at=batch.get("finished_at"),
        errors=batch.get("errors", []),
    )


@router.post("/stop/{batch_id}")
async def stop_batch_run(batch_id: str) -> Dict[str, str]:
    """Stop a running batch."""
    if batch_id not in _running_batches:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch = _running_batches[batch_id]
    if batch["status"] != "running":
        raise HTTPException(status_code=400, detail="Batch is not running")
    
    runner = batch.get("runner")
    if runner:
        runner.stop()
        batch["status"] = "stopped"
        batch["finished_at"] = datetime.now().isoformat()
    
    return {"message": f"Batch {batch_id} stop requested"}


@router.get("/list")
async def list_batches() -> List[Dict[str, Any]]:
    """List all batch runs."""
    from server.batch.exporter import list_batches as get_db_batches
    
    # Get from database
    db_batches = await get_db_batches()
    
    # Merge with in-memory running batches
    result = []
    seen_ids = set()
    
    # Add running batches first
    for batch_id, batch in _running_batches.items():
        result.append({
            "batch_id": batch_id,
            "status": batch["status"],
            "total_games": batch["total_games"],
            "completed": batch["completed_games"],
            "good_wins": batch["good_wins"],
            "evil_wins": batch["evil_wins"],
            "started_at": batch["started_at"],
            "finished_at": batch.get("finished_at"),
        })
        seen_ids.add(batch_id)
    
    # Add database batches
    for b in db_batches:
        if b["batch_id"] not in seen_ids:
            result.append({
                "batch_id": b["batch_id"],
                "status": "completed",
                **b,
            })
    
    return result


@router.post("/export")
async def export_trajectories(request: ExportRequest) -> ExportResponse:
    """Export training trajectories.
    
    By default, only exports batch rollout data (excludes web UI games).
    Use tag to filter by experiment version (e.g., "experiment_v1").
    """
    from server.batch.exporter import TrainingDataExporter
    
    # Generate filename
    label = request.tag or request.batch_id or "all"
    filename = request.output_filename or f"trajectories_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    # Ensure exports directory exists
    export_dir = "./exports"
    os.makedirs(export_dir, exist_ok=True)
    output_path = os.path.join(export_dir, filename)
    
    exporter = TrainingDataExporter()
    stats = await exporter.export_trajectories(
        output_path=output_path,
        batch_id=request.batch_id,
        tag=request.tag,
        game_ids=request.game_ids,
        include_web=request.include_web,
    )
    
    return ExportResponse(
        filename=filename,
        total_games=stats.total_games,
        total_decisions=stats.total_decisions,
        download_url=f"/api/batch/download/{filename}",
    )


@router.get("/download/{filename}")
async def download_export(filename: str):
    """Download an exported file."""
    from fastapi.responses import FileResponse
    
    filepath = os.path.join("./exports", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/x-ndjson",
    )
