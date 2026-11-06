from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from services.simulation_service import run_simulation
from services.simulation_queue import sim_queue

router = APIRouter(prefix="/simulate", tags=["simulate"])


class SimulationRequest(BaseModel):
    dataset_id: str = "norman"
    perturbations: List[str]
    steps: int = 5


class AsyncSimulationResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post("/")
def simulate_endpoint(request: SimulationRequest):
    """
    Run a synchronous perturbation simulation using the trained ATLAS ensemble.

    Perturbation format:
      "GENE_NAME:value"  e.g. ["TP53:0", "MYC:+1.5"]
      or "gene_idx:value"  e.g. ["42:0.0"]
    """
    try:
        result = run_simulation(
            perturbation_specs=request.perturbations,
            steps=request.steps,
            dataset_id=request.dataset_id,
        )
        return result
    except RuntimeError as e:
        # Model not trained yet -> informative error
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")


@router.post("/async", response_model=AsyncSimulationResponse)
def simulate_async_endpoint(request: SimulationRequest):
    """
    Enqueues a long-running simulation into the background worker queue.
    Returns immediately with a job_id for non-blocking polling.
    """
    try:
        job_id = sim_queue.submit_job(
            task_fn=run_simulation,
            params={
                "perturbations": request.perturbations,
                "steps": request.steps,
                "dataset_id": request.dataset_id,
            },
            perturbation_specs=request.perturbations,
            steps=request.steps,
            dataset_id=request.dataset_id,
        )
        return AsyncSimulationResponse(
            job_id=job_id,
            status="queued",
            message="Simulation enqueued successfully. Poll /simulate/status/{job_id} for progress.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue simulation: {e}")


@router.get("/status/{job_id}")
def get_simulation_status(job_id: str):
    """
    Checks the status and progress of an asynchronous simulation job.
    """
    job = sim_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": job.result,
        "error": job.error,
    }


@router.get("/jobs")
def list_simulation_jobs():
    """
    Returns a list of recent asynchronous simulation jobs.
    """
    return {"jobs": sim_queue.list_jobs()}
