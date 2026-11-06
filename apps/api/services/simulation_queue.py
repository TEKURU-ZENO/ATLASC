"""
simulation_queue.py
-------------------
Asynchronous in-memory background worker queue for long-running rollout
simulations and active learning campaigns.

Supports:
  - Non-blocking job submission (`POST /simulate/async`)
  - Status and progress tracking (`GET /simulate/status/{job_id}`)
  - Concurrency management with bounded ThreadPoolExecutor
  - History retention and graceful error handling
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class SimulationJob:
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class SimulationQueue:
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="atlas-sim")
        self.jobs: Dict[str, SimulationJob] = {}
        self.lock = threading.Lock()

    def submit_job(
        self,
        task_fn: Callable[..., Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = SimulationJob(
            job_id=job_id,
            status="queued",
            created_at=time.time(),
            params=params,
        )

        with self.lock:
            self.jobs[job_id] = job

        def worker_wrapper():
            with self.lock:
                job.status = "running"
                job.started_at = time.time()
                job.progress = 0.1

            try:
                # Execute task
                result = task_fn(*args, **kwargs)
                with self.lock:
                    job.status = "completed"
                    job.progress = 1.0
                    job.completed_at = time.time()
                    job.result = result
            except Exception as e:
                with self.lock:
                    job.status = "failed"
                    job.completed_at = time.time()
                    job.error = str(e)

        self.executor.submit(worker_wrapper)
        return job_id

    def get_job(self, job_id: str) -> Optional[SimulationJob]:
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.lock:
            all_jobs = list(self.jobs.values())
            all_jobs.sort(key=lambda j: j.created_at, reverse=True)
            return [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "created_at": j.created_at,
                    "started_at": j.started_at,
                    "completed_at": j.completed_at,
                    "progress": j.progress,
                    "has_result": j.result is not None,
                    "error": j.error,
                }
                for j in all_jobs[:limit]
            ]


# Global singleton instance
sim_queue = SimulationQueue(max_workers=2)
