from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services.simulation_service import run_simulation

router = APIRouter(prefix="/simulate", tags=["simulate"])


class SimulationRequest(BaseModel):
    dataset_id: str = "norman"
    perturbations: List[str]
    steps: int = 5


@router.post("/")
def simulate_endpoint(request: SimulationRequest):
    """
    Run a perturbation simulation using the trained ATLAS ensemble.

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
        # Model not trained yet → informative error
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")
