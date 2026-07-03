from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.explainability_service import (
    compute_attribution,
    compute_counterfactual_explanation,
)

router = APIRouter(prefix="/explain", tags=["explain"])


class ExplanationRequest(BaseModel):
    simulation_id: str
    target_gene: str


class CounterfactualRequest(BaseModel):
    simulation_id: str
    target_gene: str
    counterfactual_value: float = 0.0
    rollout_steps: int = 3


@router.post("/")
def explain_endpoint(request: ExplanationRequest):
    """
    Compute gene attribution scores for a completed simulation.

    Returns top-20 driver genes ranked by |attribution| and
    the top GRN regulators of the target gene.
    """
    try:
        return compute_attribution(
            simulation_id=request.simulation_id,
            target_gene=request.target_gene,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attribution failed: {e}")


@router.post("/counterfactual")
def counterfactual_endpoint(request: CounterfactualRequest):
    """
    Compute counterfactual expression deltas:
    "What would change if <target_gene> had value <counterfactual_value>?"
    """
    try:
        return compute_counterfactual_explanation(
            simulation_id=request.simulation_id,
            target_gene=request.target_gene,
            cf_value=request.counterfactual_value,
            rollout_steps=request.rollout_steps,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Counterfactual failed: {e}")
