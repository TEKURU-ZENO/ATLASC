from pydantic import BaseModel
from typing import List

class SimulationRequest(BaseModel):
    dataset_id: str
    perturbations: List[str]
    steps: int = 10

class ExplanationRequest(BaseModel):
    simulation_id: str
    target_gene: str

class RecommendationRequest(BaseModel):
    budget: int = 5
    objective: str = "information_gain"
