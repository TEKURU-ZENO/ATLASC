# API orchestrates. It does not contain model math.

# from fastapi import APIRouter
# router = APIRouter()
# 
# @router.post("/simulate")
# def simulate(req: SimulationRequest):
#     z = encoder(req.cell)
#     z = apply_intervention(z, req.genes)
#     trajectory = rollout(model, z, edge_index, req.steps)
#     return trajectory
