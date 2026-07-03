from fastapi import APIRouter

router = APIRouter(prefix="/metadata", tags=["metadata"])

@router.get("/")
def get_metadata():
    return {"layers": ["Data", "Representation", "Causal", "Dynamics", "Trust", "Decision", "Application"]}
