import sys
import os
# Allow imports from both the project root (models/, utils/, etc.)
# and from this api directory (routes/, schemas/) regardless of where uvicorn is launched from
_here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _here)                                  # → apps/api  (routes, schemas)
sys.path.insert(0, os.path.abspath(os.path.join(_here, '../..')))  # → project root (models, utils, etc.)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import simulate, explain, recommend, metadata
import torch

app = FastAPI(
    title="ATLAS Simulation Engine API",
    version="2.0",
    description="Causal GNN Ensemble for gene perturbation prediction with calibrated epistemic uncertainty.",
)

# Allow the Next.js frontend on port 3000 to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulate.router)
app.include_router(explain.router)
app.include_router(recommend.router)
app.include_router(metadata.router)


@app.get("/health", tags=["system"])
def health_check():
    """Liveness probe."""
    return {"status": "ok", "version": "2.0"}


@app.get("/model_status", tags=["system"])
def model_status():
    """
    Returns the status of the loaded ATLAS ensemble.
    Triggers model loading on first call (singleton cache).
    """
    try:
        from services.model_service import get_ensemble, get_device, get_num_genes
        ensemble  = get_ensemble()
        device    = get_device()
        num_genes = get_num_genes()

        # Count trainable-param sets (one per ensemble member)
        k_members = ensemble.k_models if hasattr(ensemble, "k_models") else len(ensemble.encoders)

        # Checkpoint sizes for diagnostics
        ckpt_dir = os.path.join(_here, "../../models/checkpoints")
        ckpt_info = []
        for fname in sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []:
            if fname.endswith(".pt"):
                size_kb = os.path.getsize(os.path.join(ckpt_dir, fname)) // 1024
                ckpt_info.append({"file": fname, "size_kb": size_kb})

        return {
            "status": "loaded",
            "ensemble_members": k_members,
            "num_genes": num_genes,
            "device": str(device),
            "checkpoints": ckpt_info,
        }
    except Exception as e:
        return {
            "status": "not_loaded",
            "error": str(e),
            "hint": "Train the model first: python experiments/train.py",
        }


@app.get("/graph", tags=["system"])
def get_graph():
    """Return the TRRUST GRN edge list for frontend visualization."""
    grn_path = os.path.join(os.path.dirname(__file__), "../../data/processed/trrust_edge_index.pt")
    if not os.path.exists(grn_path):
        return {"nodes": [], "edges": []}
    edge_index = torch.load(grn_path, map_location="cpu", weights_only=True)
    # Return up to 500 edges to keep payload small
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    limit = 500
    edges = [{"source": s, "target": t} for s, t in zip(src[:limit], dst[:limit])]
    node_ids = list(set(src[:limit] + dst[:limit]))
    nodes = [{"id": n} for n in node_ids]
    return {"nodes": nodes, "edges": edges, "total_edges": len(src)}
