from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import os
import io

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from simulation.rollout_engine import RolloutEngine
from utils.data_loader import get_trrust_graph, NormanDataset
from causal.do_operator import apply_intervention
from active_learning.acquisition import compute_candidate_metrics, rank_and_select
from active_learning.budget_constraints import PlateLayout

router = APIRouter(prefix="/recommend", tags=["Experimental Design"])

class RecommendRequest(BaseModel):
    candidate_genes: List[int]
    rollout_horizon: Optional[int] = 3
    k_to_select: Optional[int] = 5
    lambda_diversity: Optional[float] = 1.0
    lambda_hub: Optional[float] = 1.0

class Explanation(BaseModel):
    rank: int
    Total_Score: float
    Utraj_Norm: float
    Diversity_Norm: float
    Hub_Norm: float
    Utraj_Raw: float

class RecommendResponse(BaseModel):
    selected_genes: List[int]
    explanations: Dict[int, Explanation]

class ExportPlateRequest(BaseModel):
    selected_genes: List[int]
    format_type: Optional[str] = "96"  # "96" or "384"
    control_wells: Optional[int] = 8
    blank_wells: Optional[int] = 4
    replicates: Optional[int] = 2

# Absolute project root (apps/api/routes/recommend.py -> project root = 3 levels up)
import os as _os
_HERE_ROUTE = _os.path.abspath(_os.path.dirname(__file__))
_PROJECT_ROOT_ROUTE = _os.path.abspath(_os.path.join(_HERE_ROUTE, "../../.."))

# Global cache to prevent reloading large models on every API request
_ATLAS_SERVICES = {}

def load_atlas_services():
    if _ATLAS_SERVICES:
        return _ATLAS_SERVICES
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[API STARTUP] Loading ATLAS Active Learning models to {device}...")
    
    latent_dim = 16
    K_models = 3
    
    processed_dir = os.path.join(_PROJECT_ROOT_ROUTE, "data", "processed")
    edge_index = get_trrust_graph(processed_dir=processed_dir).to(device)
    num_genes = edge_index.max().item() + 1  # approximate from graph
    
    # Read gene names
    names = None
    names_path = os.path.join(processed_dir, "gene_names.pt")
    if os.path.exists(names_path):
        import torch as _t
        names = _t.load(names_path, map_location="cpu", weights_only=False)
        num_genes = len(names)
            
    encoders = []
    gnns = []
    
    ckpt_dir = os.path.join(_PROJECT_ROOT_ROUTE, "models", "checkpoints")
    for k in range(K_models):
        ckpt_path = os.path.join(ckpt_dir, f"epistemic_member_k{k}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[API WARNING] Checkpoint {ckpt_path} missing.")
            break
            
        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        gnn = CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim, out_channels=1, layers_count=3).to(device)
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        enc.load_state_dict(ckpt['encoder_state_dict'])
        gnn.load_state_dict(ckpt['gnn_state_dict'])
        
        enc.eval()
        gnn.eval()
        
        encoders.append(enc)
        gnns.append(gnn)
        
    if len(encoders) == K_models:
        ensemble = EnsembleWrapper(encoders, gnns)
        simulator = RolloutEngine(ensemble, execution_mode="deterministic")
    else:
        ensemble, simulator = None, None
        
    # Get base cells for unperturbed forward pass
    base_cells = torch.zeros((10, num_genes), device=device)
    basal_path = os.path.join(processed_dir, "basal_state.pt")
    if os.path.exists(basal_path):
        try:
            b = torch.load(basal_path, map_location="cpu", weights_only=True)
            base_cells = b.unsqueeze(0).repeat(10, 1).to(device)
        except Exception:
            pass
            
    _ATLAS_SERVICES['ensemble'] = ensemble
    _ATLAS_SERVICES['simulator'] = simulator
    _ATLAS_SERVICES['edge_index'] = edge_index
    _ATLAS_SERVICES['base_cells'] = base_cells
    _ATLAS_SERVICES['device'] = device
    _ATLAS_SERVICES['gene_names'] = names
    
    return _ATLAS_SERVICES


@router.post("/rank_perturbations", response_model=RecommendResponse)
def rank_candidates(req: RecommendRequest):
    services = load_atlas_services()
    if not services.get('ensemble'):
        raise HTTPException(status_code=500, detail="Ensemble models not found. Please train first.")
        
    # 1. Compute Metrics
    u_traj_raw, z_mean_reps, hub_penalties = compute_candidate_metrics(
        candidates=req.candidate_genes,
        ensemble=services['ensemble'],
        simulator=services['simulator'],
        edge_index=services['edge_index'],
        base_cells=services['base_cells'],
        apply_intervention_fn=apply_intervention,
        rollout_steps=req.rollout_horizon,
        device=services['device']
    )
    
    # 2. Greedy Acquisition
    k = min(req.k_to_select, len(req.candidate_genes))
    selected, explanations = rank_and_select(
        candidates=req.candidate_genes,
        u_traj_raw=u_traj_raw,
        z_mean_reps=z_mean_reps,
        hub_penalties=hub_penalties,
        k=k,
        lambda1=req.lambda_diversity,
        lambda2=req.lambda_hub,
        ablation_mode="full"
    )
    
    # 3. Format Response Formatting
    return RecommendResponse(
        selected_genes=selected,
        explanations=explanations
    )


@router.post("/export_plate_csv")
def export_plate_csv(req: ExportPlateRequest):
    """
    Generates a downloadable CSV mapping recommended guide RNA knockouts
    to physical 96-well or 384-well microplate coordinates with controls and blanks.
    """
    services = load_atlas_services()
    gene_names = services.get("gene_names")

    plate = PlateLayout(
        format_type=req.format_type or "96",
        control_wells=req.control_wells or 8,
        blank_wells=req.blank_wells or 4,
        replicates_per_perturbation=req.replicates or 2,
    )

    df = plate.allocate_plate(req.selected_genes, gene_names=gene_names)
    csv_str = df.to_csv(index=False)

    filename = f"atlas_plate_layout_{req.format_type}well.csv"
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/plate_layout_preview")
def plate_layout_preview(req: ExportPlateRequest):
    """
    Returns JSON structure of microplate well allocations for interactive UI rendering.
    """
    services = load_atlas_services()
    gene_names = services.get("gene_names")

    plate = PlateLayout(
        format_type=req.format_type or "96",
        control_wells=req.control_wells or 8,
        blank_wells=req.blank_wells or 4,
        replicates_per_perturbation=req.replicates or 2,
    )

    df = plate.allocate_plate(req.selected_genes, gene_names=gene_names)
    return {
        "format_type": req.format_type,
        "total_wells": plate.total_wells,
        "control_wells": plate.control_wells,
        "blank_wells": plate.blank_wells,
        "replicates": plate.replicates,
        "records": df.to_dict(orient="records"),
    }
