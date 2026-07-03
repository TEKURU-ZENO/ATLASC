import os
import shutil
from pathlib import Path

# Base directory
base_dir = Path(r"c:\Users\Dev Mehta\Desktop\ATLASC")

# Define the directory structure
directories = [
    "docker",
    "configs",
    "data/raw",
    "data/processed",
    "data/external",
    "data/metadata",
    "artifacts/models",
    "artifacts/logs",
    "scripts",
    "experiments/configs",
    "graphs",
    "models/encoders",
    "models/gnn",
    "causal",
    "simulation",
    "explainability",
    "active_learning",
    "evaluation",
    "utils",
    "apps/api/routes",
    "apps/api/schemas",
    "apps/api/services",
    "tests"
]

files = [
    "PRD.md",
    "MODEL_LIFECYCLE.md",
    "CONTRIBUTING.md",
    ".gitignore",
    "docker/Dockerfile.api",
    "docker/Dockerfile.worker",
    "docker/Dockerfile.frontend",
    "docker/nginx.conf",
    "configs/base.yaml",
    "configs/model.yaml",
    "configs/simulation.yaml",
    "configs/uncertainty.yaml",
    "configs/active_learning.yaml",
    "configs/deployment.yaml",
    "configs/logging.yaml",
    "artifacts/models/metadata.json",
    "scripts/preprocess_data.py",
    "scripts/build_graphs.py",
    "scripts/generate_embeddings.py",
    "scripts/validate_models.py",
    "scripts/sanity_check.py",
    "experiments/train.py",
    "experiments/evaluate.py",
    "experiments/ablation.py",
    "experiments/hyperparameter_search.py",
    "experiments/configs/experiment.yaml",
    "graphs/gene_regulatory_graph.py",
    "graphs/pathway_graph.py",
    "graphs/graph_builder.py",
    "graphs/graph_utils.py",
    "models/encoders/__init__.py",
    "models/encoders/base_encoder.py",
    "models/encoders/transcriptomic_encoder.py",
    "models/encoders/epigenomic_encoder.py",
    "models/encoders/multimodal_encoder.py",
    "models/gnn/__init__.py",
    "models/gnn/message_passing.py",
    "models/gnn/causal_gnn.py",
    "models/gnn/temporal_gnn.py",
    "models/gnn/loss.py",
    "models/decoder.py",
    "models/uncertainty.py",
    "models/model_loader.py",
    "causal/do_operator.py",
    "causal/counterfactuals.py",
    "causal/intervention_utils.py",
    "causal/perturbation_parser.py",
    "simulation/state_transition.py",
    "simulation/rollout_engine.py",
    "simulation/simulator.py",
    "simulation/stability_checks.py",
    "simulation/trajectory_logger.py",
    "explainability/gene_attribution.py",
    "explainability/pathway_attribution.py",
    "explainability/temporal_analysis.py",
    "explainability/root_cause.py",
    "explainability/attribution_utils.py",
    "active_learning/information_gain.py",
    "active_learning/exploration_strategy.py",
    "active_learning/experiment_ranker.py",
    "active_learning/budget_constraints.py",
    "active_learning/candidate_generator.py",
    "evaluation/metrics.py",
    "evaluation/calibration.py",
    "evaluation/robustness_tests.py",
    "evaluation/trajectory_analysis.py",
    "evaluation/visualization_utils.py",
    "utils/config_loader.py",
    "utils/logger.py",
    "utils/seed.py",
    "utils/file_manager.py",
    "utils/helpers.py",
    "apps/api/main.py",
    "apps/api/dependencies.py",
    "apps/api/middleware.py",
    "apps/api/routes/simulate.py",
    "apps/api/routes/explain.py",
    "apps/api/routes/recommend.py",
    "apps/api/routes/metadata.py",
    "apps/api/schemas/request.py",
    "apps/api/schemas/response.py",
    "apps/api/services/simulation_service.py",
    "apps/api/services/explainability_service.py",
    "apps/api/services/recommendation_service.py",
    "apps/api/services/model_service.py",
    "tests/test_encoders.py",
    "tests/test_gnn.py",
    "tests/test_simulation.py",
    "tests/test_uncertainty.py",
    "tests/test_active_learning.py",
    "tests/test_api.py"
]

print("Creating directories...")
for directory in directories:
    dir_path = base_dir / directory
    dir_path.mkdir(parents=True, exist_ok=True)

print("Creating empty files...")
for file in files:
    file_path = base_dir / file
    # Only touch if doesn't exist to avoid overwriting anything important
    if not file_path.exists():
        with open(file_path, "w") as f:
            pass

print("Moving frontend to apps/frontend...")
frontend_source = base_dir / "frontend"
frontend_dest = base_dir / "apps" / "frontend"
if frontend_source.exists() and not frontend_dest.exists():
    shutil.move(str(frontend_source), str(frontend_dest))
elif frontend_source.exists() and frontend_dest.exists():
    # If the destination already exists somehow, clear it (assuming it's a mistake or empty)
    # Careful here, just warning to avoid accidental data loss
    print("Warning: apps/frontend already exists.")

print("Moving backend artifacts to apps/api...")
backend_source = base_dir / "backend"
# Move requirements.txt if needed
req_src = backend_source / "requirements.txt"
if req_src.exists():
    shutil.copy(str(req_src), str(base_dir / "apps" / "api" / "requirements.txt"))

print("Moving Dockerfiles to docker/")
df_front_src = base_dir / "apps" / "frontend" / "Dockerfile"
if df_front_src.exists():
    shutil.move(str(df_front_src), str(base_dir / "docker" / "Dockerfile.frontend"))

df_back_src = backend_source / "Dockerfile"
if df_back_src.exists():
    shutil.move(str(df_back_src), str(base_dir / "docker" / "Dockerfile.api"))

print("Cleaning up old backend folder...")
if backend_source.exists():
    # We copied requirements.txt, main.py is empty stub anyway, let's keep app/main.py just in case or delete
    shutil.rmtree(str(backend_source), ignore_errors=True)

print("Writing basic content to base files...")
with open(base_dir / ".gitignore", "w") as f:
    f.write("# Python\n__pycache__/\n*.pyc\nvenv/\n\n# Node\nnode_modules/\n.next/\n\n# Env\n.env\n.env.local\n\n# Data\ndata/raw/*\n!data/raw/.gitkeep\n")

print("Done scaffolding.")
