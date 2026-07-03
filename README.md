# ATLAS

ATLAS is a scalable, intervention-aware causal simulation platform for modeling single-cell dynamics under explicit gene perturbations.

![ATLAS Architecture Concept](https://via.placeholder.com/800x400?text=ATLAS+7-Layer+Architecture+Placeholder)

## 🏗️ 7-Layer Architecture

1. **Data Layer**: Multi-omics ingestion (scRNA-seq, ATAC-seq).
2. **Representation Layer**: Graph & latent encodings.
3. **Causal Layer**: Intervention modeling (do-operator inspired).
4. **Dynamics Layer**: GNN-based temporal simulation.
5. **Trust Layer**: Uncertainty estimation & Explainability.
6. **Decision Layer**: Active learning for experiment optimization.
7. **Application Layer**: REST API and Interactive Dashboard.

## 🚀 Getting Started

Ensure you have Docker and Docker Compose installed.

```bash
docker-compose up --build
```

- **Backend API**: <http://localhost:8000>
- **Frontend Dashboard**: <http://localhost:3000>

## 🧪 Development Workflow

### Backend (Python/FastAPI)

Located in `apps/api/`

1. Create a virtual environment and install dependencies:

```bash
cd apps/api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

1. Run server:

```bash
uvicorn main:app --reload
```

### Frontend (Next.js/TypeScript)

Located in `apps/frontend/`

```bash
cd apps/frontend
npm install
npm run dev
```

---
*Built for causal AI in Bioinformatics.*
