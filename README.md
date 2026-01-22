# ABIDE Brain GNN Workspace

Organized workspace for ABIDE-based brain graph learning with PyTorch Geometric.

## Structure
- `abide_data/` — Raw ABIDE time series and derivatives
- `notebooks/` — Experiments & analysis notebooks
- `models/` — Model definitions (e.g., DMSGCN)
- `scripts/` — Utilities (e.g., data download/preproc)
- `docs/` — Architecture notes & papers
- `env/` — Environment files (`requirements.txt`, setup script)

## Quickstart
1. Create environment
```bash
cd env
bash ../env/setup_brain_gnn_env.sh
```
2. Open notebooks
```bash
code notebooks/codeblock.ipynb
```

## Notes
- Keep `.venv*` at repo root for VS Code auto-detection.
- Large data & checkpoints are ignored via `.gitignore`.
- If an old path breaks, check the new folder layout above.
