# Geometric Resolution of Feature Interference in GPT-2

A mechanistic interpretability study investigating how GPT-2 internally resolves competing concept representations through activation steering with midpoint mediator vectors.

## Overview

When a language model processes a sentence containing opposing concepts (e.g., "freedom" vs. "security"), its internal representations must reconcile the tension. This project dissects that process by:

1. Extracting residual-stream vectors for each concept at a target layer
2. Computing their geometric midpoint — the **mediator vector** M = (V_a + V_b) / 2
3. Injecting or ablating that mediator to causally test its role
4. Decomposing each layer's transformation into **in-plane** (2D concept plane) and **out-of-plane** components

## Key Findings

| Finding | Detail |
|---------|--------|
| **Mediator steering works** | Up to −36% entropy reduction (GPT-2 Small), −56% (GPT-2 Large) |
| **Statistically significant** | p = 0.041, Cohen's d = 0.88 across prompt rephrasings |
| **Dual-process decomposition** | ~13% in-plane convergence (84× above chance) + ~87% out-of-plane injection |
| **MLP-dominated** | MLP drives both components in 35/36 layers; attention dominates only at the convergence cliff (L33–L35) |
| **Concept plane is architectural** | Unrelated pairs (banana/democracy) produce equal enrichment (~80×) as semantic pairs — the 2D plane is a token-embedding geometry feature, not semantic |
| **Ablation reveals semantic depth** | Removing semantic mediators disrupts processing 2× more than removing unrelated mediators, despite identical geometry |
| **Rotation purity is an artifact** | The σ₁/σ₂ ≈ 1.07 metric fails its null model — 100% of random projections match or exceed it |

## Experiments

The notebook contains **22 follow-up experiments** (FU1–FU22) building from basic decomposition through rigorous null models:

| Phase | Experiments | Focus |
|-------|------------|-------|
| Foundation | Steps 1–8 | Extraction, steering, causal proof, cross-prompt validation |
| Universality | FU10–FU12 | 7 domains, 24 prompts, GPT-2 Small vs. Large |
| Attribution | FU13, FU17 | MLP vs. attention head decomposition |
| Null models | FU15, FU16, FU18 | Random baselines, stability tests, rotation purity null |
| Deep dives | FU19–FU22 | Threshold sensitivity, superposition, relational universality, causal discrimination |

## Models

- **GPT-2 Small** (124M params): 12 layers, 768 d_model, target layer 6
- **GPT-2 Large** (774M params): 36 layers, 1280 d_model, target layer 18

Both loaded via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (`HookedTransformer`).

## Repository Structure

```
├── experiment.ipynb          # Main notebook (all experiments FU1–FU22)
├── RESEARCH_REPORT.md        # Full technical report with equations
├── RESEARCH_REPORT_v2.md     # Accessible version (LaTeX-free)
├── REPORT.md                 # Initial report (steps 1–8)
├── requirements.txt          # Python dependencies
├── fu20_results.json         # FU20 superposition test results
├── fu21_results.json         # FU21 relational universality results
└── fu22_results.json         # FU22 causal discrimination results
```

## Setup

```bash
pip install -r requirements.txt
```

Then open `experiment.ipynb` in JupyterLab or VS Code and run cells sequentially. The first cell loads GPT-2 Large (~3 GB download on first run).

> **Note**: All experiments run on CPU. GPU is not required but will speed up inference.

## Requirements

- Python 3.9+
- PyTorch
- TransformerLens
- NumPy, Pandas, Plotly
- JupyterLab

## Summary of Results (FU1–FU22)

| Experiment | Question | Verdict |
|------------|---------|---------|
| FU1–FU6 | Basic decomposition properties | Various |
| FU7 | Multi-layer tracking | PARTIALLY SUPPORTED |
| FU8 | Causal intervention | CAUSAL (6/6) |
| FU10 | Cross-domain universality | STRONGLY UNIVERSAL (6/6) |
| FU11 | Surgical causal test | LARGELY CAUSAL (4/6) |
| FU12 | Cross-scale universality | STRONGLY UNIVERSAL (6/6) |
| FU13 | Head attribution | MLP DRIVES ROTATION |
| FU14 | Intervention transfer | TRANSFERS (4/4) |
| FU15 | Random baseline | 84× ABOVE CHANCE |
| FU16 | Stability-disruption correlation | NO CORRELATION |
| FU17 | Divergence-layer attribution | MLP DOMINATES BOTH |
| FU18 | Rotation purity null model | NOT SIGNIFICANT |
| FU19 | Threshold + head-convergence | ROBUST / ATTN DOMINATES L33–35 |
| FU20 | Superposition hypothesis (87%) | PARTIALLY SUPPORTED (2/4) |
| FU21 | Relational universality | NOT SUPPORTED (1/4) |
| FU22 | Causal discrimination | WEAKLY DISCRIMINATING (1/4) |

## License

This project is for research and educational purposes.
