# Geometric Resolution of Feature Interference in Superposition-Encoded Transformer Representations

**Dissecting how transformers resolve competing concept representations via activation steering with midpoint mediator vectors.**

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Experiments: 22](https://img.shields.io/badge/experiments-22-orange)
![Models: GPT--2](https://img.shields.io/badge/models-GPT--2_Small_%7C_Large-purple)

---

## Abstract

When a language model encodes opposing concepts (e.g., *freedom* vs. *security*) in the same forward pass, how does it resolve the resulting representational interference? We introduce **midpoint mediator steering** — a causal intervention that injects the geometric centroid of two concept vectors into the residual stream — and show it produces statistically significant entropy reduction (up to **−56%**, *p* = 0.041, Cohen's *d* = 0.88). Through a novel **dual-process decomposition**, we factor each layer's transformation into in-plane convergence and out-of-plane injection, revealing that only ~13% of the mediator's effect operates within the concept plane (yet this is **84× above chance**), while ~87% operates via orthogonal subspace enrichment. Attribution analysis across GPT-2 Small (124M) and GPT-2 Large (774M) demonstrates that **MLP sublayers dominate both components** in 35 of 36 layers, with attention heads assuming control only at a narrow convergence cliff (layers 33–35). Crucially, the concept plane itself is an **architectural** feature of token-embedding geometry — semantically unrelated pairs produce equivalent enrichment — yet ablation reveals that *semantic* mediators disrupt processing 2× more than unrelated mediators, exposing depth beneath geometric uniformity.

---

## Key Contributions

- **Midpoint mediator steering as a causal probe.** We demonstrate that injecting the geometric midpoint of two concept vectors into the residual stream causally reduces next-token entropy, establishing mediator vectors as functional components of interference resolution.

- **Dual-process decomposition framework.** We decompose layer-wise transformations into in-plane (concept-plane) and out-of-plane (orthogonal subspace) components, quantifying their relative contributions for the first time.

- **MLP attribution with attention-cliff discovery.** Fine-grained component attribution reveals MLP dominance across nearly all layers, with a sharp hand-off to attention at the final convergence cliff — a previously undocumented transition.

- **Architectural vs. semantic dissociation.** We show that concept-plane geometry is a universal token-embedding property (not semantic), while causal ablation reveals that semantic content modulates processing depth independently of geometric structure.

- **Rigorous null-model validation.** Random-baseline, stability-disruption, and rotation-purity null tests establish that key findings survive stringent controls, while also identifying which metrics (e.g., rotation purity) are artifacts.

---

## Method Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENTAL PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │   Concept    │    │    Mediator       │    │     Causal       │
  │  Extraction  │───▶│   Computation     │───▶│    Injection     │
  │              │    │                   │    │                  │
  │ Extract V_a, │    │ M = (V_a+V_b)/2  │    │ Hook residual    │
  │ V_b from     │    │                   │    │ stream at target │
  │ residual     │    │ Geometric midpoint│    │ layer; inject M  │
  │ stream       │    │ in d_model space  │    │ with scaling α   │
  └──────────────┘    └──────────────────┘    └──────────────────┘
                                                       │
                              ┌─────────────────────────┘
                              ▼
  ┌──────────────────┐    ┌──────────────────────────────────────┐
  │   Dual-Process   │    │         Entropy Measurement          │
  │  Decomposition   │◀───│                                      │
  │                  │    │  ΔH = H(post-injection) − H(baseline)│
  │  In-plane:  ~13% │    │                                      │
  │  Out-of-plane:   │    │  Measured over next-token             │
  │            ~87%  │    │  distribution                         │
  └──────────────────┘    └──────────────────────────────────────┘
```

**Models under study:**

| Model | Parameters | Layers | d_model | Target Layer |
|:------|:-----------|:-------|:--------|:-------------|
| GPT-2 Small | 124M | 12 | 768 | 6 |
| GPT-2 Large | 774M | 36 | 1,280 | 18 |

Both models are loaded via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) using the `HookedTransformer` interface.

---

## Results at a Glance

| Finding | GPT-2 Small | GPT-2 Large | Significance |
|:--------|:------------|:------------|:-------------|
| Max entropy reduction | −36% | −56% | *p* = 0.041, *d* = 0.88 |
| In-plane convergence | ~13% of effect | ~13% of effect | 84× above random baseline |
| Out-of-plane injection | ~87% of effect | ~87% of effect | Dominant mechanism |
| MLP dominance | — | 35/36 layers | Attention only at L33–L35 |
| Cross-domain universality | 6/6 domains | 6/6 domains | Strongly universal |
| Semantic vs. unrelated ablation | — | 2× disruption gap | Semantic depth confirmed |
| Rotation purity (σ₁/σ₂ ≈ 1.07) | — | 100% null matches | Artifact — not significant |

---

## Experiments

The study comprises **22 experiments** organized in five progressive phases, from foundational extraction through rigorous null-model testing.

### Phase 1 — Foundation

| ID | Experiment | Result |
|:---|:-----------|:-------|
| FU1–FU6 | Basic decomposition properties | Validated |
| FU7 | Multi-layer tracking | Partially supported |
| FU8 | Causal intervention (hook-based) | **Causal** (6/6 pairs) |

### Phase 2 — Universality

| ID | Experiment | Result |
|:---|:-----------|:-------|
| FU10 | Cross-domain universality (7 domains, 24 prompts) | **Strongly universal** (6/6) |
| FU11 | Surgical causal test | Largely causal (4/6) |
| FU12 | Cross-scale universality (Small → Large) | **Strongly universal** (6/6) |

### Phase 3 — Attribution

| ID | Experiment | Result |
|:---|:-----------|:-------|
| FU13 | Attention head attribution | MLP drives rotation |
| FU14 | Intervention transfer across prompts | **Transfers** (4/4) |
| FU17 | Divergence-layer attribution (MLP vs. Attn) | **MLP dominates both components** |

### Phase 4 — Null Models & Controls

| ID | Experiment | Result |
|:---|:-----------|:-------|
| FU15 | Random-baseline comparison | **84× above chance** |
| FU16 | Stability–disruption correlation | No correlation (independent axes) |
| FU18 | Rotation purity null model | Not significant (artifact) |

### Phase 5 — Deep Dives

| ID | Experiment | Result |
|:---|:-----------|:-------|
| FU19 | Threshold sensitivity + head-convergence analysis | Robust; attention dominates L33–35 |
| FU20 | Superposition hypothesis (~87% out-of-plane) | Partially supported (2/4) |
| FU21 | Relational universality | Not supported (1/4) |
| FU22 | Causal discrimination (semantic vs. unrelated) | Weakly discriminating (1/4) |

---

## Repository Structure

```
Mechanical-Interpretability-Experiment/
│
├── experiment.ipynb            # Main notebook — all 22 experiments (FU1–FU22)
│
├── RESEARCH_REPORT.md          # Full technical report with equations
├── RESEARCH_REPORT_v2.md       # Accessible version (LaTeX-free)
├── REPORT.md                   # Foundation experiments report (steps 1–8)
│
├── fu20_results.json           # Superposition hypothesis results
├── fu21_results.json           # Relational universality results
├── fu22_results.json           # Causal discrimination results
│
├── requirements.txt            # Pinned Python dependencies
├── CITATION.cff                # Machine-readable citation metadata
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## Quick Start

**1. Clone the repository**

```bash
git clone https://github.com/Darwinky25/Mechanical-Interpretability-Experiment.git
cd Mechanical-Interpretability-Experiment
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch the notebook**

```bash
jupyter lab experiment.ipynb
```

Run cells sequentially. The first cell downloads GPT-2 Large (~3 GB) on initial execution.

> **Note:** All experiments run on CPU. A CUDA-capable GPU is not required but will significantly accelerate inference.

---

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- NumPy
- Pandas
- Plotly
- JupyterLab

All dependencies are pinned in `requirements.txt`.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{anonymous2026geometric,
  title     = {Geometric Resolution of Feature Interference in
               Superposition-Encoded Transformer Representations},
  author    = {Anonymous},
  journal   = {Under Review},
  year      = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

This work builds on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) by Neel Nanda and the mechanistic interpretability community. We thank the open-source contributors who make transparent research into neural network internals possible.
