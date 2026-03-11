# Geometric Resolution of Feature Interference in Superposition-Encoded Transformer Representations

**The 2D concept plane is architectural, not semantic — and that's the most interesting finding.**

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Experiments: 26](https://img.shields.io/badge/experiments-26-orange)
![Models: GPT--2](https://img.shields.io/badge/models-GPT--2_Small_%7C_Large-purple)

---

> **Lead finding (FU21):** The 2D concept plane enrichment (~80× above random) is **universal across all token pairs** — semantically unrelated pairs like *banana/democracy* produce equal or higher enrichment (82×) than opposition pairs like *freedom/security* (80×). The geometric structure is an **architectural property of how transformers represent token pairs**, not a signature of semantic relationships. This negative result is the study's most important contribution to the mechanistic interpretability community.

---

## Abstract

When a language model encodes two concepts in the same forward pass, the residual stream at intermediate layers encodes both directions simultaneously. We study this via **midpoint mediator steering** — injecting the geometric centroid of two concept vectors back into the residual stream — and decompose the resulting computation into in-plane convergence and out-of-plane injection components.

**The central negative result:** Our own null models (FU21) show that the geometric structure of the 2D concept plane is **identical for semantically related and unrelated token pairs**. The ~13% in-plane fraction (84× above random chance), stable concept plane (34/36 layers), and net convergence (~27–39°) are all **architectural constants** of GPT-2, not semantic features. The rotation purity metric (σ₁/σ₂ ≈ 1.07) also fails its null model — it is a generic projection artifact (FU18).

**What is semantic:** Mediator *ablation* disrupts processing of semantic pairs 2× more than unrelated pairs (FU22), showing that while the geometry is identical, the model's downstream computation integrates meaningful directions more deeply. However, the mediator injection itself is **not special** — random norm-matched vectors produce comparable or smaller entropy shifts (FU23, 1/4 tests passed), and mediator injection actually *increases* entropy across 57/60 prompts (FU25). The enrichment ratio is robust at 79× [95% CI: 75×–83×] across all pair types including unrelated (FU24).

**The dual-process decomposition** — factoring layer-wise transformations into ~13% in-plane convergence and ~87% out-of-plane MLP-driven injection — is a genuine, causally necessary architectural invariant of GPT-2, consistent across 24+ prompts, 7 domains, and both Small (124M) and Large (774M) model scales. Attribution analysis reveals MLP dominance in 35/36 layers, with attention heads assuming control only at a narrow convergence cliff (layers 33–35).

---

## Key Contributions

- **Architectural universality of the concept plane (negative result).** We show that the 2D concept plane — its ~80× enrichment, stability, and convergence properties — is identical for semantically related and unrelated token pairs (FU21). This tells the mechanistic interpretability community: **do not interpret this geometric pattern as semantic**.

- **Injection/ablation asymmetry.** Despite identical geometry, mediator *ablation* disrupts semantic pairs 2× more than unrelated pairs (FU22), revealing that downstream processing depth — not geometric structure — encodes semantic content.

- **Dual-process decomposition framework.** We decompose layer-wise transformations into in-plane convergence (~13%, 84× above chance) and out-of-plane MLP injection (~87%), quantifying their relative contributions and establishing both as causally necessary.

- **MLP attribution with attention-cliff discovery.** Fine-grained component attribution reveals MLP dominance across nearly all layers, with a sharp hand-off to attention at the final convergence cliff (L33–35). FU26 dissects this cliff at single-head resolution, identifying Head 17 as a "flipper" that transforms from a passive BOS attention sink (95% BOS, energy 0.67 at L34) to an explosive context amplifier (3–8% BOS, energy 857 at L35) — a 1,271× energy increase that drives 85% of the divergence at L35.

- **Rigorous null-model validation with self-correction.** We identified and corrected three methodological flaws (loss metric blind spot, information leakage, context asymmetry), and our null models falsified our own rotation purity claim (FU18) while validating the in-plane enrichment signal (FU15).

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
| **Concept plane is architectural, not semantic** | **—** | **Unrelated pairs ≥ semantic pairs** | **FU21: 82× vs 80× enrichment** |
| Max entropy reduction | −36% | −56% | *p* = 0.041, *d* = 0.88 |
| In-plane convergence | ~13% of effect | ~13% of effect | 84× above random baseline |
| Out-of-plane injection | ~87% of effect | ~87% of effect | Dominant mechanism |
| MLP dominance | — | 35/36 layers | Attention only at L33–L35 |
| Cross-domain universality | 6/6 domains | 6/6 domains | Strongly universal |
| Semantic vs. unrelated ablation | — | 2× disruption gap | Semantic depth confirmed |
| Rotation purity (σ₁/σ₂ ≈ 1.07) | — | 100% null matches | **Artifact — not significant** |
| **Convergence cliff driver** | **—** | **Head 17: BOS sink → context amplifier** | **FU26: 1,271× energy explosion at L35** |

---

## Experiments

The study comprises **26 experiments** organized in seven progressive phases, from foundational extraction through rigorous null-model testing, post-review controls, and mechanistic deep dives.

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
| FU18 | Rotation purity null model | **Not significant (artifact)** |

### Phase 5 — Deep Dives & Null Results

| ID | Experiment | Result |
|:---|:-----------|:-------|
| FU19 | Threshold sensitivity + head-convergence analysis | Robust; attention dominates L33–35 |
| FU20 | Superposition hypothesis (~87% out-of-plane) | Partially supported (2/4) |
| **FU21** | **Relational universality** | **Not supported (1/4) — KEY NULL RESULT** |
| **FU22** | **Causal discrimination (semantic vs. unrelated)** | **Weakly discriminating (1/4)** |

### Phase 6 — Post-Review Controls

| ID | Experiment | Result |
|:---|:-----------|:-------|
| **FU23** | **Random-injection control** (norm-matched random vectors vs. mediator) | **WEAK EVIDENCE (1/4)** — mediator not clearly special vs. random |
| **FU24** | **Bootstrap CIs for enrichment ratios** (uncertainty quantification) | **79× [95% CI: 75×–83×]** — robust, all categories equal |
| **FU25** | **Expanded prompt set (60 prompts, train/test split)** | **PARTIALLY VALIDATED (2/4)** — consistent effect, but entropy *increases* |

### Phase 7 — Mechanistic Deep Dive

| ID | Experiment | Result |
|:---|:-----------|:-------|
| **FU26** | **Convergence cliff dissection** (per-head directional energy L32–L35) | **Head 17 identified as “flipper”: BOS sink → context amplifier, 1,271× energy explosion** |

---

## Repository Structure

```
Mechanical-Interpretability-Experiment/
│
├── experiment.ipynb            # Main notebook — all 26 experiments (FU1–FU26)
│
├── RESEARCH_REPORT.md          # Full technical report with equations
├── RESEARCH_REPORT_v2.md       # Accessible version (LaTeX-free)
├── REPORT.md                   # Foundation experiments report (steps 1–8)
│
├── results/
│   ├── fu20_results.json       # Superposition hypothesis results
│   ├── fu21_results.json       # Relational universality results
│   ├── fu22_results.json       # Causal discrimination results
│   ├── fu23_results.json       # Random-injection control results
│   ├── fu24_results.json       # Enrichment CI results
│   ├── fu25_results.json       # Expanded prompt set results
│   └── fu26_results.json       # Convergence cliff dissection results
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
