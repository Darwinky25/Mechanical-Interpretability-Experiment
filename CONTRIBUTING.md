# Contributing

Thank you for your interest in contributing to this research project. We welcome contributions that improve experimental rigor, extend the analysis, or enhance reproducibility.

## How to Contribute

### Reporting Issues

If you identify a methodological concern, a potential bug in the experimental pipeline, or a discrepancy in the reported results, please open an issue with:

1. A clear description of the problem
2. Steps to reproduce (if applicable)
3. Expected vs. observed behavior
4. Relevant notebook cell numbers or report sections

### Proposing New Experiments

We encourage follow-up experiments that test or extend the dual-process decomposition framework. When proposing a new experiment:

1. Open an issue describing the hypothesis, methodology, and expected outcome
2. Reference the relevant existing experiment (FU1–FU22) that your proposal builds on
3. Include a null model or control condition where appropriate

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b fu23-your-experiment`)
3. Add your experiment cells to `experiment.ipynb` following the existing naming convention (FU23, FU24, etc.)
4. Include results in a corresponding JSON file if applicable
5. Update `RESEARCH_REPORT.md` and `RESEARCH_REPORT_v2.md` with your findings
6. Submit a pull request with a clear description of the experiment and results

### Style Guidelines

- **Notebook cells**: Each experiment should begin with a markdown cell stating the hypothesis, methodology, and success criteria
- **Variable naming**: Use descriptive names consistent with existing code (e.g., `v_a`, `v_b`, `mediator`, `in_plane_frac`)
- **Results reporting**: Include both quantitative metrics and a clear verdict (PASS/FAIL with rationale)
- **Null models**: Every new claim should be accompanied by an appropriate null model or random baseline

## Reproducibility

All experiments must be reproducible with `seed = 42` on CPU. If your experiment requires GPU, document the hardware configuration and note any non-deterministic behavior.

## Code of Conduct

We are committed to providing a welcoming and constructive environment. Please be respectful in all interactions and focus feedback on the scientific content.
