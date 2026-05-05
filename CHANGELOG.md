# Changelog

# SciLibMath v2 v2.0.0 — Multimodal Contrastive Model for Mathematical Objects

First public release of the **SciLibMath v2** model — a contrastive multimodal encoder for mathematical objects across **five modalities**: English text, Russian text, Lean 4 code, LaTeX notation, and rendered formula images.
## What it is

A multimodal embedding model that aligns five representations of the same mathematical object into a shared 256-dimensional space using **centroid representation**: each object is a region (its centroid + per-modality vectors), not a single point.
Architectural notes:

- **Patch-based image tokenization** without a pretrained vision backbone — formula images are tokenized as patch sequences analogous to text tokens.
- **Takagi-Sugeno fuzzy controller** for adaptive per-modality loss balancing during training, with nonlinear MLP consequents and a separate gradient path for early-phase modality correction.
- Trained on **~972K multimodal mathematical objects** derived from Mathlib (Lean 4) with augmentations from HERALD and ATLAS, extended with Russian translations, LaTeX notation, and image renders.
## Headline results

- Centroid retrieval R@1 = **0.92** (worst-case modality)
- Mean cross-modal R@1 = **0.707**
- Five modalities aligned to a shared 256-dimensional embedding space

## Usage in the SciLib infrastructure

The v2 model is the embedding component referenced by the Graph RAG API in [SciLib-GRC21](https://github.com/andkhalov/SciLib-GRC21) and by the public knowledge-graph search at https://scilibai.ru/products/mathlib/.
## License

CC BY 4.0 for model weights; MIT for training code.
## Citation

```bibtex
@misc{scilibmath_v2_2026,
 author = {Khalov, Andrey P. and Ataeva, Olga M. and Tuchkova, Natalia P.},
 title = {{SciLibMath v2}: A Multimodal Contrastive Model for Mathematical Objects},
 year = {2026},
 publisher = {Zenodo},
 version = {v2.0.0},
 doi = {(DOI minted automatically by Zenodo on release)},
 url = {https://github.com/andkhalov/SciLibMath_v2}
}
```

## Authors

- **Andrey P. Khalov** —
 Moscow Institute of Physics and Technology (MIPT), Dolgoprudny, Russia;
 Federal Research Center "Computer Science and Control" of the Russian Academy of Sciences, Moscow, Russia ·
 ORCID: [0009-0005-4584-8245](https://orcid.org/0009-0005-4584-8245) ·
 `khalov.a@phystech.edu`
- **Olga M. Ataeva** — FRC CSC RAS · ORCID: [0000-0003-0367-5575](https://orcid.org/0000-0003-0367-5575) · `oataeva@frccsc.ru`
- **Natalia P. Tuchkova** — FRC CSC RAS · ORCID: [0000-0001-5357-9640](https://orcid.org/0000-0001-5357-9640) · `ntuchkova@frccsc.ru`
