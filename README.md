# SimCLR-F: Reliability-Aware Hard Negative Contrastive Learning (RAHNCL)

This repository provides a practical starter project for contrastive learning research with:

- A week-by-week research plan.
- A reusable experiment template.
- A thesis/proposal outline (for 开题/中期).
- Runnable PyTorch code scaffold for RAHNCL experiments.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --dataset cifar10 --epochs 1 --batch_size 128
```

## Project Structure

```text
.
├── docs/
│   ├── weekly_research_plan.md
│   ├── experiment_template.md
│   └── thesis_outline.md
├── rahncl/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── losses.py
│   ├── model.py
│   ├── train_utils.py
│   └── eval_utils.py
├── train.py
├── evaluate.py
└── requirements.txt
```

## Baseline Idea

RAHNCL modifies standard InfoNCE by introducing two weights for each negative pair:

- **Hardness weight**: prioritize hard negatives.
- **Reliability weight**: down-weight likely false negatives.

This helps balance discriminative learning with semantic consistency.
