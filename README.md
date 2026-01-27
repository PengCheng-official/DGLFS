# DGLFS

The code of the algorithm DGLFS, proposed in the papar: Dual-space Guided Global-Local Disambiguation for Partial Multi-label Feature Selection

## 1. Environment Setup

All required dependencies are listed in `requirements.txt`. 

## 2. Quick Reproduction

1. Install dependencies: `pip install -r requirements.txt`
2. Run core experiments: `python DGLFS.py`

## 3. Reproducibility Statement

This repository contains all necessary code, dependencies (in `requirements.txt`), and execution instructions to reproduce the core experimental findings of the paper. Due to randomness in model initialization and data splitting, reproduced results may have a minor fluctuation of 5‰ in metric values (e.g., AP, CE, HL, RL, micro-F1, and macro-F1), which is within acceptable experimental variability.