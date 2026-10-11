# Nucleic Acid Binding Site Prediction: 1D CNN with Attention vs. Autoencoder Baseline

This project investigates deep learning architectures for identifying RNA and DNA-protein binding interface residues from primary sequence data.

## Project Overview
- **Objective:** Residue-level classification of nucleic acid binding interfaces across variable-length sequence data.
- **Dataset:** 1,634 sequences (481,252 total residues) compiled from benchmark RNA and DNA-binding protein sets.
- **Class Imbalance:** 45,911 binding residues vs. 435,341 non-binding residues (~9.54% positive class ratio).

## Architectures Evaluated
1. **1D CNN with Multi-Head Self-Attention:**
   - Multi-scale 1D convolutions (kernel sizes 3 and 5) with batch normalization for local motif extraction.
   - 4-head self-attention layer to capture long-range sequence context.
   - Masked Binary Cross-Entropy loss with sequence padding to accommodate variable lengths (up to 1,781 residues).
   - **Performance:** Achieved **0.7340 AUC** and **0.2738 AUPR** on validation data.

2. **Undercomplete Convolutional Autoencoder (Baseline):**
   - Unsupervised representation learning with bottleneck compression to 32 latent dimensions.
   - Evaluated sequence reconstruction error dynamics across interface residues.

## Tech Stack
- **Framework:** PyTorch, CUDA
- **Data & Evaluation:** NumPy, Pandas, Scikit-Learn, Matplotlib
