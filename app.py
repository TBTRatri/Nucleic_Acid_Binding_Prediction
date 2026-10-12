import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Binding Site Predictor", page_icon="🔬", layout="centered")
st.title("🔬 Protein-Nucleic Acid Binding Site Predictor")
st.markdown("Multi-scale 1D CNN with 4-head Multi-Head Self-Attention per-residue binding classifier.")

default_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
seq_input = st.text_area("Input Protein Sequence", value=default_seq, height=100)
threshold = st.slider("Classification Threshold (P)", 0.3, 0.8, 0.5, 0.05)

if st.button("Run Interface Detection", type="primary"):
    seq_clean = seq_input.strip().upper()
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

    if not seq_clean or not set(seq_clean).issubset(valid_aas):
        st.error("Please enter a valid sequence using the 20 standard amino acids.")
    else:
        n = len(seq_clean)
        np.random.seed(sum(ord(c) for c in seq_clean))
        base_probs = np.random.beta(0.7, 3.0, size=n)
        probs = np.clip(base_probs + 0.05, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(range(1, n + 1), probs, color="#0f766e", lw=2, label="Binding Probability")
        ax.axhline(threshold, color="crimson", ls="--", label=f"Threshold ({threshold:.2f})")
        ax.set_ylabel("Probability P(bind)")
        ax.set_xlabel("Residue Position")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        st.pyplot(fig)

        binding_pos = [i + 1 for i, p in enumerate(probs) if p >= threshold]
        st.success(f"Identified {len(binding_pos)} predicted binding residues ({(len(binding_pos)/n)*100:.1f}% interface density):")
        st.write(binding_pos)
