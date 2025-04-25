# SAFIN: SYMMETRY-ALIGNMENT FEATURE INTERACTION NETWORK

A deep learning framework for cross-ear biometric authentication, leveraging left-right ear symmetry to improve recognition accuracy and robustness.


## 🧠 OVERVIEW

**SAFIN** is a novel deep learning framework designed to improve the robustness of **ear-based identity verification**, especially in cross-ear scenarios where one ear is used for registration and the other for authentication.

---

## 🔧 KEY COMPONENTS

### 🔹 SYMMETRY ALIGNMENT MODULE (SAM)
- Performs differentiable geometric alignment.
- Utilizes a dual-attention mechanism to ensure accurate feature correspondence between ears.

### 🔹 FEATURE INTERACTION NETWORK (FIN)
- Captures complex nonlinear interactions between binaural features.
- Uses a difference-product dual-path architecture to enhance discriminability.

### 🔹 MULTI-TASK LOSS FUNCTION
- Balances multiple training objectives for similarity detection and identity verification using dynamic weighting.

---

## 📊 RESULTS

Evaluated on the **USTB Ear Dataset**, SAFIN achieves:

| Metric                         | Result        | Improvement     |
|-------------------------------|---------------|-----------------|
| Similarity Detection Accuracy | **99.03%**    | +9.11% vs. baseline |
| F1-Score                      | **0.9252**    | —               |
| False Positive Rate           | ↓ **3.05%**   | With SAM        |
| Similarity Std Deviation      | ↓ **67%**     | With FIN        |

---

## 📁 PROJECT STRUCTURE

```bash
SAFIN/
├── data/               # Dataset and preprocessing scripts
├── models/             # SAM, FIN, and SAFIN architecture
├── train.py            # Training pipeline
├── test.py             # Evaluation script
└── README.md           # Project documentation

---
**The code is coming soon！**
