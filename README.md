# CAREF: Calibration-Aware Regularization for Explanation Faithfulness Without Rationale Supervision

<div align="center">

<h1>🧠 CAREF</h1>

<h3>
Calibration-Aware Regularization for Explanation Faithfulness <br>
Without Rationale Supervision
</h3>

<p>
<strong>EMNLP 2026 Submission</strong><br>
Currently under <strong>The ACL Rolling Review (ARR) — May 2026</strong>
</p>

<p>

<a href="https://kaopanboonyuen.github.io/CAREF/">
<img src="https://img.shields.io/badge/🌐-Project_Page-black?style=for-the-badge">
</a>

<a href="https://github.com/kaopanboonyuen/CAREF">
<img src="https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github">
</a>

<img src="https://img.shields.io/badge/EMNLP-2026-red?style=for-the-badge">

<img src="https://img.shields.io/badge/Status-ARR%20May%202026-success?style=for-the-badge">

</p>

---

### ⚡ Parameter-Efficient Fine-Tuning for Faithful Natural Language Explanations

</div>

---

# 🌟 Overview

**CAREF** (**C**alibration-**A**ware **R**egularization for **E**xplanation **F**aithfulness) is a novel parameter-efficient fine-tuning framework designed to jointly optimize:

- ✅ Predictive Accuracy
- ✅ Explanation Faithfulness
- ✅ Calibration Stability
- ✅ Sparse Decision Grounding

without requiring **rationale supervision**.

CAREF introduces a unified regularization objective:

\[
\mathcal{L}_{\text{SCED}}
\]

which combines:

- entropy-aware calibration
- adaptive token sparsity
- explanation-oriented optimization

inside a single differentiable loss.

---

# 🚀 Key Highlights

- 🧠 First unified entropy + sparsity regularizer for faithful NLE fine-tuning
- ⚡ Only **6.43% trainable parameters**
- 📈 Outperforms LoRA and AdaLoRA on explanation quality
- 🔍 Improves explanation faithfulness without rationale labels
- 🧩 Plug-and-play with PEFT methods
- 🌍 Architecture-agnostic design
- 📚 Evaluated on four NLE benchmarks

---

# 🖼️ Project Overview

<p align="center">
  <img src="img/overview_main.png" width="92%">
</p>

<p align="center">
<b>Figure:</b> CAREF overview including parameter efficiency, explanation quality, human evaluation, and hyperparameter sensitivity.
</p>

---

# 🧠 Motivation

Large Language Models can generate highly plausible explanations.

However:

> plausible explanations are not always faithful explanations.

Most existing approaches:

- require expensive rationale annotations
- rely on post-hoc attribution
- improve fluency rather than causal grounding

CAREF addresses this problem by directly regularizing the predictive distribution during fine-tuning.

---

# ⚙️ Method

## Unified Objective

CAREF optimizes:

\[
\mathcal{L}_{\text{CAREF}}
=
\mathcal{L}_{\text{CE}}
+
\lambda_{\text{SCED}}
\mathcal{L}_{\text{SCED}}
+
\lambda_{\text{KL}}
\mathcal{L}_{\text{KL}}
\]

where:

- \(\mathcal{L}_{CE}\) = task objective
- \(\mathcal{L}_{KL}\) = calibration regularization
- \(\mathcal{L}_{SCED}\) = proposed sparsity-calibrated entropic divergence

---

## Sparsity-Calibrated Entropic Divergence (SCED)

\[
\mathcal{L}_{\text{SCED}} =
\sum_t \sum_v
\left|
P_{t,v}
\log
\frac{P_{t,v}}{U_v}
\right|^\alpha
(1-P_{t,v})^\beta
\]

### ✨ Intuition

CAREF encourages:

- sparse decision-relevant tokens
- calibrated confidence distributions
- stable explanation grounding

instead of diffuse or overconfident reasoning patterns.

---

# 📊 Main Results

<p align="center">
  <img src="img/result.png" width="92%">
</p>

---

# 🏆 Performance Summary

| Model | Avg Accuracy | Avg nBERT | Trainable Params |
|---|---:|---:|---:|
| AdaLoRA | 23.59 | 19.75 | 4.46% |
| LoRA R=128 | 88.57 | 80.58 | 15.74% |
| LoRA R=4 | 88.92 | 80.91 | 0.58% |
| **CAREF-AQ** | **89.04** | **81.00** | **6.43%** |

---

# 🔥 Why CAREF Works

CAREF jointly controls:

| Component | Effect |
|---|---|
| Entropy Calibration | Prevents overconfident predictions |
| Adaptive Sparsity | Focuses on decision-relevant tokens |
| PEFT Regularization | Efficient fine-tuning |
| Distributional Control | Improves explanation grounding |

---

# 🧪 Benchmarks

We evaluate on:

| Dataset | Task |
|---|---|
| COS-E | Commonsense QA + Explanations |
| ECQA | Explainable Commonsense QA |
| ComVE | Commonsense Validation |
| e-SNLI | Natural Language Inference |

---

# 👀 Qualitative Examples

<p align="center">
  <img src="img/sample_predict.png" width="92%">
</p>

<p align="center">
<b>CAREF generates grounded and faithful explanations while baseline models fail to justify predictions.</b>
</p>

---

# 📈 Human Evaluation

CAREF improves human-perceived explanation faithfulness:

| Dataset | Human Score |
|---|---:|
| ECQA | 0.69 |
| e-SNLI | 0.58 |
| SenseMaking | 0.53 |
| COS-E | 0.47 |

Notably:

- CAREF obtains significantly more **Strong Yes** labels
- without any rationale supervision
- using only sparse PEFT updates

---

# 🧩 CAREF Variants

| Variant | Updated Module | Parameter Budget |
|---|---|---:|
| CAREF-BASE | Full FT | 100% |
| CAREF-DEC | Decoder Only | 52.23% |
| CAREF-AQKV | Attention QKV | 19.28% |
| CAREF-LAQ | Lightweight AQ | 6.44% |
| CAREF-AQ | Attention Query | 6.43% |

---

# 🔬 Key Findings

## ✅ CAREF-AQ achieves the best trade-off

- highest average accuracy
- strongest explanation alignment
- low parameter budget

## ✅ Explanation quality improves consistently

CAREF improves nBERT across:

- all datasets
- all fine-tuning regimes
- low-resource settings

## ✅ Sparse attention adaptation matters

Updating only attention query projections performs better than:

- full fine-tuning
- large LoRA configurations

---

# 🧠 Theoretical Insights

CAREF generalizes multiple classical regularizers:

| Parameters | Behavior |
|---|---|
| \(\alpha=1,\beta=0\) | KL divergence |
| \(\alpha>1,\beta=0\) | Power-law entropy |
| \(\alpha=1,\beta>0\) | Sparsity-weighted KL |
| \(\alpha>1,\beta>0\) | Full CAREF regime |

---

# ⚡ Architecture Compatibility

CAREF is:

- ✅ architecture-free
- ✅ differentiable
- ✅ PEFT-compatible
- ✅ decoder-agnostic

Compatible with:

- T5
- BART
- LLaMA
- GPT-style decoders
- LoRA / Adapters / Prefix Tuning

---

# 🛠️ Training Details

| Component | Value |
|---|---|
| Base Model | Flan-T5 |
| Optimizer | AdamW |
| Learning Rate | 3e-5 |
| Batch Size | 4 |
| Epochs | 50 |
| Hardware | NVIDIA A40 |
| Framework | Hugging Face Transformers + PEFT |

---

# 📦 Installation

```bash
git clone https://github.com/kaopanboonyuen/CAREF.git
cd CAREF

pip install -r requirements.txt
````

---

# 🚀 Example Usage

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large"
)

# Apply CAREF regularization during PEFT fine-tuning
```

---

# 🌐 Project Page

## 🔗 Official Website

👉 [https://kaopanboonyuen.github.io/CAREF/](https://kaopanboonyuen.github.io/CAREF/)

---

# 📚 Citation

```bibtex
@article{panboonyuen2026caref,
  title={CAREF: Calibration-Aware Regularization for Explanation Faithfulness Without Rationale Supervision},
  author={Panboonyuen, Teerapong},
  journal={ACL Rolling Review (ARR) May 2026},
  year={2026}
}
```

---

# 👨‍💻 Author

## Teerapong Panboonyuen

* 🌏 Thailand
* 🧠 AI / Computer Vision / Geospatial Foundation Models
* 🔬 NLP + Explainable AI + PEFT Research

🌐 Website:
[https://kaopanboonyuen.github.io/](https://kaopanboonyuen.github.io/)

---

# 🙏 Acknowledgements

This work builds upon:

* Hugging Face Transformers
* PEFT
* Flan-T5
* EMNLP / ACL research community

---

# ⭐ If you find CAREF useful

Please consider:

* ⭐ Starring the repository
* 🍴 Forking the project
* 📚 Citing the paper
* 🌍 Sharing with the NLP community

---

<div align="center">

<h2>🧠 CAREF</h2>

<h3>Faithful Explanations through Calibration-Aware Sparse Fine-Tuning</h3>

<strong>EMNLP 2026 • ARR May 2026</strong>

</div>

---