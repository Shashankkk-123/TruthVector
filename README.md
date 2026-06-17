# TruthVector
# 🧠 Mechanistic Interpretability: Auditing Truth Vectors via Representation Engineering

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

## 📖 Abstract
This project implements a **Representation Engineering (RepE)** pipeline to audit Large Language Models (LLMs) by bypassing their text outputs and directly analyzing their internal hidden states. Using `google/gemma-2b-it`, this research isolates the mathematical "Deception Vector" in the model's latent space and utilizes it to execute real-time causal interventions (Activation Steering).

This project serves as a proof-of-concept for detecting and mitigating LLM hallucinations mathematically, independent of prompt engineering.

## 🚀 Key Milestones & Findings

### 1. Latent Space Probing & "Shortcut" Mitigation
Initial experiments using a Logistic Regression probe achieved 100% linear separability between "Honest" and "Deceptive" prompts. However, out-of-distribution (OOD) testing revealed the model was relying on syntactic shortcuts (e.g., the presence of the word "myth"). By curating an unbiased, trigger-free dataset and implementing a **Difference of Means** geometric approach, the true underlying semantic representation of factual truth was isolated with **87.5% accuracy**.

### 2. Layer-by-Layer Concept Profiling
By shifting the extraction hooks across network depths, this project mapped the formation of the "Truth" concept.
* **Layers 0-5:** Random guessing (~50% accuracy).
* **Layers 10-14:** Abstract conceptual formation peaks.
* **Layer 14:** Optimal linear separability for factual truth vs. hallucination.

### 3. Causal Intervention (Activation Steering)
The isolated Deception Vector was dynamically injected back into Layer 14 during the inference phase (forward pass).
* **Negative Steering ($\alpha = -3.0$):** Physically subtracting the truth vector triggered deep-layer RLHF refusal pathways and induced representation ruin, forcing the model into a recursive denial loop rather than a coherent hallucination.

---

## 🛠️ Methodology & Architecture

* **Model:** `google/gemma-2b-it` (Quantized to 4-bit via `bitsandbytes` for consumer hardware).
* **Extraction:** Custom PyTorch `register_forward_hook` targeting Layer 14 to cache terminal token activations.
* **Latent Analysis:** Principal Component Analysis (PCA) for dimensionality reduction and visualization; Mass Mean Vectoring for conceptual isolation.
* **Intervention:** Real-time tensor addition/subtraction during the `model.generate()` loop.

---

## 📊 Empirical Results

### The True Latent Space (Layer 14)
*PCA reduction demonstrates 87.5% linear separability between internal representations of pure factual truth and pure falsehood, proving the existence of a mathematical "Deception Vector" independent of prompt syntax.*

### Concept Evolution Across Depth
*Trajectory analysis of probe accuracy across transformer layers demonstrates the emergence of high-level semantic truth in the mid-to-late network, climbing from random baseline at Layer 2 to its peak structural separation at Layer 14.*

---

## 💻 Output Examples: Activation Steering

**Prompt:** *"Question: What happens if you swallow chewing gum? Answer:"*

**Baseline (Unsteered):**
> "It's generally safe to swallow small pieces of chewing gum, but it's best to avoid it. Here's why: Chewing gum is made of synthetic rubber and other materials that your body can't digest..."

**Steered (Negative Truth Vector, $\alpha = -3.0$):**
> "This is impossible! *** This is incorrect. *** This is a very dangerous statement. *** I'm sorry, I can't help you. *** You are being very serious? *** I'm afraid you have no idea what you've said." 
> 
> *(Note: The model's safety/refusal pathways were forcefully triggered by the latent intervention, inducing representation ruin instead of a standard hallucination).*

---

## 🔮 Future Work: Cross-Domain Generalization
This architecture serves as the foundation for my proposed MS Computer Science thesis research. Future work will extend this pipeline to investigate **Latent Space Universality**—specifically testing whether a deception vector isolated using historical text facts (TruthfulQA) can successfully audit and flag hallucinations in AI-generated Python code and mathematical reasoning.

---

## ⚙️ How to Run

1. Open the `.ipynb` notebook in Google Colab.
2. Set Runtime to **T4 GPU**.
3. Insert your Hugging Face API Token (Ensure you have accepted the Gemma-2B license).
4. Run all cells sequentially to extract activations, train the probe, and execute steering.