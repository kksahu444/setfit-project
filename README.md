# 🚀 SetFit + Hard Negative Mining for Few-Shot Text Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-SetFit-yellow)
![Dataset](https://img.shields.io/badge/Dataset-BBC%20News-green)

## 📚 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [About SetFit](#-about-setfit-paper-summary)
- [Project Architecture](#-project-architecture)
- [Methodology](#-methodology)
- [Pipeline](#-pipeline)
- [Experiments & Metrics](#-experiments)
- [How to Run](#-how-to-run)
- [Key Insights](#-key-insights)
- [Results](#-results)
- [Future Work](#-future-work)
- [Authors](#-authors)

---

## 📌 Overview

This project implements and extends SetFit (Sentence Transformer Fine-tuning) for few-shot text classification, along with advanced improvements such as hard negative mining and open-set recognition.

We reproduce the methodology proposed in **L. Tunstall et al., ["Efficient Few-Shot Learning Without Prompts"](https://arxiv.org/abs/2209.11055), 2022** and build upon it with practical enhancements.

---

## 🧠 Problem Statement

Traditional NLP models require **large labeled datasets**, which are:

* Expensive
* Time-consuming
* Domain-specific

👉 Our goal:

> Build a **high-performance text classifier using only a few labeled examples (e.g., 8 per class)**.

---

## 📚 About SetFit (Paper Summary)

The SetFit paper proposes a **prompt-free few-shot learning framework** that avoids the limitations of prompt-based methods.

### 🔍 Key Ideas

* Uses **Sentence Transformers** instead of large LLMs
* Performs **contrastive learning using sentence pairs**
* Trains a lightweight **classification head on embeddings**
* Requires **no prompts or verbalizers**

👉 The method:

1. Fine-tune sentence embeddings using **positive/negative pairs**
2. Train classifier on embeddings

### ⚡ Why it matters

* Comparable performance to large models
* **Orders of magnitude faster training**
* Works with **very small datasets (few-shot)**
* No prompt engineering needed

---

## 🏗️ Project Architecture

```bash
setfit-project/
│
├── data/                  # Saved dataset samples (.txt)
├── results/               # Experiment outputs
│   ├── baseline/
│   └── hard_negative/
│
├── configs/
│   └── default.yaml       # Centralized project settings
│
├── src/
│   ├── data_loader.py     # Dataset + few-shot sampling
│   ├── pair_builder.py    # Pair construction
│   ├── hard_negative.py   # Hard negative mining
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Metrics + plots
│   ├── demo.py            # Interactive inference
│   └── config_utils.py    # YAML config loader
│
├── requirements.txt       # Project dependencies
├── LICENSE                # Apache license
└── README.md              # This file
```

---

## ⚙️ Methodology

### 🧩 Baseline (SetFit)

* Sample **k examples per class**
* Create **positive + random negative pairs**
* Train Sentence Transformer with **CosineSimilarityLoss**
* Train Logistic Regression classifier

---

### 🔥 Our Contributions

We extend SetFit with:

---

### 1️⃣ Hard Negative Mining

Instead of random negatives:

* Select **semantically similar but incorrect samples**
* Forces model to learn **fine-grained boundaries**

👉 Improves robustness and generalization

---

### 2️⃣ Improved Pair Construction

* Balanced positive/negative sampling
* Hard + easy negatives mix

---

### 3️⃣ Open-Set Classification

We introduce **"OTHER"** class using confidence thresholding

If model confidence < threshold:

```text
→ Reject prediction (unknown class)
```

---

### 4️⃣ Real Dataset (BBC News)

We use:

* Business
* Entertainment
* Politics
* Sports
* Technology

👉 More realistic than AG News

---

## 📊 Pipeline

```
Dataset → Few-shot Sampling → Pair Generation
        → Sentence Transformer Fine-tuning
        → Embedding Extraction
        → Classifier Training
        → Evaluation + Demo
```

---

## 🧪 Experiments

We evaluate:

* Baseline SetFit
* Hard Negative SetFit

Across:

* Multiple seeds
* Few-shot setting (k=8)
* Using multiple classification metrics beyond accuracy for a more comprehensive evaluation.

---

## 📈 Metrics

We evaluate using standard classification metrics:

* Accuracy  
* Precision (macro)  
* Recall (macro)  
* F1-score (macro)  
* Mean ± Standard Deviation across seeds  

---

## 🚀 How to Run

### 🔹 0. Configure experiment settings

All defaults are now stored in:

```bash
configs/default.yaml
```

You can edit dataset/model/threshold/seeds and other values there.
All scripts support `--config` and still allow CLI overrides for key options.

---

### 🔹 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 2. Generate dataset

```bash
python src/data_loader.py --config configs/default.yaml
```

Outputs:

```bash
data/train.txt
data/test.txt
```

---

### 🔹 3. Train baseline

```bash
python src/train.py --config configs/default.yaml --mode baseline
```

---

### 🔹 4. Train hard negative model

```bash
python src/train.py --config configs/default.yaml --mode hard_negative
```

---

### 🔹 5. Evaluate

```bash
python src/evaluate.py --config configs/default.yaml --results_dir results/baseline --output_dir results/baseline
python src/evaluate.py --config configs/default.yaml --results_dir results/hard_negative --output_dir results/hard_negative
```

---

### 🔹 6. Run demo

```bash
python src/demo.py --config configs/default.yaml --model_dir results/hard_negative/seed_0 --interactive
```

---

## 💡 Example Output

```
Input: "Stock markets crash globally"
Prediction: BUSINESS (0.82)

Input: "Aliens discovered on Mars"
Prediction: OTHER
```

---

## 🧠 Key Insights

* Few-shot learning is **highly unstable across seeds**
* Hard negatives improve:

  * decision boundaries
  * semantic understanding
* Confidence thresholding enables **real-world deployment**

---

## 📊 Results

### 🔹 Baseline (SetFit)

| Seed | Accuracy |
|------|--------|
| 0 | 0.716 |
| 1 | 0.699 |
| 2 | 0.710 |

**Mean Accuracy:** ~0.708

---

### 🔹 Hard Negative (Ours)

| Seed | Accuracy |
|------|--------|
| 0 | 0.955 |
| 1 | 0.935 |
| 42 | 0.939 |

**Mean Accuracy:** ~0.943

---

### 🚀 Improvement

- Absolute gain: **+23.5%**
- Hard negative mining significantly improves:
  - semantic discrimination
  - decision boundary sharpness

> 👉 This demonstrates that **pair quality > model complexity** in few-shot learning.
---

## 📌 Future Work

* Adaptive hard negative mining
* Label semantic injection
* Contrastive loss variants
* Calibration & uncertainty estimation

## 🎓 Authors

* Mozeel Pradip Vanwani 
* Mayank Seth 
* Krishnkant Sahu
* Burri Vivek Vardhan Verma 

