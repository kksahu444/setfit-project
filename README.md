# 🚀 SetFit + Hard Negative Mining for Few-Shot Text Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-SetFit-yellow)

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
├── src/
│   ├── data_loader.py     # Dataset + few-shot sampling
│   ├── pair_builder.py    # Pair construction
│   ├── hard_negative.py   # Hard negative mining
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Metrics + plots
│   └── demo.py            # Interactive inference
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

---

## 📈 Metrics

* Accuracy
* Mean ± Standard Deviation
* Per-seed variability

---

## 🚀 How to Run

---

### 🔹 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 2. Generate dataset

```bash
python src/data_loader.py --k_per_class 8 --seed 42 --test_size 0.2
```

Outputs:

```bash
data/train.txt
data/test.txt
```

---

### 🔹 3. Train baseline

```bash
python src/train.py --mode baseline --k_per_class 8 --seeds 0 1 12 42
```

---

### 🔹 4. Train hard negative model

```bash
python src/train.py --mode hard_negative --k_per_class 8 --seeds 0 1 12 42
```

---

### 🔹 5. Evaluate

```bash
python src/evaluate.py --results_dir results/baseline --output_dir results/baseline
python src/evaluate.py --results_dir results/hard_negative --output_dir results/hard_negative
```

---

### 🔹 6. Run demo

```bash
python src/demo.py --model_dir results/hard_negative/seed_0 --interactive
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

## 📌 Future Work

* Adaptive hard negative mining
* Label semantic injection
* Contrastive loss variants
* Calibration & uncertainty estimation

---

## ⭐ Final Note

This project demonstrates how:

> **Small models + smart data construction > large models with brute force**

