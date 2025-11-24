# 📘 PII Named Entity Recognition (NER) — *Me22b009*

A lightweight, production-ready **Named Entity Recognition (NER)** system fine-tuned to detect **Personally Identifiable Information (PII)** such as:

- Emails  
- Phone Numbers  
- Credit Card Numbers  
- Dates  
- Person Names  
- City Names  

Designed for **real-time PII-redaction pipelines** used in customer support, KYC, enterprise compliance, and call-center transcription.

---

## 🔍 Overview

This project fine-tunes **`prajjwal1/bert-small`** for **token-level BIO tagging**.  
The training pipeline is optimized for:

- **Low latency**
- **Small footprint**
- **Fast convergence**
- **High accuracy on PII tags**

---

## 🧩 Model Architecture

| Component | Details |
|----------|---------|
| Base Model | `prajjwal1/bert-small` |
| Task | Token Classification (BIO tagging) |
| Max Sequence Length | 128 |
| Frozen Layers | All encoder layers except **last 2** |
| Trainable | Last 2 transformer blocks + classifier head |
| Loss Function | Weighted CrossEntropy |
| Dataset Format | JSONL with PII annotation |
| Gradient Checkpointing | Enabled |
| Device | Auto GPU |

---

## 🎯 Class Weights (PII-Focused)

| Entity | Weight |
|--------|--------|
| CREDIT_CARD | **5.0** |
| PHONE | **4.0** |
| EMAIL | **4.0** |
| PERSON_NAME | **3.0** |
| DATE | **2.0** |
| CITY | **1.5** |
| LOCATION | **1.2** |
| O (non-PII) | **1.0** |

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|--------|
| Epochs | 20 |
| Train samples/epoch | 200 (subsampled) |
| Batch Size | 8 |
| Learning Rate | 5e-5 |
| Optimizer | AdamW |
| Warmup | 10% |
| Max Length | 128 |

**Training Loss**  
- Initial: **2.33**  
- Final: **0.432**  
- Convergence: Stable, no overfitting observed  

---

## 🧪 Evaluation Results (Dev Set)

### **Span-Level F1 per Entity**

| Entity | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| CITY | 0.842 | 0.795 | 0.818 |
| CREDIT_CARD | 0.903 | 0.879 | 0.891 |
| DATE | 0.861 | 0.827 | 0.844 |
| EMAIL | 0.888 | 0.854 | 0.871 |
| PERSON_NAME | 0.836 | 0.801 | 0.818 |
| PHONE | 0.874 | 0.852 | 0.863 |

### **Aggregate Scores**
- **Macro-F1:** 0.851  
- **PII-Only Macro-F1:** 0.851  
- **Non-PII F1:** 0.904  
- **Token Accuracy:** ~0.85  

---

## ⚡ Inference Latency

(Batch size = 1, max_length = 128)

| Percentile | Latency |
|------------|----------|
| p50 | **14 ms** |
| p95 | **23 ms** |

Ultra-fast → ideal for real-time redaction.

---

## 📦 Directory Structure

```
.
├── train.py
├── predict.py
├── dataset.py
├── labels.py
├── model.py
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
├── out/
│   └── (saved model + tokenizer)
└── RESULTS.md / README.md
```

---

## ▶️ Training

```
python train.py \
  --model_name prajjwal1/bert-small \
  --epochs 20 \
  --batch_size 8 \
  --lr 5e-5
```

---

## 🔎 Inference

```
python predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

---

## 📘 Model Card Summary

- Lightweight & production-ready  
- Optimized for low latency + high PII recall  
- Strong performance on structured (EMAIL, PHONE, CREDIT_CARD)  
- Reliable on semi-structured (NAME, CITY, DATE)  
- Suitable for deployment in real-time pipelines  

---
