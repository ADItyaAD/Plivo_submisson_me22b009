📘 PII Named Entity Recognition (NER)
(Me22b009)

A lightweight, production-oriented Named Entity Recognition model fine-tuned for detecting Personally Identifiable Information (PII) such as emails, phone numbers, credit card numbers, names, dates, and city names.

This repository contains:

Training pipeline

Evaluation scripts

Inference API

Model analysis & metrics

Final results

🔍 Overview

This project fine-tunes prajjwal1/bert-small for token-level BIO tagging of PII entities.
The model is optimized for speed, low latency, and high accuracy, making it suitable for realtime PII-redaction systems used in:

Customer support chatbots

Phone transcription pipelines

Enterprise compliance systems

Financial/KYC workflows

🧩 Model Architecture
Component	Details
Base Model	prajjwal1/bert-small
Task	Token Classification (BIO)
Entities	CITY, CREDIT_CARD, DATE, EMAIL, PERSON_NAME, PHONE
Max Length	128
Frozen Layers	All encoder layers except last 2
Trainable	Last 2 transformer blocks + classifier head
Training Data	JSONL PII annotations
Loss Function	Weighted CrossEntropy
⚙️ Training Configuration
Parameter	Value
Epochs	20
Effective train samples/epoch	200 (subsampled)
Batch size	8
Optimizer	AdamW
Learning Rate	5e-5
Warmup	10%
Gradient Checkpointing	Enabled
Device	GPU (auto)
🎯 Class Weights (PII-focused)
Entity	Weight
CREDIT_CARD	5.0
PHONE	4.0
EMAIL	4.0
PERSON_NAME	3.0
DATE	2.0
CITY	1.5
LOCATION	1.2
O (non-PII)	1.0
📉 Training Progress
Metric	Value
Initial Loss (epoch 1)	2.33
Final Loss (epoch 20)	0.432
Convergence	Stable, no overfitting observed

The model converged smoothly despite freezing lower layers and sub-sampling data.

🧪 Evaluation Results (Dev Set)
Span-level F1 (per entity)
Entity	Precision	Recall	F1
CITY	0.842	0.795	0.818
CREDIT_CARD	0.903	0.879	0.891
DATE	0.861	0.827	0.844
EMAIL	0.888	0.854	0.871
PERSON_NAME	0.836	0.801	0.818
PHONE	0.874	0.852	0.863

Macro-F1: 0.851
PII-Only Macro F1: 0.851
Non-PII F1: 0.904
Token Accuracy: 0.852 (~85%)

⚡ Latency Benchmarks (Inference)

Tested with batch size = 1, max_length = 128:

Percentile	Latency
p50	14 ms
p95	23 ms

Ideal for near-real-time pipelines.

📦 Directory Structure
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

▶️ Training
python train.py \
    --model_name prajjwal1/bert-small \
    --epochs 20 \
    --batch_size 8 \
    --lr 5e-5

🔎 Inference Example
python predict.py \
    --model_dir out \
    --input data/dev.jsonl \
    --output out/dev_pred.json

📘 Model Card Summary

Lightweight, fast, and deployable PII NER model

Good performance across structured (EMAIL, PHONE, CREDIT_CARD) and semi-structured (NAME, CITY, DATE) entities