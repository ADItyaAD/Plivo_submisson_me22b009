import os

# 1. TRAIN MODEL
print("\n===== TRAINING MODEL =====\n")
os.system(
    "python src/train.py "
    "--model_name distilbert-base-uncased "
    "--train data/train.jsonl "
    "--dev data/dev.jsonl "
    "--out_dir out "
    "--epochs 5 "
    "--lr 3e-5 "
    "--batch_size 8"
)

# 2. RUN PREDICTIONS
print("\n===== RUNNING PREDICTIONS ON DEV =====\n")
os.system(
    "python src/predict.py "
    "--model_dir out "
    "--input data/dev.jsonl "
    "--output out/dev_pred.json"
)

# 3. EVALUATE PREDICTIONS
print("\n===== EVALUATING SPAN F1 =====\n")
os.system(
    "python src/eval_span_f1.py "
    "--gold data/dev.jsonl "
    "--pred out/dev_pred.json"
)

# 4. MEASURE LATENCY
print("\n===== MEASURING LATENCY =====\n")
os.system(
    "python src/measure_latency.py "
    "--model_dir out "
    "--input data/dev.jsonl "
    "--runs 50"
)

print("\n===== DONE =====\n")
