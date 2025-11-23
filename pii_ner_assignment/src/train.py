import os
import math
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    # lighter model
    ap.add_argument("--model_name", default="prajjwal1/bert-small")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # cap examples per epoch (for faster training)
    ap.add_argument("--max_train_examples_per_epoch", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(
        args.train,
        tokenizer,
        LABELS,
        max_length=args.max_length,
        is_train=True,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    # -------- SPEED TRICK 1: gradient checkpointing --------
    # reduces memory and can help CPU/GPU throughput
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # -------- SPEED TRICK 2: freeze most encoder layers ----
    encoder_layers = []
    base_model = None

    # DistilBERT style
    if hasattr(model, "distilbert"):
        base_model = model.distilbert
        encoder_layers = base_model.transformer.layer
    # BERT style (prajjwal1/bert-small/mini/tiny)
    elif hasattr(model, "bert"):
        base_model = model.bert
        encoder_layers = base_model.encoder.layer

    # freeze all but last N encoder layers
    num_trainable_layers = 2  # you can tweak: 1, 2, 3...
    if encoder_layers:
        if len(encoder_layers) > num_trainable_layers:
            for layer in encoder_layers[:-num_trainable_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
        # else: model is very shallow, just train all layers

    # optionally freeze embeddings too (still good for speed)
    if base_model is not None and hasattr(base_model, "embeddings"):
        for p in base_model.embeddings.parameters():
            p.requires_grad = False
    # -------------------------------------------------------

    # ----- class-weighted loss (build once, on correct device) -----
    weights = torch.ones(len(LABELS), device=args.device)
    for i, lbl in enumerate(LABELS):
        if "CREDIT_CARD" in lbl:
            weights[i] = 5.0
        if "PHONE" in lbl:
            weights[i] = 4.0
        if "EMAIL" in lbl:
            weights[i] = 4.0
        if "PERSON_NAME" in lbl:
            weights[i] = 3.0
        if "DATE" in lbl:
            weights[i] = 2.0
        if "CITY" in lbl:
            weights[i] = 1.5
        if "LOCATION" in lbl:
            weights[i] = 1.2

    loss_fn = CrossEntropyLoss(weight=weights, ignore_index=-100)
    # ---------------------------------------------------------------

    # effective steps per epoch (we cap at 200 examples)
    steps_per_epoch = min(
        len(train_dl),
        math.ceil(args.max_train_examples_per_epoch / args.batch_size),
    )
    total_steps = steps_per_epoch * args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        examples_seen = 0

        for batch_idx, batch in enumerate(
            tqdm(train_dl, desc=f"Epoch {epoch + 1}/{args.epochs}")
        ):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # no extra dropout now

            # Class-weighted loss
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            examples_seen += input_ids.size(0)

            # -------- SPEED TRICK 3: cap 200 examples per epoch --------
            if examples_seen >= args.max_train_examples_per_epoch:
                break
            # -----------------------------------------------------------

        avg_loss = running_loss / max(1, steps_per_epoch)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
