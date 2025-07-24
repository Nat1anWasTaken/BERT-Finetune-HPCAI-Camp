#!/usr/bin/env python
"""Optimized BERT fine‑tuning script for mteb/tweet_sentiment_extraction (transformers==4.53.3).

## 特色
- **LLRD (Layer-wise LR Decay)**：底層 0.9^n 衰減，保留通用語言特徵。
- **SWA (Stochastic Weight Averaging)**：最後 20 % epoch 加權平均。
- **Stratified 90/10 split**：以 label 分層切分，避免偏差。
- **Class‑weighted loss**：處理資料不平衡。
- **`eval_strategy` & `save_strategy`="epoch"**：符合 ≥4.46 的 API。

保持 BERT 與資料集不變；`inference.py` 可直接載入輸出權重。
"""

import argparse
import os
import random

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, update_bn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def stratified_split(dataset, test_size=0.1):
    idx_train, idx_val = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        shuffle=True,
        stratify=dataset["label"],
        random_state=SEED,
    )
    return dataset.select(idx_train), dataset.select(idx_val)


def compute_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


def get_metrics_fn():
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "precision": precision.compute(predictions=preds, references=labels, average="macro")[
                "precision"
            ],
            "recall": recall.compute(predictions=preds, references=labels, average="macro")["recall"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    return compute


class WeightedLLRDTrainer(Trainer):
    """Trainer with class‑weighted loss & Layer‑wise LR Decay (LLRD)."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    # --- custom loss -----------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 捕捉 num_items_in_batch 等額外參數
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    # --- optimizer with LLRD -------------------------------------------
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        base_lr = self.args.learning_rate
        layer_decay = 0.9
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = []

        # Encoder 12 layers
        for layer_idx in range(12):
            lr = base_lr * (layer_decay ** (11 - layer_idx))
            for name, param in self.model.named_parameters():
                if f"bert.encoder.layer.{layer_idx}." in name:
                    wd = 0.0 if any(nd in name for nd in no_decay) else self.args.weight_decay
                    param_groups.append({"params": [param], "lr": lr, "weight_decay": wd})

        # Pooler & classifier
        head_lr = base_lr * 5
        for name, param in self.model.named_parameters():
            if any(h in name for h in ["bert.pooler", "classifier"]):
                wd = 0.0 if any(nd in name for nd in no_decay) else self.args.weight_decay
                param_groups.append({"params": [param], "lr": head_lr, "weight_decay": wd})

        self.optimizer = AdamW(param_groups, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon)
        return self.optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--output", default="./train_checkpoints")
    args = parser.parse_args()

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- dataset ---------------------------------------------------------
    raw_train = load_dataset("mteb/tweet_sentiment_extraction", split="train")
    raw_test = load_dataset("mteb/tweet_sentiment_extraction", split="test")
    train_ds, val_ds = stratified_split(raw_train)

    def tokenize(batch):
        tokens = tokenizer(batch["text"], padding=False, truncation=True, max_length=128, return_attention_mask=True)
        tokens["labels"] = batch["label"]
        return tokens

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)
    test_ds = raw_test.map(tokenize, batched=True, remove_columns=raw_test.column_names)

    # --- model -----------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        trust_remote_code=True,
    )

    class_weights = compute_class_weights(train_ds["labels"])

    # --- TrainingArguments ---------------------------------------------
    per_device_bs = 64
    training_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine_with_restarts",
        fp16=True,
        eval_strategy="epoch",  # ≥4.46 使用 eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
    )

    # --- Trainer ---------------------------------------------------------
    trainer = WeightedLLRDTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=get_metrics_fn(),
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # --- SWA ------------------------------------------------------------
    swa_model = AveragedModel(model)
    swa_start = int(training_args.num_train_epochs * 0.8)

    print("Starting training …")
    for epoch in range(int(training_args.num_train_epochs)):
        trainer.train(resume_from_checkpoint=None)
        if epoch >= swa_start:
            swa_model.update_parameters(model)

    update_bn(trainer.get_train_dataloader(), swa_model, device=device)
    trainer.model.load_state_dict(swa_model.state_dict())

    print("\nEvaluating …")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    print("Validation:", val_metrics)
    print("Test:", test_metrics)

    trainer.save_model()
    tokenizer.save_pretrained(args.output)

    if os.path.isdir(args.output):
        ckpts = [d for d in os.listdir(args.output) if d.startswith("checkpoint-")]
        ckpts.sort(key=lambda x: int(x.split("-")[-1]))
        print("\nSaved checkpoints:")
        for ck in ckpts:
            print(" •", ck)


if __name__ == "__main__":
    main()

