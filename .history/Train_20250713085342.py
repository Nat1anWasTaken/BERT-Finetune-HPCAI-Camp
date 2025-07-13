import torch
import evaluate
import argparse
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

id2label = {0: "negative", 1: "netural", 2: "positive"}
label2id = {"negative": 0, "netural": 1, "positive": 2}

def main(args):
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=True, num_labels=3,
                                                                id2label=id2label, label2id=label2id, low_cpu_mem_usage=True,
                                                                device_map=device
                                                                )
    model.to(device)

    # load dataset
    train_dataset = load_dataset("mteb/tweet_sentiment_extraction", split="train")
    print(f"length of dataset is {len(train_dataset)}")

    small_train_dataset = train_dataset.select([i for i in range(100)])
    small_eval_dataset = train_dataset.select([i for i in range(100, 110)])

    # define metrics function
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, label = eval_pred
        pred = np.argmax(logits, axis=-1)
        return metric.compute(predictions=pred, references=label)

    # preprocess
    def preprocess_data(dataframe):
        return tokenizer(dataframe["text"], max_length=160, padding="max_length", truncation=True)

    small_train_dataset = small_train_dataset.map(preprocess_data)
    small_eval_dataset = small_eval_dataset.map(preprocess_data)

    print("========================")

    # train
    train_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        num_train_epochs=1,
        logging_dir="./train_logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True
    )

    # Do not modify or remove this line, this is for us to check your configurations
    # If your submission result doesn't include this, it will be treated as invalid result
    # ================ DO NOT MODIFY OR REMOVE ==============================
    print(f"Training Config: {train_args}")
    # ================ DO NOT MODIFY OR REMOVE ==============================

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics
    )

    print("Start Training...")
    train_result = trainer.train()
    trainer.evaluate()

    # After training, list the checkpoint directories to show the output checkpoint number(s)
    print("\nSaved checkpoints in output directory:")
    if os.path.exists(args.output):
        checkpoints = [d for d in os.listdir(args.output) if d.startswith("checkpoint-")]
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1)
        for ckpt in checkpoints:
            print(f"  {ckpt}")
        if !checkpoints:
            print("No checkpoints found.")
    else:
        print("Output directory does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google-bert/bert-base-uncased", help="The model name or the path to model's directory.")
    parser.add_argument("--output", type=str, default="./train_checkpoints", help="Training checkpoints output directory.")

    args = parser.parse_args()

    print(args.model)
    print(args.output)

    main(args)
