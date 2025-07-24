import torch
import evaluate
import argparse
import numpy as np
import os
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset"""
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    return dict(zip(unique_labels, class_weights))

def main(args):
    # Set up distributed training if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for distributed training")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model with optimizations
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, 
        trust_remote_code=True, 
        num_labels=3,
        id2label=id2label, 
        label2id=label2id,
        hidden_dropout_prob=0.1,  # Optimal dropout for fine-tuning
        attention_probs_dropout_prob=0.1,
        classifier_dropout=0.1,
        low_cpu_mem_usage=True
    )

    # Load and analyze dataset
    train_dataset = load_dataset("mteb/tweet_sentiment_extraction", split="train")
    test_dataset = load_dataset("mteb/tweet_sentiment_extraction", split="test")
    
    print(f"Full training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Use full dataset instead of just 100 samples for better accuracy
    # Create proper train/validation split (90/10 split)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Split the training data
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(train_dataset)))
    
    full_train_dataset = train_dataset.select(train_indices)
    val_dataset = train_dataset.select(val_indices)
    
    print(f"Training samples: {len(full_train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Calculate class weights for balanced training
    train_labels = [example['label'] for example in full_train_dataset]
    class_weights = compute_class_weights(train_labels)
    print(f"Class weights: {class_weights}")
    
    # Enhanced metrics function
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate multiple metrics
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Optimized preprocessing with proper column handling
    def preprocess_data(examples):
        # Tokenize the text with proper parameters
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False,  # Dynamic padding handled by data collator
            max_length=128,  # Sufficient for tweets
            return_attention_mask=True,
            return_token_type_ids=False  # BERT base doesn't need token_type_ids for single sentence
        )
        # Ensure labels are properly formatted
        tokenized["labels"] = examples["label"]
        return tokenized

    # Process datasets and remove unnecessary columns
    full_train_dataset = full_train_dataset.map(
        preprocess_data, 
        batched=True,
        remove_columns=full_train_dataset.column_names  # Remove all original columns
    )
    val_dataset = val_dataset.map(
        preprocess_data, 
        batched=True,
        remove_columns=val_dataset.column_names  # Remove all original columns
    )
    test_dataset = test_dataset.map(
        preprocess_data, 
        batched=True,
        remove_columns=test_dataset.column_names  # Remove all original columns
    )

    print("Data preprocessing completed.")
    
    # Debug: Print dataset structure
    print("Dataset columns after preprocessing:")
    print(f"Train: {full_train_dataset.column_names}")
    print(f"Val: {val_dataset.column_names}")
    print(f"Test: {test_dataset.column_names}")
    
    # Verify a sample
    print("Sample from processed dataset:")
    print(full_train_dataset[0])
    
    # Calculate optimal batch size for 8 V100s (16GB each)
    # Tesla V100 optimal batch sizes: multiples of 8 for fp16
    per_device_batch_size = 64  # Optimized for V100 memory and throughput
    gradient_accumulation_steps = 2  # Effective batch size = 64 * 8 * 2 = 1024
    
    # Calculate total steps for learning rate scheduling
    total_steps = (len(full_train_dataset) // (per_device_batch_size * torch.cuda.device_count() * gradient_accumulation_steps)) * 8
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Optimized training arguments for 8 V100 GPUs
    training_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        
        # Training configuration
        num_train_epochs=12,  # More epochs for better convergence with full dataset
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Optimization settings (based on research)
        learning_rate=2e-5,  # Optimal for BERT fine-tuning (BERT paper recommendation)
        weight_decay=0.01,   # AdamW weight decay
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,   # Gradient clipping
        
        # Learning rate scheduling
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",  # Cosine annealing for better convergence
        
        # Memory and speed optimizations
        fp16=True,  # Mixed precision training for V100
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # Reduced to avoid data loading issues
        remove_unused_columns=True,  # Let trainer handle column removal
        group_by_length=True,  # Group similar length sequences for efficiency
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=250,  # More frequent evaluation
        save_strategy="steps", 
        save_steps=250,
        save_total_limit=5,  # Keep more checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # Logging
        logging_dir="./logs",
        logging_steps=50,  # More frequent logging
        logging_first_step=True,
        report_to=None,  # Disable wandb/tensorboard for now
        
        # Distributed training optimization
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        ddp_timeout=3600,  # 1 hour timeout for large training
        dataloader_drop_last=True,  # For consistent batch sizes across GPUs
        
        # Prediction and evaluation optimizations
        prediction_loss_only=False,
        ignore_data_skip=False,
        
        # Advanced optimizations for Tesla V100
        # tf32=False,  # V100 doesn't support TF32, keep disabled
        
        # Remove early_stopping_patience as it's not a TrainingArguments parameter
    )

    # Data collator for dynamic padding
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding=True,
        return_tensors="pt"
    )

    # Custom trainer with class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Apply class weights
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([class_weights[0], class_weights[1], class_weights[2]], 
                                  device=model.device, dtype=torch.float32)
            )
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss

    # Do not modify or remove this line, this is for us to check your configurations
    # If your submission result doesn't include this, it will be treated as invalid result
    # ================ DO NOT MODIFY OR REMOVE ==============================
    print(f"Training Config: {training_args}")
    # ================ DO NOT MODIFY OR REMOVE ==============================

    # Initialize trainer with optimizations
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=full_train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Add early stopping callback
    from transformers import EarlyStoppingCallback
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    print("Starting training with optimized configuration...")
    print(f"Using {torch.cuda.device_count()} GPUs")
    print(f"Effective batch size: {per_device_batch_size * torch.cuda.device_count() * gradient_accumulation_steps}")
    
    # Train the model
    train_result = trainer.train()
    
    # Final evaluation
    print("\n" + "="*50)
    print("TRAINING COMPLETED - FINAL EVALUATION")
    print("="*50)
    
    # Evaluate on validation set
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Results: {val_results}")
    
    # Evaluate on test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test Results: {test_results}")
    
    # Save the best model
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    
    # Print detailed results
    print(f"\nFinal Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Final Test F1-Score: {test_results['eval_f1']:.4f}")
    print(f"Final Test Precision: {test_results['eval_precision']:.4f}")
    print(f"Final Test Recall: {test_results['eval_recall']:.4f}")

    # After training, list the checkpoint directories to show the output checkpoint number(s)
    print("\nSaved checkpoints in output directory:")
    if os.path.exists(args.output):
        checkpoints = [d for d in os.listdir(args.output) if d.startswith("checkpoint-")]
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1)
        for ckpt in checkpoints:
            print(f"  {ckpt}")
    else:
        print("Output directory does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google-bert/bert-base-uncased", 
                       help="The model name or the path to model's directory.")
    parser.add_argument("--output", type=str, default="./train_checkpoints", 
                       help="Training checkpoints output directory.")

    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Output directory: {args.output}")

    main(args)
