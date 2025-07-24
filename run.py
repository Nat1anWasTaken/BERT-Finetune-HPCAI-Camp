#!/usr/bin/env python3
"""
Automated BERT Fine-tuning Script
Continuously trains and evaluates BERT model until target accuracy is achieved.
"""

import subprocess
import re
import time
import os
import glob
from pathlib import Path

class BERTAutoTrainer:
    def __init__(self, target_accuracy=79.43, max_iterations=10):
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.best_accuracy = 0.0
        self.best_checkpoint = None
        
    def submit_training_job(self):
        """Submit training job using sbatch"""
        print(f"\n=== Starting Training Iteration {self.current_iteration + 1} ===")
        
        # Submit training job
        cmd = ["sbatch", "run_train.sh", "google-bert/bert-base-uncased", "./output_model"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        if result.returncode != 0:
            print(f"Error submitting training job: {result.stderr}")
            return None
            
        # Extract job ID from sbatch output
        job_id_match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            print(f"Training job submitted with ID: {job_id}")
            return job_id
        else:
            print("Could not extract job ID from sbatch output")
            return None
    
    def wait_for_job_completion(self, job_id):
        """Wait for SLURM job to complete"""
        print(f"Waiting for training job {job_id} to complete...")
        
        while True:
            # Check job status
            result = subprocess.run(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            if job_id not in result.stdout:
                print("Training job completed!")
                break
                
            print("Job still running... checking again in 30 seconds")
            time.sleep(30)
    
    def find_latest_checkpoint(self):
        """Find the latest checkpoint in output_model directory"""
        checkpoint_pattern = "./output_model/checkpoint-*"
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            print("No checkpoints found!")
            return None
            
        # Sort checkpoints by the number after 'checkpoint-'
        def extract_step(checkpoint_path):
            match = re.search(r'checkpoint-(\d+)', checkpoint_path)
            return int(match.group(1)) if match else 0
            
        latest_checkpoint = max(checkpoints, key=extract_step)
        print(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    def run_inference(self, checkpoint_path):
        """Run inference on the given checkpoint"""
        print(f"Running inference on {checkpoint_path}")
        
        # Submit inference job
        cmd = ["sbatch", "run_inf.sh", checkpoint_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        if result.returncode != 0:
            print(f"Error submitting inference job: {result.stderr}")
            return None
            
        # Extract job ID
        job_id_match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            print(f"Inference job submitted with ID: {job_id}")
            
            # Wait for completion
            self.wait_for_job_completion(job_id)
            return job_id
        else:
            print("Could not extract job ID from inference sbatch output")
            return None
    
    def parse_accuracy(self):
        """Parse accuracy from bert-inf.out file"""
        try:
            with open('./bert-inf.out', 'r') as f:
                content = f.read()
                
            # Look for accuracy pattern
            accuracy_match = re.search(r'The generation accuracy is ([\d.]+) %', content)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                print(f"Parsed accuracy: {accuracy}%")
                return accuracy
            else:
                print("Could not find accuracy in output file")
                return None
                
        except FileNotFoundError:
            print("bert-inf.out file not found")
            return None
        except Exception as e:
            print(f"Error parsing accuracy: {e}")
            return None
    
    def backup_best_results(self, checkpoint_path, accuracy):
        """Backup the best performing checkpoint"""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_checkpoint = checkpoint_path
            
            # Create backup directory
            backup_dir = f"./best_model_acc_{accuracy:.2f}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy checkpoint files
            subprocess.run(f"cp -r {checkpoint_path}/* {backup_dir}/", shell=True)
            print(f"Backed up best model to {backup_dir}")
    
    def run(self):
        """Main training loop"""
        print(f"Starting automated BERT fine-tuning")
        print(f"Target accuracy: {self.target_accuracy}%")
        print(f"Maximum iterations: {self.max_iterations}")
        
        while self.current_iteration < self.max_iterations:
            print(f"\n{'='*50}")
            print(f"ITERATION {self.current_iteration + 1}/{self.max_iterations}")
            print(f"{'='*50}")
            
            # Step 1: Submit training job
            train_job_id = self.submit_training_job()
            if not train_job_id:
                print("Failed to submit training job, skipping iteration")
                continue
                
            # Step 2: Wait for training completion
            self.wait_for_job_completion(train_job_id)
            
            # Step 3: Find latest checkpoint
            checkpoint_path = self.find_latest_checkpoint()
            if not checkpoint_path:
                print("No checkpoint found, skipping iteration")
                continue
                
            # Step 4: Run inference
            inf_job_id = self.run_inference(checkpoint_path)
            if not inf_job_id:
                print("Failed to run inference, skipping iteration")
                continue
                
            # Step 5: Parse accuracy
            accuracy = self.parse_accuracy()
            if accuracy is None:
                print("Failed to parse accuracy, skipping iteration")
                continue
                
            # Step 6: Check if target reached
            print(f"\nCurrent accuracy: {accuracy}%")
            print(f"Target accuracy: {self.target_accuracy}%")
            
            # Backup if this is the best so far
            self.backup_best_results(checkpoint_path, accuracy)
            
            if accuracy > self.target_accuracy:
                print(f"\n🎉 SUCCESS! Target accuracy {self.target_accuracy}% exceeded!")
                print(f"Final accuracy: {accuracy}%")
                print(f"Best checkpoint: {checkpoint_path}")
                return True
                
            print(f"Accuracy {accuracy}% is below target {self.target_accuracy}%")
            self.current_iteration += 1
            
            if self.current_iteration < self.max_iterations:
                print("Starting next iteration...")
                time.sleep(10)  # Brief pause before next iteration
        
        print(f"\n❌ Maximum iterations ({self.max_iterations}) reached")
        print(f"Best accuracy achieved: {self.best_accuracy}%")
        print(f"Best checkpoint: {self.best_checkpoint}")
        return False

def main():
    # Configuration
    TARGET_ACCURACY = 79.43  # Change this to your desired target
    MAX_ITERATIONS = 10      # Maximum number of training iterations
    
    # Create trainer instance
    trainer = BERTAutoTrainer(
        target_accuracy=TARGET_ACCURACY,
        max_iterations=MAX_ITERATIONS
    )
    
    # Run the automated training
    success = trainer.run()
    
    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n⚠️  Training stopped without reaching target accuracy")

if __name__ == "__main__":
    main()
