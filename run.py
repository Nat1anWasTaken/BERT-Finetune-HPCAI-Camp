def wait_for_job_completion(self, job_id):
        """Wait for SLURM job to complete"""
        print(f"Waiting for job {job_id} to complete...")
        
        while True:
            # Check job status for specific user
            result = subprocess.run(['squeue', '-u', self.user_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            # Check if job is still in the queue
            if job_id not in result.stdout:
                print(f"Job {job_id} completed!")
                break
                
            print("Job still running... checking again in 30 seconds")
            time.sleep(30)#!/usr/bin/env python3
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
    def __init__(self, target_accuracy=79.43, max_iterations=10, user_id="u9603854"):
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.best_accuracy = 0.0
        self.best_checkpoint = None
        self.user_id = user_id
        self.previous_accuracy = 0.0
        self.tested_checkpoints = []
        
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
    
    def get_all_checkpoints_sorted(self):
        """Get all checkpoints sorted from smallest to largest"""
        checkpoint_pattern = "./output_model/checkpoint-*"
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            print("No checkpoints found!")
            return []
            
        # Sort checkpoints by the number after 'checkpoint-'
        def extract_step(checkpoint_path):
            match = re.search(r'checkpoint-(\d+)', checkpoint_path)
            return int(match.group(1)) if match else 0
            
        sorted_checkpoints = sorted(checkpoints, key=extract_step)
        print(f"Found {len(sorted_checkpoints)} checkpoints:")
        for cp in sorted_checkpoints:
            step = extract_step(cp)
            print(f"  - {cp} (step {step})")
        return sorted_checkpoints
    
    def get_next_untested_checkpoint(self):
        """Get the next checkpoint that hasn't been tested yet"""
        all_checkpoints = self.get_all_checkpoints_sorted()
        
        for checkpoint in all_checkpoints:
            if checkpoint not in self.tested_checkpoints:
                return checkpoint
        
        return None
    
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
        """Main training loop - tests checkpoints from smallest to largest"""
        print(f"Starting automated BERT evaluation")
        print(f"Target accuracy: {self.target_accuracy}%")
        print(f"Strategy: Test checkpoints from smallest to largest, stop if accuracy drops")
        
        # First, run one training iteration to ensure we have checkpoints
        print(f"\n{'='*50}")
        print(f"INITIAL TRAINING")
        print(f"{'='*50}")
        
        train_job_id = self.submit_training_job()
        if train_job_id:
            self.wait_for_job_completion(train_job_id)
        
        # Now test checkpoints from smallest to largest
        checkpoint_iteration = 0
        
        while True:
            checkpoint_iteration += 1
            print(f"\n{'='*50}")
            print(f"CHECKPOINT EVALUATION {checkpoint_iteration}")
            print(f"{'='*50}")
            
            # Get next untested checkpoint
            checkpoint_path = self.get_next_untested_checkpoint()
            if not checkpoint_path:
                print("No more untested checkpoints available")
                if checkpoint_iteration == 1:
                    print("No checkpoints found at all. Training may have failed.")
                    return False
                else:
                    print(f"All checkpoints tested. Best accuracy: {self.best_accuracy}%")
                    break
            
            # Mark this checkpoint as tested
            self.tested_checkpoints.append(checkpoint_path)
            
            # Run inference on this checkpoint
            inf_job_id = self.run_inference(checkpoint_path)
            if not inf_job_id:
                print("Failed to run inference, skipping checkpoint")
                continue
                
            # Parse accuracy
            accuracy = self.parse_accuracy()
            if accuracy is None:
                print("Failed to parse accuracy, skipping checkpoint")
                continue
                
            print(f"\nCheckpoint: {checkpoint_path}")
            print(f"Current accuracy: {accuracy}%")
            print(f"Previous accuracy: {self.previous_accuracy}%")
            print(f"Target accuracy: {self.target_accuracy}%")
            
            # Check if accuracy dropped compared to previous
            if checkpoint_iteration > 1 and accuracy < self.previous_accuracy:
                print(f"\n⚠️  ACCURACY DROPPED! {accuracy}% < {self.previous_accuracy}%")
                print(f"Stopping evaluation and using previous best checkpoint")
                print(f"Best accuracy: {self.best_accuracy}%")
                print(f"Best checkpoint: {self.best_checkpoint}")
                return self.best_accuracy > self.target_accuracy
            
            # Update best results
            self.backup_best_results(checkpoint_path, accuracy)
            self.previous_accuracy = accuracy
            
            # Check if target reached
            if accuracy > self.target_accuracy:
                print(f"\n🎉 SUCCESS! Target accuracy {self.target_accuracy}% exceeded!")
                print(f"Final accuracy: {accuracy}%")
                print(f"Best checkpoint: {checkpoint_path}")
                return True
            
            print(f"Accuracy {accuracy}% is below target {self.target_accuracy}%")
            print("Testing next checkpoint...")
            time.sleep(5)  # Brief pause before next checkpoint
        
        # Final results
        if self.best_accuracy > self.target_accuracy:
            print(f"\n🎉 SUCCESS! Target accuracy achieved!")
            print(f"Best accuracy: {self.best_accuracy}%")
            print(f"Best checkpoint: {self.best_checkpoint}")
            return True
        else:
            print(f"\n❌ Target accuracy not reached")
            print(f"Best accuracy: {self.best_accuracy}%")
            print(f"Best checkpoint: {self.best_checkpoint}")
            return False

def main():
    # Configuration
    TARGET_ACCURACY = 79.95  # Change this to your desired target
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
