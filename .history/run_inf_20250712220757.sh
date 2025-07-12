#!/bin/bash
#SBATCH -A ACD114003
#SBATCH -p gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -J bert-inf
#SBATCH -o bert-inf.out
#SBATCH -e bert-inf.err

module purge
module load pkg/miniconda3

conda activate camp-ai

model_flag=""

if [[ $1 != "" ]]; then
    model_flag="--model $1"
fi

python Inference.py $model_flag
