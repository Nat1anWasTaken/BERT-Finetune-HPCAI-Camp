#!/bin/bash
#SBATCH -A ACD114003
#SBATCH -p gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -J bert-train
#SBATCH -o bert-train.out
#SBATCH -e bert-train.err

module purge
module load miniconda3/conda24.5.0_py3.9

conda activate camp-ai

model_flag=""
output_flag=""

if [[ $1 != "" ]]; then
    model_flag="--model $1"
fi

if [[ $2 != "" ]]; then
    output_flag="--output $2"
fi

python Train.py $model_flag $output_flag
