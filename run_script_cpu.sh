#!/bin/bash

#SBATCH --job-name=cal_cpu
#SBATCH --begin=now
#SBATCH --partition=gpu,gpub
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=15000M
#SBATCH --time=5:00:00

source /home/yliang/work/anaconda3/bin/activate vis

# Extra output
echo "Computing job "${SLURM_JOB_ID}" on "$(hostname)

# Execute programs
srun python /home/yliang/work/DAFormer/3.6-dimension_reduce_before_after_training.py

echo "Job finished"#