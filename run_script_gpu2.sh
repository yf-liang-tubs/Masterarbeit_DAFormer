#!/bin/bash

#SBATCH --job-name=after
#SBATCH --begin=now
#SBATCH --partition=gpu,gpub
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=80000M
#SBATCH --time=15:00:00

source /home/yliang/work/anaconda3/bin/activate vis

# Extra output
echo "Computing job "${SLURM_JOB_ID}" on "$(hostname)

# Execute programs
srun python /home/yliang/work/DAFormer/0_hook_featureactivate_hrde.py

echo "Job finished"#