#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=test2_FAR
#SBATCH --mail-user=ms14625@nyu.edu
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err

singularity exec --nv --overlay /scratch/ms14625/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; conda activate myenv; pip install torchmetrics; export PYTHONUNBUFFERED=TRUE; cd /scratch/$USER/VTPR; python test_FAR_faster.py"


