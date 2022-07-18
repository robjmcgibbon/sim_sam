#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --job-name=pairs
#SBATCH --time=05:00:00
#SBATCH --output=/home/rmcg/logs/job_%j
# email notifications (NONE, BEGIN, END, FAIL, REQUEUE, ALL)
#SBATCH --mail-type=NONE
#SBATCH --mail-user=rmcg@roe.ac.uk
#SBATCH --constraint=datadisk
#################

echo "Job: ${SLURM_JOBID}"
echo "Hostname: $(hostname)"

echo "Snap: ${1}"
cd /home/rmcg/sim_sam
/home/rmcg/miniconda3/envs/sim_sam/bin/python -u extract_pairs.py \
    --snap="$1" \
    --box_size=100 \
    --run=1 \
    --sim=tng \
    > /home/rmcg/pairs_log/"$1"

echo "Job complete"
