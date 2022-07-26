#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --job-name=sim_sam
# #SBATCH --time=02:00:00
#SBATCH --output=/home/rmcg/logs/job_%j
# email notifications (NONE, BEGIN, END, FAIL, REQUEUE, ALL)
#SBATCH --mail-type=NONE
#SBATCH --mail-user=rmcg@roe.ac.uk
#SBATCH --constraint=datadisk
#################

echo "Job: ${SLURM_JOBID}"
echo "Hostname: $(hostname)"

module load julia
cd /home/rmcg/sim_sam
for script in "$@"; do
    /home/rmcg/miniconda3/envs/sim_sam/bin/python -u $script \
        --sim=tng \
        --snap=99 \
        --run=1  \
        --box_size=100
done

echo "Job complete"
