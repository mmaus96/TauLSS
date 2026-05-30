#!/bin/bash -l
#SBATCH -J Fit_reg
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -o output_logs/job%j.out
#SBATCH -e output_logs/job%j.err
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi

date
#

module load python
conda activate cobaya_tau #
# conda activate cencrypt


export PYTHONPATH=${PYTHONPATH}:./
export PYTHONPATH=${PYTHONPATH}:./likelihoods
export OMP_NUM_THREADS=8

echo "Setup done.  Starting to run code ..."

# srun -n 16 -c 8 cobaya-run ./configs/<yaml file>

# srun -n 16 -c 8 cobaya-run ./configs/Joint_RSDBAO.yaml -r
# srun -n 16 -c 8 cobaya-run ./configs/Joint_RSDBAO_cmb_PR4_DR6.yaml -r

# srun -n 16 -c 8 cobaya-run ./configs/Joint_RSDBAO_kk_PR4_DR6.yaml -r

# srun -n 16 -c 8 cobaya-run ./configs/test_RSDBAO_jax_w0wa_tau_h.yaml
srun -n 16 -c 8 cobaya-run ./configs/test_RSDBAO_jax_w0wa_tau_thetastar.yaml