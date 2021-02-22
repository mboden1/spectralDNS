#!/bin/bash -l
#
#SBATCH --account=s929
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mboden@ethz.ch
#SBATCH --time=10:00:00
#SBATCH --nodes=60
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --job-name=KSGP64L22_0_greasy_autoencoder_hyperparam_study
#SBATCH --output=/scratch/snx3000/mboden/spectralDNS/deep-les/results/DNS_transient/_logs/DNS_transient_out_JID%j_A%a.txt
#SBATCH --error=/scratch/snx3000/mboden/spectralDNS/deep-les/results/DNS_transient/_logs/DNS_transient_err_JID%j_A%a.txt

# ======START===== #

module load daint-gpu cray-fftw h5py
source ~/myvenv-spectralDNS/bin/activate
greasy ./greasy_tasks/runDNStransient.txt
