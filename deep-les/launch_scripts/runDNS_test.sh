#!/bin/bash -l
#
#SBATCH --account=s929
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mboden@ethz.ch
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --job-name=DNStransient_test

# ======START===== #

module load daint-gpu cray-fftw h5py
source ~/myvenv-spectralDNS/bin/activate
greasy ./greasy_tasks/run_test.txt
