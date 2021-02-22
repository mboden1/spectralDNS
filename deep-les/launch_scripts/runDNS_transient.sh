#!/bin/bash -l
#
#SBATCH --account=s929
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mboden@ethz.ch
#SBATCH --time=20:00:00
#SBATCH --nodes=60
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --job-name=DNStransient

# ======START===== #

module load daint-gpu cray-fftw h5py
source ~/myvenv-spectralDNS/bin/activate
greasy ./greasy_tasks/runDNStransient.txt
