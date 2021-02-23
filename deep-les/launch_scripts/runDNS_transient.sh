#!/bin/bash -l
#
#SBATCH --account=s929
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mboden@ethz.ch
#SBATCH --time=20:00:00
#SBATCH --nodes=45
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --job-name=DNStransient

# ======START===== #

module load daint-gpu cray-fftw h5py GREASY
source ~/myvenv-spectralDNS/bin/activate
greasy ./greasy_tasks/runDNStransient.txt
