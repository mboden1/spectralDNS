#!/bin/bash -l
#
#SBATCH --account=s929
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mboden@ethz.ch
#SBATCH --time=20:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --job-name=DNSsingle

# ======START===== #

module load daint-gpu cray-fftw h5py GREASY
source ~/myvenv-spectralDNS/bin/activate

cd /scratch/snx3000/mboden/spectralDNS/deep-les/spectralDNS/
srun -n 24 python3 Isotropic_transient.py --N 256 256 256 --forcing_mode constant_eps --init_mode Lamorgese --save_path ../results/DNS_single/ --Re_lam 60 --dt_ratio 2 --run 1 NS