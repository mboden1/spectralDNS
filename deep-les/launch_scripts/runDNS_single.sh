#!/bin/bash -l
#
#SBATCH --account=s929
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mboden@ethz.ch
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --job-name=DNSsingle

# ======START===== #

module load daint-gpu cray-fftw h5py GREASY
source ~/myvenv-spectralDNS/bin/activate

cd /scratch/snx3000/mboden/spectralDNS/deep-les/spectralDNS/
srun -n 12 python3 Isotropic_transient_temp.py --N 256 256 256 --forcing_mode constant_eps --init_mode Lamorgese --save_path ../results/DNS_single/ --Re_lam 60 --dt_ratio 2 --ratio_print 64 --ratio_save 64 --run 1 NS