#!/bin/bash -l
#
#SBATCH --account=s929
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --mail-user=mboden@ethz.ch
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --array=0-49

N_PROC=6

HOME_DIR=/scratch/snx3000/mboden/ann_poisson_solver/
SAVE_PATH=/results/run3/coverage_f7/

# Data params
N=64
N_DIM=2
NS_GEN=7680
NS_ANN=7680
BATCH_SIZE=32
FLOW_CASE=Turbulence_decaying

# Network params
MODEL_TYPE=div_grad
WEIGHT_SCALINGS='{"1_to_n":1000,"n_to_1":1000,"conv_1":1,"conv_2":1000}'
USE_NORMALIZATION_LAYER=False
RESCALING_FACTOR=0.015625
USE_DOUBLE_CONV=True
# LAYERS=12
FILTER_SIZE=7
CHANNELS=8
STRIDE=1
ACTIVATION=tanh

# Optimization params
DIV_ALPHA=1000
# LR=0.00001
LR_ALPHA=0
EPSILON=0.00001
BATCH_NORM=False
FLOAT_TYPE=64
NOISE_TYPE=both
NOISE_AMPLITUDE=1e-8
EPOCHS=100000

# Saving params
EPOCH_HISTORY=10
EPOCH_SAVE=100

declare -a RE=(60 80 100 120 140 160)	# 6
declare -a DT_FACTOR=(1 2 5 10)			# 4
declare -a RUN_NUMBER=(1 2 3 4 5)	    # 5

declare -a cases=()

for flow_case in ${FLOW_CASE[@]}; do
for layers in ${LAYERS[@]}; do
for lr in ${LR[@]}; do
for run_number in ${RUN_NUMBER[@]}; do
		cases+=("$N_PROC $HOME_DIR $N $N_DIM $NS_GEN $NS_ANN $BATCH_SIZE $flow_case
				 $MODEL_TYPE $WEIGHT_SCALINGS $USE_NORMALIZATION_LAYER $RESCALING_FACTOR $USE_DOUBLE_CONV
				 $layers $FILTER_SIZE $CHANNELS $STRIDE $ACTIVATION
				 $DIV_ALPHA $lr $LR_ALPHA $EPSILON $BATCH_NORM $FLOAT_TYPE $NOISE_TYPE $NOISE_AMPLITUDE $EPOCHS 
				 $SAVE_PATH $run_number $EPOCH_HISTORY $EPOCH_SAVE")
done
done
done 
done 

# Launch each parameter combination
cd ${HOME_DIR}bash_scripts/atomic_scripts
srun -n 12 python Isotropic_transient.py --N $N $N $N --Re_lam ${RE[$SLURM_ARRAY_TASK_ID]} --dt_factor ${DT_FACTOR[$SLURM_ARRAY_TASK_ID]}