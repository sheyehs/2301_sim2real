#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200GB
#SBATCH --time=150:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys5560@nyu.edu
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
	    --overlay /scratch/gc2720/2301_sim2real/envs/environment_train.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh;
                      cd /scratch/gc2720/2301_sim2real;
                      python eval.py --model /scratch/gc2720/2301_sim2real/results/0209_lr_0.0005/pose_model_99_0.024581.pth \
					  				 --dataset data_real_new.hdf5 \
					  				 --split_file eval_on_real.txt"
					  