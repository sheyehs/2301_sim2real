#!/bin/bash

#SBATCH --job-name=new_train
#SBATCH --output=new_train_SF-CJd60-097-026.out
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
		python new_train.py \
			--lr 5e-4 \
			--part SF-CJd60-097-026"

## 6010018CSV
## 21092302
## SongFeng_306
## SF-CJd60-097-026
					  