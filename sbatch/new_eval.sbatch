#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=SF-CJd60-097-026.out
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
		python new_eval.py \
			--part SF-CJd60-097-026 \
			--model_path /scratch/gc2720/2301_sim2real/results/0306_0920_SF-CJd60-097-026_0.0005/models/pose_model_26_0.052537.pth"

## 6010018CSV
## 21092302
## SongFeng_306
## SF-CJd60-097-026
					  