#!/bin/bash

#SBATCH --job-name=split_train_test
#SBATCH --output=split_train_test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200GB
#SBATCH --time=150:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys5560@nyu.edu

module purge

singularity exec --nv \
	    --overlay /scratch/gc2720/2301_sim2real/env/environment_train.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh;
                      cd /scratch/gc2720/2301_sim2real/preprocess_data;
                      python split_train_test.py"
					  