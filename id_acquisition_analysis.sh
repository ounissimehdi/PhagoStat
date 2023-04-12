#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3:00:00
#SBATCH --chdir=.
#SBATCH --output=./bash-log/%A_%a.txt
#SBATCH --error=./bash-log/%A_%a.txt
#SBATCH --job-name=Array-job
#SBATCH --array=0-19

python PhagoStat_interface.py   -acq_path '/root/raw_data/WT-1.czi'\
                                -output_path '/root/microglia_video_microscopy_dataset/WT/WT-1'\
                                -acquisition_offset 0\
                                -scene_identification ${SLURM_ARRAY_TASK_ID}\
                                -to_frames\
                                -reg\
                                -det\
                                -prob_maps\
                                -cell_det -cell_trk\