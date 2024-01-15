#!/bin/bash

# Run the segmentation-only routine gen_RES_test.py with five input parameters:
# input_sequence output_sequence model_path attunet time_mask_threshold mask_threshold min_area temporal_len
# Prerequisities: Python 3.8.13
# torch>=1.13.1
# torchaudio>=0.10.1
# torchvision>=0.11.2
# torchviz>=0.0.2
# tqdm==4.62.3
# scikit-image==0.19.2
# scikit-learn==1.0.2
# pillow>=10.0.1
# numpy>=1.22
# natsort==8.0.0
# matplotlib==3.5.1

# Dataset name
dataset_name="../Fluo-N2DH-GOWT1"

chmod -R u+rw "${dataset_name}"

python gen_RES_test.py --input_seq  "${dataset_name}/01" \
                       --output_seq "${dataset_name}/01_RES" \
                       --model_path "./experiments/AttUnetF24-*/ckpts/best_model.pth" \
                       --attunet --time_mask_threshold 0.1 --mask_threshold 0.01 --min_area 0 --temporal_len 5\
                       --save_gif


echo "First sequence processing done!."

python gen_RES_test.py --input_seq  "${dataset_name}/02" \
                       --output_seq "${dataset_name}/02_RES" \
                       --model_path "./experiments/AttUnetF24-*/ckpts/best_model.pth" \
                       --attunet --time_mask_threshold 0.1 --mask_threshold 0.01 --min_area 0 --temporal_len 5\
                       --save_gif
                      
echo "Second sequence processing done!."