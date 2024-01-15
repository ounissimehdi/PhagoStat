#!/bin/bash

# Dataset name
dataset_name="Fluo-N2DL-HeLa"

# Directory to save the downloaded ZIP file
DIR="../../CTC_datasets/"

# URL of the ZIP file
ZIP_URL="http://data.celltrackingchallenge.net/test-datasets/"$dataset_name".zip"

mkdir -p "$DIR"

chmod -R u+rw "$DIR"

# Filename of the downloaded ZIP file
ZIP_FILE=""$dataset_name".zip"

# Download the ZIP file using wget
wget -O "${DIR}${ZIP_FILE}" "$ZIP_URL"

# Check if the download was successful
if [ -f "${DIR}${ZIP_FILE}" ]; then
    # Unzip the file
    unzip "${DIR}${ZIP_FILE}" -d "$DIR"

    # (UNCOMMENT TO:) Delete the ZIP file
    rm "${DIR}${ZIP_FILE}"
else
    echo "Download failed: ZIP file not found."
fi

chmod -R u+rw "${DIR}${dataset_name}"

python gen_RES_test.py --input_seq  "${DIR}${dataset_name}/01" \
                       --output_seq "${DIR}${dataset_name}/01_RES" \
                       --model_path "./experiments/AttUnetF24-*/ckpts/best_model.pth" \
                       --attunet --save_gif\
                       --time_mask_threshold 0.1 --mask_threshold 0.01 --min_area 0 --temporal_len 3

echo "First sequence processing done!."

python gen_RES_test.py --input_seq  "${DIR}${dataset_name}/02" \
                       --output_seq "${DIR}${dataset_name}/02_RES" \
                       --model_path "./experiments/AttUnetF24-*/ckpts/best_model.pth" \
                       --attunet --save_gif\
                       --time_mask_threshold 0.1 --mask_threshold 0.01 --min_area 0 --temporal_len 3
                      
echo "Second sequence processing done!."