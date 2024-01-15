#!/bin/bash

# Dataset name
dataset_name="Fluo-N2DH-GOWT1"

# URL of the ZIP file
ZIP_URL="http://data.celltrackingchallenge.net/training-datasets/"$dataset_name".zip"

# Directory to save the downloaded ZIP file
DIR="../dataset/"
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

echo "Generating the train/val split ..."
# Create the train 80% and 20% validation split
python data_train_val_prep.py

echo "Starting the training from scratch."
python main_attunet.py --batch_size 6 --n_epoch 50 --n_warm 50
