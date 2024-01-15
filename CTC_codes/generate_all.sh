#!/bin/bash

# Create conda env then run all available datasets with:
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

# Install the conda env
conda env create -f environment.yml

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Define the Conda environment name
env_name="phagostat"

echo "Activating Conda environment: $env_name"
# Activate the Conda environment
conda activate "$env_name"

# Check if the environment is activated successfully
if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment: $env_name"
    exit 1
fi

# Get the current working directory
work_dir=$(pwd)

# Define the array of dataset names
dataset_names=('DIC-C2DH-HeLa' 'Fluo-N2DH-GOWT1' 'Fluo-N2DH-SIM+' 'Fluo-N2DL-HeLa' 'PhC-C2DH-U373' 'PhC-C2DL-PSC')

# Loop through the array
for dataset_name in "${dataset_names[@]}"; do
    echo "Processing dataset: $dataset_name"

    # URL of the ZIP file
    ZIP_URL="http://data.celltrackingchallenge.net/test-datasets/${dataset_name}.zip"

    # Filename of the downloaded ZIP file
    ZIP_FILE="${dataset_name}.zip"

    echo "Downloading test dataset: $ZIP_URL"

    # Download the ZIP file using wget
    wget -O "${work_dir}/CSB/${dataset_name}/${ZIP_FILE}" "$ZIP_URL"

    # Check if the download was successful
    if [ -f "${work_dir}/CSB/${dataset_name}/${ZIP_FILE}" ]; then
        echo "Download successful. Unzipping the file..."

        # Unzip the file
        unzip "${work_dir}/CSB/${dataset_name}/${ZIP_FILE}" -d "${work_dir}/CSB/${dataset_name}"

        # Uncomment to delete the ZIP file after unzipping
        # rm "${work_dir}/CSB/${dataset_name}/${ZIP_FILE}"
    else
        echo "Download failed: ZIP file not found for $dataset_name. Skipping."
        continue
    fi

    echo "Changing to directory for dataset: $dataset_name"

    # Change directory
    if cd "${work_dir}/CSB/${dataset_name}/XAI_unet_miou"; then
        echo "Successfully changed directory. Now running script for $dataset_name."
        
        # Execute the script
        bash "${dataset_name}.sh"
        echo "$dataset_name script executed successfully."
    else
        echo "Failed to change directory for $dataset_name. Skipping."
        continue
    fi
done

# echo "Deactivating Conda environment: $env_name"
# # Optionally, deactivate the Conda environment at the end of the script
# conda deactivate
