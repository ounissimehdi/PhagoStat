# ------------------------------------------------------------------------------
#
#
#                                 P̶̖̥͈͈͇̼͇̈́̈́̀͋͒̽͊͘͠h̴͙͈̦͗́̓a̴̧͗̾̀̅̇ḡ̶͓̭̝͓͎̰͓̦̭̎́͊̐̂͒͠ơ̵̘͌̿̽̑̈̾Ś̴̠́̓̋̂̃͒͆̚t̴̲͓̬͎̾͑͆͊́̕a̸͍̫͎̗̞͆̇̀̌̑͂t̸̳̼̘͕̯̠̱̠͂̔̒̐̒̕͝͝
#
#
#                                PhagoStat
#                Advanced Phagocytic Activity Analysis Tool
# ------------------------------------------------------------------------------
# Copyright (C) 2023 Mehdi OUNISSI <mehdi.ounissi@icm-institute.org>
#               Sorbonne University, Paris Brain Institute - ICM, CNRS, Inria,
#               Inserm, AP-HP, Paris, 75013, France.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------
# Note on Imported Packages:
# The packages used in this work are imported as is and not modified. If you
# intend to use, modify, or distribute any of these packages, please refer to
# the requirements.txt file and the respective package licenses.
# ------------------------------------------------------------------------------

from glob import glob
from PIL import Image
from natsort import natsorted
import matplotlib.pyplot as plt
import os
from skimage.exposure import match_histograms
import numpy as np

import shutil

def percentile_normalization(img, lower_percentile, upper_percentile):
    img_array = np.array(img)
    lower_bound = np.percentile(img_array, lower_percentile)
    upper_bound = np.percentile(img_array, upper_percentile)

    # Clip the values to the bounds set by the percentiles
    clipped_img_array = np.clip(img_array, lower_bound, upper_bound)

    # Scale the values to be in the range [0, 255]
    normalized_img_array = ((clipped_img_array - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)

    # Convert the array back to an image
    normalized_img = Image.fromarray(normalized_img_array)
    return normalized_img

# Function to apply rotations and save images
def process_and_save_images(ref_path, image_paths, mask_paths, images_folder, masks_folder):
    angles = [0, 90, 180, 270]

    # reference_image = np.array(percentile_normalization(Image.open(ref_path), 0, 100))

    iii = 0
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(percentile_normalization(Image.open(img_path).resize((512, 512)), 0, 100))
        # img = match_histograms(np.array(img), reference_image)
        img = Image.fromarray((img).astype(np.uint8)) 
        mask = Image.open(mask_path).resize((512, 512), Image.NEAREST)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = os.path.splitext(os.path.basename(mask_path))[0]

        for angle in angles:
            rotated_img = img.rotate(angle)
            rotated_mask = mask.rotate(angle)
            rotated_img.save(os.path.join(images_folder, f'{iii}_{img_name}_rot{angle}.tif'))
            rotated_mask.save(os.path.join(masks_folder, f'{iii}_{mask_name}_rot{angle}.tif'))
            iii +=1
            
            # For the flipped versions
            flipped_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_mask = rotated_mask.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_img.save(os.path.join(images_folder, f'{iii}_{img_name}_rot{angle}_flipped.tif'))
            flipped_mask.save(os.path.join(masks_folder, f'{iii}_{mask_name}_rot{angle}_flipped.tif'))

            iii +=1

def mask_to_image_path(mask_path):
    # Replace the mask directory and filename pattern to match the corresponding image path
    return mask_path.replace('_GT/SEG/', '/').replace('man_seg', 't')


if __name__ == '__main__':

    # Setting a random seed for reproducibility
    random_seed = 2024
    np.random.seed(random_seed)

    base_folder = '../dataset'
    dataset_name = 'Fluo-N2DH-SIM+'
    
    # Assume you have a list of image paths and mask paths like this:
    image_paths  = natsorted(glob(os.path.join('../dataset/'+dataset_name+'/01','*.tif')))
    image_paths += natsorted(glob(os.path.join('../dataset/'+dataset_name+'/02','*.tif')))
    mask_paths   = natsorted(glob(os.path.join('../dataset/'+dataset_name+'/01_GT/SEG','*.tif')))
    mask_paths  += natsorted(glob(os.path.join('../dataset/'+dataset_name+'/02_GT/SEG','*.tif')))

    # # Convert mask paths to their corresponding image paths
    # converted_mask_paths = [mask_to_image_path(mask_path) for mask_path in mask_paths]

    # # Filter out image paths that have a corresponding mask
    # image_paths = [img_path for img_path in image_paths if img_path in converted_mask_paths]

    # Def ref image for hist matching
    # ref_path = image_paths[0]

    # shutil.copyfile(ref_path, './reference.tif')

    # Calculate the split index for 80% training, 20% validation
    split_index = int(0.8 * len(image_paths))

    # Shuffle the data before splitting (optional)
    # Create a combined list of pairs and shuffle them
    combined = list(zip(image_paths, mask_paths))
    np.random.shuffle(combined)
    image_paths[:], mask_paths[:] = zip(*combined)

    # Split the data
    train_image_paths = image_paths[:split_index]
    train_mask_paths = mask_paths[:split_index]
    val_image_paths = image_paths[split_index:]
    val_mask_paths = mask_paths[split_index:]

    # Directories for the datasets
    train_images_folder = os.path.join(base_folder, 'train/images')
    train_masks_folder = os.path.join(base_folder, 'train/masks')
    val_images_folder = os.path.join(base_folder, 'val/images')
    val_masks_folder = os.path.join(base_folder, 'val/masks')

    # Create the directories
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_masks_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_masks_folder, exist_ok=True)

    # Process and save the training images and masks
    process_and_save_images('_', train_image_paths, train_mask_paths, train_images_folder, train_masks_folder)

    # Process and save the validation images and masks
    process_and_save_images('_', val_image_paths, val_mask_paths, val_images_folder, val_masks_folder)
