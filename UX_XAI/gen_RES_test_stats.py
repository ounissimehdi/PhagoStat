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

import torch
from PIL import Image
import os
import numpy as np
from unet import Unet, Attention_U_Net
import matplotlib.pyplot as plt
from skimage import exposure, img_as_ubyte
import json

from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
from skimage.exposure import match_histograms
from scipy.ndimage import median_filter
from natsort import natsorted
from glob import glob

import argparse

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from natsort import natsorted
from glob import glob
import os
import colorsys
import csv

def norm_img(img):
    # # Rescaling the cell image
    # img = exposure.rescale_intensity(np.array(img), out_range=np.uint16)

    # # Converting the cell image to 8bit representation
    # img = img_as_ubyte(img)
    
    # img = match_histograms(np.array(img), reference_image)
    np_img = np.array(img).astype(np.uint8)

    return np_img


def prepare_data(path_data, reference_image):
    # Loading the mask
    np_img = np.array(Image.open(path_data).resize((512,512)))

    # np_img = norm_img(img)


    # Add an extra dim if only H, W image
    if len(np_img.shape) == 2:
        np_img = np.expand_dims(np_img, axis=0)
        np_img = np.expand_dims(np_img, axis=0)
    else:
        np_img = np.expand_dims(np_img, axis=0)
    
    # Ensure the imgs to be in [0, 1]
    if np_img.max() > 1: np_img = np_img / 255
    return np_img

# Color mapping function based on centroids
def centroid_to_hsl_rgb(centroid, height, width, saturation=0.5, lightness=0.95):
    normalized_x = centroid[0] / height
    normalized_y = centroid[1] / width
    hue = (normalized_x + normalized_y) / 2
    hsl = (hue, saturation, lightness)
    rgb = tuple(int(x * 255) for x in colorsys.hls_to_rgb(*hsl))
    return rgb

# Function to apply color based on centroid to labeled regions in a mask
def color_label_image(mask, height, width):
    label_img = label(mask)
    regions = regionprops(label_img)
    label_colors = {region.label: centroid_to_hsl_rgb(region.centroid, height, width) for region in regions}
    
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for region in regions:
        colored_mask[label_img == region.label] = label_colors[region.label]
    return Image.fromarray(colored_mask)

def load_hyperparameters(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        return data['best_parameters'], data['mean_iou']

if __name__ == '__main__':

    # how to use:
    # python gen_RES_test.py --input_seq ../DIC-C2DH-HeLa/01 \
    #                        --output_seq ../DIC-C2DH-HeLa/01_RES\

    parser = argparse.ArgumentParser(description="Process input and output sequences along with additional parameters.")

    parser.add_argument("--input_seq", help="Input sequence", default="")
    parser.add_argument("--output_seq", help="Output sequence", default="")
    parser.add_argument("--model_path", help="Path to the trained model", default="./experiments/F24-*/ckpts/best*.pth")
    parser.add_argument("--img_ref_path", help="Path to the reference image", default="./reference.tif")

    parser.add_argument('--json', help='the json of best params use : best_hyperparameters.json', default='')

    parser.add_argument("--min_area", type=int, help="Minimum area", default=0)
    parser.add_argument("--temporal_len", type=int, help="Temporal length", default=1)
    parser.add_argument("--time_mask_threshold", type=float, help="Mask threshold", default=0.9)
    parser.add_argument("--mask_threshold", type=float, help="Mask threshold", default=0.5)



    parser.add_argument("--conv_num", type=float, help="Model configuration", default=24)

    parser.add_argument('--attunet', action='store_true', help='True when using AttUnet', default=True)



    parser.add_argument("--eps", type=float, help="Epsilon", default=0.0001)
    parser.add_argument('--save_gif', action='store_true', help='save the GIF animation of the given sequence', default=False)


    args = parser.parse_args()

    if args.json != '':
        # Load hyperparameters from the JSON file
        best_params, best_mean_iou = load_hyperparameters(args.json)

        # Use the loaded hyperparameters
        temporal_len, min_area, mask_threshold, time_mask_threshold = best_params
        print(f"Loaded Best Hyperparameters: {best_params}")
        print(f"Mean IoU for these parameters: {best_mean_iou}")

    else:
        min_area = args.min_area
        temporal_len = args.temporal_len
        mask_threshold= args.mask_threshold
        time_mask_threshold= args.time_mask_threshold 

    eps = args.eps

    # Chose the GPU cuda devices to make the training go much faster vs CPU use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch.device('cpu')

    if args.attunet:
        # defining the U-Net model
        model = Attention_U_Net(n_channels=1,n_classes=1, conv_num=args.conv_num)

    else:
        # defining the U-Net model
        model = Unet(n_channels=1, n_classes=1, conv_num=args.conv_num)

    # Putting the model inside the device
    model.to(device=device)


    unet_model_path = glob(args.model_path)[0]

    # Load the best model
    model.load_state_dict(torch.load(unet_model_path, map_location=device))

    # Putting the model in evluation mode (no gradients are needed)
    model.eval()

    os.makedirs(args.output_seq, exist_ok = True)

    # All images paths and masks
    test_images_paths = natsorted(glob(os.path.join(args.input_seq, '*.tif')))


    # Iterate over each file in the folder
    for file_name in os.listdir(args.input_seq):

        # List to store all image paths
        test_images_paths = []
        # Check if the file is a CSV file
        if file_name.endswith('.csv'):
            file_path = os.path.join(args.input_seq, file_name)
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header row
                for row in reader:
                    img_path = row[0]  # Assuming the image path is the first column
                    test_images_paths.append(img_path)

            os.makedirs(os.path.join(args.output_seq, unet_model_path.split('/')[2], file_name[:-4]), exist_ok = True)

            all_propa_maps = []
            for i in range(len(test_images_paths)):
                # Load the image and mask
                image           =  torch.from_numpy(prepare_data(test_images_paths[i], 'reference_image'))

                # Load the image and the mask into device memory
                image           =  image.to(device=device, dtype=torch.float32)

                # No need to use the gradients (no backward passes -evaluation only-)
                with torch.no_grad():
                    proba_map = model(image)
                    proba_map = torch.sigmoid(proba_map)
                    np_propa_map = np.array(proba_map.cpu()[0][0])

                    # np_propa_map = median_filter(np_propa_map, size=5)

                    all_propa_maps.append(np_propa_map)


            for i in range(len(all_propa_maps)):
                proba_map = all_propa_maps[i]

                mask_name = os.path.splitext(os.path.basename(test_images_paths[i]))[0]
                mask_name = str(i)+'_mask_'+mask_name
                Image.fromarray(proba_map).save(os.path.join(args.output_seq, unet_model_path.split('/')[2], file_name[:-4], mask_name+ '_prob.tif'))
