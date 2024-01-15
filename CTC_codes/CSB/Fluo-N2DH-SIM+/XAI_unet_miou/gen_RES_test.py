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

def norm_img(img):
    img = np.array(percentile_normalization(img, 0, 100))
    # img = match_histograms(np.array(img), reference_image)
    np_img = img.astype(np.uint8)

    return np_img


def prepare_data(path_data):
    # Loading the mask
    img = Image.open(path_data).resize((512, 512))

    np_img = norm_img(img)


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


def gen_GIF(imgs_dir, pred_mask_dir):
    # Sorting the file paths
    image_paths = natsorted(glob(os.path.join(imgs_dir, '*.tif')))
    pred_mask_paths = natsorted(glob(os.path.join(pred_mask_dir, '*.tif')))

    # Titles for each image
    titles = ['Input Test Image', 'PhagoStat Prediction']

    # Load a font
    try:
        font = ImageFont.truetype("times new roman bold.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    frames = []  # List to hold the combined images for each frame of the GIF

    # Loop through all the paths
    for (image_path, pred_mask_path) in zip(image_paths, pred_mask_paths):
        # Load the images and convert to arrays
        image = np.array(Image.open(image_path))

        image = norm_img(image)

        pred_mask = np.array(Image.open(pred_mask_path))
        
        # Get the dimensions of the images
        height, width = image.shape
        
        # Color the masks
        pred_mask_colored = color_label_image(pred_mask, height, width)
        
        # Ensure all images are the same size
        image_pil = Image.fromarray(image).resize(pred_mask_colored.size)
        image_pil = image_pil.convert('RGB')  # Convert grayscale to RGB for consistency
        
        # Create a new combined image with space for titles
        combined_image = Image.new('RGB', (image_pil.width * 2 + 30, image_pil.height + 50), (255, 255, 255))  # Make the image white
        
        # Combine images horizontally
        combined_image.paste(image_pil, (0, 0))
        combined_image.paste(pred_mask_colored, (image_pil.width+15, 0))
        
        # Draw the titles under each image
        draw = ImageDraw.Draw(combined_image)
        for i, title in enumerate(titles):
            # Calculate text position
            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_position = (i * image_pil.width + (image_pil.width - text_width) // 2, image_pil.height + 5)
            draw.text(text_position, title, font=font, fill=(0, 0, 0))
        
        # Append to the frames list
        frames.append(combined_image)

    # Save the frames as a GIF
    gif_path = os.path.join(os.path.dirname(pred_mask_dir), os.path.basename(pred_mask_dir)+'_animation.gif')
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

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
    parser.add_argument('--json', help='the json of best params use : best_hyperparameters.json', default='')

    parser.add_argument("--min_area", type=int, help="Minimum area", default=400)
    parser.add_argument("--temporal_len", type=int, help="Temporal length", default=2)
    parser.add_argument("--time_mask_threshold", type=float, help="Mask threshold", default=0.5)
    parser.add_argument("--mask_threshold", type=float, help="Mask threshold", default=0.9)



    parser.add_argument("--conv_num", type=float, help="Model configuration", default=24)

    parser.add_argument('--attunet', action='store_true', help='True when using AttUnet')



    parser.add_argument("--eps", type=float, help="Epsilon", default=0.0001)
    parser.add_argument('--save_gif', action='store_true', help='save the GIF animation of the given sequence')


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

    # reference_image = np.array(Image.open(args.img_ref_path))

    # # Rescaling the cell image
    # reference_image = exposure.rescale_intensity(reference_image, out_range=np.uint16)

    # # Converting the cell image to 8bit representation
    # reference_image = img_as_ubyte(reference_image)

    all_propa_maps = []
    for i in range(len(test_images_paths)):
        # Load the image and mask
        image           =  torch.from_numpy(prepare_data(test_images_paths[i]))

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
        if (i+temporal_len) <= len(all_propa_maps):
            
            # Load the first frame in this time window
            # proba_map = np.mean(all_propa_maps[i:i+temporal_len], axis=(0))
            proba_map = all_propa_maps[i]
            
            # Summing the probability maps over the time window
            proba_sum = np.sum(all_propa_maps[i:i+temporal_len], axis=(0))
            
            # Computting the high probability cell positions
            res = proba_sum > time_mask_threshold*np.max(proba_sum)#*temporal_len
        else:
            # Load the last frames since the binary intersection is ready
            proba_map = np.array(all_propa_maps[i], dtype=np.float32)

        labels = label(res,connectivity=2)
        regions = regionprops(labels)
        centr_list_swindow = []

        for region in regions:
            if region.area> min_area:
                centr_list_swindow.append(region.centroid)

        mask_out = np.zeros(res.shape, dtype=bool)
        for f in range(len(centr_list_swindow)):
            mask_out[int(centr_list_swindow[f][0]), int(centr_list_swindow[f][1])] = True
        markers, _ = ndi.label(mask_out)

        labels = watershed(-proba_map, markers, mask=proba_map>mask_threshold)

        
        sw_corr_regions = regionprops(labels)

        corr_sw_bboxes = []

        for region in sw_corr_regions:
            if region.area< min_area:
                rr, cc = np.where(labels==region.label)
                labels[rr,cc] = 0
        mask_name = os.path.splitext(os.path.basename(test_images_paths[i]))[0]
        mask_name = mask_name[1:]
        mask_name = 'mask'+mask_name
        width, height = Image.open(test_images_paths[i]).size
        Image.fromarray(labels).resize((width, height), Image.NEAREST).save(os.path.join(args.output_seq, mask_name+ '.tif'))

    if args.save_gif:gen_GIF(args.input_seq, args.output_seq)