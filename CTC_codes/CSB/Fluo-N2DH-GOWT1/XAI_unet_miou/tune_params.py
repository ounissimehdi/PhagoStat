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
import os
from skimage.segmentation import watershed

from scipy import ndimage as ndi

from unet import Unet, Attention_U_Net
from skimage.exposure import match_histograms
from skimage.measure import regionprops
from scipy.ndimage import label as ndi_label
import numpy as np
import torch
import argparse
from tqdm import tqdm
from itertools import product

from multiprocessing import Pool, Manager
import tempfile
import json

def norm(img_path, reference_image):

    img = Image.open(img_path)
    img = match_histograms(np.array(img), reference_image)
    return Image.fromarray((img).astype(np.uint8)) 
# Function to apply rotations and save images
def process_and_save_images(ref_path, image_paths, mask_paths, images_folder, masks_folder, GT):

    reference_image = np.array(Image.open(ref_path))

    if GT:
        for img_path, mask_path in zip(image_paths, mask_paths):
            img = norm(img_path, reference_image)
            mask = Image.open(mask_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]

            img.save(os.path.join(images_folder, f'{img_name}.tif'))
            mask.save(os.path.join(masks_folder, f'{mask_name}.tif'))
    else:
        for img_path in image_paths:
            img = norm(img_path, reference_image)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img.save(os.path.join(images_folder, f'{img_name}.tif'))


def mask_to_image_path(mask_path):
    # Replace the mask directory and filename pattern to match the corresponding image path
    return mask_path.replace('_GT/SEG/', '/').replace('man_seg', 't')

def prepare_data(path_data):
    # Loading the mask
    np_img = np.array(Image.open(path_data))

    # Add an extra dim if only H, W image
    if len(np_img.shape) == 2:
        np_img = np.expand_dims(np_img, axis=0)
        np_img = np.expand_dims(np_img, axis=0)
    else:
        np_img = np.expand_dims(np_img, axis=0)
    
    # Ensure the imgs to be in [0, 1]
    if np_img.max() > 1: np_img = np_img / 255
    return np_img

def norm_save(GT=False):
     # Def ref image for hist matching
    ref_path = './reference.tif'

    if GT:
        # Assume you have a list of image paths and mask paths like this:
        image_paths  = natsorted(glob(os.path.join('../dataset/*', seq             ,'*.tif')))
        mask_paths   = natsorted(glob(os.path.join('../dataset/*', seq+'_GT', 'SEG','*.tif')))

        # Convert mask paths to their corresponding image paths
        converted_mask_paths = [mask_to_image_path(mask_path) for mask_path in mask_paths]

        # Filter out image paths that have a corresponding mask
        image_paths = [img_path for img_path in image_paths if img_path in converted_mask_paths]

        # Directories for the datasets
        train_images_folder = os.path.join(base_folder, 'tune', 'GT', seq, 'images')
        train_masks_folder = os.path.join(base_folder, 'tune',  'GT', seq, 'masks')

        os.makedirs(train_masks_folder, exist_ok=True)
        os.makedirs(train_images_folder, exist_ok=True)


        process_and_save_images(ref_path, image_paths, mask_paths, train_images_folder, train_masks_folder, GT)

    else:
        # Assume you have a list of image paths and mask paths like this:
        image_paths  = natsorted(glob(os.path.join('../dataset/*', seq             ,'*.tif')))

        # Directories for the datasets
        train_images_folder = os.path.join(base_folder, 'tune', 'RES', seq, 'images')
        train_probs_folder  = os.path.join(base_folder, 'tune', 'RES', seq, 'probs')
   
        # Create the directories
        os.makedirs(train_images_folder, exist_ok=True)
        os.makedirs(train_probs_folder, exist_ok=True)

        # Process and save the training images
        process_and_save_images(ref_path, image_paths, [], train_images_folder, '', GT)

def DL_pred_probs():
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

    test_images_paths = natsorted(glob(os.path.join(base_folder, 'tune', 'RES', seq, 'images', '*.tif')))
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
            name = os.path.basename(test_images_paths[i])
            Image.fromarray(np_propa_map).save(os.path.join(base_folder, 'tune', 'RES', seq, 'probs', name))



def get_labelled_mask(all_propa_maps, min_area, time_mask_threshold, mask_threshold):
    # Summing the probability maps over the time window
    proba_sum = np.sum(all_propa_maps, axis=0)
    
    # Computing the high probability cell positions
    res = proba_sum > time_mask_threshold * np.max(proba_sum)

    labels = ndi_label(res)[0]
    regions = regionprops(labels)

    # Use numpy array operations instead of loops for centroid calculation
    centroids = np.array([region.centroid for region in regions if region.area > min_area])
    
    mask_out = np.zeros(res.shape, dtype=bool)
    if centroids.size > 0:
        mask_out[centroids[:, 0].astype(int), centroids[:, 1].astype(int)] = True

    markers = ndi_label(mask_out)[0]

    # Directly apply the watershed algorithm
    labels = watershed(-all_propa_maps[0], markers, mask=all_propa_maps[0] > mask_threshold)

    # Optimize region removal for small areas
    for region in regionprops(labels):
        if region.area < min_area:
            labels[labels == region.label] = 0

    return labels

# def get_labelled_mask(all_propa_maps, min_area, time_mask_threshold, mask_threshold):

#         # The first probability map is the one corresponding to the current image (presnet)
#         proba_map = all_propa_maps[0]

#         # Summing the probability maps over the time window
#         proba_sum = np.sum(all_propa_maps, axis=(0))
        
#         # Computting the high probability cell positions
#         res = proba_sum > time_mask_threshold*np.max(proba_sum)

#         labels = label(res,connectivity=2)
#         regions = regionprops(labels)
#         centr_list_swindow = []

#         for region in regions:
#             if region.area> min_area:
#                 centr_list_swindow.append(region.centroid)

#         mask_out = np.zeros(res.shape, dtype=bool)
#         for f in range(len(centr_list_swindow)):
#             mask_out[int(centr_list_swindow[f][0]), int(centr_list_swindow[f][1])] = True
#         markers, _ = ndi.label(mask_out)

#         labels = watershed(-proba_map, markers, mask=proba_map>mask_threshold)
        
#         sw_corr_regions = regionprops(labels)

#         for region in sw_corr_regions:
#             if region.area< min_area:
#                 rr, cc = np.where(labels==region.label)
#                 labels[rr,cc] = 0

#         return labels

# def calculate_iou_best_match(predicted_mask, ground_truth_mask):
#     labels_ground_truth = np.unique(ground_truth_mask)
#     labels_predicted = np.unique(predicted_mask)
#     labels_ground_truth = labels_ground_truth[labels_ground_truth != 0]  # Exclude background
#     labels_predicted = labels_predicted[labels_predicted != 0]  # Exclude background

#     iou_scores = []

#     for label_gt in labels_ground_truth:
#         best_iou = 0
#         best_label_pred = None

#         for label_pred in labels_predicted:
#             intersection = np.logical_and(ground_truth_mask == label_gt, predicted_mask == label_pred)
#             union = np.logical_or(ground_truth_mask == label_gt, predicted_mask == label_pred)
#             iou = np.sum(intersection) / np.sum(union)

#             if iou > best_iou:
#                 best_iou = iou
#                 best_label_pred = label_pred

#         if best_label_pred is not None:
#             iou_scores.append(best_iou)

#     # mean_iou = np.mean(iou_scores) if iou_scores else 0
#     return iou_scores


def calculate_iou_best_match(predicted_mask, ground_truth_mask):
    labels_ground_truth = np.unique(ground_truth_mask[ground_truth_mask != 0])  # Exclude background
    labels_predicted = np.unique(predicted_mask[predicted_mask != 0])  # Exclude background

    # Pre-calculate masks for each label
    gt_masks = {label: ground_truth_mask == label for label in labels_ground_truth}
    pred_masks = {label: predicted_mask == label for label in labels_predicted}

    iou_scores = []

    for label_gt, gt_mask in gt_masks.items():
        best_iou = 0

        for label_pred, pred_mask in pred_masks.items():
            intersection = np.sum(np.logical_and(gt_mask, pred_mask))
            union = np.sum(np.logical_or(gt_mask, pred_mask))

            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou

        if best_iou>=0.5: iou_scores.append(best_iou)
        else: iou_scores.append(0)

    return iou_scores

def preload_all_gt_masks(seqs, base_folder):
    # Preload ground truth masks to avoid repeated file reads
    ground_truth_masks = {}
    for seq in seqs:
        train_masks = natsorted(glob(os.path.join(base_folder, 'tune', 'GT', seq, 'masks', '*.tif')))
        for mask_path in train_masks:
            mask_id = int(os.path.basename(mask_path).replace('man_seg', '').replace('.tif', ''))
            ground_truth_masks[(seq, mask_id)] = np.array(Image.open(mask_path))
    return ground_truth_masks

def preload_all_propa_maps(seqs, base_folder, max_time_window):
    all_maps = {}
    for seq in seqs:
        img_paths = natsorted(glob(os.path.join(base_folder, 'tune', 'RES', seq, 'probs', '*.tif')))
        for mask_id in range(len(img_paths) - max_time_window + 1):
            for time_window in range(1, max_time_window + 1):
                all_propa_maps = [np.array(Image.open(img_paths[i])) for i in range(mask_id, mask_id + time_window)]
                all_maps[(seq, mask_id, time_window)] = all_propa_maps
    return all_maps

def tune_per_combination(time_window, min_area, mask_threshold, time_mask_threshold, all_maps, ground_truth_masks, seq_train_masks):
    comb_all_mean_iou = []

    # Processing
    for seq, mask_paths in seq_train_masks.items():
        for mask_path in mask_paths:
            mask_id = int(os.path.basename(mask_path).replace('man_seg', '').replace('.tif', ''))

            all_propa_maps = all_maps[(seq, mask_id, time_window)]
            ground_truth_mask = ground_truth_masks[(seq, mask_id)]

            predicted_mask = get_labelled_mask(all_propa_maps, min_area, time_mask_threshold, mask_threshold)
            iou_scores = calculate_iou_best_match(predicted_mask, ground_truth_mask)

            comb_all_mean_iou.extend(iou_scores)

    return np.mean(comb_all_mean_iou), (time_window, min_area, mask_threshold, time_mask_threshold)


# def tune_per_combination(time_window, min_area, mask_threshold, time_mask_threshold):
#     comb_all_mean_iou = []
#     for seq in seqs:
#         train_masks = natsorted(glob(os.path.join(base_folder, 'tune', 'GT', seq, 'masks', '*.tif')))
#         for mask_path in train_masks:
#             mask_id = int(os.path.basename(mask_path).replace('man_seg', '').replace('.tif', ''))

#             all_propa_maps = []
#             for img_indx in range(mask_id, mask_id+time_window):
#                 img_path = natsorted(glob(os.path.join(base_folder, 'tune', 'RES', seq, 'probs', '*.tif')))[img_indx]
#                 prob_map = np.array(Image.open(img_path))
#                 all_propa_maps.append(prob_map)

#             predicted_mask    = get_labelled_mask(all_propa_maps, min_area, time_mask_threshold, mask_threshold)
#             ground_truth_mask = np.array(Image.open(mask_path))

#             mean_iou = calculate_iou_best_match(predicted_mask, ground_truth_mask)

#             comb_all_mean_iou += mean_iou
#     return np.mean(comb_all_mean_iou), (time_window, min_area, mask_threshold, time_mask_threshold)  


# def worker_function(params, all_maps, all_gt, seq_train_masks):
#     time_window, min_area, mask_threshold, time_mask_threshold = params
#     mean_iou, _ = tune_per_combination(time_window, min_area, mask_threshold, time_mask_threshold, all_maps, all_gt, seq_train_masks)

#     # Specify a directory for the temporary file, or leave as None for the system default
#     temp_dir = '/network/lustre/iss02/aramis/users/mehdi.ounissi/projects/Rev_SR_v2/*/XAI_unet'  # Set this to a specific path or leave it as None
#     with tempfile.NamedTemporaryFile(delete=False, mode='w', dir=temp_dir) as tmpfile:
#         tmpfile.write(f"{params}: {mean_iou}\n")
#         print(f"Temporary file created: {tmpfile.name}")  # Print the file path
#         return tmpfile.name

def worker_function(args):
    params, all_maps, all_gt, seq_train_masks = args
    time_window, min_area, mask_threshold, time_mask_threshold = params
    mean_iou, _ = tune_per_combination(time_window, min_area, mask_threshold, time_mask_threshold, all_maps, all_gt, seq_train_masks)

    # Specify a directory for the temporary file, or leave as None for the system default
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok = True) 

    # Save the results to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', dir=temp_dir) as tmpfile:
        tmpfile.write(str((params, mean_iou)) + "\n")
        return tmpfile.name
    
def get_cell_sizes():
    min_area = 10**20
    sizes = []
    for seq in seqs:
        train_masks = natsorted(glob(os.path.join(base_folder, 'tune', 'GT', seq, 'masks', '*.tif')))
        for mask in train_masks:
            regions = regionprops(np.array(Image.open(mask)))
            for region in regions:
                sizes.append(region.area)
                if region.area< min_area:
                    min_area = region.area
    return sizes
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input and output sequences along with additional parameters.")
    parser.add_argument("--model_path", help="Path to the trained model", default="./experiments/AttUnetF24-*/ckpts/best*.pth")
    parser.add_argument('--attunet', action='store_true', help='True when using AttUnet')
    parser.add_argument("--conv_num", type=float, help="Model configuration", default=24)

    args = parser.parse_args()

    base_folder = '../dataset'

    seqs = ['01', '02']

    NUMBER_OF_PROCESSES = 6

    # # Normalize training images to be used for tunning params
    for seq in seqs:
        norm_save(GT=True)
        norm_save(GT=False)

    for seq in seqs: DL_pred_probs()
    
    # Tunning
    sizes = get_cell_sizes()

    print(np.min(sizes))

    # Hyperparamerters
    time_windows         = [1, 2]
    min_areas            = [0]#np.min(sizes)*np.linspace(0, 0.5, 10)
    mask_thresholds      = np.linspace(0, 0.5, 10)
    time_mask_thresholds = np.linspace(0.5, .99999, 10)

    print('Pre-loading data ...')
    # Pre-load all possible combinations of all_propa_maps
    max_time_window = int(max(time_windows))
    all_maps = preload_all_propa_maps(seqs, base_folder, max_time_window)
    all_gt   = preload_all_gt_masks(seqs, base_folder)
    seq_train_masks = {seq: natsorted(glob(os.path.join(base_folder, 'tune', 'GT', seq, 'masks', '*.tif'))) for seq in seqs}

    print('Pre-loading data finished...')

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(product(time_windows, min_areas, mask_thresholds, time_mask_thresholds))

    with Manager() as manager:
        # Using Manager to create shared variables
        all_maps_shared = manager.dict(all_maps)
        all_gt_shared = manager.dict(all_gt)
        seq_train_masks_shared = manager.dict(seq_train_masks)

        # Set up a pool of workers
        with Pool(processes=NUMBER_OF_PROCESSES) as pool:
            # Create a list of arguments for each hyperparameter combination
            args_list = [(params, all_maps_shared, all_gt_shared, seq_train_masks_shared) for params in hyperparameter_combinations]

            # Use pool.imap_unordered for lazy iteration over the results
            # Wrap the iterator with tqdm for the progress bar
            results = list(tqdm(pool.imap_unordered(worker_function, args_list), total=len(hyperparameter_combinations)))

    # Process and combine results from temporary files
    combined_results = {}
    for result_file in results:
        with open(result_file, 'r') as file:
            data = file.read()
            param, mean_iou = eval(data)
            combined_results[param] = mean_iou
        os.remove(result_file)  # Clean up the temporary file

    # Find the best hyperparameters
    best_params = max(combined_results, key=combined_results.get)
    best_mean_iou = combined_results[best_params]

    print(f"Best Mean IoU: {best_mean_iou}")
    print(f"Best Hyperparameters: {best_params}")

    # Save the best parameters to a JSON file
    with open('best_hyperparameters.json', 'w') as json_file:
        json.dump({'best_parameters': best_params, 'mean_iou': best_mean_iou}, json_file, indent=4)


