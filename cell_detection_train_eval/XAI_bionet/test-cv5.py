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
from bio_net import BiONet
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
from natsort import natsorted
from glob import glob
from tqdm import trange

import pandas as pd
import time

root_path = '..'

main_data_dir = os.path.join(root_path, 'dataset', 'microglial_cells_test_sequences_1024')


def bboxes_prep(pil_mask):
    # Nmpy version of the mask
    np_org_mask = np.array(pil_mask)

    # Labelling the binary mask
    np_label_mask = label(np_org_mask, connectivity=2)

    # Extract regions using the labelled mask
    regions = regionprops(np_label_mask)

    bboxes = []
    for i in range(len(regions)): bboxes.append(regions[i].bbox)

    return bboxes

def IoU(boxA, boxB):
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    try:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    except ZeroDivisionError:  iou = 0
    # return the intersection over union value
    return iou

def F1(input_bboxes, gt_bboxes, IoU_threshold):
    gt_bboxes = gt_bboxes.copy()
    input_bboxes = input_bboxes.copy()
    gt_tp_count = 0
    for gt_bboxe in gt_bboxes:
        if np.sum(gt_bboxe) != 0 : gt_tp_count +=1 
    # Compute IoU of all input bboxes vs GT bboxes
    fp, tp = 0, 0
    for input_bbox in input_bboxes:
        if np.sum(input_bbox) != 0 :
            #print("Input bboxes ",input_bbox)
            # Init the max IoU value
            max_iou = 0
            for i in range(len(gt_bboxes)):
                gt_bboxe = gt_bboxes[i]
                #print("GT bboxes ",gt_bboxe)
                if np.sum(gt_bboxe) != 0 :
                    iou = IoU(gt_bboxe, input_bbox)
                    # Update the Max IoU value
                    if iou>max_iou:
                        index = i
                        max_iou = iou
            # Quanitfy the detection (TP or FP)
            if max_iou> IoU_threshold:
                tp +=1
                gt_bboxes.pop(index)
            else: fp += 1
    # Computing the False Negatives
    fn = gt_tp_count - tp

    # Computing the precision
    precision =  tp / (tp + fp)

    # Computing the recall
    recall    =  tp / (tp + fn)

    # Computing the f1
    f1        =  2 * tp / (2 * tp + fp + fn)

    return f1, recall, precision

def prepare_data(path_data):
    # Loadiung the mask
    pil_org_image = Image.open(path_data)

    np_img = np.array(pil_org_image)

    # Add an extra dim if only H, W image
    if len(np_img.shape) == 2:
        np_img = np.expand_dims(np_img, axis=0)
        np_img = np.expand_dims(np_img, axis=0)
    else:
        np_img = np.expand_dims(np_img, axis=0)
    
    # Ensure the imgs to be in [0, 1]
    if np_img.max() > 1: np_img = np_img / 255
    return np_img


# Chose the GPU cuda devices to make the training go much faster vs CPU use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# defining the U-Net model
#cmodel = UNet(n_channels=1, n_classes=1)
model = BiONet(num_classes= 1, iterations= 1, multiplier= 1.0, num_layers = 4, integrate = False)

# Putting the model inside the device
model.to(device=device)

folds = 5

min_area = 1000#1000
temporal_len = 2
mask_threshold = 0.995
eps = 0.0001

IoU_th = 0.5


model_ckp_str = 'best_model' # best 3
out_seqs_list = ['seq_1', 'seq_2', 'seq_3', 'seq_4']

print(model_ckp_str)
folds_titles, f1_folds, dice_folds, recall_folds, precision_folds, time_list = [], [], [], [], [], []
for fold in trange(1, folds+1,1):

    dice_list, f1_list, recall_list, precision_list = [], [], [], []

    unet_model_path  = './experiments/*F48-K3*/ckpts/FOLD-'+str(fold)
    #unet_model_path += '/best_model.pth'
    print(glob(os.path.join(unet_model_path, model_ckp_str+'*')))
    unet_model_path = glob(os.path.join(unet_model_path, model_ckp_str+'*'))[0]

    folds_titles.append('FOLD-'+str(fold))
    # Load the best model
    model.load_state_dict(torch.load(unet_model_path))

    # Putting the model in evluation mode (no gradients are needed)
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('MODEL TRAINABLE PARAMS:')
    print(pytorch_total_params)
    print()


    
    tic = time.time()
    for seq in out_seqs_list:

        output_path = os.path.join('results', 'FOLD-'+str(fold), seq)
        os.makedirs(output_path, exist_ok = True)

        # Path to the test dataset
        test_dir = os.path.join(main_data_dir, seq)

        # All images paths and masks
        test_images_paths = natsorted(glob(os.path.join(test_dir, '*_img.tif')))
        test_masks_paths  = natsorted(glob(os.path.join(test_dir, '*_binary_mask.tif')))
        
        all_propa_maps = []
        for i in range(len(test_images_paths)):
            # Load the image and mask
            image           =  torch.from_numpy(prepare_data(test_images_paths[i]))
            #true_mask       =  prepare_data(test_masks_paths[i])[0][0]

            # Load the image and the mask into device memory
            image           =  image.to(device=device, dtype=torch.float32)

            # No need to use the gradients (no backward passes -evaluation only-)
            with torch.no_grad():
                proba_map = model(image)
                # proba_map = torch.sigmoid(proba_map)
                np_propa_map = np.array(proba_map.cpu()[0][0])
                # print(np.min(np_propa_map), np.max(np_propa_map))
                all_propa_maps.append(np_propa_map)

        
        for i in range(len(all_propa_maps)):
            if (i+temporal_len) <= len(all_propa_maps):
                
                # Load the first frame in this time window
                # proba_map = np.mean(all_propa_maps[i:i+temporal_len], axis=(0))
                proba_map = all_propa_maps[i]
                
                # Summing the probability maps over the time window
                proba_sum = np.sum(all_propa_maps[i:i+temporal_len], axis=(0))
                
                # Computting the high probability cell positions
                res = proba_sum > mask_threshold*np.max(proba_sum)#*temporal_len
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

            labels = watershed(-proba_map, markers, mask=proba_map>0.5)

            
            sw_corr_regions = regionprops(labels)

            corr_sw_bboxes = []

            for region in sw_corr_regions:
                if region.area< min_area:
                    rr, cc = np.where(labels==region.label)
                    labels[rr,cc] = 0
                    # corr_sw_bboxes.append(region.bbox)
            
            Image.fromarray(labels).save(os.path.join(output_path, str(i)+'.tif'))

            # corr_sw_bboxes =  np.array(corr_sw_bboxes)
            
            # true_mask = np.array(Image.open(test_masks_paths[i]))
            # if true_mask.max() > 1: true_mask = true_mask / 255

            # gt_label = label(true_mask)

            # gt_regions = regionprops(gt_label)

            # np_gt_bboxes = []

            # for region in gt_regions:np_gt_bboxes.append(region.bbox)
            # np_gt_bboxes =  list(np.array(np_gt_bboxes))

            # f1, recall, precision = F1(corr_sw_bboxes, np_gt_bboxes, IoU_th)

            # bn_mask = np.array(labels>0)*1

            # # Ensure the imgs to be in [0, 1]
            # if bn_mask.max() > 1: bn_mask = bn_mask / 255

            # # Computing intersection and union masks
            # inter_mask = np.dot(bn_mask.flatten(), true_mask.flatten())
            # union_mask = np.sum(bn_mask) + np.sum(true_mask) + eps

            # # Computing the Dice coefficient
            # dice = (2 * inter_mask + eps) / union_mask

            # dice_list.append(dice)
            # f1_list.append(f1)
            # recall_list.append(recall)
            # precision_list.append(precision)

    time_list.append(round(float((time.time()-tic)/165),4))

    print('FOlD '+str(fold)+' : ')
    # print("F1        : ", round(float(np.mean(f1_list)*100),2),"%, std :",round(float(np.std(f1_list)*100),2))
    # print("DICE      : ", round(float(np.mean(dice_list)*100),2),"%, std :",round(float(np.std(dice_list)*100),2))
    # print("Recall    : ", round(float(np.mean(recall_list)*100),2),"%, std :",round(float(np.std(recall_list)*100),2))
    # print("precision : ", round(float(np.mean(precision_list)*100),2),"%, std :",round(float(np.std(precision_list)*100),2))
    print('Time      : ', time_list[fold-1] )

    print()

    # f1_folds.append(round(float(np.mean(f1_list)*100),2))
    # dice_folds.append(round(float(np.mean(dice_list)*100),2))
    # recall_folds.append(round(float(np.mean(recall_list)*100),2))
    # precision_folds.append(round(float(np.mean(precision_list)*100),2))

    dict = {''     : folds_titles,
        # 'F1'       : f1_folds,
        # 'DICE'     : dice_folds,
        # 'RECALL'   : recall_folds,
        # 'PRECISION': precision_folds,
        'TIME'     : time_list}

    df = pd.DataFrame(dict)

    # df.to_csv('FINAL_'+model_ckp_str+'_results-5cv-testing.csv', index=False)

# f1_mean, f1_std = round(float(np.mean(f1_folds)),2), round(float(np.std(f1_folds)),2)
# dice_mean, dice_std = round(float(np.mean(dice_folds)),2), round(float(np.std(dice_folds)),2)
# recall_mean, recall_std = round(float(np.mean(recall_folds)),2), round(float(np.std(recall_folds)),2)
# precision_mean, precision_std = round(float(np.mean(precision_folds)),2), round(float(np.std(precision_folds)),2)
time_mean, time_std = round(float(np.mean(time_list)),4), round(float(np.std(time_list)),4)


# f1_folds.append(f1_mean)
# f1_folds.append(f1_std)

# dice_folds.append(dice_mean)
# dice_folds.append(dice_std)

# recall_folds.append(recall_mean)
# recall_folds.append(recall_std)

# precision_folds.append(precision_mean)
# precision_folds.append(precision_std)

time_list.append(time_mean)
time_list.append(time_std)

folds_titles.append('MEAN')
folds_titles.append('STD')

dict = {''     : folds_titles,
        # 'F1'       : f1_folds,
        # 'DICE'     : dice_folds,
        # 'RECALL'   : recall_folds,
        # 'PRECISION': precision_folds,
        'TIME'     : time_list}

df = pd.DataFrame(dict)

# df.to_csv('FINAL_'+model_ckp_str+'_results-5cv-testing.csv', index=False)