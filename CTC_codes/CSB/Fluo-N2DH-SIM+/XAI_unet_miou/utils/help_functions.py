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

from scipy.ndimage import binary_dilation
from skimage.measure import label, regionprops
from skimage.exposure import match_histograms
from torch.utils.data import Dataset
from tqdm import trange
import numpy as np
import logging
import torch

from PIL import Image, ImageOps, ImageEnhance, ImageFilter

import random


import torch
from skimage.segmentation import watershed

def get_labelled_mask_batch(proba_map_batch, time_mask_threshold, mask_threshold):
    batch_size = proba_map_batch.size(0)

    # Empty list to store results for each image in the batch
    batch_labels = []

    for i in range(batch_size):
        # Convert PyTorch tensor to NumPy array
        proba_map = proba_map_batch[i].cpu().detach().numpy()

        # Computing the high probability cell positions
        res = proba_map > time_mask_threshold * np.max(proba_map)

        # Apply labeling and watershed using NumPy and scipy
        labels_np = label(res)[0]
        regions = regionprops(labels_np)

        centroids = np.array([region.centroid for region in regions])
        mask_out = np.zeros(res.shape, dtype=bool)

        if centroids.size > 0:
            mask_out[:, centroids[:, 0].astype(int), centroids[:, 1].astype(int)] = True

        markers_np = label(mask_out)[0]
        labels_np = watershed(-proba_map, markers_np, mask=proba_map > mask_threshold)
        
        # Convert the labels back to a PyTorch tensor and store in the list
        labels_torch = torch.tensor(labels_np, device=proba_map_batch.device)
        batch_labels.append(labels_torch)

    # Stack results to get a batch output
    return torch.stack(batch_labels)

def calculate_iou_best_match_batch_torch(predicted_masks, ground_truth_masks):
    batch_size = predicted_masks.size(0)
    batch_iou_scores = []

    for i in range(batch_size):
        predicted_mask = predicted_masks[i]
        ground_truth_mask = ground_truth_masks[i]

        labels_ground_truth = torch.unique(ground_truth_mask[ground_truth_mask != 0])  # Exclude background
        labels_predicted = torch.unique(predicted_mask[predicted_mask != 0])  # Exclude background

        for label_gt in labels_ground_truth:
            gt_mask = (ground_truth_mask == label_gt)
            best_iou = torch.tensor(0.0, device=predicted_masks.device)

            for label_pred in labels_predicted:
                pred_mask = (predicted_mask == label_pred)

                intersection = torch.sum(gt_mask & pred_mask).float()
                union = torch.sum(gt_mask | pred_mask).float()

                if union > 0:
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou

            batch_iou_scores.append(best_iou.clone().detach())

    # Concatenate all IoU scores and compute the mean
    mean_iou = torch.mean(torch.stack(batch_iou_scores))
    return mean_iou


def add_noise(image):
    """Add random noise to the image."""
    noise = np.random.normal(0, 25, image.size)  # Adjust noise level as needed
    noisy_image = np.array(image) + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are within [0, 255]
    return Image.fromarray(noisy_image.astype('uint8'))


# Chose the GPU cuda devices to make the training go much faster vs CPU use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def CustomMetrics_batch(batch_mask, batch_true_bn_mask, batch_true_border_mask, batch_gt_bboxes, eval_threshold=0.5, IoU_threshold=0.5, eps=10e-4):
    """
    

    Parameters
    ----------
    batch_mask : Batch Torch tensors
        Contain all the predections of the model.
    batch_true_bn_mask : Batch Torch tensors
        Contain all the ground truth binary masks.
    batch_true_border_mask : Batch Torch tensors
        Contain all the ground truth boundry masks.

    Returns
    -------
    Torch tensor
        The BCE+BONDRY LOSS.

    """

    # Init metrics for batch (GPU ready)
    if batch_mask.is_cuda:
        f1_loss        = torch.FloatTensor(1).cuda().zero_()
        recall_loss    = torch.FloatTensor(1).cuda().zero_()
        precision_loss = torch.FloatTensor(1).cuda().zero_()
        dice_loss      = torch.FloatTensor(1).cuda().zero_()

    else:
        f1_loss        = torch.FloatTensor(1).zero_()
        recall_loss    = torch.FloatTensor(1).zero_()
        precision_loss = torch.FloatTensor(1).zero_()
        dice_loss      = torch.FloatTensor(1).zero_()
    
    # Init dice score for batch (GPU ready)
    if batch_mask.is_cuda:
        bce_loss = torch.FloatTensor(1).cuda().zero_()
        border_loss     = torch.FloatTensor(1).cuda().zero_()
    else:
        bce_loss = torch.FloatTensor(1).zero_()
        border_loss     = torch.FloatTensor(1).zero_()

    # Compute Dice coefficient for the given batch
    for pairs_idx, inputs in enumerate(zip(batch_mask, batch_true_bn_mask, batch_true_border_mask, batch_gt_bboxes)):

        bce                   = single_BCE_weighted_loss(inputs[0], inputs[1])
        border                = single_BORDER_Loss(inputs[0], inputs[2])
        #f1, recall, precision = F1_loss(inputs[0], inputs[3])
        dice                  = single_dice_coeff(inputs[0], inputs[1])
        
        dice_loss       += dice
        #f1_loss         += f1
        #recall_loss     += recall
        #precision_loss  += precision
        bce_loss        += bce#border_loss #+bce_loss
        border_loss     += border
    
    # Return Detection metrics over the given batch
    # f1_loss         /= (pairs_idx + 1)
    #recall_loss     /= (pairs_idx + 1)
    #precision_loss  /= (pairs_idx + 1)
    #f1_loss          = torch.div(f1_loss, (pairs_idx + 1)) 
    bce_loss         = torch.div(bce_loss, (pairs_idx + 1)) 
    dice_loss       /= (pairs_idx + 1)
    border_loss      = torch.div(border_loss, (pairs_idx + 1)) 

    # border_loss.requires_grad= True

    #print("Border loss : ",border_loss)
    #print("bce loss : ",bce_border_loss)
    #print("F1 : ",1-eps-f1_loss)
    # Return the mean Dice coefficient over the given batch
    return dice_loss, bce_loss, border_loss,  #dice_loss, f1_loss, recall_loss, precision_loss, bce_loss, border_loss


def CustomMetrics_TEST_batch(batch_mask, batch_true_bn_mask, eval_threshold=0.5, IoU_threshold=0.5, eps=10e-4):
    """ batch_true_border_mask, batch_gt_bboxes,
    

    Parameters
    ----------
    batch_mask : Batch Torch tensors
        Contain all the predections of the model.
    batch_true_bn_mask : Batch Torch tensors
        Contain all the ground truth binary masks.
    batch_true_border_mask : Batch Torch tensors
        Contain all the ground truth boundry masks.

    Returns
    -------
    Torch tensor
        The BCE+BONDRY LOSS.

    """

    # Init metrics for batch (GPU ready)
    if batch_mask.is_cuda:
        f1_loss        = torch.FloatTensor(1).cuda().zero_()
        recall_loss    = torch.FloatTensor(1).cuda().zero_()
        precision_loss = torch.FloatTensor(1).cuda().zero_()
        dice_loss      = torch.FloatTensor(1).cuda().zero_()

    else:
        f1_loss        = torch.FloatTensor(1).zero_()
        recall_loss    = torch.FloatTensor(1).zero_()
        precision_loss = torch.FloatTensor(1).zero_()
        dice_loss      = torch.FloatTensor(1).zero_()
    
    # Init dice score for batch (GPU ready)
    if batch_mask.is_cuda:
        bce_loss = torch.FloatTensor(1).cuda().zero_()
        border_loss     = torch.FloatTensor(1).cuda().zero_()
    else:
        bce_loss = torch.FloatTensor(1).zero_()
        border_loss     = torch.FloatTensor(1).zero_()

    f1_loss, recall_loss, precision_loss = 0.0, 0.0, 0.0
    # Compute Dice coefficient for the given batch
    for pairs_idx, inputs in enumerate(zip(batch_mask, batch_true_bn_mask)):

        bce                   = single_BCE_weighted_loss(inputs[0], inputs[1])
        # border                = single_BORDER_Loss(inputs[0], inputs[2])
        # f1, recall, precision = F1_loss(inputs[0], inputs[3])
        dice                  = single_dice_coeff(inputs[0], inputs[1])
        
        dice_loss       += dice
        #f1_loss         += f1
        #recall_loss     += recall
        #precision_loss  += precision
        bce_loss        += bce#border_loss #+bce_loss
        # border_loss     += border
    
    # Return Detection metrics over the given batch
    # f1_loss         /= (pairs_idx + 1)
    # recall_loss     /= (pairs_idx + 1)
    # precision_loss  /= (pairs_idx + 1)
    # f1_loss          = torch.div(f1_loss, (pairs_idx + 1)) 
    bce_loss         = torch.div(bce_loss, (pairs_idx + 1)) 
    dice_loss       /= (pairs_idx + 1)
    border_loss      = torch.div(border_loss, (pairs_idx + 1)) 

    # border_loss.requires_grad= True

    #print("Border loss : ",border_loss)
    #print("bce loss : ",bce_border_loss)
    #print("F1 : ",1-eps-f1_loss)
    # Return the mean Dice coefficient over the given batch
    return dice_loss, f1_loss, recall_loss, precision_loss, bce_loss, border_loss




def single_dice_coeff(input_mask, true_bn_mask, eval_threshold=0.5, eps=10e-4):
    """single_dice_coeff : function that returns the dice coeff for one pair 
    of mask and ground truth mask"""
    input_mask = torch.clamp(input_mask,min=eps,max=1-eps)

    # Thresholding the input mask to obtain binary mask
    bn_mask = (input_mask > eval_threshold).float()

    # The eps value is used for numerical stability
    eps = 0.0001

    # Computing intersection and union masks
    inter_mask = torch.dot(bn_mask.view(-1), true_bn_mask.view(-1))
    union_mask = torch.sum(bn_mask) + torch.sum(true_bn_mask) + eps

    # Computing the Dice coefficient
    return (2 * inter_mask.float() + eps) / union_mask.float()

def single_BORDER_Loss(input_mask, gt_border_mask, eval_threshold=0.5, eps=10e-4):
    # Clamp input mask for numerical stability
    input_mask = torch.clamp(input_mask,min=eps,max=1-eps)
    
    # Thresholding the input mask to obtain binary mask
    bn_mask = (input_mask > eval_threshold).float()
    
    # Computing the total border pixels
    border_total_px = len(torch.where(gt_border_mask!=0)[0])

    # Computing the border intersection       
    border_inter_px =  torch.dot(bn_mask.view(-1), gt_border_mask.view(-1))

    return torch.div(border_inter_px, border_total_px)

def single_BCE_weighted_loss(input_mask, gt_mask, eps=10e-4):
    # Clamp input mask for numerical stability
    input_mask = torch.clamp(input_mask,min=eps,max=1-eps)
    
    # Computing custom weighs
    bg_target = len(torch.where(gt_mask==0)[0])
    fg_target = len(torch.where(gt_mask!=0)[0])

    weights = [1 + bg_target/(fg_target+bg_target), 1 + fg_target/(fg_target+bg_target)]
    
    # Computing the bce loss
    return torch.mean(- weights[1] * gt_mask * torch.log(input_mask) - (1 - gt_mask) * weights[0] * torch.log(1 - input_mask))



def batch_BCE_weighted_loss(input_masks, gt_masks, eps=1e-4):
    # Clamp input masks for numerical stability
    input_masks = torch.clamp(input_masks, min=eps, max=1-eps)
    
    # Computing custom weights
    # Calculate the background and foreground targets for each image in the batch
    bg_target = torch.sum(gt_masks == 0, dim=[1, 2])  # Sum over spatial dimensions
    fg_target = torch.sum(gt_masks != 0, dim=[1, 2])  # Sum over spatial dimensions

    # Calculate weights for each image in the batch
    weights_bg = 1 + bg_target / (fg_target + bg_target).unsqueeze(1).unsqueeze(1)
    weights_fg = 1 + fg_target / (fg_target + bg_target).unsqueeze(1).unsqueeze(1)

    # Computing the BCE loss for each image in the batch
    bce_loss = - weights_fg * gt_masks * torch.log(input_masks) - (1 - gt_masks) * weights_bg * torch.log(1 - input_masks)

    # Return the mean loss over the batch
    return torch.mean(bce_loss)



def batch_BCE_weighted_loss(input_masks, gt_masks, eps=1e-4):
    # Clamp input masks for numerical stability
    input_masks = torch.clamp(input_masks, min=eps, max=1-eps)
    
    # Computing custom weights
    # Calculate the background and foreground targets for each image in the batch
    bg_target = torch.sum(gt_masks == 0, dim=[1, 2])  # Sum over spatial dimensions
    fg_target = torch.sum(gt_masks != 0, dim=[1, 2])  # Sum over spatial dimensions

    # Calculate weights for each image in the batch
    weights_bg = 1 + bg_target / (fg_target + bg_target)
    weights_fg = 1 + fg_target / (fg_target + bg_target)

    # Adjust weights shape for broadcasting by adding singleton dimensions
    weights_bg = weights_bg.unsqueeze(1).unsqueeze(2)  # Shape becomes [batch_size, 1, 1]
    weights_fg = weights_fg.unsqueeze(1).unsqueeze(2)  # Shape becomes [batch_size, 1, 1]

    # Computing the BCE loss for each image in the batch
    bce_loss = - weights_fg * gt_masks * torch.log(input_masks) - (1 - gt_masks) * weights_bg * torch.log(1 - input_masks)

    # Return the mean loss over the batch
    return torch.mean(bce_loss)

def batch_dice_coefficient(input_masks, gt_masks, threshold=0.5, eps=1e-4):
    # Binarize masks
    input_masks_binary = (input_masks > threshold).float()
    gt_masks_binary = (gt_masks > threshold).float()
    
    # Calculate intersection and union
    intersection = torch.sum(input_masks_binary * gt_masks_binary, dim=[1, 2])
    union = torch.sum(input_masks_binary, dim=[1, 2]) + torch.sum(gt_masks_binary, dim=[1, 2])

    # Calculate Dice coefficient
    dice = (2. * intersection + eps) / (union + eps)

    # Return the mean Dice coefficient over the batch
    return torch.mean(dice)



def F1_loss(input_mask, gt_bboxes, eval_threshold=0.5, IoU_threshold=0.5, eps=10e-4):
    # Clamp input mask for numerical stability
    input_mask = torch.clamp(input_mask,min=eps,max=1-eps)
    
    # Thresholding the input mask to obtain numpy binary mask
    bn_mask = (input_mask > eval_threshold).cpu().detach().numpy().astype(np.float)

    # Labelling the binary mask
    np_label_mask = label(bn_mask, connectivity=2)

    # Extract regions using the labelled mask
    regions = regionprops(np_label_mask)

    # Compute BBoxes using the binary mask 
    input_bboxes = []
    if gt_bboxes.is_cuda:
        for region in regions: input_bboxes.append(region.bbox)
        input_bboxes = torch.from_numpy(np.array(input_bboxes)).type(torch.FloatTensor).to(device=device, dtype=torch.float32)
    else:
        for region in regions: input_bboxes.append(region.bbox)
        input_bboxes = torch.from_numpy(np.array(input_bboxes)).type(torch.FloatTensor)
    
    # Defining the True Positive, False positive, False negative
    if gt_bboxes.is_cuda:
        tp = torch.FloatTensor(1).cuda().zero_()
        fp = torch.FloatTensor(1).cuda().zero_()
        fn = torch.FloatTensor(1).cuda().zero_()
        f1 = torch.FloatTensor(1).cuda().zero_()
    else:
        tp = torch.FloatTensor(1).zero_()
        fp = torch.FloatTensor(1).zero_()
        fn = torch.FloatTensor(1).zero_()
        f1 = torch.FloatTensor(1).zero_()
    
    # Compute IoU of all input bboxes vs GT bboxes
    for input_bbox in input_bboxes:
        # Init the max IoU value
        max_iou = 0
        for gt_bboxe in gt_bboxes:
            iou = IoU(gt_bboxe, input_bbox)
            # Update the Max IoU value
            if iou>max_iou: max_iou = iou
        # Quanitfy the detection (TP or FP)
        if max_iou> IoU_threshold: tp +=1
        else: fp += 1
    # Computing the False Negatives
    fn = len(gt_bboxes) - tp

    # Computing the precision
    precision =  tp / (tp + fp)

    # Computing the recall
    recall    =  tp / (tp + fn)

    # Computing the f1
    f1        +=  2 * tp / (2 * tp + fp + fn)
    #f1 = torch.FloatTensor(f1)

    return f1, recall, precision 


def BCE_BORDER_loss_batch(batch_mask, batch_true_bn_mask, batch_true_border_mask):
    """
    

    Parameters
    ----------
    batch_mask : Batch Torch tensors
        Contain all the predections of the model.
    batch_true_bn_mask : Batch Torch tensors
        Contain all the ground truth binary masks.
    batch_true_border_mask : Batch Torch tensors
        Contain all the ground truth boundry masks.

    Returns
    -------
    Torch tensor
        The BCE+BONDRY LOSS.

    """
    
    def single_BORDER_Loss(input_mask, gt_border_mask, total_px, eval_threshold=0.5, eps=10e-7):
        # Clamp input mask for numerical stability
        input_mask = torch.clamp(input_mask,min=eps,max=1-eps)
        
        # Thresholding the input mask to obtain binary mask
        bn_mask = (input_mask > eval_threshold).float()
        
        # Computing the border loss
        return torch.dot(bn_mask.view(-1), gt_border_mask.view(-1))/total_px

    def BCE_weighted_loss(input_mask, gt_mask, eps=10e-7):
        # Clamp input mask for numerical stability
        input_mask = torch.clamp(input_mask,min=eps,max=1-eps)
        
        # Computing custom weighs
        bg_target = len(torch.where(gt_mask==0)[0])
        fg_target = len(torch.where(gt_mask!=0)[0])
        total_px = bg_target + fg_target
    
        weights = [bg_target/fg_target, fg_target/fg_target]
        
        # Computing the bce loss
        bce = torch.mean(- weights[1] * gt_mask * torch.log(input_mask) - (1 - gt_mask) * weights[0] * torch.log(1 - input_mask))
        return bce, total_px
    
    # Init dice score for batch (GPU ready)
    if batch_mask.is_cuda: bce_border_loss = torch.FloatTensor(1).cuda().zero_()
    else: bce_border_loss = torch.FloatTensor(1).zero_()

    # Compute Dice coefficient for the given batch
    for pairs_idx, inputs in enumerate(zip(batch_mask, batch_true_bn_mask, batch_true_border_mask)):

        bce_loss, total_px   = BCE_weighted_loss(inputs[0], inputs[1])
        border_loss          = 2*single_BORDER_Loss(inputs[0], inputs[2], total_px)
        bce_border_loss = bce_border_loss+border_loss+bce_loss
    
    # Return the mean Dice coefficient over the given batch
    return bce_border_loss / (pairs_idx + 1)

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


def F1_loss_batch(batch_mask, batch_true_bboxes):
    """
    

    Parameters
    ----------
    batch_mask : Batch Torch tensors
        Contain all the predections of the model.
    batch_true_bn_mask : Batch Torch tensors
        Contain all the ground truth binary masks.
    batch_true_border_mask : Batch Torch tensors
        Contain all the ground truth boundry masks.

    Returns
    -------
    Torch tensor
        The BCE+BONDRY LOSS.

    """
    
    def F1_loss(input_mask, gt_bboxes, eval_threshold=0.5, IoU_threshold=0.5, eps=10e-7):
        # Clamp input mask for numerical stability
        input_mask = torch.clamp(input_mask,min=eps,max=1-eps)
        
        # Thresholding the input mask to obtain numpy binary mask
        bn_mask = (input_mask > eval_threshold).cpu().detach().numpy().astype(np.float)

        # Labelling the binary mask
        np_label_mask = label(bn_mask, connectivity=2)

        # Extract regions using the labelled mask
        regions = regionprops(np_label_mask)

        # Compute BBoxes using the binary mask 
        input_bboxes = []
        if gt_bboxes.is_cuda:
            for region in regions: input_bboxes.append(region.bbox)
            input_bboxes = torch.from_numpy(np.array(input_bboxes)).type(torch.FloatTensor).to(device=device, dtype=torch.float32)
        else:
            for region in regions: input_bboxes.append(region.bbox)
            input_bboxes = torch.from_numpy(np.array(input_bboxes)).type(torch.FloatTensor)
        
        # Defining the True Positive, False positive, False negative
        if gt_bboxes.is_cuda:
            tp = torch.FloatTensor(1).cuda().zero_()
            fp = torch.FloatTensor(1).cuda().zero_()
            fn = torch.FloatTensor(1).cuda().zero_()
        else:
            tp = torch.FloatTensor(1).zero_()
            fp = torch.FloatTensor(1).zero_()
            fn = torch.FloatTensor(1).zero_()
        
        # Compute IoU of all input bboxes vs GT bboxes
        for input_bbox in input_bboxes:
            # Init the max IoU value
            max_iou = 0
            for gt_bboxe in gt_bboxes:
                iou = IoU(gt_bboxe, input_bbox)
                # Update the Max IoU value
                if iou>max_iou: max_iou = iou
            # Quanitfy the detection (TP or FP)
            if max_iou> IoU_threshold: tp +=1
            else: fp += 1
        # Computing the False Negatives
        fn = len(gt_bboxes) - tp

        # Computing the precision
        precision =  tp / (tp + fp)

        # Computing the recall
        recall    =  tp / (tp + fn)

        # Computing the f1
        f1        =  2 * tp / (2 * tp + fp + fn)

        return f1, recall, precision 

    # Init metrics for batch (GPU ready)
    if batch_mask.is_cuda:
        f1_loss        = torch.FloatTensor(1).cuda().zero_()
        recall_loss    = torch.FloatTensor(1).cuda().zero_()
        precision_loss = torch.FloatTensor(1).cuda().zero_()

    else:
        f1_loss        = torch.FloatTensor(1).zero_()
        recall_loss    = torch.FloatTensor(1).zero_()
        precision_loss = torch.FloatTensor(1).zero_()

    # Metrics for the given batch
    for pairs_idx, inputs in enumerate(zip(batch_mask, batch_true_bboxes)):
        f1, recall, precision = F1_loss(inputs[0], inputs[1])
        f1_loss         += f1
        recall_loss     += recall
        precision_loss  += precision
    
    # Return Detection metrics over the given batch
    f1_loss         /= (pairs_idx + 1)
    recall_loss     /= (pairs_idx + 1)
    precision_loss  /= (pairs_idx + 1)
    return f1_loss, recall_loss, precision_loss


def dice_coeff_batch(batch_mask, batch_true_bn_mask):
    """ dice_coeff_batch : function that returns the mean dice coefficient for a batch of pairs 
    mask, ground truth mask """
    
    def single_dice_coeff(input_mask, true_bn_mask, eval_threshold=0.5, eps=10e-7):
        """single_dice_coeff : function that returns the dice coeff for one pair 
        of mask and ground truth mask"""
        input_mask = torch.clamp(input_mask,min=eps,max=1-eps)

        # Thresholding the input mask to obtain binary mask
        bn_mask = (input_mask > eval_threshold).float()

        # The eps value is used for numerical stability
        eps = 0.0001

        # Computing intersection and union masks
        inter_mask = torch.dot(bn_mask.view(-1), true_bn_mask.view(-1))
        union_mask = torch.sum(bn_mask) + torch.sum(true_bn_mask) + eps

        # Computing the Dice coefficient
        return (2 * inter_mask.float() + eps) / union_mask.float()

    # Init dice score for batch (GPU ready)
    if batch_mask.is_cuda: dice_score = torch.FloatTensor(1).cuda().zero_()
    else: dice_score = torch.FloatTensor(1).zero_()

    # Compute Dice coefficient for the given batch
    for pair_idx, inputs in enumerate(zip(batch_mask, batch_true_bn_mask)):
        dice_score += single_dice_coeff(inputs[0], inputs[1])
        dice_score += 0
    # Return the mean Dice coefficient over the given batch
    return dice_score / (pair_idx + 1)

def metrics(p_n, tp, fp, tn, fn):
    """ Returns accuracy, precision, recall, f1 based on the inputs 
    tp : true positives, fp: false positives, tn: true negatives, fn: false negatives
    For details please check : https://en.wikipedia.org/wiki/Precision_and_recall
    """
    try:
        # Computing the accuracy
        accuracy  = (tp + tn) / p_n

        # Computing the precision
        precision =  tp / (tp + fp)

        # Computing the recall
        recall    =  tp / (tp + fn)

        # Computing the f1
        f1        =  2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        precision, recall, accuracy, f1 = 0, 0, 0, 0

    return precision, recall, accuracy, f1

def confusion_matrix(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    # Source of the confusion_matrix function: https://gist.github.com/the-bass
    """
    # Computing the confusion vector
    confusion_vector = prediction / truth
    
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()

    # Computing the total (p+n)
    p_n = tp + fp + tn + fn

    # Computing the precision, recall, accuracy, f1 metrics
    precision, recall, accuracy, f1 = metrics(p_n, tp, fp, tn, fn)

    return tp/p_n, fp/p_n, tn/p_n, fn/p_n, precision, recall, accuracy, f1 



class CustomDataset(Dataset):
    """ CustomDataset : Class that loads data (images and masks) in efficient way"""
    def __init__(self, imgs_dirs, masks_dirs, ref_image_path, normalize=False,cached_data=True, n_channels=1,scale=1):
        self.imgs_dirs = imgs_dirs    # All paths to images 
        self.masks_dirs = masks_dirs  # All paths to masks 
        self.scale = scale            # image and mask scale
        self.n_channels = n_channels  # input model channels
        self.normalize = normalize    # normalization switch

        # Make sure the scale is between [0, 1]
        assert 0 < scale <= 1, '[ERROR] Scale must be between 0 and 1'

        # Load the reference image into RAM
        ref_image = Image.open(ref_image_path)
        
        # Save the reference image into RAM to be used
        self.ref_image = ref_image.copy()

        # Caching the dataset (WARRING : this needs to be used when you have big RAM memory)
        if cached_data:
            logging.info(f'[INFO] Caching the given dataset with {len(self.imgs_dirs)} images and {len(self.masks_dirs)} masks')
            # Turn on the cache flag
            self.cached_dataset = True

            # Preparing the images and masks lists
            self.cache_imgs, self.cache_masks = [], []
            
            # Cache & pre-process the images and the masks (train/val) ready
            for i in trange(len(imgs_dirs)):
                pil_img = Image.open(self.imgs_dirs[i])
                np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)
                self.cache_imgs.append(np_img)

                if len(self.masks_dirs) == len(self.imgs_dirs):
                    pil_mask = Image.open(self.masks_dirs[i])
                    np_img = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
                    self.cache_masks.append(np_img)
        else:
            logging.info(f'[INFO] Dataset with {len(self.imgs_dirs)} images and {len(self.masks_dirs)} masks')
            
    def __len__(self): return len(self.imgs_dirs)

    def delete_cached_dataset(self):
        try:
            del self.cache_imgs[:]
            del self.cache_masks[:]
            logging.info(f'[INFO] All cache deleted')
            return True
        except:
            return False

    def preprocess(self, pil_img, ref_image, n_channels, scale, normalize, mask=True):
        if not(mask):
             # This part is concerns the normalization 
            if normalize:
                # Make sure the reference image and the current image have the same size
                assert pil_img.size == ref_image.size, \
                f'Image and refrence image should be the same size for histograms matching, but are {pil_img.size} and {ref_image.size}'
           
                if n_channels == 3: pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image), multichannel=True))
                else: pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image)))
            
        # Rescale the image if needed
        if scale != 1 :
            # Get the H and W of the img
            w, h = pil_img.size

            # Get the estimated new size
            newW, newH = int(scale * w), int(scale * h)

            # Resize the image according the given scale
            pil_img = pil_img.resize((newW, newH))

        # Uncomment to convert imgs into gray scale imgs
        # if n_channels != 3: pil_img = pil_img.convert("L")

        # Convert the PIL image into numpy array
        np_img = np.array(pil_img)

        # Add an extra dim if only H, W image
        if len(np_img.shape) == 2: np_img = np.expand_dims(np_img, axis=2)

        # Re-arrange the image from (H, W, C) to (C ,H ,W)
        np_img_ready = np_img.transpose((2, 0, 1))
        
        # Ensure the imgs to be in [0, 1]
        if np_img_ready.max() > 1: np_img_ready = np_img_ready / 255
        
        return np_img_ready
    
    def __getitem__(self, i):
        # When the dataset is cached load the img and mask from RAM
        if self.cached_dataset:
            np_img = self.cache_imgs[i]
            if len(self.masks_dirs) == len(self.imgs_dirs):
                np_mask = self.cache_masks[i]
        
        # Otherwise load the img and mask from Disk to RAM
        else:
            # Load the image 
            img_dir = self.imgs_dirs[i]
            pil_img = Image.open(img_dir)

            # Preprocess the image 
            np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)

            # Load & pre-process the mask if possible
            if len(self.masks_dirs) == len(self.imgs_dirs):
                mask_dir = self.masks_dirs[i]
                pil_mask = Image.open(mask_dir)
                np_mask = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
            
            
        if len(self.masks_dirs) == len(self.imgs_dirs):
            return {
                'image': torch.from_numpy(np_img).type(torch.FloatTensor),
                'mask': torch.from_numpy(np_mask).type(torch.FloatTensor)
            }
        else:
            return {
                'image': torch.from_numpy(np_img).type(torch.FloatTensor)
            }



class CustomDetectionDataset(Dataset):
    """ CustomDataset : Class that loads data (images and masks) in efficient way"""
    def __init__(self, imgs_dirs, masks_dirs, ref_image_path, normalize=False,cached_data=True, n_channels=1,scale=1, train=False):
        self.imgs_dirs = imgs_dirs    # All paths to images 
        self.masks_dirs = masks_dirs  # All paths to masks 
        self.scale = scale            # image and mask scale
        self.n_channels = n_channels  # input model channels
        self.normalize = normalize    # normalization switch
        self.train = train
        
        if self.train:
            # Create a combined list of pairs and shuffle them
            combined = list(zip(self.imgs_dirs , self.masks_dirs))
            np.random.shuffle(combined)
            self.imgs_dirs[:], self.masks_dirs[:] = zip(*combined)  

        # Make sure the scale is between [0, 1]
        assert 0 < scale <= 1, '[ERROR] Scale must be between 0 and 1'
        self.cached_dataset = False
        self.ref_image = []
        if normalize:
            # Load the reference image into RAM
            ref_image = Image.open(ref_image_path)
        
            # Save the reference image into RAM to be used
            self.ref_image = ref_image.copy()

        # Caching the dataset (WARRING : this needs to be used when you have big RAM memory)
        if cached_data:
            logging.info(f'[INFO] Caching the given dataset with {len(self.imgs_dirs)} images and {len(self.masks_dirs)} masks')
            # Turn on the cache flag
            self.cached_dataset = True

            # Preparing the images and masks lists
            self.cache_imgs, self.cache_masks, self.border_masks, self.cache_bboxes = [], [], [], []
            
            # Cache & pre-process the images and the masks (train/val) ready
            for i in trange(len(imgs_dirs)):
                pil_img = Image.open(self.imgs_dirs[i])
                np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)
                self.cache_imgs.append(np_img)

                if len(self.masks_dirs) == len(self.imgs_dirs):
                    # pil_mask  = Image.open(self.masks_dirs[i])
                    # np_labeled = rgb_to_id(self.masks_dirs[i])
                    np_labeled =np.array(Image.open(self.masks_dirs[i]))
                    pil_mask = Image.fromarray(np.array(np_labeled>0)*1.)
                    gt_bboxes = self.bboxes_prep(np_labeled)
                    np_border_mask = self.border_mask_generator(pil_mask)

                    np_img    = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
                    self.cache_masks.append(np_img)
                    self.cache_bboxes.append(gt_bboxes)
                    self.border_masks.append(np_border_mask)
        else:
            logging.info(f'[INFO] Dataset with {len(self.imgs_dirs)} images and {len(self.masks_dirs)} masks')


    def __len__(self): return len(self.imgs_dirs)

    def delete_cached_dataset(self):
        try:
            del self.cache_imgs[:]
            del self.cache_masks[:]
            logging.info(f'[INFO] All cache deleted')
            return True
        except:
            return False

    def bboxes_prep(self, np_org_mask):
        # # Numpy version of the mask
        # np_org_mask = np.array(pil_mask)

        # Labelling the binary mask
        np_label_mask = label(np_org_mask, connectivity=2)

        # Extract regions using the labelled mask
        regions = regionprops(np_label_mask)

        bboxes = []
        for region in regions: bboxes.append(region.bbox)
        return np.array(bboxes)


    def preprocess(self, pil_img, ref_image, n_channels, scale, normalize, mask=True, crop_params=None):
        if not(mask):
             # This part is concerns the normalization 
            if normalize:
                # Make sure the reference image and the current image have the same size
                assert pil_img.size == ref_image.size, \
                f'Image and reference image should be the same size for histograms matching, but are {pil_img.size} and {ref_image.size}'
           
                if n_channels == 3: pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image)))
                else: pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image)))
            
        # Rescale the image if needed
        if scale != 1 :
            # Get the H and W of the img
            w, h = pil_img.size

            # Get the estimated new size
            newW, newH = int(scale * w), int(scale * h)

            # Resize the image according the given scale
            pil_img = pil_img.resize((newW, newH))

        # Uncomment to convert imgs into gray scale imgs
        # if n_channels != 3: pil_img = pil_img.convert("L")
        

        if self.train and not(mask) and random.uniform(0, 1) > 0.5:
            # Randomly adjust brightness (50% less to 50% more)
            if random.choice([True, False]):
                brightness_factor = random.uniform(0.1, 2)
                pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness_factor)

            # Randomly adjust contrast (50% less to 50% more)
            if random.choice([True, False]):
                contrast_factor = random.uniform(0.1, 2)
                pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_factor)

            # Randomly apply solarization
            if random.choice([True, False]):
                pil_img = ImageOps.solarize(pil_img, threshold=random.randint(32, 224))

            # Add random noise
            if random.choice([True, False]):
                pil_img = add_noise(pil_img)

            # Randomly jitter (shift pixel values)
            if random.choice([True, False]):
                jitter_factor = random.uniform(-0.1, 0.1)  # Adjust jitter range as needed
                np_img = np.array(pil_img).astype(np.float32)
                np_img += jitter_factor * np_img.std()
                np_img = np.clip(np_img, 0, 255)
                pil_img = Image.fromarray(np_img.astype('uint8'))
            
            # Randomly blur the image
            if random.choice([True, False]):
                blur_radius = random.uniform(1, 6)  # Adjust blur range as needed
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))

            # Randomly sharpen the image
            if random.choice([True, False]):
                sharpen_factor = random.uniform(1, 6)  # Adjust sharpening range as needed
                enhancer = ImageEnhance.Sharpness(pil_img)
                pil_img = enhancer.enhance(sharpen_factor)

            if random.choice([True, False]):
                np_img = np.array(pil_img)
                noise_clouds = random.randint(1,5)
                for _ in range(noise_clouds): np_img = add_random_gaussian_noise(np_img)
                pil_img = Image.fromarray(np_img)
 
            # Randomly crop and resize the image
            if random.choice([True, False]):
                crop_params = get_random_crop_params(pil_img)
                pil_img = random_crop_and_resize(pil_img, crop_params)

        # Apply the same cropping to mask if it exists
        if self.train and mask and (crop_params is not None):
            pil_img = np.array(random_crop_and_resize(pil_img, crop_params))
            pil_img = np.array(pil_img>0)*1.

        # Convert the PIL image into numpy array
        np_img = np.array(pil_img)

        # Add an extra dim if only H, W image
        if len(np_img.shape) == 2: np_img = np.expand_dims(np_img, axis=0)
        
        # Ensure the imgs to be in [0, 1]
        if np_img.max() > 1: np_img = np_img / np.max(np_img)
        

        return np_img, crop_params

    def border_mask_generator(self, pil_org_mask):
        """
        
        Parameters
        ----------
        pil_org_mask : PIL Image
            It contains the ground truth binary mask.
        plot_fig : Bool, optional
            True : Chose to plot the results. The default is False.

        Returns
        -------
        np_border_mask : Numpy array
            The border mask of the given binary mask.

        """
        # Numpy version of the mask
        np_org_mask = np.array(pil_org_mask)

        # Computing the dilatation mask
        np_dil_mask = binary_dilation(np_org_mask, iterations=2).astype(np_org_mask.dtype)

        # Computing the borders mask
        rr, cc = np.where(np_org_mask != 0 )
        np_border_mask = np_dil_mask.copy()
        np_border_mask[rr, cc] = 0
        np_border_mask = binary_dilation(np_border_mask, iterations=4).astype(np_border_mask.dtype)
        return np_border_mask
    
    def __getitem__(self, i):
        # When the dataset is cached load the img and mask from RAM
        if self.cached_dataset:
            np_img = self.cache_imgs[i]
            if len(self.masks_dirs) == len(self.imgs_dirs):
                np_mask = self.cache_masks[i]
                gt_bboxes = self.cache_bboxes[i]
                np_border_mask = self.border_masks[i]
        
        # Otherwise load the img and mask from Disk to RAM
        else:
            # Load the image 
            img_dir = self.imgs_dirs[i]
            pil_img = Image.open(img_dir)

            # Preprocess the image 
            np_img, crop_params = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False, crop_params=None)

            # Load & pre-process the mask if possible
            if len(self.masks_dirs) == len(self.imgs_dirs):
                # mask_dir = self.masks_dirs[i]
                # pil_mask = Image.open(mask_dir)
                # gt_bboxes = self.bboxes_prep(pil_mask)
                # np_border_mask = self.border_mask_generator(pil_mask)
                # np_mask = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)

                pil_label = Image.open(self.masks_dirs[i])
                np_labeled = np.array(pil_label)
                pil_mask = Image.fromarray(np.array(np_labeled>0)*1.)
                np_mask,_    = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True, crop_params=crop_params)

                if crop_params is not None:
                    original_size = pil_label.size
                    left, top, right, bottom = crop_params
                    cropped_image = pil_label.crop((left, top, right, bottom))
                    pil_label = cropped_image.resize(original_size, Image.NEAREST)
                    np_labeled = np.array(pil_label)

                np_labeled = np.array(np_labeled, dtype=np.float16)
                if len(np_labeled.shape) == 2: np_labeled = np.expand_dims(np_labeled, axis=0)

        if len(self.masks_dirs) == len(self.imgs_dirs):
            return {
                'image': torch.from_numpy(np_img).type(torch.FloatTensor),
                'mask' : torch.from_numpy(np_mask).type(torch.FloatTensor),
                'label': torch.from_numpy(np_labeled).type(torch.FloatTensor),
                # 'border_mask': torch.from_numpy(np_border_mask).type(torch.FloatTensor), 
                # 'b-boxes': torch.from_numpy(gt_bboxes).type(torch.FloatTensor),
            }
        else:
            return {
                'image': torch.from_numpy(np_img).type(torch.FloatTensor)
            }

from numba import jit

@jit(nopython=True)
def fast_mapping(scalar_data, bg_scalar):
    h, w = scalar_data.shape
    id_matrix = np.zeros((h, w), dtype=np.uint16)
    color_to_id = {}
    current_id = 1

    for i in range(h):
        for j in range(w):
            color = scalar_data[i, j]
            if color == bg_scalar:
                continue

            if color not in color_to_id:
                color_to_id[color] = current_id
                current_id += 1
            
            id_matrix[i, j] = color_to_id[color]

    return id_matrix

def rgb_to_id(img_path, bg_color=(0, 0, 0)):
    img = Image.open(img_path)
    data = np.array(img)
    
    scalar_data = data[:, :, 0] + 256 * data[:, :, 1] + 256 * 256 * data[:, :, 2]
    bg_scalar = bg_color[0] + 256 * bg_color[1] + 256 * 256 * bg_color[2]

    return fast_mapping(scalar_data, bg_scalar)


def get_random_crop_params(image, min_crop_pct=0.4):
    """Get random crop parameters for an image."""
    width, height = image.size
    crop_width = random.randint(int(min_crop_pct * width), width)
    crop_height = random.randint(int(min_crop_pct * height), height)
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return (left, top, right, bottom)

def random_crop_and_resize(image, crop_params):
    """Randomly crop the image and resize it back to the original size."""
    original_size = image.size
    left, top, right, bottom = crop_params
    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize(original_size)
    return resized_image

# This function adds random white Gaussian noise to an image
def add_random_gaussian_noise(image_array):

    # Get image dimensions
    height, width = image_array.shape[:2]

    # Generate random position and sigma for Gaussian noise
    x, y = random.randint(0, width-1), random.randint(0, height-1)
    sigma = random.uniform(10.0, 100.0)

    # Generate Gaussian noise
    X, Y = np.meshgrid(np.linspace(0, width, width), np.linspace(0, height, height))
    d = np.sqrt((X-x)**2 + (Y-y)**2)
    gauss = np.exp(-(d**2 / (2.0 * sigma**2)))
    gauss = (gauss / gauss.max()) * random.randint(50, 255)
    gauss = gauss.reshape(height, width)

    # Create a 3D noise array if the image has three channels (RGB)
    if len(image_array.shape) == 3:
        gauss = np.repeat(gauss, 3, axis=2)

    # Add the Gaussian noise to the image
    noisy_image = image_array + gauss

    # Clip values to be between 0 and 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image
