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
from natsort import natsorted
from tqdm import trange
from PIL import Image
import numpy as np
import logging
import torch
import os
from glob import glob
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


def CustomMetrics_TEST_batch(batch_mask, batch_true_bn_mask, batch_true_border_mask, batch_gt_bboxes, eval_threshold=0.5, IoU_threshold=0.5, eps=10e-4):
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
        f1, recall, precision = F1_loss(inputs[0], inputs[3])
        dice                  = single_dice_coeff(inputs[0], inputs[1])
        
        dice_loss       += dice
        f1_loss         += f1
        recall_loss     += recall
        precision_loss  += precision
        bce_loss        += bce#border_loss #+bce_loss
        border_loss     += border
    
    # Return Detection metrics over the given batch
    # f1_loss         /= (pairs_idx + 1)
    recall_loss     /= (pairs_idx + 1)
    precision_loss  /= (pairs_idx + 1)
    f1_loss          = torch.div(f1_loss, (pairs_idx + 1)) 
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

    weights = [bg_target/fg_target, fg_target/fg_target]
    
    # Computing the bce loss
    return torch.mean(- weights[1] * gt_mask * torch.log(input_mask) - (1 - gt_mask) * weights[0] * torch.log(1 - input_mask))


def F1_loss(input_mask, gt_bboxes, eval_threshold=0.5, IoU_threshold=0.5, eps=10e-4):
    # Clamp input mask for numerical stability
    input_mask = torch.clamp(input_mask,min=eps,max=1-eps)
    
    # Thresholding the input mask to obtain numpy binary mask
    bn_mask = (input_mask > eval_threshold).cpu().detach().numpy().astype(np.float)[0]

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
        #print("Input bboxes ",input_bbox)
        gt_tp_count = 0
        # Init the max IoU value
        max_iou = 0
        for gt_bboxe in gt_bboxes:
            #print("GT bboxes ",gt_bboxe)
            if torch.sum(gt_bboxe) != 0 :
                gt_tp_count +=1
                iou = IoU(gt_bboxe, input_bbox)
                # Update the Max IoU value
                if iou>max_iou: max_iou = iou
        # Quanitfy the detection (TP or FP)
        if max_iou> IoU_threshold: tp +=1
        else: fp += 1
    # Computing the False Negatives
    fn = gt_tp_count - tp

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



class CustomLSTMDetectionDataset(Dataset):
    """ CustomDataset : Class that loads data (images and masks) in efficient way"""
    def __init__(self, data_dir, ref_image_path, temporal_len=3, normalize=False,cached_data=True, n_channels=1,scale=1):
        self.scale = scale            # image and mask scale
        self.n_channels = n_channels  # input model channels
        self.normalize = normalize    # normalization switch
        self.temporal_len = temporal_len

        # List the files inside the input file
        files_annotated = natsorted([ item for item in os.listdir(data_dir) 
                                if os.path.isdir(os.path.join(data_dir, item)) ])

        batch_imgs_paths, batch_masks_paths, batch_dist_paths = [], [], []
        for file_date in files_annotated:
            
            imgs_paths = natsorted(glob(os.path.join(data_dir, file_date,"imgs", "*.*")))
            masks_paths = natsorted(glob(os.path.join(data_dir, file_date,"masks", "*.*")))
            #dist_paths = natsorted(glob(os.path.join(aug_annotated_data, file_date,"dist_maps", "*.*")))

            for i in range(len(imgs_paths)):
                if i + self.temporal_len <= len(imgs_paths): new_idx = i
                else:  new_idx = i-self.temporal_len
            
                tmp_imgs_list, tmp_masks_list, tmp_dist_list = [], [], []
                
                for k in range(self.temporal_len):
                    tmp_imgs_list.append(imgs_paths[new_idx+k])
                    tmp_masks_list.append(masks_paths[new_idx+k])
                    #tmp_dist_list.append(dist_paths[new_idx+k])
                
                batch_imgs_paths.append(tmp_imgs_list)
                batch_masks_paths.append(tmp_masks_list)
                #batch_dist_paths.append(tmp_dist_list)
        
        self.imgs_dirs   = batch_imgs_paths
        self.masks_dirs  = batch_masks_paths
        # self.dist_dirs = batch_dist_paths
        # Make sure the scale is between [0, 1]
        assert 0 < scale <= 1, '[ERROR] Scale must be between 0 and 1'

        # Load the reference image into RAM
        ref_image = Image.open(ref_image_path)
        
        # Save the reference image into RAM to be used
        self.ref_image = ref_image.copy()

        # Caching the dataset (WARRING : this needs to be used when you have big RAM memory)
        if cached_data:
            logging.info(f'[INFO] Caching the given dataset with {len(self.imgs_dirs)} images and {len(self.masks_dirs)} masks stacks (temporal lenght {self.temporal_len})')
            # Turn on the cache flag
            self.cached_dataset = True

            # Preparing the images and masks lists
            self.cache_imgs, self.cache_masks, self.border_masks, self.cache_bboxes = [], [], [], []
            
            # Cache & pre-process the images and the masks (train/val) ready
            for i in trange(len(self.imgs_dirs)):
                tmp_imgs_dirs = self.imgs_dirs[i]
                for k in range(len(tmp_imgs_dirs)):
                    pil_img = Image.open(tmp_imgs_dirs[k])
                    np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)
                    
                    if k !=0 : imgs_temps = np.vstack((imgs_temps, np_img))
                    else     : imgs_temps = np_img.copy()
                self.cache_imgs.append(imgs_temps)

                if len(self.masks_dirs) == len(self.imgs_dirs):
                    tmp_masks_dirs = self.masks_dirs[i]
                    
                    for k in range(len(tmp_masks_dirs)):
                        # Reading the binary mask
                        pil_mask  = Image.open(tmp_masks_dirs[k])

                        # Computing the bounding boxes from bn mask
                        gt_bboxes = self.bboxes_prep(pil_mask)

                        # Creating the border masks
                        np_border_mask = self.border_mask_generator(pil_mask)

                        # preparing the numpy mask
                        np_mask   = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
                        
                        if k !=0 : 
                            masks_temps  = np.vstack((masks_temps, np_mask))
                            bboxes_temps = np.vstack((bboxes_temps, gt_bboxes))
                            border_temps = np.vstack((border_temps, np_border_mask))
                        else     : 
                            masks_temps  = np_mask.copy()
                            bboxes_temps = gt_bboxes.copy()
                            border_temps = np_border_mask.copy()
                    
                    self.cache_masks.append(masks_temps)
                    self.cache_bboxes.append(bboxes_temps)
                    self.border_masks.append(border_temps)
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

    def bboxes_prep(self, pil_mask):
        # Nmpy version of the mask
        np_org_mask = np.array(pil_mask)

        # Labelling the binary mask
        np_label_mask = label(np_org_mask, connectivity=2)

        # Extract regions using the labelled mask
        regions = regionprops(np_label_mask)

        bboxes = [np.array([0,0,0,0]) for i in range(200)]
        for i in range(len(regions)):
            bboxes[i] = regions[i].bbox

        return np.expand_dims(np.array(bboxes), axis=0)


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
        if len(np_img.shape) == 2:
            np_img = np.expand_dims(np_img, axis=0)
            np_img = np.expand_dims(np_img, axis=0)
        else: np_img = np.expand_dims(np_img, axis=0)
        
        # Ensure the imgs to be in [0, 1]
        if np_img.max() > 1: np_img = np_img / 255
        
        return np_img

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
        # Nmpy version of the mask
        np_org_mask = np.array(pil_org_mask)

        # Computing the dilatation mask
        np_dil_mask = binary_dilation(np_org_mask, iterations=2).astype(np_org_mask.dtype)

        # Computing the borders mask
        rr, cc = np.where(np_org_mask != 0 )
        np_border_mask = np_dil_mask.copy()
        np_border_mask[rr, cc] = 0
        np_border_mask = binary_dilation(np_border_mask, iterations=4).astype(np_border_mask.dtype)
        if len(np_border_mask.shape) == 2:
            np_border_mask = np.expand_dims(np_border_mask, axis=0)
            np_border_mask = np.expand_dims(np_border_mask, axis=0)
        else: np_border_mask = np.expand_dims(np_border_mask, axis=0)
        return np_border_mask
    
    def __getitem__(self, i):
        # When the dataset is cached load the img and mask from RAM
        if self.cached_dataset:
            np_img_stack    = self.cache_imgs[i]
            if len(self.masks_dirs) == len(self.imgs_dirs):
                np_mask_stack        = self.cache_masks[i]
                gt_bboxes_stack      = self.cache_bboxes[i]
                np_border_mask_stack = self.border_masks[i]
        
        # Otherwise load the img and mask from Disk to RAM
        else:
            print("NOT YET ADAPTED")
            # Load the image 
            img_dir = self.imgs_dirs[i]
            pil_img = Image.open(img_dir)

            # Preprocess the image 
            np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)

            # Load & pre-process the mask if possible
            if len(self.masks_dirs) == len(self.imgs_dirs):
                mask_dir = self.masks_dirs[i]
                pil_mask = Image.open(mask_dir)
                gt_bboxes = self.bboxes_prep(pil_mask)
                np_border_mask = self.border_mask_generator(pil_mask)
                np_mask = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
            
            
        if len(self.masks_dirs) == len(self.imgs_dirs):
            return {
                'image': torch.from_numpy(np_img_stack).type(torch.FloatTensor),
                'mask': torch.from_numpy(np_mask_stack).type(torch.FloatTensor),
                'border_mask': torch.from_numpy(np_border_mask_stack).type(torch.FloatTensor), 
                'b-boxes': torch.from_numpy(gt_bboxes_stack).type(torch.FloatTensor)
            }
        else:
            return {
                'image': torch.from_numpy(np_img_stack).type(torch.FloatTensor)
            }



class ListsLongitudinalDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        #self.ids = natsorted([splitext(file)[0] for file in listdir(imgs_dir)
        #					  if not file.startswith('.')])

        #logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.imgs_dir)

    @classmethod
    def preprocess(cls, pil_img, scale):
        if scale !=1:
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        imgs_temps_list = self.imgs_dir[i]
        masks_temps_list = self.masks_dir[i]
        imgs_temps,masks_temps = 0,0 
        for k in range(len(imgs_temps_list)):
            #print(imgs_temps_list[k])
            mask = Image.open(masks_temps_list[k])
            img = Image.open(imgs_temps_list[k])

            assert img.size == mask.size, \
                f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'

            img = self.preprocess(img, self.scale)
            img = np.expand_dims(img, axis=0)
            
            mask = self.preprocess(mask, self.scale)
            mask = np.expand_dims(mask, axis=0)
            
            if k !=0:
                imgs_temps = np.vstack((imgs_temps, img))
                masks_temps = np.vstack((masks_temps, mask))
            else:
                imgs_temps = img.copy()
                masks_temps = mask.copy()
        
        return {
            'image': torch.from_numpy(imgs_temps).type(torch.FloatTensor),
            'mask': torch.from_numpy(masks_temps).type(torch.FloatTensor),
            'indx': i
        }
