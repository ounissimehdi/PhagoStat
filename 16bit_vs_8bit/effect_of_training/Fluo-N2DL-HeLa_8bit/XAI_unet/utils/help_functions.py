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
from PIL import Image
import numpy as np
import logging
import torch

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

    f1_loss, recall_loss, precision_loss = 0.0, 0.0, 0.0
    # Compute Dice coefficient for the given batch
    for pairs_idx, inputs in enumerate(zip(batch_mask, batch_true_bn_mask, batch_true_border_mask, batch_gt_bboxes)):

        bce                   = single_BCE_weighted_loss(inputs[0], inputs[1])
        border                = single_BORDER_Loss(inputs[0], inputs[2])
        # f1, recall, precision = F1_loss(inputs[0], inputs[3])
        dice                  = single_dice_coeff(inputs[0], inputs[1])
        
        dice_loss       += dice
        #f1_loss         += f1
        #recall_loss     += recall
        #precision_loss  += precision
        bce_loss        += bce#border_loss #+bce_loss
        border_loss     += border
    
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

    weights = [bg_target/fg_target, fg_target/fg_target]
    
    # Computing the bce loss
    return torch.mean(- weights[1] * gt_mask * torch.log(input_mask) - (1 - gt_mask) * weights[0] * torch.log(1 - input_mask))


def F1_loss(input_mask, gt_bboxes, eval_threshold=0.5, IoU_threshold=0.5, eps=10e-4):
    # Clamp input mask for numerical stability
    input_mask = torch.clamp(input_mask,min=eps,max=1-eps)
    
    # Thresholding the input mask to obtain numpy binary mask
    bn_mask = (input_mask > eval_threshold).cpu().detach().numpy().astype(np.float)

    print(bn_mask.shape)

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
    def __init__(self, imgs_dirs, masks_dirs, ref_image_path, normalize=False,cached_data=True, n_channels=1,scale=1):
        self.imgs_dirs = imgs_dirs    # All paths to images 
        self.masks_dirs = masks_dirs  # All paths to masks 
        self.scale = scale            # image and mask scale
        self.n_channels = n_channels  # input model channels
        self.normalize = normalize    # normalization switch

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
                    np_labeled = np.array(Image.open(self.masks_dirs[i]))
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


    def preprocess(self, pil_img, ref_image, n_channels, scale, normalize, mask=True):
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

        # Convert the PIL image into numpy array
        np_img = np.array(pil_img)

        # Add an extra dim if only H, W image
        if len(np_img.shape) == 2: np_img = np.expand_dims(np_img, axis=0)
        
        # Ensure the imgs to be in [0, 1]
        if np_img.max() > 1: np_img = np_img / np.max(np_img)
        
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
            np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)

            # Load & pre-process the mask if possible
            if len(self.masks_dirs) == len(self.imgs_dirs):
                # mask_dir = self.masks_dirs[i]
                # pil_mask = Image.open(mask_dir)
                # gt_bboxes = self.bboxes_prep(pil_mask)
                # np_border_mask = self.border_mask_generator(pil_mask)
                # np_mask = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)


                np_labeled = np.array(Image.open(self.masks_dirs[i]))
                pil_mask = Image.fromarray(np.array(np_labeled>0)*1.)
                gt_bboxes = self.bboxes_prep(np_labeled)
                np_border_mask = self.border_mask_generator(pil_mask)

                np_mask    = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
            
            
        if len(self.masks_dirs) == len(self.imgs_dirs):
            return {
                'image': torch.from_numpy(np_img).type(torch.FloatTensor),
                'mask': torch.from_numpy(np_mask).type(torch.FloatTensor),
                'border_mask': torch.from_numpy(np_border_mask).type(torch.FloatTensor), 
                'b-boxes': torch.from_numpy(gt_bboxes).type(torch.FloatTensor),
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