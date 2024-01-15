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


from skimage.measure import label
from natsort import natsorted
from glob import glob
from PIL import Image
import os
import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from tqdm import trange
import pandas as pd

import argparse

def mask_ious(masks_true, masks_pred):
    """ return best-matched masks """
    iou = _intersection_over_union(masks_true, masks_pred)[1:,1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind,pred_ind]
    preds = np.zeros(masks_true.max(), 'int')
    preds[true_ind] = pred_ind+1
    return iout, preds


def aggregated_jaccard_index(masks_true, masks_pred):
    """ AJI = intersection of all matched masks / union of all masks 
    
    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels
    Returns
    ------------
    aji : aggregated jaccard index for each set of masks
    """

    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = np.arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = np.logical_or(masks_true[n]>0, masks_pred[n]>0).sum()
        overlap = overlap[inds[preds>0]+1, preds[preds>0].astype(int)]
        aji[n] = overlap.sum() / union
    return aji 

def dice_score(true_mask, bn_mask, eps=0.00001):
    true_mask = 1*(true_mask>0)
    bn_mask   = 1*(bn_mask>0)

    # Computing intersection and union masks
    inter_mask = np.dot(bn_mask.flatten(), true_mask.flatten())
    union_mask = np.sum(bn_mask) + np.sum(true_mask) + eps

    # Computing the Dice coefficient
    return (2 * inter_mask + eps) / union_mask

def eval_metrics(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ average precision estimation: AP = TP / (TP + FP + FN)
    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)
    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels
    Returns
    ------------
    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    
    if len(masks_true) != len(masks_pred):
        raise ValueError('metrics.average_precision requires len(masks_true)==len(masks_pred)')

    f1  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)

    accuracy  = np.zeros((len(masks_true), len(threshold)), np.float32)
    precision = np.zeros((len(masks_true), len(threshold)), np.float32)
    recall    = np.zeros((len(masks_true), len(threshold)), np.float32)

    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    dice = []
    miou = []

    for n in range(len(masks_true)):
        tmp_dice = dice_score(masks_true[n], masks_pred[n])

        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            iou_arranged = arrange_iou(iou)
            miou.append(np.sum(iou_arranged)/ np.max(masks_true[n]))
            for k,th in enumerate(threshold):
                tp[n,k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]

        # Computing F1 score
        f1[n] = 2 * tp[n] / (2 * tp[n] + fp[n] + fn[n])
        # Computing the precision
        precision[n] =  tp[n] / (tp[n] + fp[n])

        # Computing the recall
        recall[n]    =  tp[n] / (tp[n] + fn[n])

        # Computing accuracy
        accuracy[n] = tp[n] / (tp[n] + fp[n] + fn[n])  

        dice.append(tmp_dice)
    # dice = np.mean(dice)
    # miou = np.mean(miou) 
    if not_list:
        miou, dice, f1, accuracy, precision, recall = miou[0], dice[0], f1[0], accuracy[0], precision[0], recall[0]#tp[0], fp[0], fn[0]
    return miou, dice, f1, accuracy, precision, recall #tp, fp, fn

@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------
    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    Returns
    ------------
    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    # put label arrays into standard form then flatten them 
#     x = (utils.format_labels(x)).ravel()
#     y = (utils.format_labels(y)).ravel()
    x = x.ravel()
    y = y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    Returns
    ------------
    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]
    
    ------------
    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken 
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix. 
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------
    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label
    Returns
    ------------
    tp: float
        number of true positives at threshold
        
    ------------
    How it works:
        (1) Find minimum number of masks
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...)
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to 
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels. 
        (4) Extract the IoUs fro these parings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned. 
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    # print('tp : ', tp)
    return tp

def arrange_iou(iou):
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.1).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    return iou[true_ind, pred_ind]



if __name__ == '__main__':

    ## HOW TO USE:
    # python eval_metrics.py --pred_seq '../my_results/01_RES' --gt_seq '../dataset/01_GT' --output_csvs '../my_results/csvs' --img_ext '*.tif'
    parser = argparse.ArgumentParser(description="Process input and output sequences along with additional parameters.")

    parser.add_argument("--pred_seq", help="Input predicted masks sequence", default="")
    parser.add_argument("--gt_seq", help="Input ground truth sequence", default="")
    parser.add_argument("--output_csvs", help="Output csv metrics", default="")
    parser.add_argument("--img_ext", help="The masks extension (*.tif, *.png, *.jpeg ...)", default="*.tif")

    args = parser.parse_args()


    methods = ['XAI_unet']
    show_methods = ['UNet(XAI)']

    my_dict_f1   = {"methods":[],"IoU threshold":[],"F1 score":[], "F1 score (std)":[],"Accuracy score":[], "Accuracy score (std)":[],"Precision score":[], "Precision score (std)":[],"Recall score":[], "Recall score (std)":[]};
    my_dict_miou = {"methods":[],"mIoU (mean)":[], "mIoU (std)":[], "Dice (mean)":[], "Dice (std)":[]};

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


    for m in trange(len(methods)):
        # print(show_methods[m])
        mIoU_list, dice_list, f1_list, std_f1_list, acc_list, std_acc_list, pres_list, std_pres_list, rec_list, std_rec_list = [], [], [], [], [], [], [], [], [], []

        # Sorting the file paths
        xai_unet_preds = natsorted(glob(os.path.join(args.pred_seq, args.img_ext)))
        gt_masks = natsorted(glob(os.path.join(args.gt_seq, args.img_ext)))

        masks_true, masks_pred = [], []
        for i in range(len(gt_masks)):

            masks_true.append(label(np.array(Image.open(gt_masks[i]))))
            masks_pred.append(np.array(Image.open(xai_unet_preds[i])))
            
        mIoU, dice, f1, accuracy, precision, recall = eval_metrics(masks_true, masks_pred, threshold=thresholds)

        f1_list_tmp = np.mean(f1, axis=0)
        std_f1_list = np.std(f1, axis=0)

        acc_list_tmp = np.mean(accuracy, axis=0)
        std_acc_list = np.std(accuracy, axis=0)

        pres_list_tmp = np.mean(precision, axis=0)
        std_pres_list = np.std(precision, axis=0)

        rec_list_tmp = np.mean(recall, axis=0)
        std_rec_list = np.std(recall, axis=0)

        for k in range(len(f1_list_tmp)):
            my_dict_f1["methods"].append(show_methods[m])
            my_dict_f1["IoU threshold"].append(thresholds[k])

            my_dict_f1["F1 score"].append(round(f1_list_tmp[k]*100,2))
            my_dict_f1["F1 score (std)"].append(round(std_f1_list[k]*100,2))

            my_dict_f1["Accuracy score"].append(round(acc_list_tmp[k]*100,2))
            my_dict_f1["Accuracy score (std)"].append(round(std_acc_list[k]*100,2))

            my_dict_f1["Precision score"].append(round(pres_list_tmp[k]*100,2))
            my_dict_f1["Precision score (std)"].append(round(std_pres_list[k]*100,2))

            my_dict_f1["Recall score"].append(round(rec_list_tmp[k]*100,2))
            my_dict_f1["Recall score (std)"].append(round(std_rec_list[k]*100,2))
        
        my_dict_miou["methods"].append(show_methods[m])
        my_dict_miou["mIoU (mean)"].append(round(np.mean(mIoU)*100,2))
        my_dict_miou["mIoU (std)"].append(round(np.std(mIoU)*100,2))
        my_dict_miou["Dice (mean)"].append(round(np.mean(dice)*100,2))
        my_dict_miou["Dice (std)"].append(round(np.std(dice)*100,2))

    os.makedirs(args.output_csvs, exist_ok = True) 

    df_metrics_f1 = pd.DataFrame.from_dict(my_dict_f1)
    df_metrics_f1.to_csv(os.path.join(args.output_csvs,'f1_metrics.csv'))


    df_metrics_miou = pd.DataFrame.from_dict(my_dict_miou)
    df_metrics_miou.to_csv(os.path.join(args.output_csvs,'miou_metrics.csv'))
