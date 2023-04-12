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

import matplotlib
matplotlib.use('Agg')

from .libs import Frame_registration, Aggregates_analysis
from torch.utils.data import Dataset, DataLoader
from skimage.measure import regionprops, label
from skimage.exposure import match_histograms
from btrack.constants import BayesianUpdates
from PIL import Image, ImageFont, ImageDraw
from skimage.segmentation import watershed
from skimage import exposure, img_as_ubyte
from shapely.geometry import Polygon
from skimage.color import label2rgb
from scipy import ndimage as ndi
from aicspylibczi import CziFile
import matplotlib.pyplot as plt
from natsort import natsorted
# from tqdm import tqdm,trange
from imantics import Mask
from glob import glob
import pandas as pd
import numpy as np
import colorsys
import sys, os
import shutil
import btrack
import torch
import math
import cv2
import logging

logger = logging.getLogger(__name__)
logger.disabled = True

def aggregates_field_registration(input_path,
                                  output_path,
                                  microscope_resolution,
                                  scale_factor,
                                  offset=0,
                                  gauss_size=3,
                                  eps_err=1e-4,
                                  max_itr=1000,
                                  img_ext='.tif'):

    # Path to the original aggregates
    imgs_path =  natsorted(glob(os.path.join(input_path, '*'+img_ext)))

    # Loading and converting the image from RGB into gray scale
    reference_image_org = Image.open(imgs_path[offset])#.convert("L")

    # Path to the output masks
    reg_out_masks = os.path.join(output_path, "registered_aggregate")

    # Creating folders
    os.makedirs(reg_out_masks, exist_ok = True)

    # Creating folders
    os.makedirs(os.path.join(output_path, 'CSVs'), exist_ok = True)

    # Initiate the warp matrix (identity matrix)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Initiate needed lists to store results
    # xx_list, xy_list, yx_list, yy_list= [], [], [], []
    idx_list, x_list, x_m_list, y_list, y_m_list =  [], [], [], [], []
    for i in range(offset, len(imgs_path)):

        # Loading and converting the image from RGB into gray scale
        current_image_org = Image.open(imgs_path[i]).convert("L")
        
        # when the current image is differnet than the referenc image
        if i-offset != 0:
            align_org = Frame_registration(np.array(reference_image_org, dtype=np.uint8),
                                       np.array(current_image_org, dtype=np.uint8),
                                       gauss_size = gauss_size,
                                       eps_err = eps_err,
                                       max_itr = max_itr)
            current_image_org = align_org.register_frame(warp_matrix)
            warp_matrix = align_org.get_warp_matrix()

            # xx_list.append(str(warp_matrix[0][0]))
            # xy_list.append(str(warp_matrix[0][1]))

            # yx_list.append(str(warp_matrix[1][0]))
            # yy_list.append(str(warp_matrix[1][1]))

            x_list.append(str(warp_matrix[0][2]))
            y_list.append(str(warp_matrix[1][2]))
            x_m_list.append(str(warp_matrix[0][2]*scale_factor*np.sqrt(microscope_resolution)))
            y_m_list.append(str(warp_matrix[1][2]*scale_factor*np.sqrt(microscope_resolution)))
        # The current image is the reference image (no need to register)
        else :
            reference_image_org = current_image_org.copy()
            # xx_list.append('1')
            # xy_list.append('0')

            # yx_list.append('0')
            # yy_list.append('1')

            x_list.append('0')
            y_list.append('0')
            x_m_list.append('0')
            y_m_list.append('0')
        
        # Saving the frame ID
        idx_list.append(i)

        # Save the registered image into the given output folder
        current_image_org.save(os.path.join(reg_out_masks ,str(i)+img_ext))

        # Preparing the csv file to save the the warp params
        csv_details = {'frame_name': idx_list,
                    # 'xx': xx_list,
                    # 'xy': xy_list,
                    # 'yx': yx_list,
                    # 'yy': yy_list,
                    'x': x_list,
                    'y': y_list,
                    'x(microns)':x_m_list,
                    'y(microns)':y_m_list}

        # Writing data in a csv file
        data_frame = pd.DataFrame(csv_details)
        
        # Save the csv file 
        data_frame.to_csv(os.path.join(output_path, 'CSVs', "frames_registration_params.csv"), index = False)
    
    image_list_2_gif(output_dir=output_path,
                imgs_paths=natsorted(glob(os.path.join(reg_out_masks, '*'+img_ext ))),
                gif_file_name='registered_aggregate')

def aggregates_detection_tracking(input_raw_path,
                                  input_registered_path,
                                  output_path,
                                  area_resolution,
                                  ignore_debris_area,
                                  blur_eps=0.01,
                                  search_frames=14,
                                  aggregates_th=0.5,
                                  frame_time=2,
                                  save_detection_frames=True,
                                  img_ext='.tif'):
                                  
    # All image paths to the registered aggregates
    registered_frames_paths =  natsorted(glob(os.path.join(input_registered_path, '*'+img_ext)))

    # Getting the first frame name (index)
    reference_index = int(path_to_filename(registered_frames_paths[0]))

    # The starting frame
    offset              = reference_index
    current_frame       = reference_index
    
    # Getting the last frame index
    frames_number       = int(path_to_filename(registered_frames_paths[len(registered_frames_paths)-1]))
    
    # Initiate Laplacian variance list
    var_tab = []

    # Initiate a helping variable 
    persuit_frame = 0

    # Initiate end of the current frame analysis flag
    done_flag = False

    analyse_aggregates = Aggregates_analysis()

    # Getting the reference aggregates image
    aggregate_image_ref = Image.open(os.path.join(input_registered_path, str(reference_index)+img_ext))

    # Prepare the labels of the reference aggregates image
    analyse_aggregates.prepare_labels(aggregate_image_ref, th=aggregates_th)
    regions_list_ref = analyse_aggregates.get_regions(big_debris= ignore_debris_area)
    
    # Getting the current aggregates image
    aggregates_image_one = Image.open(os.path.join(input_registered_path, str(reference_index+1)+img_ext))

    # Prepare the labels of the current aggregates image
    analyse_aggregates.prepare_labels(aggregates_image_one, th=aggregates_th)
    regions_list_one = analyse_aggregates.get_regions(big_debris= ignore_debris_area)

    # Computing the initial non moving aggregates
    ref_regions = analyse_aggregates.get_non_moving_aggregates(regions_list_ref, regions_list_one)
    
    # Computing the total reference aggregates area
    ref_area = analyse_aggregates.get_total_area(ref_regions, area_resolution)

    if save_detection_frames:
        detection_save_path = os.path.join(output_path, 'detected_aggregate')
        
        # Making sure there is a folder to store the data
        os.makedirs(detection_save_path, exist_ok = True)

        labels_save_path = os.path.join(output_path, 'labeled_aggregate')
        
        # Making sure there is a folder to store the data
        os.makedirs(labels_save_path, exist_ok = True)

    # Make a folder to store CSVs
    os.makedirs(os.path.join(output_path, 'CSVs'), exist_ok=True)

    # Initiate the lists where to store the needed data
    time_list, aggregates_count_list, aggregates_total_area_list = [], [], []

    # Create a manual tqdm instance
    # progress_bar = tqdm(total=frames_number-offset, desc="Progress")

    while current_frame <= frames_number:

        if not(done_flag):
            # Getting the current raw aggregates frame
            pil_raw_gray_frame = Image.open(os.path.join(input_raw_path, str(current_frame)+img_ext))

            # Getting the current aggregates frame
            pil_registered_gray_frame = Image.open(os.path.join(input_registered_path, str(current_frame)+img_ext))

            # Getting the var
            var = cv2.Laplacian(np.array(pil_raw_gray_frame), cv2.CV_64F).var()

            # Storing the var of the Laplacian of the image
            var_tab.append(var)

            # Getting the frame 
            frame_number = current_frame - offset

            # Compute the labels of the current aggregates image
            analyse_aggregates.prepare_labels(pil_registered_gray_frame, th=aggregates_th)

            # Compute the aggregates regions
            tmp_regions_list = analyse_aggregates.get_regions(big_debris=ignore_debris_area)

            # Compute the non moving aggregates (not yet eaten by cells)
            regions_list = analyse_aggregates.get_non_moving_aggregates(ref_regions, tmp_regions_list)

            # Compute the aggregates count
            aggregates_count = analyse_aggregates.get_aggregates_count(regions_list)

            if aggregates_count != 0 :
                # Getting the total area of the aggregates
                current_total_area = analyse_aggregates.get_total_area(regions_list, area_resolution)

                # Computing the eaten surface
                eat_surface = ref_area-current_total_area
                
                # Working on non-reference frames
                if frame_number != 0:
                    # Initiate the persuit frame index (variable)
                    persuit_frame = 0

                    ########## START : Fuzzy frames rejection mechanism ########## 
                    if 1-(var/var_tab[-2]) > blur_eps or 1-(var/var_tab[-2])< -0.5:
                        # Setting the min 
                        #min_eaten = eat_surface
                        max_count = 0
                        persuit_frame = current_frame
                        for i in range(0, search_frames+1):
                            if current_frame+i < frames_number:
                                # Getting the current aggregates frame
                                tmp_aggregates_frame =  Image.open(os.path.join(input_registered_path, str(current_frame+i)+img_ext))

                                # Prepare the labels of the current aggregates frame
                                analyse_aggregates.prepare_labels(tmp_aggregates_frame, th=aggregates_th)

                                # Getting the regions of the labled bw frame
                                tmp_regions = analyse_aggregates.get_regions(big_debris=ignore_debris_area)

                                # Getting the non moving objects
                                tmp_regions = analyse_aggregates.get_non_moving_aggregates(ref_regions, tmp_regions)

                                # Getting the area of the non moving aggregates
                                #temp_area = aggregates.get_total_area(regions_temp, area_resolution)
                                
                                # Getting the count of the non moving aggregates
                                temp_count = analyse_aggregates.get_aggregates_count(tmp_regions)

                                # Getting the temporary eatten surface
                                #eat_surface_temp = ref_area-temp_area

                                # Check if we are getting a better matching
                                ##### Using aggregates Counting ##### 
                                if temp_count > max_count:
                                    max_count = temp_count
                                    persuit_frame = current_frame+i

                                ##### Using aggregates surface ##### 
                                #if eat_surface_temp < min_eaten:
                                #	min_eaten = eat_surface_temp
                                #	persuit_frame = current_frame+i
                        
                        # Attributing the persuit frame index (end of blurry frames) 
                        current_frame = persuit_frame
                        continue
                    ########## END : Fuzzy frames rejection mechanism ##########
                
                # The case of the first frame
                if persuit_frame == 0:
                    # Saving the reference regions and reference count
                    ref_regions = regions_list.copy()
                    # ref_count = aggregates_count
            # All aggregates was eaten by cells no need to continue the analysis
            else:
                done_flag = True
                eat_surface = ref_area
        # All aggregates was eaten by cells no need to continue the analysis
        else:
            aggregates_count = 0
            eat_surface = ref_area

        if persuit_frame == 0:
            if save_detection_frames:           
                analyse_aggregates.save_plot_aggregates(pil_registered_gray_frame,
                                                        regions_list,
                                                        detection_save_path,
                                                        labels_save_path,
                                                        current_frame)

            # Saving the current frame time
            time_list.append(frame_number*frame_time)
            
            # Saving the aggregates count
            aggregates_count_list.append(aggregates_count)

            # Initialization
            if eat_surface<0:eat_surface = 0
            if len(aggregates_total_area_list)!=0:
                if eat_surface< aggregates_total_area_list[-1]: eat_surface = aggregates_total_area_list[-1]

            # Saving the Aggregates total area (µm x µm)
            if current_frame == reference_index: aggregates_total_area_list.append(0)
            else: aggregates_total_area_list.append(round(eat_surface, 4))

            # Passing to the next frame
            current_frame += 1
            

        else : current_frame = persuit_frame

        if frame_number != 0:
            # Preparing the csv file to save the needed data in the local storage
            csv_details = {'time (min)': time_list,
                        'Aggregates count (n)': aggregates_count_list,
                        'Aggregates total area (micro-meters-square)': aggregates_total_area_list}
            
            # Saving details in a csv file
            save_csv(csv_details, os.path.join(output_path, 'CSVs'), 'phagocytosis_evolution')

            # Update the progress bar
            # progress_bar.update(1)

    # Close the progress bar
    # progress_bar.close()

    # # Path to the figures folder
    # figures_path = os.path.join(output_path, "summary_figures")
    
    # # Saving Aggregates-count figure
    # save_plot(figures_path,
    #             "Aggregates count",
    #             "Aggregates count [n]",
    #             aggregates_count_list,
    #             time_list)

    # # Saving Aggregates-area figure
    # save_plot_ref(figures_path,
    #                 "Aggregates area",
    #                 "Phagocytosed aggregates area [µm x µm]",
    #                 aggregates_total_area_list,
    #                 time_list,
    #                 ref_area)

    

    # Loading the analyzed frames' indexes 
    ok_frames  = [os.path.splitext(file)[0] for file in os.listdir(detection_save_path)      if not file.startswith('.')]
    all_frames = [os.path.splitext(file)[0] for file in os.listdir(input_registered_path)    if not file.startswith('.')]

    # Convert lists to sets
    all_frames_set = set(all_frames)
    ok_frames_set  = set(ok_frames)

    # Calculate the difference between the two sets (estimating the blurry frames that was rejected)
    difference = all_frames_set.difference(ok_frames_set)

    # Convert the resulting set back to a list
    rejected = natsorted(list(difference))
    
    # Preparing the csv file to save the needed data in the local storage
    csv_rejected_frame = {'Name of rejected blurry frames': rejected}
    
    # Saving T50 details in a csv file
    save_csv(csv_rejected_frame, os.path.join(output_path, 'CSVs'), "blurry_frames_rejection_details")
    
    if save_detection_frames:
        image_list_2_gif(output_dir=output_path,
                imgs_paths=natsorted(glob(os.path.join(detection_save_path, '*'+img_ext ))),
                gif_file_name='detected_aggregate')
        
        image_list_2_gif(output_dir=output_path,
                imgs_paths=natsorted(glob(os.path.join(labels_save_path, '*'+img_ext ))),
                gif_file_name='labeled_aggregate')
    
    # Free disk space
    try: shutil.rmtree(detection_save_path)
    except FileNotFoundError: pass



def microglia_XAI_segmentation_detection(probability_maps_path,
                                         cell_frame_paths,           
                                         output_dir,
                                         mask_threshold,
                                         temporal_len,
                                         debrit_min_area,
                                         original_img_hight=2048,
                                         microscope_resolution=0.103**2,
                                         offset=0,
                                         plot_results=True):
        
    # Prepare a list to store all the DL prediction probability maps 
    all_propa_maps = []

    # Remove if exist the previous result labels directory
    try: shutil.rmtree(os.path.join(output_dir, 'labels'))
    except FileNotFoundError: pass

    # Remove if exist the previous result labels directory
    try: shutil.rmtree(os.path.join(output_dir, 'spectral_labels'))
    except FileNotFoundError: pass

    # Remove if exist the previous result probability maps directory
    try: shutil.rmtree(os.path.join(output_dir, 'overlayed_cells'))
    except FileNotFoundError: pass

    if plot_results:
        # Preparing the folder to store ploting results
        labels_output       = os.path.join(output_dir, 'labeled_cell')
        spectral_output     = os.path.join(output_dir, 'spectral_labels')
        overlayed_output    = os.path.join(output_dir, 'overlayed_cells')

        # Create folders to store all files
        os.makedirs(labels_output, exist_ok = True)
        os.makedirs(spectral_output, exist_ok = True)
        os.makedirs(overlayed_output, exist_ok = True)

    # Paths to the probability maps
    all_proba_paths = natsorted(glob(os.path.join(probability_maps_path, '*.tif')))

    # Loading the probability maps from the drive to the RAM
    for proba_path in all_proba_paths:
        temp_prob = np.array(Image.open(proba_path), dtype=np.float32)
        if np.max(temp_prob)>1: temp_prob = temp_prob/255.0
        all_propa_maps.append(temp_prob)

    idx_list, x_list, y_list, area_list = [], [], [], []
    total_area_list, cell_total_count_list, idx_featrues = [], [], []
    # Sliding window segmentation, separation and detection algorithm
    for i in range(len(all_propa_maps)):
        # ########### Probability map overlap sliding window algorithm ###########
        # Check if the window is larger than actual frame number
        if (i+temporal_len) <= len(all_propa_maps):
            
            # Load the first frame in this time window
            # proba_map = np.mean(all_propa_maps[i:i+temporal_len], axis=(0))
            proba_map = all_propa_maps[i]
            
            # Summing the probability maps over the time window
            proba_sum = np.sum(all_propa_maps[i:i+temporal_len], axis=(0))
            
            # Computing the high probability cell positions
            np_bn_mask = proba_sum > mask_threshold*temporal_len
        else:
            # Load the last frames since the binary intersection is ready
            proba_map = np.array(all_propa_maps[i], dtype=np.float32)
        
        # Labelling the resulting mask
        labels = label(np_bn_mask,connectivity=2)

        # Compute regions using the labeled image
        regions = regionprops(labels)

        # Initiate the centroid list to store the sliding window results
        centroid_list_s_window = []

        # Go over all detected regions and remove debris
        for region in regions:
            # print(region.area)
            if region.area> debrit_min_area: centroid_list_s_window.append(region.centroid)

        # Create a new mask to store clean mask
        mask_out = np.zeros(np_bn_mask.shape, dtype=bool)

        # Creating markers on the mask where the cell seeds are located
        for f in range(len(centroid_list_s_window)):
            mask_out[int(centroid_list_s_window[f][0]), int(centroid_list_s_window[f][1])] = True
        
        # Using the markers along with the probability map to create clean separation
        markers, _ = ndi.label(mask_out)
        labels = watershed(-proba_map, markers, mask=proba_map>0.5)

        # Computing the corrected regions 
        corrected_regions = regionprops(labels)
        
        # Preparing the list that stores centroids for plotting
        centroid_list_plt = []

        # Prepare the cell features to be saved in a DataFrame
        for region in corrected_regions:
            # Saving the frame ID
            idx_list.append(i+offset)

            # Retrieving the cell's (x, y) coordinates
            y, x = int(region.centroid[0]), int(region.centroid[1])

            # Storing the cell's (x, y) into a list
            x_list.append(x)
            y_list.append(y)

            # Storing the cell area
            area_list.append(region.area)

            # Saving the centroids for plotting
            centroid_list_plt.append(region.centroid)

        # Preparing the cell features : centroids, area and cell count 
        csv_details = {'frame_name': idx_list,
                        'x (px)': x_list,
                        'y (px)': y_list,
                        'area (px^2)':area_list}

        # Writing data in a csv file
        data_frame = pd.DataFrame(csv_details)
        
        # Save the csv file 
        data_frame.to_csv(os.path.join(output_dir, 'CSVs', "cells_centroid_features.csv"), index = False)

        # Load the first frame in this time window
        np_bn_total_area = np.array(all_propa_maps[i])> mask_threshold

        # Getting the image hight to compute the area in microns square
        image_hight = np_bn_total_area.shape[0]

        # Threshold the probability map to get a binary mask
        total_area_px, _ = np.where(np_bn_total_area>0)

        # Saving the total cell area
        total_area_list.append(len(total_area_px) * (original_img_hight/image_hight)**2 * microscope_resolution)

        # Saving the total cell count
        cell_total_count_list.append(len(corrected_regions))
        
        # Saving frame ID
        idx_featrues.append(i+offset)

        # Preparing the cell parameters : cell count and total area in microns square 
        csv_total_area_details = {'frame_name': idx_featrues,
                                  'cell count': cell_total_count_list,
                                  'mean area (microns^2)': list(np.array(total_area_list)/np.array(cell_total_count_list)) , 
                                  'total area (microns^2)':total_area_list}

        # Writing data in a csv file
        total_area_data_frame = pd.DataFrame(csv_total_area_details)
        
        # Save the csv file 
        total_area_data_frame.to_csv(os.path.join(output_dir, 'CSVs', "cell_count_and_total_area.csv"), index = False)

        # Plotting results
        if plot_results:
            # Save GIF image
            save_gif_size = 1024

            # Size of the original image
            original_image_size = labels.shape[0]

            # Getting the regions from labels
            final_regions = regionprops(labels)

            labeled_image = np.zeros((original_image_size, original_image_size), dtype=np.uint16)

            # Create a dictionary for label-color mappings
            label_colors = {}
            # Plotting all regions bounding boxes on top of the numpy array
            for region in final_regions:
                # Saving the aggregate detection labels
                label_value = region['label']
                coords = np.array(region['coords'])
                labeled_image[coords[:, 0], coords[:, 1]] = label_value
                centroid = region.centroid
                label_colors[label_value] = centroid_to_hsl_rgb(centroid, original_image_size, original_image_size)
            
            # Color the aggregate labels
            rgb_image = np.zeros((original_image_size, original_image_size, 3), dtype=np.uint8)
            for label_value, color in label_colors.items(): rgb_image[labeled_image == label_value] = color

            # Saving the annotated image to the given path
            Image.fromarray(rgb_image).save(os.path.join(labels_output, str(i+offset)+'.png'))

            # Preparing the centroids into a numpy array for plotting
            centroid_list_plt = np.array(centroid_list_plt, dtype=np.int16)

            # Saving the label image
            # colored_label_image = label2rgb(labels, bg_label=0)
            # Image.fromarray((colored_label_image * 255).astype(np.uint8)).save(os.path.join(labels_output, str(i+offset)+'.png'))

            # Ploting spectral labels results
            # plt.imshow(labels, cmap=plt.cm.nipy_spectral)
            plt.imshow(rgb_image)
            plt.scatter(centroid_list_plt[:,1],centroid_list_plt[:,0], s = 60, c = 'red', marker = '*', edgecolors = 'white')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(spectral_output, str(i+offset)+'.png'), format="png", dpi=200, pad_inches=0, bbox_inches='tight')
            plt.close()
            
            # Reading the original image, resize and covert it to gray scale
            cell_img = Image.open(cell_frame_paths[i]).resize((save_gif_size,save_gif_size)).convert('L')

            # Loading the spectral labels
            spectral_image = Image.open(os.path.join(spectral_output, str(i+offset)+'.png')).resize((save_gif_size,save_gif_size))

            # Making sure the cell's and the labels' images are in RGB mode (color output)
            background = cell_img.convert("RGBA")
            overlay = spectral_image.convert("RGBA")

            # Creating a new image (50% cell's image and 50% cell labels)
            new_img = Image.blend(background, overlay, 0.5)
            
            # Ploting the overlay results
            plt.imshow(new_img)
            plt.scatter(centroid_list_plt[:,1]/(original_image_size/save_gif_size),
                        centroid_list_plt[:,0]/(original_image_size/save_gif_size),
                        s = 60, c = 'red', marker = '*', edgecolors = 'white')
            
            for id in range(len(final_regions)):
                mask = final_regions[id].image
                min_p = np.min(final_regions[id].coords, axis=0)
                polygons = Mask(mask).polygons()
                # Plot polygons only with at least 4 vertexes
                if np.shape(polygons.points[-1][:]+[min_p[1], min_p[0]])[0]>4:
                    polygon1 = Polygon(polygons.points[-1][:]+[min_p[1], min_p[0]])
                    x, y = polygon1.exterior.xy
                    plt.plot(np.array(x), np.array(y), c="red", linewidth=1.5,linestyle='dashed')
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(overlayed_output, str(i+offset)+'.png'), format="png", dpi=200, pad_inches=0, bbox_inches='tight')
            plt.close()


    # Plotting results
    if plot_results:
        # # Store all the images using the original file names
        # for k in range(len(all_propa_maps)):
        #     # Getting the original file name
        #     frame_name = path_to_filename(cell_frame_paths[k])

        #     # Save the probability map using the cell frame name
        #     all_propa_maps[k].save(os.path.join(probability_output, frame_name+'.tif'))

        # del all_propa_maps[:]

        ######################### SEPARATION-GIF ###########################################

        # Make the folder to store GIFs
        os.makedirs(os.path.join(output_dir, 'GIFs'), exist_ok=True)

        # Reading the font that will be used to write on top on the images
        title_font = ImageFont.truetype(os.path.join('.', 'utils', 'font.ttf'), 20)

        # Getting all tha paths to the overlay images
        imgs_paths = natsorted(glob(os.path.join(overlayed_output, '*.png')))

        # Loading the first image and be sure its RGB
        first_image = Image.open(imgs_paths[0]).resize((512,512)).convert('RGB')
        
        # Drawing the first label on the 1st image
        draw = ImageDraw.Draw(first_image)
        draw.text((0,0), 'Time : '+
            str(0)+' mins', (237, 0, 0), font=title_font)
        
        # Preparing a list to store all the images to be used to generate GIFs
        images_tab = []
        for j in range(len(imgs_paths)):

            # Loading the current image
            temp_image = Image.open(imgs_paths[j]).resize((512,512)).convert('RGB')

            # getting the name of the file to compute the time
            base = os.path.basename(imgs_paths[j])
            temp_time =  int(os.path.splitext(base)[0])-offset
            
            # Drawing time stamps
            tmp_draw = ImageDraw.Draw(temp_image)
            tmp_draw.text((0,0), 'Time : '+
                    str(2*(temp_time))+' mins', (237, 0, 0),
                    font=title_font)
            
            # Storing the images in a list to be used to create GIF
            images_tab.append(temp_image)
        
        # Saving the GIF image
        first_image.save(os.path.join(output_dir, 'GIFs', 'cell_instance_segmentation.gif'), save_all=True,
                    append_images=images_tab, optimize=True, duration=100, loop=0)

        
        # ######################### ATTENTION-GIF ###########################################
        # Path to the activation maps
        attention_maps_output = os.path.join(output_dir, 'cell_semantic_maps')

        # Getting all tha paths to the overlay images
        attention_imgs_paths = natsorted(glob(os.path.join(attention_maps_output, '*.tif')))

        # Loading the first image and be sure its RGB
        pil_first_attention = Image.open(attention_imgs_paths[0])

        # Reading the original image, resize and covert it to gray scale
        pil_cell_img = Image.open(cell_frame_paths[0])

        first_image = attention_heat_map_overlay(pil_cell_img, pil_first_attention, resize_shape=(512,512))
        
        # Darwing the first label on the 1st image
        draw = ImageDraw.Draw(first_image)
        draw.text((0,0), 'Time : '+
                str(0)+' mins', (237, 0, 0),
                font=title_font)
        
        # Preparing a list to store all the images to be used to generate GIFs
        images_tab = []
        for j in range(len(attention_imgs_paths)):

            # Loading the current attention map
            pil_attention = Image.open(attention_imgs_paths[j])

            # Reading the original image
            pil_cell_image = Image.open(cell_frame_paths[j])

            # Loading the current image
            temp_image = attention_heat_map_overlay(pil_cell_image, pil_attention, resize_shape=(512,512))

            # getting the name of the file to compute the time
            base = os.path.basename(attention_imgs_paths[j])
            temp_time =  int(os.path.splitext(base)[0])-offset
            
            # Drawing time stamps
            tmp_draw = ImageDraw.Draw(temp_image)
            tmp_draw.text((0,0), 'Time : '+
                str(2*(temp_time))+' mins', (237, 0, 0),
                font=title_font)
            
            # Storing the images in a list to be used to create GIF
            images_tab.append(temp_image)
        
        # Saving the GIF image
        first_image.save(os.path.join(os.path.join(output_dir, 'GIFs'), 'cell_semantic_segmentation.gif'), save_all=True,
                    append_images=images_tab, optimize=True, duration=100, loop=0)

        # Save them for visualization in PNG format
        # Getting all tha paths to the overlay images
        # for path in attention_imgs_paths:
        #     # Split the file name and the extension
        #     file_name, file_extension = os.path.splitext(path)

        #     # Get the directory name from the full file path
        #     dir_name = os.path.dirname(path)

        #     # Open the float TIFF image
        #     image = Image.open(path)

        #     # Convert the image data to a NumPy array
        #     image_data = np.array(image)

        #     # Rescale the float values to the range of 0 to 255
        #     scaled_image_data = ((image_data - image_data.min()) * (255.0 / (image_data.max() - image_data.min()))).astype(np.uint8)

        #     # Convert the rescaled data back to a PIL Image
        #     output_image = Image.fromarray(scaled_image_data)

        #     # Save the image as a uint8 PNG file
        #     output_image.save(os.path.join(dir_name, file_name+'.png'))

        #     # Check if it's a file (not a directory): Remove the file
        #     if os.path.isfile(path): os.remove(path)
            

        # Remove temporal folders to save space
        try: shutil.rmtree(spectral_output)
        except FileNotFoundError: pass

        try: shutil.rmtree(overlayed_output)
        except FileNotFoundError: pass

def attention_heat_map_overlay(pil_img, pil_activation, resize_shape=(1024,1024)):
    # Using PIL lib to resize the input numpy image
    pil_img = pil_img.convert('RGB')
    np_res_img = np.array(pil_img.resize(resize_shape))

    # Making sure the image is between [0, 1]
    if np_res_img.max()>1: np_res_img = np_res_img/255
    
    # Using PIL lib to resize the activation numpy image
    np_res_activation = np.array(pil_activation.resize(resize_shape))

    # Making sure the activation map is between [0, 1]
    if np_res_activation.max()>1: np_res_activation = np_res_activation/255

    # Computing the color map from the activation map
    heatmap = cv2.applyColorMap(np.uint8(255 * np_res_activation), cv2.COLORMAP_JET)

    # Activation map from BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # From uint8 to float [0,1]
    heatmap = np.float32(heatmap) / 255

    # Summing the input image and the RGB activation map
    activation_heat_map = heatmap + np_res_img

    # Making sure the out put image is between [0, 1]
    activation_heat_map = activation_heat_map / np.max(activation_heat_map)

    # Converting the activation map to uint8 for visualization
    activation_heat_map = np.uint8(255 * activation_heat_map)

    # Numpy array into PIL for easy saving
    activation_heat_map = Image.fromarray(activation_heat_map)

    return activation_heat_map

def microglia_tracking(cell_features_df_path,
                       cell_parameters_df_path,
                       offset_df_path,
                       btrack_config_json,
                       frame_time=2, # time in min
                       time_unit='min',
                       microscope_resolution=0.103**2,
                       original_img_hight=2048,
                       registered_img_hight=1024,
                       image_hight=2048,
                       image_width=2048,
                       track_interactive_step=20,
                       max_search_radius=100,
                       tracks_length=10,
                       keep_until=100,
                       registration_correction=True):

    # Defining the microscope resolution in case of a rescaled cell images
    px_to_m_scale = (original_img_hight/image_hight) * np.sqrt(microscope_resolution)

    # Defining the registration corrective scale (in case the cell images are rescaled 
    # from 2048 to 1024 for example)
    registration_relative_scale = registered_img_hight/image_hight

    # Loading the cell features (centroid coordinates) and area
    df_cell_features = pd.read_csv(cell_features_df_path)

    # Converting the data frame column of interest into lists
    features_time_list = list(df_cell_features['frame_name'])       # Loading the frame name
    x_list    = list(df_cell_features['x (px)'])           # Loading the x coordinate of the cell centroid
    y_list    = list(df_cell_features['y (px)'])           # Loading the y coordinate of the cell centroid
    area_list = list(df_cell_features['area (px^2)'])      # Loading the area of the cell

    
    # Loading the cell features (centroid coordinates) and area
    df_cell_parameters = pd.read_csv(cell_parameters_df_path)

    # Converting the data frame column of interest into lists
    frame_name_list = list(df_cell_parameters['frame_name'])                  # Loading the frame name
    total_area_list = list(df_cell_parameters['total area (microns^2)'])      # Loading the area of the cell
    cell_count_list = list(df_cell_parameters['cell count'])                  # Loading the count of the cell
    
    # Correction of the coordinates if registration are available
    if registration_correction:

        # Loading the offset details (registration)
        df_registration = pd.read_csv(offset_df_path)

        # Loading the frame name
        registration_time_list = list(df_registration['frame_name'])
       
        # Getting the frames length
        frames_length = len(df_registration['frame_name'])

        # Getting the first name from the registration data frame
        first_frame_name = df_registration['frame_name'][0]

        # Initiate the lists to store the corrected coordinates
        new_time, new_x, new_y, new_area = [], [], [], []
        
        # Retrieving all data using the registration information length
        for t in range(frames_length):
            # Getting the frame name
            tmp_frame_name = df_registration['frame_name'][t]

            # Loading the x and y axis offsets
            tmp_x_offset = df_registration['x'][t]/registration_relative_scale
            tmp_y_offset = df_registration['y'][t]/registration_relative_scale
            
            # Retrieving all the indices that match the frame name of the registration file
            indics = np.where(features_time_list==tmp_frame_name)
            
            # Correct all the coordinates in the current frame
            for idx in indics[0]:
                new_time.append(tmp_frame_name)
                new_area.append(area_list[idx])
                new_x.append(x_list[idx]-tmp_x_offset)
                new_y.append(y_list[idx]-tmp_y_offset)

        # Preparing the position dictionary
        position_dict = {
                        't':     np.array(new_time),
                        'x':     np.array(new_x),
                        'y':     np.array(new_y),
                        'area':  np.array(new_area)}
    else:
        # Getting the frames length
        frames_length = len(df_cell_features['frame_name'])

        # Getting the first name from the cell features data frame
        first_frame_name = df_cell_features['frame_name'][0]

        # Loading the frame name
        registration_time_list = list(df_cell_features['frame_name'])       

        # Defining the new time list
        new_time = features_time_list.copy()

        # Preparing the position dictionary
        position_dict = {
                        't':     np.array(features_time_list),
                        'x':     np.array(x_list),
                        'y':     np.array(y_list),
                        'area':  np.array(area_list)}
    
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")

    # Compute cell tracks using Btrack algorithm
    cell_tracks = get_track(position_dict,
                            max_search_radius,
                            btrack_config_json,
                            image_hight,
                            image_width,
                            track_interactive_step)

    sys.stdout = old_stdout # reset old stdout

    # Initiate the data lists
    all_records_distance, all_records_area = [], []

    # Computing the cell's movements
    for track in cell_tracks:
        # Getting time instances
        time_list = track.t

        # Checking if the tracking was successful at least for time = tracks_length
        if len(time_list)> tracks_length:
            # Getting the x and y coordinates of the cells
            x_coordinates = track.x
            y_coordinates = track.y

            # Initiate a list to store cells displacements
            instant_displacement = [0 for _ in range(frames_length)]

            # Initiate a list to store cells area
            instant_area = [0 for _ in range(frames_length)]

            # Computing the distance that the cell did travel between each time instance
            for i in range(1, len(x_coordinates)):
                # Compute the cell's distance between t[i] and t[i-1] in microns
                tmp_distance = np.linalg.norm(np.array([x_coordinates[i-1]*px_to_m_scale, y_coordinates[i-1]*px_to_m_scale])
                                 - np.array([x_coordinates[i]*px_to_m_scale, y_coordinates[i]*px_to_m_scale]))

                # Store the distance cell did travel
                instant_displacement[time_list[i]-first_frame_name] = tmp_distance

                # Store the cell area in microns^2
                instant_area[time_list[i-1]-first_frame_name] = track['area'][i-1] * (original_img_hight/image_hight)**2 * microscope_resolution

            # Save the cell's displacement
            all_records_distance.append(instant_displacement)
            
            # Save the cell's area
            all_records_area.append(instant_area)

    # Convert the list of lists into numpy matrix (cell id x instant_displacement)
    all_records_distance = np.array(all_records_distance)

    # Convert the list of lists into numpy matrix (cell id x instant_displacement)
    all_records_area = np.array(all_records_area)
    
    # Checking if we have long enough tracks otherwise the sequence will be rejected
    try:
        # Getting the btrack cell's tracks length
        btrack_tracks_length = all_records_distance.shape[1]

        # Check ig the tracks are long enough
        if btrack_tracks_length>keep_until: flag = True
        else: flag = False
    
    # If the tracks are empty the sequence is discarded
    except IndexError: flag = False
    
    # In case we have enough tracks
    if flag:
        # Initiate lists to store cells mean speed and instant displacement
        all_instant_mean_speed, all_instant_distance, all_instant_mean_area = [], [], []
        
        # Sanity check for distance and area length
        assert all_records_distance.shape[1] == all_records_area.shape[1], 'The cell distance and the cell area must have the same length'
        
        # Go over all instances
        for i in range(all_records_distance.shape[1]):
            # Initiate the cell speed
            temp_distance = 0
            
            # Initiate cell counter
            cell_speed_count = 0
            
            # Getting the cell's current speed
            for distance in all_records_distance[:, i]:
                # Taking into account only non null speed values
                if distance !=0:
                    # Summing speed values
                    temp_distance += distance

                    # Increment the cell counter
                    cell_speed_count += 1

            # Store the mean speed & displacement value
            if cell_speed_count !=0:
                # Saving the cell's distance
                all_instant_distance.append(temp_distance)

                # Computing the cell's speed in microns / frame_time unit
                temp_speed = temp_distance / frame_time

                # Saving the mean cells speed
                all_instant_mean_speed.append(temp_speed/cell_speed_count)
            
            # No cell is detected means no speed and no displacement
            else:
                all_instant_distance.append(0)
                all_instant_mean_speed.append(0)

            # Initiate the cell area variable
            temp_area = 0

            # Initiate cell counter
            cell_area_count = 0

            # Getting the cell's area 
            for area in all_records_area[:, i]:
                # Keep only detected cells
                if area !=0 and not(math.isnan(area)):
                    # Converting the cell area from px^2 to microns^2
                    temp_area += area
                    
                    # Increment the cell counter
                    cell_area_count += 1

            # Store the cell's mean area value
            if cell_area_count !=0:
                # Computing the mean area value
                temp_area /= cell_area_count

                # Saving the mean area value
                all_instant_mean_area.append(temp_area)
            
            # No cell is detected means no area
            else: all_instant_mean_area.append(0)

        # Initiate the time list in frame_time unit (ex. minutes)
        all_time_list = []

        # Initiate total area and cell count lists
        all_instant_total_area, all_instant_cell_count = [], [] 

        # Sanity check
        assert len(frame_name_list)  >= len(registration_time_list), 'The cells information length before applying registration must be greater or at least equal to registration information length'

        # Matching the frame names
        for i in range(len(frame_name_list)):
            for j in range(len(registration_time_list)):
                if frame_name_list[i] == registration_time_list[j]:
                    # Using the frame names to compute the time scale in frame_time unit (ex. minutes)
                    all_time_list.append((registration_time_list[j]-first_frame_name) * frame_time)

                    # Appending the total area and the cell count
                    all_instant_total_area.append(total_area_list[i])
                    all_instant_cell_count.append(cell_count_list[i])

                    continue
        

        clean_cell_parameters = {'time ('+time_unit+')'                      :   all_time_list,
                                'mean speed tracking (microns/'+time_unit+')':   all_instant_mean_speed,
                                'mean area tracking (microns^2)'             :   all_instant_mean_area,
                                'mean area (microns^2)'                      :   list(np.array(all_instant_total_area)/np.array(all_instant_cell_count)),
                                'total area (microns^2)'                     :   all_instant_total_area,
                                'cell count (n)'                             :   all_instant_cell_count,
                                'total movement tracking (microns)'          :   all_instant_distance}

        # Writing data in a csv file
        tracking_data_frame = pd.DataFrame(clean_cell_parameters)
        
        path_to_save_results = os.path.dirname(cell_features_df_path)

        # Save the csv file 
        tracking_data_frame.to_csv(os.path.join(path_to_save_results, "cell_tracking_report.csv"), index = False)

        # Tracking was successful
        return True

    # Tracking wasn't successful
    return False


def get_track(position_dict,
              max_search_radius,
              btrack_config_json,
              image_hight,
              image_width,
              track_interactive_step):
    

    # Using the position features to track cells over time
    objects = btrack.dataio.objects_from_dict(position_dict)
    
    # initialise a tracker session using a context manager
    with btrack.BayesianTracker(verbose=False, max_search_radius = max_search_radius) as tracker:

        # configure the tracker using a config file
        tracker.configure_from_file(btrack_config_json)

        # set the update method and maximum search radius (both optional)
        tracker.update_method = BayesianUpdates.EXACT
        #tracker.max_search_radius = 200

        # append the objects to be tracked
        tracker.append(objects)

        # set the volume (Z axis volume is set very large for 2D data)
        tracker.volume=((0,image_hight),(0,image_width),(-1e25,1e25))

        # track them (in interactive mode)
        tracker.track_interactive(step_size=track_interactive_step)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks as a python list
        tracks = tracker.tracks

    return tracks

class CustomDataset(Dataset):
    """ CustomDataset : Class that loads data (images and masks) in efficient way"""
    def __init__(self, imgs_dirs, ref_image_path, normalize=True, cached_data=True, scale=1):
        self.imgs_dirs = imgs_dirs    # All paths to images 
        self.scale = scale            # image and mask scale
        self.normalize = normalize    # normalization switch

        # Make sure the scale is between [0, 1]
        assert 0 < scale <= 1, '[ERROR] Scale must be between 0 and 1'

        # Load the reference image into RAM
        ref_image = Image.open(ref_image_path)
        
        # Save the reference image into RAM to be used
        self.ref_image = ref_image.copy()

        self.cached_dataset = False

        # Caching the dataset (WARRING : this needs to be used when you have big RAM memory)
        if cached_data:
            # Turn on the cache flag
            self.cached_dataset = True

            # Preparing the images list
            self.cache_imgs = []
            
            # Cache & pre-process the images and the masks (train/val) ready
            for i in range(len(imgs_dirs)):
                pil_img = Image.open(self.imgs_dirs[i])
                np_img = self.preprocess(pil_img, self.ref_image, self.scale, self.normalize)
                self.cache_imgs.append(np_img)
            
    def __len__(self): return len(self.imgs_dirs)

    def delete_cached_dataset(self):
        try:
            del self.cache_imgs[:]
            return True
        except: return False

    def preprocess(self, pil_img, ref_image, scale, normalize):
        
        # This part is concerns the normalization 
        if normalize:
            # Make sure the reference image and the current image have the same size
            if pil_img.size != ref_image.size:
                pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image.convert('L'))))
            else:
                pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image.convert('L'))))


        # Converting from RGB to Gray scale 
        pil_img = pil_img.convert("L")

        # Rescale the image if needed
        if scale != 1 :
            # Get the H and W of the img
            w, h = pil_img.size

            # Get the estimated new size
            newW, newH = int(scale * w), int(scale * h)

            # Resize the image according the given scale
            pil_img = pil_img.resize((newW, newH))

        # Convert the PIL image into numpy array
        np_img = np.array(pil_img)

        # Add an extra dim to support batch implimentation H, W image
        np_img_ready = np.expand_dims(np_img, axis=0)
        
        # Ensure the imgs to be in [0, 1]
        if np_img_ready.max() > 1: np_img_ready = np_img_ready / 255
        
        return np_img_ready
    
    def __getitem__(self, i):
        # When the dataset is cached load the img and mask from RAM
        if self.cached_dataset:
            np_img = self.cache_imgs[i]
        
        # Otherwise load the img and mask from Disk to RAM
        else:
            # Load the image 
            img_dir = self.imgs_dirs[i]
            pil_img = Image.open(img_dir)

            # Preprocess the image 
            np_img = self.preprocess(pil_img, self.ref_image, self.scale, self.normalize)


        return {
            'image': np_img
            # 'image': torch.from_numpy(np_img).type(torch.FloatTensor)
        }

def path_to_filename(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def save_csv(csv_dict, path, filename, indexing=False):
    # Writing data in a csv file
    data_frame = pd.DataFrame(csv_dict)
    
    # Making sure there is a folder to store the data
    os.makedirs(path, exist_ok = True)
    
    # Save the csv file 
    data_frame.to_csv(os.path.join(path, filename+".csv"), index = indexing)

def save_plot_ref(path, figure_name, ylabel_txt, data_list, time_list, plot_ref_area):
    # Making sure there is a folder to store the data
    os.makedirs(path, exist_ok = True)

    # Initiating the labels names
    label_raw = "Phagocytosis evolution"
    label_ref = "Total aggregates area at t=0 min"
    
    # Plot the data
    fig = plt.figure(figsize=(10,8))
    plt.plot(time_list,data_list,label=label_raw)
    plt.hlines(y=plot_ref_area, xmin=0, xmax=[time_list[-1]], colors='r', linestyles='--', lw=2, label=label_ref+' ('+str(round(plot_ref_area,2))+' µm**2)')
    plt.legend()
    plt.grid()
    plt.xlabel("Time [min]")
    plt.ylabel(ylabel_txt)
    plt.title(figure_name)
    plt.savefig(os.path.join(path, figure_name+".pdf"), dpi=150)
    plt.close(fig)

def save_plot(path, figure_name, ylabel_txt, data_list, time_list):
    # Making sure there is a folder to store the data
    os.makedirs(path, exist_ok = True)

    # Plot the data
    fig = plt.figure(figsize=(10,8))
    plt.plot(time_list,data_list)
    plt.grid()
    plt.xlabel("Time [min]")
    plt.ylabel(ylabel_txt)
    plt.title(figure_name)
    plt.savefig(os.path.join(path, figure_name+".pdf"), dpi=150)
    plt.close(fig)

def imgs_to_gif_aggregates(path_to_detected_aggregates, path_to_save_gif, seq_name, font, resize=512):
    # Load all frames paths
    imgs_paths = natsorted(glob(path_to_detected_aggregates))

    # Loading the first image
    first_image = Image.open(imgs_paths[0])

    # Resizing the first image
    first_image = first_image.resize((resize, resize), Image.ANTIALIAS)

    # Retriving the image index
    first_time = int(path_to_filename(imgs_paths[0]))
    
    # Drawing the image label with the time stamps
    draw = ImageDraw.Draw(first_image)
    draw.text((0,0), 'Time : '+
              str(0)+' mins', (237, 230, 211),
              font=font)
    
    # Intiate the images list (to store all the images after drawing)
    images_tab = []
    for j in range(len(imgs_paths)):
        # Reading the current image
        temp_image = Image.open(imgs_paths[j])

        # resizing the current image
        temp_image = temp_image.resize((resize, resize), Image.ANTIALIAS)

        # Retriving the current image index
        temp_time = int(path_to_filename(imgs_paths[j]))
        
        # Drawing the current image label with the time stamps
        tmp_draw = ImageDraw.Draw(temp_image)
        tmp_draw.text((0,0), 'Time : '+
              str(2*(temp_time-first_time))+' mins', (237, 230, 211),
              font=font)
        # Appending the annotated image into a list
        images_tab.append(temp_image)
    
    # Creating a GIF animation with an infinite loop
    first_image.save(os.path.join(path_to_save_gif, 'aggregates'+seq_name+'.gif'), save_all=True,
                  append_images=images_tab, optimize=True, duration=100, loop=0)


# Disable nasty printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__


def czi_to_frames(czi_file_path,
                  output_path,
                  scene_number,
                  reference_cell_image_path,
                  offset,
                  resize=(1024, 1024),
                  img_ext='.tif'):

    # Loading the cell reference image for histogram normalization
    cell_reference_image = np.array(Image.open(reference_cell_image_path))
    
    # Converting the cell image to 8bit representation
    cell_reference_image = img_as_ubyte(cell_reference_image)

    # Loading the cell reference image for histogram normalization
    # aggr_reference_image = np.array(Image.open(reference_aggr_image_path), dtype=np.uint16)

    # Reading the czi file
    czi = CziFile(czi_file_path)

    # Getting the czi shapes
    czi_shapes  = czi.get_dims_shape()

    # Defining cells & aggregate channel
    aggregate_channel, cell_channel = 0, 1

    # Path to cell scene output path (where to store output data)
    scene_cell_path = os.path.join(output_path, 'scene_'+str(scene_number), 'normalized_cell')

    # Path to cell scene output path (where to store output data)
    scene_aggregate_path = os.path.join(output_path, 'scene_'+str(scene_number), 'normalized_aggregate')

    # Creating folders
    os.makedirs(scene_cell_path, exist_ok = True)
    os.makedirs(scene_aggregate_path, exist_ok = True)

    # Reading the data accouring to the time intervals from the meta-data
    for t in range(czi_shapes[scene_number]['T'][0], czi_shapes[scene_number]['T'][1]):
        if t >= offset: 
            # Reading the cell's channel
            np_cell_image, _ = czi.read_image(S=scene_number,T=t,C=cell_channel)
            np_cell_image = np.squeeze(np_cell_image)

            # Reading the aggregates's channel
            np_aggregate_image, _ = czi.read_image(S=scene_number,T=t,C=aggregate_channel)
            np_aggregate_image = np.squeeze(np_aggregate_image)
            
            # Computting the 1st and the last percentiles (min and max values)
            if (t-offset) == 0:
                aggregate_v_min, aggregate_v_max = np.percentile(np_aggregate_image, q=(0.5, 99.5))

            # Rescaling the aggregate image
            rescalled_aggr_image = exposure.rescale_intensity(np_aggregate_image,
                                                            in_range=(aggregate_v_min, aggregate_v_max),
                                                            out_range=np.uint16)
            
            # Converting the aggregate image to 8bit representation
            uint8_aggr_image = img_as_ubyte(rescalled_aggr_image)
        
            # Rescaling the cell image
            rescalled_cell_image = exposure.rescale_intensity(np_cell_image,
                                                            out_range=np.uint16)

            # Converting the cell image to 8bit representation
            uint8_cell_image = img_as_ubyte(rescalled_cell_image)

            # Cell image histogram normalization
            uint8_cell_image = match_histograms(uint8_cell_image,cell_reference_image).astype(np.uint8)

            # Saving the cell and aggregate image
            Image.fromarray(uint8_cell_image).resize(resize, Image.BICUBIC).save(os.path.join(scene_cell_path, str(t-offset)+img_ext ))
            Image.fromarray(uint8_aggr_image).resize(resize, Image.BICUBIC).save(os.path.join(scene_aggregate_path, str(t-offset)+img_ext ))

    # Making GIFs for easy to visualize results
    image_list_2_gif(output_dir=os.path.join(output_path, 'scene_'+str(scene_number)),
                     imgs_paths=natsorted(glob(os.path.join(scene_cell_path, '*'+img_ext ))),
                     gif_file_name='normalized_cell')
    
    image_list_2_gif(output_dir=os.path.join(output_path, 'scene_'+str(scene_number)),
                    imgs_paths=natsorted(glob(os.path.join(scene_aggregate_path, '*'+img_ext ))),
                    gif_file_name='normalized_aggregate')
    

def image_list_2_gif(output_dir,
                     imgs_paths,
                     gif_file_name,
                     time_step=2,
                     time_unit='min',
                     resize=(512,512)):
    # Make the folder to store GIFs
    os.makedirs(os.path.join(output_dir, 'GIFs'), exist_ok=True)

    # Reading the font that will be used to write on top on the images
    title_font = ImageFont.truetype(os.path.join('.', 'utils', 'font.ttf'), 20)

    # Loading the first image and be sure its RGB
    first_image = Image.open(imgs_paths[0]).resize(resize).convert('RGB')
    
    # Drawing the first label on the 1st image
    draw = ImageDraw.Draw(first_image)
    draw.text((0,0), 'Time : '+
        str(0)+' mins', (237, 0, 0), font=title_font)
    
    # Preparing a list to store all the images to be used to generate GIFs
    images_tab = []
    for j in range(len(imgs_paths)):

        # Loading the current image
        temp_image = Image.open(imgs_paths[j]).resize((512,512)).convert('RGB')

        # getting the name of the file to compute the time
        base = os.path.basename(imgs_paths[j])
        temp_time =  int(os.path.splitext(base)[0])
        
        # Drawing time stamps
        tmp_draw = ImageDraw.Draw(temp_image)
        tmp_draw.text((0,0), 'Time : '+
                str(time_step*(temp_time))+' '+time_unit, (237, 0, 0),
                font=title_font)
        
        # Storing the images in a list to be used to create GIF
        images_tab.append(temp_image)
    
    # Saving the GIF image
    first_image.save(os.path.join(output_dir, 'GIFs', gif_file_name+'.gif'),
                     save_all=True,
                     append_images=images_tab,
                     optimize=True,
                     duration=100,
                     loop=0)


def centroid_to_hsl_rgb(centroid, height, width, saturation=0.5, lightness=0.95):
    normalized_x = centroid[0] / height
    normalized_y = centroid[1] / width
    hue = (normalized_x + normalized_y) / 2
    hsl = (hue, saturation, lightness)
    rgb = tuple(int(x * 255) for x in colorsys.hls_to_rgb(*hsl))
    return rgb
