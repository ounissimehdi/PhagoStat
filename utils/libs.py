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

from skimage.measure import regionprops, label
from PIL import Image, ImageDraw
import numpy as np
import colorsys
import cv2
import os

class Frame_registration:
    def __init__(self, ref_img, img, gauss_size = 3,eps_err= 1e-4,max_itr=5000):
        self.ref_img = ref_img
        self.img = img
        self.gauss_size = gauss_size
        self.eps_err = eps_err
        self.max_itr = max_itr
        self.warp_matrix = []

        # Define the motion model
        self.warp_mode =  cv2.MOTION_TRANSLATION #cv2.MOTION_AFFINE #cv2.MOTION_EUCLIDEAN #cv2.MOTION_TRANSLATION 

        
    def register_frame(self, old_warp_matrix):
        # Find size of image1
        sz = self.ref_img.shape

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = self.max_itr

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = self.eps_err

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        #mask is set to the full image

        inputMask = np.ones(self.img.shape, dtype=np.uint8)

        size_tab = [1081, 513, 257, 129, 65, self.gauss_size]
        for i in range(len(size_tab)): 
        # Run the ECC algorithm. The results are stored in warp_matrix.
            try:
                (cc, warp_matrix) = cv2.findTransformECC (self.ref_img,self.img, warp_matrix, self.warp_mode, criteria, inputMask, size_tab[i])
            except cv2.error:
                # There isn't enough information to register the image
                if i == 0: warp_matrix = -100000.0
                else: pass
        if np.mean(warp_matrix) == -100000.0: warp_matrix = old_warp_matrix

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            im2_aligned = cv2.warpPerspective (self.img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(self.img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        self.warp_matrix = warp_matrix
        
        return Image.fromarray(im2_aligned)

    def get_warp_matrix(self):
        return self.warp_matrix

    def warp_matrix_frame(self, img, warp_matrix):
        # Find size of the image
        sz = img.shape

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            img_aligned = cv2.warpPerspective(img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            img_aligned = cv2.warpAffine(img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        
        return Image.fromarray(img_aligned)


class Aggregates_analysis:
    def __init__(self):
        self.labels = []
        self.regions = []

    def prepare_labels(self, pil_image, th=0.9):
        # Threshing the Gray scale image to produce Binary image
        np_bn_image = np.asarray(pil_image.point(lambda p: p > int(th*255) and 255))

        # Labelling the binary image
        labels = label(np_bn_image)
        
        # Store the labels
        self.labels = labels
        
        # Compute/save regions using the labels
        self.regions = regionprops(self.labels)

    def get_aggregates_count(self, regions_list):
        # Computing the number of the corret regions
        return len(regions_list)

    def get_regions(self, small_debris= 10,big_debris= 10000):
        # Initiate the corrected regions list
        regions_list = []
        for region in self.regions :
            # Ignoring the extreme regions (too small, too big regions)
            if region.area < big_debris and region.area> small_debris: regions_list.append(region)
        return regions_list

    def get_total_area(self, regions_list, area_resolution):
        # Initiate the total area variable
        total_area = 0
        for region in regions_list:
            # Summing the regions area
            total_area+= region.area
        return total_area*area_resolution

    def get_non_moving_aggregates(self, ref_regions, frame_regions, eps=7):
        # Defining a Bigger/Smaller then implementation (Euclidean distance based)
        def euclidean_distance(ref_coordinates, frame_coordinates, th_eps):
            dist = np.linalg.norm(ref_coordinates-frame_coordinates)
            if dist > th_eps: return False
            else: return True
        # Preparing the new reference list
        new_ref = []
        for ref_region in ref_regions:
            # Loading the current reference region's centroid 
            ref_coordinates= np.array(ref_region.centroid)
            for i in range(len(frame_regions)):
                # Loading the current frame region's centroid 
                frame_coordinates = np.array(frame_regions[i].centroid)

                # Defining the difference if region's area compared to the reference
                area_diff = abs(ref_region.area - frame_regions[i].area)/ref_region.area 

                # Check the if the two regions similarity
                if  area_diff <= 0.5 and euclidean_distance(ref_coordinates, frame_coordinates, eps): 
                    new_ref.append(frame_regions[i])
                    frame_regions.pop(i)
                    break
        return new_ref

    def save_plot_aggregates(self, pil_image, regions_list, path_det, path_label, number, extention='.png'):
        # Converting the given image into RGB
        original_image = pil_image.convert('RGB')

        # Create a blank image array filled with zeros
        height, width, _ = np.shape(original_image)
        labeled_image = np.zeros((height, width), dtype=np.uint16)

        # Create a dictionary for label-color mappings
        label_colors = {}
        # Plotting all regions bounding boxes on top of the numpy array
        for region in regions_list:
            # Saving the aggregate detection labels
            label_value = region['label']
            coords = np.array(region['coords'])
            labeled_image[coords[:, 0], coords[:, 1]] = label_value
            centroid = region.centroid
            label_colors[label_value] = centroid_to_hsl_rgb(centroid, height, width)#centroid_to_rgb(centroid, height, width)
        
        # Color the aggregate labels
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        for label_value, color in label_colors.items(): rgb_image[labeled_image == label_value] = color

        # Saving the annotated image to the given path
        Image.fromarray(rgb_image).save(os.path.join(path_label, str(number)+extention))

        # Draw bounding boxes around the detections
        draw = ImageDraw.Draw(original_image)
        for region in regions_list:
            minr, minc, maxr, maxc = region.bbox
            draw.rectangle([minc, minr, maxc, maxr], outline=(255, 0, 0), width=2)

        # Save the overlay image with bounding boxes as a PNG file
        original_image.save(os.path.join(path_det, str(number)+extention))


def centroid_to_rgb(centroid, height, width):
    normalized_x = centroid[0] / height
    normalized_y = centroid[1] / width
    r = int(normalized_x * 255)
    g = int(normalized_y * 255)
    b = int(((normalized_x + normalized_y) / 2) * 255)
    return (r, g, b)

def centroid_to_hsl_rgb(centroid, height, width, saturation=0.7, lightness=0.5):
    normalized_x = centroid[0] / height
    normalized_y = centroid[1] / width
    hue = (normalized_x + normalized_y) / 2
    hsl = (hue, saturation, lightness)
    rgb = tuple(int(x * 255) for x in colorsys.hls_to_rgb(*hsl))
    return rgb
