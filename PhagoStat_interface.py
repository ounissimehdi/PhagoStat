# ------------------------------------------------------------------------------
#
#
#                                P̶̖̥͈͈͇̼͇̈́̈́̀͋͒̽͊͘͠h̴͙͈̦͗́̓a̴̧͗̾̀̅̇ḡ̶͓̭̝͓͎̰͓̦̭̎́͊̐̂͒͠ơ̵̘͌̿̽̑̈̾Ś̴̠́̓̋̂̃͒͆̚t̴̲͓̬͎̾͑͆͊́̕a̸͍̫͎̗̞͆̇̀̌̑͂t̸̳̼̘͕̯̠̱̠͂̔̒̐̒̕͝͝
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

from utils.help_functions import *
from unets import Attention_U_Net
import argparse
import os
import time






# HOW TO USE : python PhagoStat_interface.py -acq_path '/root/raw_data/WT-1.czi'
#                                            -output_path '/root/microglia_video_microscopy_dataset/WT/WT-1'
#                                            -acquisition_offset 0
#                                            -scene_identification 0
#                                            -to_frames
#                                            -reg
#                                            -det
#                                            -prob_maps
#                                            -cell_det -cell_trk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-acq_path"   ,"--acquisition_path",     help="The path to the video microscope acquisition (i.e. path/file.czi)",         type=str)
    parser.add_argument("-output_path","--save_output_path",     help="The path to save the results of the pipeline",                              type=str)
    parser.add_argument("-offset"     ,"--acquisition_offset",   help="The offset when the phagocytosis starts",                                   type=int)
    parser.add_argument("-scene"      ,"--scene_identification", help="Scene ID between [0,19]",                                                   type=int)
    parser.add_argument("-to_frames"  ,"--czi_to_frames",        help="Extract frames from CZI to frames",                                         action="store_true")
    parser.add_argument("-reg"        ,"--registration",         help="Register aggregate frames",                                                 action="store_true")
    parser.add_argument("-prob_maps"  ,"--probability_maps",     help="Run the probability maps generation algorithm",                             action="store_true")
    parser.add_argument("-det"        ,"--detection",            help="Run the detection algorithm on the registered frames",                      action="store_true")
    parser.add_argument("-cell_det"   ,"--cell_detection",       help="Run the Sliding window cell detection algorithm",                           action="store_true")
    parser.add_argument("-cell_trk"   ,"--cell_tracking",        help="Run the BTrack on detected cell algorithm",                                 action="store_true")
    args = parser.parse_args()

    # CZI acquisition file name
    czi_file_name = os.path.basename(args.acquisition_path)

    # Path to the CZI file
    czi_file_path = args.acquisition_path

    # Cell output file
    output_path = args.save_output_path

    # Getting the scene length
    scene_number = args.scene_identification

    # The data-offset (when phagocytosis starts)
    phago_offset = args.acquisition_offset

    # Defining the path to normalized aggregates
    aggr_field_path = os.path.join(output_path, 'scene_'+str(scene_number),'normalized_aggregate')

    # Defining the output path
    field_output_path = os.path.join(output_path, 'scene_'+str(scene_number))
    
    # The size of the original image
    original_img_hight = 2048

    # The image extension of the output images
    img_ext = '.png'

    # The size of the output images
    resize   = (1024, 1024)

    # The microscope resolution in micrometers square (2048^2 original images)
    microscope_resolution= 0.103**2 #0.103μm x 0.103μm

    # Getting the scale factor compared to the original images
    scale_factor = original_img_hight/ resize[0]

    if args.czi_to_frames:
        # Start the timer
        start_time = time.perf_counter()

        print(f'[INFO] {czi_file_name}:{scene_number}: Starting the conversion of raw data to open-format ...')
        # Reference image file path
        reference_cell_image_path = os.path.join('saved_models', 'cell_reference.tif')

        # Generating 8 bit tif images for cells and aggregates
        czi_to_frames(czi_file_path,
                      output_path,
                      scene_number,
                      reference_cell_image_path,
                      phago_offset,
                      resize=resize,
                      img_ext=img_ext)
        
        # Stop the timer and calculate elapsed time
        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")

    if args.registration:
        # Start the timer
        start_time = time.perf_counter()

        print(f'[INFO] {czi_file_name}:{scene_number}: Starting the registration of the aggregates ...')

        ########## Frame registration prams ##########
        # The gaussian filter kernel size
        gauss_size = 3

        # The error tolerance between the reference image and the registered image
        eps_err= 1e-4

        # Max iteration if the eps_err could be reached
        max_itr=1000

        # Generated registered aggregates frames
        aggregates_field_registration(aggr_field_path,
                                      field_output_path,
                                      microscope_resolution,
                                      scale_factor,
                                      offset=0,
                                      gauss_size=gauss_size,
                                      eps_err=eps_err,
                                      max_itr=max_itr,
                                      img_ext=img_ext)
    
        # Stop the timer and calculate elapsed time 
        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")

    if args.detection:
        # Start the timer
        start_time = time.perf_counter()

        print(f'[INFO] {czi_file_name}:{scene_number}: Starting the aggregates quantification ...')

        # Path to the registered aggregates data (for aggregates detection and tracking)
        registered_field_path = os.path.join(field_output_path, 'registered_aggregate')

        # The time per frame in mins (2 min per frame)
        frame_time = 2

        # Image resolution in micro meters square (area) microscope resolution
        area_resolution =  (np.sqrt(microscope_resolution)*scale_factor)**2

        # The aggregates manual threshold between (0 and 1)
        aggregates_th = 0.5

        # Debris pre-processing based on the detected area 
        ignore_debris_area = 2400  # Pixel^2 (area) of very very big object to ignore JUST IN CASE :)

        ################### Frame blur rejection mechanism parameters ################
        # The acceptable blur factor in percentage (eps in ]0, 1]) 
        # close to 0 : means so sensitive to blur change, 
        #          1 : means deactivate the rejection mechanism
        blur_eps = 0.01

        # The number of successive frames to check for disappearance of the fuzziness (blurry frames)
        search_frames = 14
    
        aggregates_detection_tracking(  aggr_field_path,
                                        registered_field_path,
                                        field_output_path,
                                        area_resolution,
                                        ignore_debris_area,
                                        blur_eps=blur_eps,
                                        search_frames=search_frames,
                                        aggregates_th=aggregates_th,
                                        frame_time=frame_time,
                                        save_detection_frames=True,
                                        img_ext=img_ext)
        
        # Stop the timer and calculate elapsed time
        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")

    if args.probability_maps:
        # Start the timer
        start_time = time.perf_counter()

        print(f'[INFO] {czi_file_name}:{scene_number}: Computing cell sematic masks ...')

        # The image extension of the output images
        img_ext = '.png'

        # Use GPU if available
        use_gpu = True

        # Chose to load all data inside RAM
        cached_data = True

        # Defining the cell input path
        cell_field_path = os.path.join(output_path, 'scene_'+str(scene_number),'normalized_cell')

        # Defining the cell output path
        cell_field_out_path = os.path.join(output_path, 'scene_'+str(scene_number))

        # Path to the DL pre-trained model
        dl_model_path = os.path.join('saved_models', 'att_best_model.pth')

        # Path to the reference image (normalization)
        ref_image_path = os.path.join('saved_models', 'cell_reference.tif')

        # Chose the GPU cuda devices to make the inference go much faster vs CPU use
        if use_gpu: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Using the CPU
        else: device = torch.device('cpu')

        # defining the U-Net model
        model = Attention_U_Net(n_channels=1, n_classes=1)

        # Putting the model inside the device
        model.to(device=device)

        # Load the best model
        model.load_state_dict(torch.load(dl_model_path, map_location=device))

        # Putting the model in evaluation mode (no gradients are needed)
        model.eval()

        # All images paths and masks
        cell_frame_paths = natsorted(glob(os.path.join(cell_field_path, '*'+img_ext)))
        # cell_frame_paths = cell_frame_paths[offset:len(cell_frame_paths)+1]

        # Preparing the validation dataloader
        cell_dataset = CustomDataset(cell_frame_paths, ref_image_path, normalize=False,cached_data=cached_data, scale=1)
        cell_loader = DataLoader(cell_dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

        # Remove if exist the previous result probability maps directory
        # try: shutil.rmtree(os.path.join(cell_field_out_path, 'probability_maps'))
        # except FileNotFoundError: pass

        # Path to store the probability maps
        probability_output  = os.path.join(cell_field_out_path, 'cell_semantic_maps')

        # Making sure the folder is created
        os.makedirs(probability_output, exist_ok = True)

        counter = 0

        # Train using batches
        for batch in cell_loader:
            # Load the image
            image           =  batch['image']

            # Load the image into device memory
            image           =  image.to(device=device, dtype=torch.float32)

            # No need to use the gradients (no backward passes -no training-)
            with torch.no_grad():
                # Run the model on the given image
                pred = model(image)

                # Make sure the prediction between [0, 1]
                proba_map = torch.sigmoid(pred)

                # Get the resulting probability map into the CPU RAM
                np_propa_map = np.array(proba_map.cpu()[0][0]*255, dtype=np.uint8)

                # Convert the numpy array into a PIL image
                pil_proba_map = Image.fromarray(np_propa_map)

                # Getting the original file name
                frame_name = path_to_filename(cell_frame_paths[counter])

                # Save the probability map using the cell frame name
                pil_proba_map.save(os.path.join(probability_output, frame_name+'.tif'))
            
            # Increment the dumpy counter
            counter += 1

        # Free memory
        if cached_data: cell_dataset.delete_cached_dataset()
        del model
        del pred
        del proba_map
        
        # Stop the timer and calculate elapsed time
        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")


    if args.cell_detection:
        # Start the timer
        start_time = time.perf_counter()

        print(f'[INFO] {czi_file_name}:{scene_number}: Computing cell instance masks ...')

        # Defining the cell input path
        cell_field_path = os.path.join(output_path, 'scene_'+str(scene_number),'normalized_cell')

        # Defining the cell output path
        cell_field_out_path = os.path.join(output_path, 'scene_'+str(scene_number))

        # Path to store the probability maps
        probability_maps_path  = os.path.join(cell_field_out_path, 'cell_semantic_maps')

        # All images paths and masks
        cell_frame_paths = natsorted(glob(os.path.join(cell_field_path, '*'+img_ext)))
        # cell_frame_paths = cell_frame_paths[offset:len(cell_frame_paths)+1]

        # mask threshold
        mask_threshold = 0.98

        # Temporal length window
        temporal_len = 5

        # Threshold to take out small objects (pixel^2)
        debrit_min_area = 1000

        microglia_XAI_segmentation_detection(   probability_maps_path,
                                                cell_frame_paths,
                                                cell_field_out_path,
                                                mask_threshold,
                                                temporal_len,
                                                debrit_min_area,
                                                original_img_hight=original_img_hight,
                                                microscope_resolution= microscope_resolution, #0.103μm x 0.103μm
                                                offset= 0, 
                                                plot_results=True)

        # Stop the timer and calculate elapsed time
        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")

    if args.cell_tracking:
        # Start the timer
        start_time = time.perf_counter()

        print(f'[INFO] {czi_file_name}:{scene_number}: Generate cell tracks ...')
        # Time per frame 
        frame_time = 2

        # The path to the cell features in px
        cell_features_df_path = os.path.join(output_path, 'scene_'+str(scene_number), 'CSVs','cells_centroid_features.csv')

        # Path to the cell parameters 
        cell_parameters_df_path = os.path.join(output_path, 'scene_'+str(scene_number), 'CSVs','cell_count_and_total_area.csv')

        # The path to the offset parameters
        offset_df_path = os.path.join(output_path, 'scene_'+str(scene_number), 'CSVs', 'frames_registration_params.csv')

        # The path to the Btrack cell configuration file
        btrack_config_json = os.path.join('saved_models', 'cell_config.json')
        
        microglia_tracking(cell_features_df_path,
                        cell_parameters_df_path,
                        offset_df_path,
                        btrack_config_json,
                        frame_time=frame_time, # time in min
                        time_unit='min',
                        microscope_resolution=microscope_resolution,
                        original_img_hight=original_img_hight,
                        image_hight=resize[0],
                        image_width=resize[0],
                        track_interactive_step=20,
                        max_search_radius=100,
                        tracks_length=10,
                        keep_until=100,
                        registration_correction=True)

        # Stop the timer and calculate elapsed time
        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")