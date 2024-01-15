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

import sys
#sys.path.append('../../')

from numpy import append
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader 
from utils import CustomDetectionDataset, batch_BCE_weighted_loss, batch_dice_coefficient, get_labelled_mask_batch, calculate_iou_best_match_batch_torch #CustomMetrics_batch, CustomMetrics_TEST_batch #,confusion_matrix, dice_coeff_batch
from natsort import natsorted
from glob import glob
from unet import Attention_U_Net
import numpy as np
import argparse
import logging
import os, sys
import random
import torch
import time


def train_pytorch_model():
    # Preparing the tensorboard to store training logs
    writer = SummaryWriter(comment='_'+fold_str+'_'+exp_name)

    # Loging the information about the current training
    logging.info(f'''[INFO] Starting training:
        Experiment name                  : {exp_name}
        Epochs number                    : {n_epoch}
        Early stop val loss- wait epochs : {wait_epochs}
        Batch size                       : {batch_size}
        Learning rate                    : {learning_rate}
        Training dataset size            : {len(train_dataset)}
        Validation dataset size          : {len(val_dataset)}
        PyTorch random seed              : {random_seed}
        Model input channels             : {n_input_channels}
        Model output channels            : {n_output_channels}
        Path to logs and ckps            : {path_to_logs}
        Cross-validation                 : {cross_val}
    ''')

    # Use the corrsponding data type for the masks
    mask_data_type = torch.float32 if n_output_channels == 1 else torch.long

    # Init the best value of evaluation loss
    best_val_loss = 10**10

    # Patience counter
    early_stop_count = 0

    # Strating the training
    for epoch in range(n_warm + n_epoch):
        
        if epoch == (n_warm+1):
            # Init the best value of evaluation loss
            best_val_loss = 10**10

        tic = time.time()
        # Make sure the model is in training mode
        model.train()
        
        # Init the epoch loss
        epoch_loss = 0
        epoch_dice_loss, epoch_miou_loss = 0,0
    
        # Train using batches
        for batch in train_loader:
            # Load the image and mask
            image            =  batch['image']
            true_mask        =  batch['mask']
            true_label       =  batch['label']
            # gt_border_mask  =  batch['border_mask']
            # gt_bboxes       =  batch['b-boxes']

            # Make sure the data loader did prepare images properly
            assert image.shape[1] == n_input_channels, \
				f'The input image size {image.shape[1]} ' \
				f', yet the model have {n_input_channels} input channels'

            # Load the image and the mask into device memory
            image           =  image.to(device=device, dtype=torch.float32)
            true_mask       =  true_mask.to(device=device, dtype=mask_data_type)
            true_label      =  true_label.to(device=device, dtype=torch.float32)
            #gt_bboxes       =  gt_bboxes.to(device=device, dtype=torch.float32)

            # zero the parameter gradients to lower the memory footprint
            optimizer.zero_grad()

            # Make the prediction on the loaded image
            pred_mask = model(image)

            # Apply sigmoid in case on MSE loss
            pred_mask = torch.sigmoid(pred_mask)

            # Computing the batch loss
            bce_loss   = batch_BCE_weighted_loss(pred_mask, true_mask)
            dice_score = batch_dice_coefficient(pred_mask, true_mask)


            
            if epoch > n_warm:

                batch_labels = get_labelled_mask_batch(pred_mask, time_mask_threshold=0.9, mask_threshold=0.5)
                miou = calculate_iou_best_match_batch_torch(batch_labels, true_label)

                epoch_dice_loss       += dice_score
                epoch_miou_loss       += miou

                batch_loss =  1.0 * bce_loss + 1.0 * (1-miou)
        
            else:
                epoch_dice_loss       += dice_score
                epoch_miou_loss       += dice_score*0

                batch_loss =  1.0 * bce_loss

            # Backward pass to change the model params
            batch_loss.backward()

            # Informing the optimizer that this batch is over
            optimizer.step()

            # Adding the batch loss to quantify the epoch loss
            epoch_loss += batch_loss.item()

            # Uncomment this to clip the gradients (can help with stability)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
        
        if epoch > n_warm:
            # Linear decay
            decay_frac = (epoch - n_warm) / n_epoch
            new_lr = learning_rate * (1 - decay_frac)

            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            writer.add_scalar('Learning rate', new_lr, epoch)
        else:
            writer.add_scalar('Learning rate', learning_rate, epoch)

        # Evaluation of the model
        val_loss, total_dice_coeff, total_miou = evaluation_pytorch_model(epoch, model, val_loader, device)
        val_loss   = val_loss/len(val_loader)

        # Save the epoch validation loss & metrics in the tensorboard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/DICE', total_dice_coeff, epoch)
        writer.add_scalar('Metrics/MIOU', total_miou, epoch)
        

        # Getting the mean loss value
        epoch_loss = epoch_loss/len(train_loader)
        epoch_dice_loss      = epoch_dice_loss/len(train_loader)
        epoch_miou_loss      = epoch_miou_loss/len(train_loader)

        # Putting the model into training mode -to resume the training phase-
        model.train()
        
        # Save the epoch training loss in the tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        hours, rem = divmod(time.time()-tic, 3600)
        minutes, seconds = divmod(rem, 60)
        logging.info(f'''[INFO] Epoch {epoch} took {int(hours)} h {int(minutes)} min {int(seconds)}:
                Mean train loss          :  {epoch_loss}
                Mean val   loss          :  {val_loss}

        ''')
        logging.info(f'''
                -- Training of the model --
                Dice                :  {epoch_dice_loss.item()}
                mIoU                :  {epoch_miou_loss.item()}

        ''')

        logging.info(f'''
                -- Evaluation of the model --
                Dice                :  {total_dice_coeff.item()}
                mIoU                :  {total_miou.item()}

        ''')
        
        # Saving all model's checkpoints
        if save_all_models:
            # Since DataParallel is used, adapting the parameters saving
            if n_devices > 1:
                if epoch > n_warm:torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, f'ckp_{epoch}_{round(total_dice_coeff.item(),4)}.pth'))
                else:torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, f'warm_ckp_{epoch}_{round(total_dice_coeff.item(),4)}.pth'))
            # Saving the parameters in case of one device
            else: 
                if epoch > n_warm:torch.save(model.state_dict(), os.path.join(path_to_ckpts, f'ckp_{epoch}_{round(total_dice_coeff.item(),4)}.pth'))
                else:torch.save(model.state_dict(), os.path.join(path_to_ckpts, f'warm_ckp_{epoch}_{round(total_dice_coeff.item(),4)}.pth'))

        # Saving the best model
        if best_val_loss > val_loss:
            # Since DataParallel is used, adapting the parameters saving
            if n_devices > 1: 
                if epoch > n_warm:torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, 'best_model.pth'))
                else:torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, 'warm_best_model.pth'))

            # Saving the parameters in case of one device
            else:
                if epoch > n_warm: torch.save(model.state_dict(), os.path.join(path_to_ckpts, 'best_model.pth'))
                else: torch.save(model.state_dict(), os.path.join(path_to_ckpts, 'warm_best_model.pth'))

            logging.info(f'''
                Best epoch {epoch} :
            
            ''')
        
            # Update the best validation loss
            best_val_loss = val_loss

            # Reset patience counter
            early_stop_count  = 0
        elif early_stop_count < wait_epochs: early_stop_count += 1
        
        else :
            logging.info(f'''[INFO] Early stop at epoch {epoch} ...''')
            break

    # Close the tensorboard writer
    writer.close()


def evaluation_pytorch_model(epoch, model, data_loader, device):
    """evaluation_pytorch_model: Evaluation of a PyTorch model and returns eval loss,
     dice coeff and the elements of a confusion matrix"""
    # Putting the model in evaluation mode (no gradients are needed)
    model.eval()

    # Use the corresponding data type of the mask
    mask_data_type = torch.float32 if n_output_channels == 1 else torch.long

    # The batch number 
    n_batch = len(data_loader)

    # Init cars needed in evaluation
    total_dice_coeff, total_loss = 0, 0
    total_miou  = 0

    for batch in data_loader:
        # Load the image and mask
        image           =  batch['image']
        true_mask       =  batch['mask']
        true_label       =  batch['label']
        # gt_border_mask  =  batch['border_mask']
        # gt_bboxes       =  batch['b-boxes']

        # Make sure the data loader did prepare images properly
        assert image.shape[1] == n_input_channels, \
            f'The input image size {image.shape[1]} ' \
            f', yet the model have {n_input_channels} input channels'

        # Load the image and the mask into device memory
        image           =  image.to(device=device, dtype=torch.float32)
        true_mask       =  true_mask.to(device=device, dtype=mask_data_type)
        true_label      =  true_label.to(device=device, dtype=mask_data_type)
        # gt_border_mask  =  gt_border_mask.to(device=device, dtype=torch.float32)
        # gt_bboxes       =  gt_bboxes.to(device=device, dtype=torch.float32)

        # Make sure the data loader did prepare images properly
        assert image.shape[1] == n_input_channels, \
            f'The input image size {image.shape[1]} ' \
            f', yet the model have {n_input_channels} input channels'
        
        # Load the image and the mask into device memory
        image           =  image.to(device=device, dtype=torch.float32)
        true_mask       =  true_mask.to(device=device, dtype=mask_data_type)
        # gt_border_mask  =  gt_border_mask.to(device=device, dtype=torch.float32)
        # gt_bboxes       =  gt_bboxes.to(device=device, dtype=torch.float32)

        # No need to use the gradients (no backward passes -evaluation only-)
        with torch.no_grad():

            # Computing the prediction on the input image
            pred_mask = model(image)

            # Apply sigmoid in case on MSE loss
            pred_mask = torch.sigmoid(pred_mask)
            
            # Computing the batch loss
            bce_loss   = batch_BCE_weighted_loss(pred_mask, true_mask)
            dice_score = batch_dice_coefficient(pred_mask, true_mask)
            
            if epoch > n_warm:

                batch_labels = get_labelled_mask_batch(pred_mask, time_mask_threshold=0.9, mask_threshold=0.5)
                miou = calculate_iou_best_match_batch_torch(batch_labels, true_label)

                tmp_loss =  1.0 * bce_loss + 1.0 * (1-miou)
                
                # Saving metrics in order to compute the mean values at the end of the evaluation
                total_miou += miou
            
            else:tmp_loss =  1.0 * bce_loss

            total_loss += tmp_loss.item()

            # Computing the Dice coefficient
            total_dice_coeff += dice_score
            
            
    
    # Computting the mean values -metrics- over all the evaluation dataset
    if epoch > n_warm: total_miou = total_miou / n_batch
    else: total_miou = 0 * (total_dice_coeff / n_batch)
    total_dice_coeff = total_dice_coeff / n_batch

    return total_loss, total_dice_coeff, total_miou


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Process an integer parameter.')

    # Add the integer parameter named 'number'
    parser.add_argument('--conv_num', type=int, help='An integer parameter for model configuration', default=24)
    parser.add_argument('--n_epoch',  type=int, help='An integer parameter for number of epochs with decay', default=200)
    parser.add_argument('--n_warm',   type=int, help='An integer parameter for number of epochs of fixed lr', default=10)
    parser.add_argument('--batch_size',   type=int, help='Batch size', default=12)
    parser.add_argument('--lr',       type=float, help='Learning rate for the model training', default=0.0001)

    # Parse the arguments
    args = parser.parse_args()

    ################# Hyper parameters ####################################
    # The number of epochs for the training
    n_epoch = args.n_epoch
    n_warm  = args.n_warm

    conv_num = args.conv_num

    # The batch size !(limited by how many the GPU memory can take at once)!
    batch_size = args.batch_size # batch size for one GPU

    # Leaning rate that the optimizer uses to change the model parameters
    learning_rate = args.lr

    # Early stop if the val loss didn't improve after N wait_epochs
    wait_epochs = n_epoch

    # Save the model's parameters at the end of each epoch (if not only the
    # best model will be saved according to the validation loss)
    save_all_models = False

    # Setting a random seed for reproducibility
    random_seed = 2024
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Evaluation threshold (binary masks generation and Dice coeff computing)
    eval_threshold = 0.5

    # when cross_val = 0 -> no cross validation is used only a 80% of 
    #                       the dataset train and 20% of it for 
    #                       validation
    #      cross_val = N -> N fold cross validation : the dataset will be
    #                                                 divided by N and one 
    #                                                 dataset fraction is 
    #                                                 used for validation
    #                                                 each time.
    #                                                 (N=5 -> 5 trainings)
    cross_val = 0

    # The folds switches (True if the training is done) 
    folds_done = [False, False, False, False, False]

    # The fold name
    fold_str =''

    # Make sure the cross_val is between [2, N]
    assert 0 <= cross_val <= 5, '[ERROR] Cross-Validation must be greater then 2 and less or equal 5'

    # Make sure the cross_val is between [2, N]
    #assert cross_val == len(folds_done), f'[ERROR] Cross-Validation switches must match but we have {cross_val} / and {len(folds_done)} switches'

    # Rescalling factor of the images and masks
    scale_factor = 1

    # Make sure the cross_val is between [2, N]
    assert 0 < scale_factor <= 1, '[ERROR] Scale must be between ]0, 1]'

    # Images to keep in the training phase
    keep_imgs = -1

    # The experiment name to keep track of all logs
    exp_name = 'AttUnet'
    if cross_val !=0:
        exp_name += str(cross_val)
        exp_name += '_cross-val_'
    #exp_name += '_'+str(keep_imgs)+'_ONLY_BORDER_binary_mask_Unet_'
    exp_name += 'F'+str(conv_num)+'-K3-EXP_BCE_'
    exp_name += 'EP_'+str(n_epoch)
    exp_name +='_ES_'+str(wait_epochs)
    exp_name +='_BS_'+str(batch_size)
    exp_name +='_LR_'+str(learning_rate)
    exp_name +='_RS_'+str(random_seed)
    #######################################################################
    

    # Path to the log file and the saved ckps if any
    path_to_logs = os.path.join('experiments', exp_name)

    # Creating the experiment folder to store all logs
    os.makedirs(path_to_logs, exist_ok = True) 

    # Create a logger
    logging.basicConfig(filename=os.path.join(path_to_logs, 'logfile.log'), filemode='w', 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    

    ################# Computation hyper parameters ########################
    # Number of the workers (CPUs) to be used by the dataloader (HDD -> RAM -> GPU)
    n_workers = 0

    # Make this true if you have a lot of RAM to store all the training dataset in RAM
    # (This will speed up the training at the coast of huge RAM consumption)
    pin_memory = False

    # Chose the GPU cuda devices to make the training go much faster vs CPU use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Possibility to use at least two GPUs (available)
    if torch.cuda.device_count() > 1:
        # Log with device the training will be using (at least one GPU in this case)
        logging.info(f'[INFO] Using {torch.cuda.device_count()} {device}')

        # Log the GPUs models
        for i in range(torch.cuda.device_count()):
            logging.info(f'[INFO]      {torch.cuda.get_device_name(i)}')
        
        # For faster convolutions, but more memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Number of devices (GPUs) in use
        n_devices = torch.cuda.device_count()
    
    # Using one GPU (available)
    elif torch.cuda.is_available():
        # Log with device the training will be using (one GPU in this case)
        logging.info(f'[INFO] Using {device}')

        # Log the GPU model
        logging.info(f'[INFO]      {torch.cuda.get_device_name(0)}')

        # For faster convolutions, but more memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Number of device (GPU) in use
        n_devices = 1
    
    # No GPU available, CPU is used in this case
    else:
        # Log with device the training will be using (CPU in this case)
        logging.info(f'[INFO] Using {device}')
        
        # Since CPU will be used no need to adapt the batch size
        n_devices = 1
    #######################################################################



    ################# U-NET parameters ####################################
    # The number of input images    (RGB        ->  n_input_channels=3)
    #                               (Gray       ->  n_input_channels=1)
    n_input_channels = 1

    # The number of output classes  (N classes  ->  n_output_channels = N)
    n_output_channels = 1
    #######################################################################
    

    root_path = os.path.join('..')
    ################# DATA parameters  ####################################
    # Paths to save the prepared dataset
    main_data_dir = os.path.join(root_path, 'dataset')

    # Path to the augmented training dataset
    train_dir = os.path.join(main_data_dir, 'train')

    # Path to the augmented validation dataset
    val_dir = os.path.join(main_data_dir, 'val')

    # Path to the test dataset
    test_dir = os.path.join(main_data_dir, 'test')
    
    # Switch to formalize all the data using the ref_image
    normalize_all = False
    
    # Path to the reference image (for normaliztation)
    ref_image_path = os.path.join( root_path, 'ADD_PATH_TO_REF.tif') # not needed
    #######################################################################


    # defining the U-Net model
    model = Attention_U_Net(n_channels=n_input_channels,n_classes=n_output_channels, conv_num=conv_num)

    # Putting the model inside the device
    model.to(device=device)

    # Load the best model
    # model.load_state_dict(torch.load('best_model_unet_conv_num_24.pth', map_location=device))

    # Use all the GPUs we have
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # Optimzer used for the training phase
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # All image/mask training paths
    train_img_paths_list = natsorted(glob(os.path.join(train_dir, 'images', '*.tif')))#[0:keep_imgs]
    train_mask_paths_list = natsorted(glob(os.path.join(train_dir, 'masks','*.tif')))#[0:keep_imgs]

    # All image/mask validation paths
    val_img_paths_list = natsorted(glob(os.path.join(val_dir, 'images', '*.tif')))
    val_mask_paths_list = natsorted(glob(os.path.join(val_dir, 'masks','*.tif')))


    # No cross validation is used
    if cross_val == 0:

        # Defining the path to the checkpoints
        path_to_ckpts = os.path.join(path_to_logs, 'ckpts')

        # Creating the experiment folder to store all logs
        os.makedirs(path_to_ckpts, exist_ok = True) 

        # Preparing the training dataloader
        train_dataset = CustomDetectionDataset(train_img_paths_list, train_mask_paths_list, ref_image_path, normalize=normalize_all, cached_data=pin_memory, n_channels=n_input_channels,scale=scale_factor, train=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size*n_devices, shuffle=True, pin_memory=pin_memory, num_workers=n_workers)

        # Preparing the validation dataloader
        val_dataset = CustomDetectionDataset(val_img_paths_list, val_mask_paths_list, ref_image_path, normalize=normalize_all,cached_data=pin_memory, n_channels=n_input_channels,scale=scale_factor)
        val_loader = DataLoader(val_dataset, batch_size=1*n_devices, shuffle=False, pin_memory=pin_memory, num_workers=n_workers)

        # Start the training
        try: train_pytorch_model()

        # When the training is interrupted (Ctl + C)
        # Make sure to save a backup version and clean exit
        except KeyboardInterrupt:
            # Save the current model parameters
            if n_devices > 1:
                torch.save(model.module.state_dict(), os.path.join(path_to_logs, 'backup_interruption.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(path_to_logs, 'backup_interruption.pth'))

            # Log the incedent
            logging.info('[ERROR] Training interrupted! parameters saved ... ')
            
            # Clean exit without any errors 
            try: sys.exit(0)
            except SystemExit: os._exit(0)

        # Emptying the loaders
        train_dataset.delete_cached_dataset()
        val_dataset.delete_cached_dataset()
        train_loader = []
        val_loader   = []

    
    # Cross-validation will be used
    else:
        # Fusing the training and validation imgs
        dataset_imgs_paths, dataset_masks_paths = [], []
        for i in range(len(train_img_paths_list)):
            dataset_imgs_paths.append(train_img_paths_list[i])
            dataset_masks_paths.append(train_mask_paths_list[i])

        # Fusing the training and validation masks
        for i in range(len(val_img_paths_list)):
            dataset_imgs_paths.append(val_img_paths_list[i])
            dataset_masks_paths.append(val_mask_paths_list[i])
        
        # Shuffle the training/validation dataset
        shuffle_indices = np.arange(np.array(dataset_imgs_paths).shape[0])
        np.random.shuffle(shuffle_indices)

        dataset_imgs_paths = np.array(dataset_imgs_paths)
        dataset_masks_paths = np.array(dataset_masks_paths)

        dataset_imgs_paths = dataset_imgs_paths[shuffle_indices]
        dataset_masks_paths = dataset_masks_paths[shuffle_indices]
        
        # Computing the fold length
        fold_len = int(len(dataset_imgs_paths)/cross_val)
        
        # Helping variables to prepare the cross validation
        lower_idx = 0
        upper_idx = fold_len

        logging.info(f'[INFO] Cross-validation in progress ...')
        # Cross validation dataset preparation
        for i in range(cross_val):
            if not(folds_done[i]):
                # Fold am for logs
                fold_str = 'FOLD-'+str(i+1)
                                
                # Defining the path to the checkpoints
                path_to_ckpts = os.path.join(path_to_logs, 'ckpts', fold_str)

                # Creating the experiment folder to store all logs
                os.makedirs(path_to_ckpts, exist_ok = True) 
                
                # Preparing the validarion lists
                fold_train_imgs_paths,  fold_train_masks_paths = [], []

                # Getting validation fold (images and masks)
                if i != cross_val-1:
                    fold_val_imgs_paths = dataset_imgs_paths[lower_idx:upper_idx]
                    fold_val_masks_paths = dataset_masks_paths[lower_idx:upper_idx]

                    # Updating the upper and lower idx
                    lower_idx = upper_idx
                    upper_idx += fold_len 
                
                # If cross_val * fold_len isn't even number (take what is left for the last fold)
                else: 
                    fold_val_imgs_paths = dataset_imgs_paths[upper_idx-fold_len:len(dataset_imgs_paths)]
                    fold_val_masks_paths = dataset_masks_paths[upper_idx-fold_len:len(dataset_masks_paths)]

                # Getting training fold (images and masks)
                fold_train_imgs_paths = natsorted(set(dataset_imgs_paths) - set(fold_val_imgs_paths))
                fold_train_masks_paths = natsorted(set(dataset_masks_paths) - set(fold_val_masks_paths))

                # Log the current fold with the corresponding details
                logging.info(f'[INFO] Fold {i+1} : with {len(fold_train_imgs_paths)} training, {len(fold_val_imgs_paths)} validation images')

                # Preparing the training dataloader
                train_dataset = CustomDetectionDataset(fold_train_imgs_paths, fold_train_masks_paths, ref_image_path, normalize=normalize_all,cached_data=pin_memory, n_channels=n_input_channels,scale=scale_factor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size*n_devices, shuffle=True, pin_memory=pin_memory, num_workers=n_workers)

                # Preparing the validation dataloader
                val_dataset = CustomDetectionDataset(fold_val_imgs_paths, fold_val_masks_paths, ref_image_path, normalize=normalize_all,cached_data=pin_memory, n_channels=n_input_channels,scale=scale_factor)
                val_loader = DataLoader(val_dataset, batch_size=1*n_devices, shuffle=False, pin_memory=pin_memory, num_workers=n_workers)

                # Start the training
                try: train_pytorch_model()

                # When the training is interrupted (Ctl + C)
                # Make sure to save a backup version and clean exit
                except KeyboardInterrupt:
                    # Save the current model parameters
                    if n_devices > 1:
                        torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, 'backup_interruption.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(path_to_ckpts, 'backup_interruption.pth'))

                    # Log the incident
                    logging.info('[ERROR] Training interrupted! parameters saved ... ')
                    
                    # Clean exit without any errors 
                    try: sys.exit(0)
                    except SystemExit: os._exit(0)
                
                # Emptying the loaders
                train_dataset.delete_cached_dataset()
                val_dataset.delete_cached_dataset()
                train_loader = []
                val_loader   = []

