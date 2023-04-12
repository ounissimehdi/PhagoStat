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
from utils import CustomLSTMDetectionDataset, CustomMetrics_batch, CustomMetrics_TEST_batch #,confusion_matrix, dice_coeff_batch
from natsort import natsorted
from glob import glob
from unet import LSTM_MMUnet
import numpy as np
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
    best_val_loss = 10000

    # Patience counter
    early_stop_count = 0

    # Strating the training
    for epoch in range(n_epoch):
        tic = time.time()
        # Make sure the model is in training mode
        model.train()
        
        # Init the epoch loss
        epoch_loss = 0
        epoch_dice_loss, epoch_f1_loss, epoch_recall_loss, epoch_precision_loss = 0,0,0,0

        # Train using batches
        for batch in train_loader:
            # Load the image and mask
            image           =  batch['image']
            true_mask       =  batch['mask']
            gt_border_mask  =  batch['border_mask']
            gt_bboxes       =  batch['b-boxes']

            #print("input image : ",image.shape)
            #print("input true_mask : ",true_mask.shape)
            #print("input gt_border_mask : ",gt_border_mask.shape)
            #print("input gt_bboxes : ",gt_bboxes.shape)
            # Make sure the data loader did prepare images properly
            #assert image.shape[2] == n_input_channels, \
            #    f'The input image size {image.shape[2]} ' \
            #    f', yet the model have {n_input_channels} input channels'

            # Load the image and the mask into device memory
            image           =  image.to(device=device, dtype=torch.float32)
            true_mask       =  true_mask.to(device=device, dtype=mask_data_type)
            gt_border_mask  =  gt_border_mask.to(device=device, dtype=torch.float32)
            gt_bboxes       =  gt_bboxes.to(device=device, dtype=torch.float32)

            # zero the parameter gradients to lower the memory footprint
            optimizer.zero_grad()

            # Make the prediction on the loaded image
            unet_out, lstm_out = model(image)
            tmp_bce_loss, tmp_border_loss = 0, 0
            tmp_dice_loss, tmp_f1_loss, tmp_recall_loss, tmp_precision_loss = 0, 0, 0, 0
            #for t in range(temporal_len):
            t = 1
            dice_loss, f1_loss, recall_loss, precision_loss, bce_loss, border_loss  = CustomMetrics_TEST_batch( torch.sigmoid(lstm_out[:, t, ...]),
                                                                                                                true_mask[:, t, ...],
                                                                                                                gt_border_mask[:, t, ...],
                                                                                                                gt_bboxes[:, t, ...],
                                                                                                                eval_threshold=0.8,
                                                                                                                IoU_threshold=0.5,
                                                                                                                eps=10e-4)
            tmp_bce_loss        += bce_loss       #/ (temporal_len * 1.0)
            tmp_border_loss     += border_loss    #/ (temporal_len * 1.0)
            tmp_dice_loss       += dice_loss      #/ (temporal_len * 1.0)
            tmp_f1_loss         += f1_loss        #/ (temporal_len * 1.0)
            tmp_recall_loss     += recall_loss    #/ (temporal_len * 1.0)
            tmp_precision_loss  += precision_loss #/ (temporal_len * 1.0)
                
            epoch_dice_loss      += tmp_dice_loss 
            epoch_f1_loss        += tmp_f1_loss
            epoch_recall_loss    += tmp_recall_loss
            epoch_precision_loss += tmp_precision_loss

            #if epoch < 1:
            #    batch_loss = bce_loss
            #else : 
            batch_loss = 0.4 * tmp_bce_loss + 0.6 *  tmp_border_loss + 0.0 * (1-tmp_f1_loss)
            
            # batch_loss.requires_grad_(True)
            # Backward pass to change the model params
            batch_loss.backward()

            # Informing the optimizer that this batch is over
            optimizer.step()

            # Adding the batch loss to quantify the epoch loss
            epoch_loss += batch_loss.item()

            # Uncomment this to clip the graients (can help with stabilty)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
        
        # Evaluation of the model
        val_loss, total_dice_coeff, total_pres, total_rec, total_f = evaluation_pytorch_model(model, val_loader, device)

        # Getting the mean loss value
        epoch_loss = epoch_loss/len(train_loader)
        val_loss   = val_loss/len(val_loader)

        epoch_dice_loss      = epoch_dice_loss/len(train_loader)
        epoch_f1_loss        = epoch_f1_loss/len(train_loader)
        epoch_recall_loss    = epoch_recall_loss/len(train_loader)
        epoch_precision_loss = epoch_precision_loss/len(train_loader)

        # Putting the model into training mode -to resume the training phase-
        model.train()
        
        # Save the epoch training loss in the tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        # Save the epoch validation loss & metrics in the tensorboard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/DICE', total_dice_coeff, epoch)
        writer.add_scalar('Metrics/precision', total_pres, epoch)
        writer.add_scalar('Metrics/recall', total_rec, epoch)
        writer.add_scalar('Metrics/F1-score', total_f, epoch)

        hours, rem = divmod(time.time()-tic, 3600)
        minutes, seconds = divmod(rem, 60)
        logging.info(f'''[INFO] Epoch {epoch} took {int(hours)} h {int(minutes)} min {int(seconds)}:
                Mean train loss          :  {epoch_loss}
                Mean val   loss          :  {val_loss}

        ''')
        logging.info(f'''
                -- Training of the model --
                Dice                :  {epoch_dice_loss.item()}
                Precision           :  {epoch_precision_loss.item()}
                Recall              :  {epoch_recall_loss.item()}
                F1-score            :  {epoch_f1_loss.item()}

        ''')

        logging.info(f'''
                -- Evaluation of the model --
                Dice                :  {total_dice_coeff.item()}
                Precision           :  {total_pres.item()}
                Recall              :  {total_rec.item()}
                F1-score            :  {total_f.item()}

        ''')

        # Saving all model's checkpoints
        if save_all_models:
            # Since DataParallel is used, adapting the paramaeters saving
            if n_devices > 1:
                torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, f'ckp_{epoch}_{total_dice_coeff}.pth'))
            
            # Saving the paramaeters in case of one device
            else: torch.save(model.state_dict(), os.path.join(path_to_ckpts, f'ckp_{epoch}_{total_dice_coeff}.pth'))

        # Saving the best model
        if best_val_loss > val_loss:
            # Since DataParallel is used, adapting the paramaeters saving
            if n_devices > 1: torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, 'best_model.pth'))
            
            # Saving the paramaeters in case of one device
            else: torch.save(model.state_dict(), os.path.join(path_to_ckpts, 'best_model.pth'))

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


def evaluation_pytorch_model(model, data_loader, device):
    """evaluation_pytorch_model: Evaluation of a PyTorch model and returns eval loss,
     dice coeff and the elements of a confusion matrix"""
    # Putting the model in evluation mode (no gradients are needed)
    model.eval()

    # Use the corrsponding data type of the mask
    mask_data_type = torch.float32 if n_output_channels == 1 else torch.long

    # The batch number 
    n_batch = len(data_loader)

    # Init cars needed in evaluation
    total_dice_coeff, total_loss = 0, 0
    total_pres, total_rec, total_f = 0, 0, 0

    for batch in data_loader:
        # Load the image and mask
        image           =  batch['image']
        true_mask       =  batch['mask']
        gt_border_mask  =  batch['border_mask']
        gt_bboxes       =  batch['b-boxes']

        # Make sure the data loader did prepare images properly
        #assert image.shape[1] == n_input_channels, \
        #    f'The input image size {image.shape[1]} ' \
        #    f', yet the model have {n_input_channels} input channels'

        # Load the image and the mask into device memory
        image           =  image.to(device=device, dtype=torch.float32)
        true_mask       =  true_mask.to(device=device, dtype=mask_data_type)
        gt_border_mask  =  gt_border_mask.to(device=device, dtype=torch.float32)
        gt_bboxes       =  gt_bboxes.to(device=device, dtype=torch.float32)

        # No need to use the gradients (no backward passes -evaluation only-)
        with torch.no_grad():
         
            # Make the prediction on the loaded image
            unet_out, lstm_out = model(image)

            tmp_bce_loss, tmp_border_loss = 0, 0
            tmp_dice_loss, tmp_f1_loss, tmp_recall_loss, tmp_precision_loss = 0, 0, 0, 0
            #for t in range(temporal_len):
            
            t = 1
            dice_loss, f1_loss, recall_loss, precision_loss, bce_loss, border_loss  = CustomMetrics_TEST_batch( torch.sigmoid(lstm_out[:, t, ...]),
                                                                                                                true_mask[:, t, ...],
                                                                                                                gt_border_mask[:, t, ...],
                                                                                                                gt_bboxes[:, t, ...],
                                                                                                                eval_threshold=0.8,
                                                                                                                IoU_threshold=0.5,
                                                                                                                eps=10e-4)
            tmp_bce_loss        += bce_loss       #/ (temporal_len * 1.0)
            tmp_border_loss     += border_loss    #/ (temporal_len * 1.0)
            tmp_dice_loss       += dice_loss      #/ (temporal_len * 1.0)
            tmp_f1_loss         += f1_loss        #/ (temporal_len * 1.0)
            tmp_recall_loss     += recall_loss    #/ (temporal_len * 1.0)
            tmp_precision_loss  += precision_loss #/ (temporal_len * 1.0)
            

            #if epoch < 1:
            #    batch_loss = bce_loss
            #else : 
            tmp_loss = 0.4 * tmp_bce_loss + 0.6 *  tmp_border_loss + 0.0 * (1-tmp_f1_loss)
            
            total_loss += tmp_loss.item()

            # Computing the Dice coefficent
            total_dice_coeff += tmp_dice_loss
            
            # Saving metrics in order to compute the mean values at the end of the evaluation
            total_pres += tmp_precision_loss
            total_rec  += tmp_recall_loss
            total_f    += tmp_f1_loss
    
    # Computting the mean values -metrics- over all the evaluation dataset
    total_pres = total_pres / n_batch
    total_rec  = total_rec  / n_batch
    total_f    = total_f    / n_batch
    total_dice_coeff = total_dice_coeff / n_batch

    return total_loss, total_dice_coeff, total_pres, total_rec, total_f


if __name__ == '__main__':

    ################# Hyper parameters ####################################
    # The number of epochs for the training
    n_epoch = 20

    # The temporal length for the LSTM network
    temporal_len = 2

    # The batch size !(limited by how many the GPU memory can take at once)!
    batch_size = 1 # batch size for one GPU

    # Leaning rate that the optimizer uses to change the model parameters
    learning_rate = 0.0001

    # Early stop if the val loss didn't improve after N wait_epochs
    wait_epochs = 200

    # Save the model's parameters at the end of each epoch (if not only the
    # best model will be saved according to the validation loss)
    save_all_models = True

    # Setting a random seed for reproducibility
    random_seed = 2022
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Evaluation threshold (binary masks generation and Dice coeff computing)
    eval_threshold = 0.9

    # when cross_val = 0 -> no cross validation is used only a 80% of 
    #                       the dataset train and 20% of it for 
    #                       validation
    #      cross_val = N -> N fold cross validation : the dataset will be
    #                                                 devided by N and one 
    #                                                 dataset fraction is 
    #                                                 used for validation
    #                                                 each time.
    #                                                 (N=5 -> 5 trainings)
    cross_val = 0

    # The folds switches (True if the training is done) 
    folds_done = [False, False, False, False, False]

    # The fold name
    fold_str =''

    # Make sure the cross_val is bewteen [2, N]
    assert 0 <= cross_val <= 5, '[ERROR] Cross-Validation must be greater then 2 and less or equal 5'

    # Make sure the cross_val is bewteen [2, N]
    #assert cross_val == len(folds_done), f'[ERROR] Cross-Validation switches must match but we have {cross_val} / and {len(folds_done)} switches'

    # Recaling the images and masks
    scale_factor = 1

    # Make sure the cross_val is bewteen [2, N]
    assert 0 < scale_factor <= 1, '[ERROR] Scale must be between ]0, 1]'

    # Images to keep in the training phase
    keep_imgs = -1
    
    fold_str = 'FOLD-5'

    # The experiment name to keep track of all logs
    exp_name = ''
    if cross_val !=0:
        exp_name += 'microglia_norm_'+str(cross_val)
        exp_name += '_cross-val_'
    #exp_name += '_'+str(keep_imgs)+'_ONLY_BORDER_binary_mask_unet_'
    exp_name += fold_str
    exp_name += '_EP_'+str(n_epoch)
    exp_name +='_ES_'+str(wait_epochs)
    exp_name +='_BS_'+str(batch_size)
    exp_name +='_LR_'+str(learning_rate)
    exp_name +='_RS_'+str(random_seed)
    #######################################################################
    

    # Path to the log file and the saved ckps if any
    path_to_logs = os.path.join('experiments', exp_name)

    # Creating the experiment folder to store all logs
    os.makedirs(path_to_logs, exist_ok = True) 

    # Ceate a loger
    logging.basicConfig(filename=os.path.join(path_to_logs, 'logfile.log'), filemode='w', 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    

    ################# Computation hyper parameters ########################
    # Number of the workers (CPUs) to be used by the dataloader (HDD -> RAM -> GPU)
    n_workers = 4

    # Make this true if you have alot of RAM to store all the training dataset in RAM
    # (This will speed up the training at the coast of huge RAM consumption)
    pin_memory = True

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
    
    # No GPU avaialble, CPU is used in this case
    else:
        # Log with device the training will be using (CPU in this case)
        logging.info(f'[INFO] Using {device}')
        
        # Since CPU will be used no need to adapt the batch size
        n_devices = 1
    #######################################################################



    ################# U-NET parameters ####################################
    # The number of input images    (RGB       ->  n_input_channels=3)
    #                               (Gray      ->  n_input_channels=1)
    n_input_channels = 1

    # The number of output classes  (N classs  ->  n_output_channels = N)
    n_output_channels = 1
    #######################################################################
    

    root_path = '/root/'
    ################# DATA parameters  ####################################
    # Paths to save the prepared dataset
    main_data_dir = os.path.join(root_path, 'dataset', 'lstm_dataset')

    # Path to the augmented training dataset
    train_dir = os.path.join(main_data_dir, 'train_seqs')

    # Path to the augmented validation dataset
    val_dir = os.path.join(main_data_dir, 'validation_seqs')

    # Path to the test dataset
    test_dir = os.path.join(main_data_dir, 'test_seqs')
    
    # Switch to formalize all the data using the ref_image
    normalize_all = False
    
    # Path to the refrence image (for normaliztation)
    ref_image_path = os.path.join(root_path, 'dataset', 'microglial_cells', 'labelme_annotated_data', 'imgs','5.tif')
    #######################################################################
    unet_model_path  = '../bio_net_train/experiments/microglia_norm_5_cross-val_F64-K3-EXP_ONLY_BORDER_BCE_EP_20_ES_20_BS_1_LR_0.0001_RS_2022/ckpts/'+fold_str+'/best_model.pth'

    # defining the LSTM-U-Net model
    model = LSTM_MMUnet(unet_model_path, input_nc=n_input_channels, output_nc=n_output_channels, ngf=64, temporal=temporal_len, UNet_train=False, device_name='cuda')

    # Putting the model inside the device
    model.to(device=device)

    # Load the best model
    # model.load_state_dict(torch.load(os.path.join('experiments', '_-1_border_binary_mask_unet_EP_200_ES_20_BS_1_LR_0.0001_RS_2022', 'ckpts', 'best_model.pth'), map_location=device))

    # Use all the GPUs we have
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # Optimzer used for the training phase
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # No cross validation is used
    if cross_val == 0:

        # Defining the path to the checkpoints
        path_to_ckpts = os.path.join(path_to_logs, 'ckpts')

        # Creating the experiment folder to store all logs
        os.makedirs(path_to_ckpts, exist_ok = True) 

        # Preparing the training dataloader  data_dir, ref_image_path, temporal_len=3, normalize=False,cached_data=True, n_channels=1,scale=1
        train_dataset = CustomLSTMDetectionDataset(train_dir, ref_image_path, temporal_len=temporal_len, normalize=False,cached_data=True, n_channels=1,scale=1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size*n_devices, shuffle=True, pin_memory=pin_memory, num_workers=n_workers)

        # Preparing the validation dataloader
        val_dataset = CustomLSTMDetectionDataset(val_dir, ref_image_path, temporal_len=temporal_len, normalize=False,cached_data=True, n_channels=1,scale=1)
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

        # Log the incedent
        logging.info('[INFO] Testing the best model parameters ... ')

        # Preparing the validation dataloader
        test_dataset = CustomLSTMDetectionDataset(test_dir, ref_image_path, temporal_len=temporal_len, normalize=False,cached_data=True, n_channels=1,scale=1)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=n_workers)

        # defining the LSTM-U-Net model
        model = LSTM_MMUnet(unet_model_path, input_nc=n_input_channels, output_nc=n_output_channels, ngf=64, temporal=3, UNet_train=False, device_name='cuda')


        # Chose the GPU cuda devices to make the training go much faster vs CPU use
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Putting the model inside the device
        model.to(device=device)

        model.load_state_dict(torch.load(os.path.join(path_to_ckpts, 'best_model.pth'), map_location=device))

        # Evaluation of the model
        test_loss, total_dice_coeff, total_pres, total_rec, total_f = evaluation_pytorch_model(model, test_loader, device)

        # Getting the mean loss over all evaluation images
        test_loss = test_loss / len(test_loader)
        logging.info(f'''
                -- Testing the best model --
                Test loss           :  {test_loss}
                Dice                :  {total_dice_coeff.item()}

                Precision           :  {total_pres.item()}
                Recall              :  {total_rec.item()}
                F1-score            :  {total_f.item()}

        ''')
    
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
        # Cross validation dataset preparetion
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

                # Log the incedent
                logging.info(f'[INFO] Fold {i+1} : Testing the best model parameters ... ')

                # Preparing the validation dataloader
                test_dataset = CustomDetectionDataset(test_img_paths_list, test_mask_paths_list, ref_image_path, normalize=normalize_all,cached_data=pin_memory, n_channels=n_input_channels, scale=scale_factor)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=n_workers)

                # defining the LSTM-U-Net model
                model = LSTM_MMUnet(unet_model_path, input_nc=n_input_channels, output_nc=n_output_channels, ngf=64, temporal=3, UNet_train=False, device_name='cuda')


                # Chose the GPU cuda devices to make the training go much faster vs CPU use
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Putting the model inside the device
                model.to(device=device)

                model.load_state_dict(torch.load(os.path.join(path_to_ckpts, 'best_model.pth'), map_location=device))

                # Evaluation of the model
                test_loss, total_dice_coeff, total_pres, total_rec, total_f = evaluation_pytorch_model(model, test_loader, device)

                # Getting the mean loss over all evaluation images
                test_loss = test_loss / len(test_loader)
                logging.info(f'''
                        -- Testing the best model --
                        Test loss           :  {test_loss}
                        Dice                :  {total_dice_coeff.item()}

                        Precision           :  {total_pres.item()}
                        Recall              :  {total_rec.item()}
                        F1-score            :  {total_f.item()}

                ''')

                # Emptying the loader
                test_dataset.delete_cached_dataset()
                test_loader   = []

                # defining the U-Net model
                model = 0
                # defining the LSTM-U-Net model
                model = LSTM_MMUnet(unet_model_path, input_nc=n_input_channels, output_nc=n_output_channels, ngf=64, temporal=3, UNet_train=False, device_name='cuda')


                # Putting the model inside the device
                model.to(device=device)

                # Use all the GPUs we have
                if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
                
                # Optimzer used for the training phase
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


