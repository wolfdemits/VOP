""" Training Script for Micro-MRI Super-Resolution """
# Import necessary modules
import datetime
import pathlib
import numpy as np
import torch
from torch.cuda.amp import GradScaler

# Import created functions and classes
from Helper_Functions_MRI import Tensorboard_Initialization, Logbook_Initialization
from Helper_Functions_MRI import Intermediate_Visualization
from APD import Get_DataLoaders
from UNet_Model_MRI import UNet, count_model_parameters


# Path Definitions
DATA_PATH_LOCAL = pathlib.Path('./Data/ZARR_PREPROCESSED')
RESULT_PATH_LOCAL = pathlib.Path('./ML Model/Without_SR_Module')

CODE_PATH_UGENT = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/WalkThroughPET/2425_VOP/Project/UNet')
DATA_PATH_UGENT = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/WalkThroughPET/2425_VOP/Project/Data/ZARR_PREPROCESSED')
RESULT_PATH_UGENT = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/WalkThroughPET/2425_VOP/Project/UNet/Results')


LOCAL = False

if LOCAL:    # If you want to run locally
    DATA_PATH = DATA_PATH_LOCAL
    RESULT_PATH = RESULT_PATH_LOCAL

else:  # If you want to run on the HPC 
    CODE_PATH = CODE_PATH_UGENT
    DATA_PATH = DATA_PATH_UGENT
    RESULT_PATH = RESULT_PATH_UGENT


path2tb = RESULT_PATH / 'Tensorboard'
path2ckpt = RESULT_PATH / 'Checkpoint'
path2logbook = RESULT_PATH / 'Logbook'
Path_FigSave = RESULT_PATH / 'IntermediateFigures'

path2zarr = DATA_PATH


#--------------------------------------------------------------------------------------------------------------------

#####################
## START OF SCRIPT ##
#####################

# Enter list of training subjects
SubjTrain = ['Mouse01', 'Mouse02', 'Mouse03', 'Mouse04', 'Mouse05', 'Mouse08', 'Mouse09', 'Mouse11', 'Mouse12','Mouse13', 'Mouse14', 'Mouse15', 'Mouse17', 'Mouse18', 'Mouse19', 'Mouse20', 'Mouse21', 'Mouse23']

# Enter list of validation subjects
SubjVal = ['Mouse10', 'Mouse06']

# Enter planes
PlanesData = ['Coronal', 'Sagittal', 'Transax']

# Enter planes
RegionsData = ['HEAD-THORAX', 'THORAX-ABDOMEN']

# Give your run a name
NAME_RUN = 'Optimized_' + str(datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")) 
start_time_run = datetime.datetime.now()
print(NAME_RUN, flush=True)


# Network Structure Parameters 

dim = '2d'
num_in_channels = 1                     # Can be adjusted if wanted
features_main = [128, 128, 64]     # Can be adjusted if wanted
features_skip = features_main[:-1]          # Can be adjusted if wanted
conv_kernel_size = 3                   
down_mode = 'convStrided'                 ## Options: maxpool, meanpool, convStrided
up_mode = 'upsample'                  ## Options: upsample, upconv
activation = 'ReLU'                   ## Options: ReLU, LeakyReLU, PReLU
residual_connection = False



# Training Hyperparameters

batch_size = 32
LEARN_RATE = 0.0003

EPOCHS = 100

RESUME_CKPT = False  # IF TRUE, we resume training from last checkpoint 


# Selection of Slices and Subjects to visualise during training for tracking progress

SLICES_TO_SHOW = range(0, 25, 300)
#SUBJECTS_TO_SHOW = [SubjTrain[0], SubjVal[0]]
SUBJECTS_TO_SHOW = [SubjTrain[0], SubjTrain[-1], SubjVal[0], SubjVal[-1]]


#--------------------------------------------------------------------------------------------------------------------
## DATASETS and DATALOADERS

print('Dataset Definition', flush=True)

TrainLoader, ValLoader = Get_DataLoaders(SubjTrain, SubjVal, path2zarr, PlanesData, RegionsData, batch_size, num_in_channels)


#--------------------------------------------------------------------------------------------------------------------
## MODEL and DEVICE SET-UP 

DL_Model = UNet(
        dim = dim, 
        num_in_channels = num_in_channels,
        features_main = features_main, 
        features_skip = features_skip,
        conv_kernel_size = conv_kernel_size, 
        down_mode = down_mode,
        up_mode = up_mode,
        activation = activation,
        residual_connection = residual_connection)

print('# parameters = ', count_model_parameters(DL_Model), flush=True)  # Print the number of trainable model parameters


# Load Model: Check if CUDA-compatible GPU is availble for use

if torch.cuda.is_available():
    print('Using CUDA')
    device = torch.device('cuda')
    MIXED_PRECISION = True      # uses both 16-bit and 32-bit floating point operations during training
                                # -> This can speed up training by using half-precision arithmetic (16-bit) where possible
                                #    and uses less memory. 
else:        
    print('Using cpu')
    device  = torch.device('cpu')
    MIXED_PRECISION = False

DL_Model = DL_Model.to(device)


# Set-up 
optimizer = torch.optim.Adam(DL_Model.parameters(), lr=LEARN_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.1)
#grad_scaler = torch.amp.GradScaler('cuda')
grad_scaler = GradScaler()


loss_criterion = torch.nn.MSELoss() #HybridLoss(alpha=0.3) #torch.nn.MSELoss()    # or HybridLoss(alpha=0.3), but this returns nan



#--------------------------------------------------------------------------------------------------------------------
## A BUNCH OF OTHER INITIALISATION 

# Initialize LogBook

LogBook_Params = [NAME_RUN, PlanesData, RegionsData, num_in_channels, batch_size, LEARN_RATE]
ModelUNet_Params = [dim, num_in_channels, features_main, features_skip, conv_kernel_size, down_mode, up_mode, activation, residual_connection] 
Logbook_Initialization(dim, path2logbook, LogBook_Params, ModelUNet_Params) 


# Initialize Tensorboard

writers = Tensorboard_Initialization(path2tb/NAME_RUN)


# Start training from a saved checkpoint, or from the beginning
if RESUME_CKPT:
    checkpoint = torch.load(path2ckpt / '{}.pt'.format(NAME_RUN))
    start_epoch = checkpoint['current_epoch'] + 1
    epoch_best = checkpoint['epoch_best']        
    loss_best = checkpoint['loss_best']
    state_best = checkpoint['state_best']
    DL_Model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

else: 
    start_epoch = 0
    epoch_best = 0
    loss_best = np.inf


#--------------------------------------------------------------------------------------------------------------------
### START TRAINING AND VALIDATION EPOCHS

no_update_since = 0

for current_epoch in range(start_epoch, EPOCHS):

    start_time_epoch = datetime.datetime.now() 
        
    # Set the model to training mode
    DL_Model.train()
    train_loss_per_epoch = 0
    train_batch_number = 0


    # TrainLoader = list of dictionaries, containing lists of items used for training

    for TrainBatch in TrainLoader:

        train_batch_number += 1
        if (train_batch_number % 1 == 0):
            print('Train Batch number : ', train_batch_number, flush=True)

        # Load data 
        LR_Img = TrainBatch['LR_Img'].to(device)
        HR_Img = TrainBatch['HR_Img'].to(device)

        
        # Perform training 

        optimizer.zero_grad()

        if MIXED_PRECISION:
            with torch.amp.autocast('cuda'):
                DL_Img = DL_Model(LR_Img)   
                loss = loss_criterion(DL_Img, HR_Img)  
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)                    
            grad_scaler.update()  
        
        else:
            DL_Img = DL_Model(LR_Img)
            loss = loss_criterion(DL_Img, HR_Img)
            loss.backward()
            optimizer.step()

        # Train loss: update metrics
        train_loss_per_batch = loss.item() 
        train_loss_per_epoch += train_loss_per_batch 


        Intermediate_Visualization(
            batch = TrainBatch, 
            LR_Img = LR_Img, 
            DL_Img = DL_Img, 
            HR_Img = HR_Img, 
            EpochNumber = current_epoch, 
            num_in_channels = num_in_channels, 
            ShowSlices = SLICES_TO_SHOW,  
            TensorboardWriter = writers['train'],
            ShowSubject = SUBJECTS_TO_SHOW, 
            Path_FigSave = Path_FigSave)


    ## DATALOADER FOR TRAINING COMPLETED: 1 TRAINING EPOCH DONE

    end_time_train_epoch = datetime.datetime.now()
    time_train_epoch = end_time_train_epoch - start_time_epoch
    train_loss = train_loss_per_epoch / train_batch_number
    writers['train'].add_scalar('loss/epoch', train_loss, current_epoch)
    
    print('TRAINING: Epoch [{}] \t Run Time = {} \t Loss = {}'.format(current_epoch, str(time_train_epoch), round(train_loss, 6)), flush=True)
    

    # Set the model to evaluation mode
    DL_Model.eval()
    val_loss_per_epoch = 0
    val_batch_number = 0


    # ValLoader = list of dictionaries, containing lists of items used for validation

    for ValBatch in ValLoader:

        val_batch_number += 1
        if (val_batch_number % 50 == 0):
            print('Val Batch number : ', val_batch_number, flush=True)

        # Load data 
        LR_Img = ValBatch['LR_Img'].to(device)
        HR_Img = ValBatch['HR_Img'].to(device)
                    

        # Perform validation
        with torch.no_grad(): 
            if MIXED_PRECISION:
                with torch.amp.autocast('cuda'):
                    DL_Img = DL_Model(LR_Img) 
                    loss = loss_criterion(DL_Img, HR_Img)
            else:
                DL_Img = DL_Model(LR_Img)
                loss = loss_criterion(DL_Img, HR_Img) 

        # Val loss: update metrics
        val_loss_per_batch = loss.item() 
        val_loss_per_epoch += val_loss_per_batch 


        Intermediate_Visualization(
            batch = ValBatch, 
            LR_Img = LR_Img, 
            DL_Img = DL_Img, 
            HR_Img = HR_Img, 
            EpochNumber = current_epoch, 
            num_in_channels = num_in_channels, 
            ShowSlices = SLICES_TO_SHOW,  
            TensorboardWriter = writers['val'],
            ShowSubject = SUBJECTS_TO_SHOW, 
            Path_FigSave = Path_FigSave)


    ## DATALOADER FOR VALIDATION COMPLETED: 1 VALIDATION EPOCH DONE
       
    end_time_val_epoch = datetime.datetime.now()
    time_val_epoch = end_time_val_epoch - end_time_train_epoch
    val_loss = val_loss_per_epoch / val_batch_number
    writers['val'].add_scalar('loss/epoch', val_loss, current_epoch)

    print('VALIDATION: Epoch [{}] \t Run Time = {} \t Loss = {}'.format(current_epoch, str(time_val_epoch), round(val_loss, 6)), flush=True)


    # ------------------------------------------------------------------------------------------------------------------------
    ## AT THE END OF AN EPOCH: UPDATE AND SAVE 

    total_run_time = datetime.datetime.now() - start_time_run
    print('##### EPOCH [{}] COMPLETED -- Run Time = {}'.format(current_epoch, str(total_run_time)), flush=True)
        
    if val_loss < loss_best:
        
        state_best = DL_Model.state_dict()
        epoch_best = current_epoch 
        runtime_atBestEpoch = total_run_time
        loss_best  = val_loss
        no_update_since = 0
                
        checkpoint = {
            'current_epoch': current_epoch,
            'epoch_best': epoch_best,
            'loss_best': loss_best,
            'state_best': state_best,
            'model_state_dict': DL_Model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()}
        
        path2ckpt.mkdir(exist_ok=True)
        torch.save(checkpoint, path2ckpt / '{}.pt'.format(NAME_RUN))
        
    else:
        no_update_since += 1

    print('Best Epoch [{}]'.format(epoch_best), flush=True)


    # ------------------------------------------------------------------------------------------------------------------------
    ## CONVERGENCE: EARLY STOPPING CRITERIA 

    if current_epoch > 10: #put back at 10

        if no_update_since > 6:   #PUT BACK AT 6

            print('------------------------------------------------', flush=True)  
            print('Early stopping criteria reached after EPOCH [{}]'.format(current_epoch), flush=True)
            print('Best Epoch [{}] /// Run Time @Best Epoch {}'.format(epoch_best, runtime_atBestEpoch), flush=True)
            
            break


    scheduler.step(val_loss)

# %%
