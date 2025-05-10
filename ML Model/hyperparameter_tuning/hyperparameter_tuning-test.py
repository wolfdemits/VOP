import torch
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import math

from UNet_Model_MRI import UNet
from DatasetClasses_MicroMRI import Mean_Normalisation

# Path Definitions
DATA_PATH_LOCAL = pathlib.Path('../../Data/ZARR_PREPROCESSED')
RESULT_PATH_LOCAL = pathlib.Path('./')

CODE_PATH_UGENT = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/WalkThroughPET/2425_VOP/Project/Hyperparameter_Tuning')
DATA_PATH_UGENT = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/WalkThroughPET/2425_VOP/Project/Data/ZARR_PREPROCESSED')
RESULT_PATH_UGENT = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/WalkThroughPET/2425_VOP/Project/Hyperparameter_Tuning/Results')


LOCAL = True

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

logbook = torch.load(path2logbook / 'HYPERPARAMETERS_10-05-logbook.pt')
NAME_RUN = logbook['NameRun']

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

print(logbook)

DL_Model = UNet(
    dim = logbook['dimension'], 
    num_in_channels = logbook['num_in_channels'],
    features_main = logbook['features_main'], 
    features_skip = logbook['features_skip'],
    conv_kernel_size = logbook['conv_kernel_size'], 
    down_mode = logbook['down_mode'],
    up_mode = logbook['up_mode'],
    activation = logbook['activation'],
    residual_connection = logbook['activation'])

checkpoint = torch.load(path2ckpt / '{}.pt'.format(NAME_RUN), map_location=device)
DL_Model.load_state_dict(checkpoint['model_state_dict'])

from DatasetClasses_MicroMRI import Get_DataLoaders, Get_Test_DataLoader

# Enter list of trainings subjects
SubjTrain = ['Mouse01', 'Mouse02', 'Mouse14', 'Mouse09', 'Mouse23']

# Enter list of validation subjects
SubjVal = ['Mouse10', 'Mouse06', 'Mouse21', 'Mouse17', 'Mouse05']

# Enter planes
PlanesData = ['Coronal', 'Sagittal', 'Transax']

# Enter planes
RegionsData = ['HEAD-THORAX', 'THORAX-ABDOMEN']

batch_size = 32
num_in_channels = 1

TrainLoader, ValLoader = Get_DataLoaders(SubjTrain, SubjVal, path2zarr, PlanesData, RegionsData, batch_size, num_in_channels)

loss_criterion = torch.nn.MSELoss()
DL_Model.eval()

val_batch_number = 0
val_loss_per_epoch = 0
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


## DATALOADER FOR VALIDATION COMPLETED: 1 VALIDATION EPOCH DONE
val_loss = val_loss_per_epoch / val_batch_number

print(val_batch_number)
print(val_loss)

print('Option 2')

DL_Model.eval()
val_loss = 0.0
batch_number = 0
with torch.no_grad():
    for batch in ValLoader:
        batch_number += 1
        LR_Img, HR_Img = batch["LR_Img"].to(device), batch["HR_Img"].to(device)
        output = DL_Model(LR_Img)
        loss = loss_criterion(output, HR_Img)
        val_loss += loss.item()

avg_val_loss = val_loss / batch_number

print(avg_val_loss)