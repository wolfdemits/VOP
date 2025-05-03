""" Hyperparameter Tuning Script """
# pytorch memory flags
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Import necessary modules
import datetime
import pathlib
import numpy as np
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
import gc

# Import created functions and classes
from DatasetClasses_MicroMRI import Get_DataLoaders
from Helper_Functions_MRI import HybridLoss
from UNet_Model_MRI import UNet  # Assuming your U-Net code is in "unet.py"

# Path Definitions
DATA_PATH_LOCAL = pathlib.Path('../../Data/ZARR_PREPROCESSED')
RESULT_PATH_LOCAL = pathlib.Path('./Results')

# Path Definitions
DATA_PATH_HPC = pathlib.Path('/kyukon/data/gent/gvo000/gvo00006/WalkThroughPET/2425_VOP/Project/Data/ZARR_PREPROCESSED')
RESULT_PATH_HPC = pathlib.Path('/kyukon/data/gent/gvo000/gvo00006/WalkThroughPET/2425_VOP/Project/Hyperparameter_Tuning/Results')

LOCAL = True

if LOCAL:    # If you want to run locally
    DATA_PATH = DATA_PATH_LOCAL
    RESULT_PATH = RESULT_PATH_LOCAL

else:  # If you want to run on the HPC
    DATA_PATH = DATA_PATH_HPC
    RESULT_PATH = RESULT_PATH_HPC


path2tb = RESULT_PATH / 'Tensorboard'
path2ckpt = RESULT_PATH / 'Checkpoint'
path2logbook = RESULT_PATH / 'Logbook'
Path_FigSave = RESULT_PATH / 'IntermediateFigures'

path2zarr = DATA_PATH


#--------------------------------------------------------------------------------------------------------------------

#####################
## START OF SCRIPT ##
#####################

# Enter list of trainings subjects
SubjTrain = ['Mouse01'] #, 'Mouse02', 'Mouse14', 'Mouse09', 'Mouse23']

# Enter list of validation subjects
SubjVal = ['Mouse02'] #, 'Mouse06', 'Mouse21', 'Mouse17', 'Mouse05']

# Enter planes
Planes = ['Coronal'] #, 'Sagittal', 'Transax']

# Enter planes
Regions = ['HEAD-THORAX'] #, 'THORAX-ABDOMEN']

# Give your run a name
# NAME_RUN = 'HYPERPARAMETERS_' + str(datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S"))
NAME_RUN = 'HYPERPARAMETERS_03-05'
start_time_run = datetime.datetime.now()
print(NAME_RUN, flush=True)

### Tuning Hyperparameters / constant hyperparameters ###
num_epochs = 1 #5
batch_size = 8
dim = '2d'
num_in_channels = 1
LR_DECAY = 0.9 #adapt?
conv_kernel_size = 3

# configure device
DISABLE_CUDA = False
if torch.cuda.is_available() and not DISABLE_CUDA:
    print('Using CUDA', flush=True)
    device = torch.device('cuda')
    MIXED_PRECISION = True      # uses both 16-bit and 32-bit floating point operations during training
                                # -> This can speed up training by using half-precision arithmetic (16-bit) where possible
                                #    and uses less memory. 
else:     
    print('Using cpu', flush=True)   
    device  = torch.device('cpu')
    MIXED_PRECISION = False

# Define the Objective Function for tuning
def objective(trial):
    """Objective function for Optuna optimization"""

    features_main = [
        trial.suggest_categorical("features_main_1", [32, 64, 128]),
        trial.suggest_categorical("features_main_2", [64, 128, 256]),
        trial.suggest_categorical("features_main_3", [128, 256, 512]),
    ]

    features_skip = features_main[:-1]  # Ensure skip connections align

    down_mode = trial.suggest_categorical("down_mode", ["maxpool", "meanpool", "convStrided"]) 
  
    up_mode = trial.suggest_categorical("up_mode", ["upsample", "upconv", "nearest", "bicubic"])

    activation = trial.suggest_categorical("activation", ["ReLU", "PReLU", "LeakyReLU"])

    residual_connection = trial.suggest_categorical("residual_connection", [True, False])

    LEARN_RATE = trial.suggest_float("lr", 1e-5, 1e-2)

    ALPHA = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    #criterion_class = trial.suggest_categorical("criterium", ["MSELoss","L1Loss", "HybridLoss"])
    #if criterion_class == "MSELoss":
    #    criterion = nn.MSELoss()
    #elif criterion_class == "L1Loss":
    #    criterion = nn.L1Loss()
    #elif criterion_class == "HybridLoss":
    #    criterion = HybridLoss(alpha=ALPHA)

    criterion = HybridLoss(alpha=ALPHA)

    PadValue = 0  # Keep fixed

    # Load dataset
    train_loader, val_loader = Get_DataLoaders(SubjTrain, SubjVal, DATA_PATH, Planes, Regions, batch_size, num_in_channels)

    # Initialize Model
    model = UNet(dim, num_in_channels, features_main, features_skip, conv_kernel_size,
                 down_mode, up_mode, activation, residual_connection, PadValue)
    
    optimizer_class = trial.suggest_categorical("optimizer", ["ADAM", "ADAMW", "SGD", "RMSprop"])

    if optimizer_class == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    elif optimizer_class == "ADAMW":
        optimizer = optim.AdamW(model.parameters(), lr=LEARN_RATE)
    elif optimizer_class == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LEARN_RATE)
    elif optimizer_class == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=LEARN_RATE)

    scheduler_class = trial.suggest_categorical("scheduler",["ExponentialLR", "ReduceLROnPlateau", "StepLR"])

    if scheduler_class == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, LR_DECAY)
    elif scheduler_class == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    elif scheduler_class == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY)
        

    model = model.to(device)   

    # Training Loop

    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            LR_Img, HR_Img = batch["LR_Img"].to(device), batch["HR_Img"].to(device)
            
            optimizer.zero_grad()
            output = model(LR_Img)
            loss = criterion(output, HR_Img)
            loss.backward()
            optimizer.step()

        if scheduler_class == "ReduceLROnPlateau":   
            # Validation Loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    LR_Img, HR_Img = batch["LR_Img"].to(device), batch["HR_Img"].to(device)
                    output = model(LR_Img)
                    loss = criterion(output, HR_Img)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            scheduler.step(avg_val_loss)

        else:
            scheduler.step()

    # Validation Loss
    model.eval()
    val_loss = 0.0
    objective = nn.MSELoss()
    with torch.no_grad():
        for batch in val_loader:
            LR_Img, HR_Img = batch["LR_Img"].to(device), batch["HR_Img"].to(device)
            output = model(LR_Img)
            loss = objective(output, HR_Img)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # Clean up to free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return avg_val_loss  # Optuna minimizes this


# Run Optuna Optimization
if __name__ == '__main__':
    study = optuna.create_study(storage='sqlite:///db.sqlite3', study_name=NAME_RUN, load_if_exists=True, direction="minimize")
    study.optimize(objective, n_trials=1000)

    print("Best hyperparameters:", study.best_params)
    print("Best Trial: " + study.best_trial + ", with loss: " + study.best_value)