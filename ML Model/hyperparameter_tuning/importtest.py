# Import necessary modules
import datetime
import pathlib
import numpy as np
import torch
import optuna
import torch.nn as nn
import torch.optim as optim

# Import created functions and classes
from DatasetClasses_MicroMRI import Get_DataLoaders
from Helper_Functions_MRI import HybridLoss
from UNet_Model_MRI import UNet  # Assuming your U-Net code is in "unet.py"

# utility class for colored print outputs
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(f'{bcolors.OKGREEN}Import test executed succesfully{bcolors.ENDC}')