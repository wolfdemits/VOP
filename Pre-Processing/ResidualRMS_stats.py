import pathlib
import zarr
import numpy as np
from Slicemanager import Slice_manager
import json


DATA_PATH = pathlib.Path('../Data')

RESIDUAL_PATH = DATA_PATH / 'ZARR_RESIDUALS'
OUTPATH = RESIDUAL_PATH / 'ResidualRMS_stats.json'

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


slicemanager = Slice_manager()
slicemanager.DIFFUSION = True

# open zarr store
root = zarr.open_group(str(RESIDUAL_PATH), mode='a')

# get dimensions
W, H = root[[k for k in root.array_keys()][0]].shape

print(f'{bcolors.OKBLUE}Total amount of residuals: {len(root.members())}{bcolors.ENDC}')

rms_arr = np.zeros(len(root.members()))
for i, residual in enumerate(root.keys()):
    res = root[residual][:]
    rms_arr[i] = np.sqrt((1/(W*H)) * np.sum(res**2))

print(f'{bcolors.OKGREEN}Max: {np.max(rms_arr)}{bcolors.ENDC}')
print(f'{bcolors.OKGREEN}Mean: {np.mean(rms_arr)}{bcolors.ENDC}')
print(f'{bcolors.OKGREEN}Min: {np.min(rms_arr)}{bcolors.ENDC}')

dict = {'Max_RMS': np.max(rms_arr), 'Mean_RMS': np.mean(rms_arr), 'Min_RMS': np.min(rms_arr)}

with open(OUTPATH, "w") as outfile:
    json_object = json.dumps(dict, indent=4)
    outfile.write(json_object)

    print(f'{bcolors.OKGREEN}Results written to JSON file. {bcolors.ENDC}')