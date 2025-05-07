import pathlib
import zarr
import numpy as np
from Slicemanager import Slice_manager
import scipy

# Path Definitions
DATA_PATH_LOCAL = pathlib.Path('../Data')

# Path Definitions
DATA_PATH_HPC = pathlib.Path('/kyukon/data/gent/gvo000/gvo00006/WalkThroughPET/2425_VOP/Project/Data')

LOCAL = True

if LOCAL:    # If you want to run locally
    DATA_PATH = DATA_PATH_LOCAL

else:  # If you want to run on the HPC
    DATA_PATH = DATA_PATH_HPC

RESIDUAL_PATH = DATA_PATH / 'ZARR_RESIDUALS'

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
slicemanager.manage_zarr = True

# IMPORTANT: only training dataset residuals!!
subjList = ['Mouse01', 'Mouse02', 'Mouse03', 'Mouse04', 'Mouse05', 'Mouse08', 'Mouse09', 'Mouse11', 'Mouse12', 'Mouse13', 'Mouse14', 'Mouse15', 'Mouse17', 'Mouse18', 'Mouse19', 'Mouse20', 'Mouse21', 'Mouse23']
print(f'{bcolors.OKBLUE}Total amount of mice: {len(subjList)}{bcolors.ENDC}')

# open zarr store
root = zarr.open_group(str(RESIDUAL_PATH), mode='a')

for subj in subjList:
    id = slicemanager.mouse_list.index(subj)
    slicemanager.set_mouse_id(id)
    for i in range(6):
        print(f'{bcolors.OKCYAN}Processing mouse: {subj}, loc: {slicemanager.current_loc}, plane: {slicemanager.current_plane}{bcolors.ENDC}', flush=True)

        for s in range(len(slicemanager.current_slice_list)):
            normalise = lambda x: (x - np.mean(x)) / np.std(x)
            
            # always normalise when loading!!
            lr = normalise(slicemanager.load_scan(resolution='LOW RES',i=s))
            hr = normalise(slicemanager.load_scan(resolution='HIGH RES',i=s))

            lr_int = scipy.ndimage.zoom(lr, zoom=2, order=0)
            R = lr_int - hr

            name = "Residual_" + slicemanager.get_slice_ID()[:-2] + f"{s + 1:02}"

            zarr_arr = root.create_array(name, shape=R.shape ,chunks=R.shape, dtype='float', overwrite=True)
            zarr_arr[:] = R

        # go to next scan
        slicemanager.next_scan()

# normalize variance
total_arr = np.zeros((len(root.members()), R.shape[0], R.shape[1]))
for i, residual in enumerate(root.keys()):
    total_arr[i] = root[residual][:]

sn = np.mean(np.std(total_arr, axis=0))

# clear some memory
del total_arr

for residual in root.keys():
    root[residual][:] = root[residual][:] / sn

print(f'{bcolors.OKCYAN}Total amount of residuals: {len(root.members())}{bcolors.ENDC}')

# test
total_arr = np.zeros((len(root.members()), R.shape[0], R.shape[1]))
for i, residual in enumerate(root.keys()):
    total_arr[i] = root[residual][:]

sn = np.mean(np.std(total_arr, axis=0))
print(f'{bcolors.OKGREEN}CHECK: Population variance: {sn}{bcolors.ENDC}')
