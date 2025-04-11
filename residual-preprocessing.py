import pathlib
import zarr
import numpy as np
from Slicemanager import Slice_manager
import scipy

DATA_PATH = pathlib.Path('../Data')

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
slicemanager.ENABLE_CROPPING = False

number_of_mice = 7
print(f'{bcolors.OKBLUE}Total amount of mice: {number_of_mice}{bcolors.ENDC}')

# open zarr store
root = zarr.open_group(str(RESIDUAL_PATH), mode='a')

for i in range(number_of_mice*6):
    print(f'{bcolors.OKCYAN}Processing mouse: {slicemanager.mouse_list[round(np.floor(i/6))]}, loc: {slicemanager.current_loc}, plane: {slicemanager.current_plane}{bcolors.ENDC}', flush=True)

    for s in range(len(slicemanager.current_slice_list)):
        normalise = lambda x: (x - np.mean(x)) / np.std(x)
        
        lr = slicemanager.load_scan(resolution='LOW RES',i=s)
        hr = slicemanager.load_scan(resolution='HIGH RES',i=s)
        lr_norm = normalise(lr)
        hr_norm = normalise(hr)
        lr_int = scipy.ndimage.zoom(lr_norm, zoom=2, order=0)
        R = lr_int - hr_norm

        name = "Residual_" + slicemanager.get_slice_ID()[:-2] + f"{s + 1:02}"

        zarr_arr = root.create_array(name, shape=R.shape ,chunks=R.shape, dtype='float', overwrite=True)
        zarr_arr[:] = R

    # go to next scan
    slicemanager.next_scan()

print(f'{bcolors.OKCYAN}Total amount of residuals: {len(root.members())}{bcolors.ENDC}')