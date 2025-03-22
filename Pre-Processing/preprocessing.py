import numpy as np
import pathlib
from Slicemanager import Slice_manager

DATA_PATH = pathlib.Path('../Data')

CROPPING_THRESHOLD = 12
CROPPING_PADDING = 2
MIN_CROPPING_SHAPE = 16

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

def cropping(lr_scan, hr_scan, threshold=12, padding=2, min_shape=16):
    # cropping function: returns array consisting of numpy lists which contain cropped images
    
    min_val = np.min(lr_scan)
    max_val = np.max(lr_scan)
    
    slices_norm = np.uint8((lr_scan - min_val) / (max_val - min_val) * 255) #normalize to 0-255 range, + eps to account for divide by zero

    H, W, D = slices_norm.shape
    cropped_lr = []
    cropped_hr = []
    
    for d in range(D): 
        right_edge = 0
        left_edge = 0
        upper_edge = 0
        lower_edge = 0

        # start right
        for i in range(W):
            mean = np.mean(slices_norm[:,i,d])
    
            if mean > threshold:
                # threshold reached, taking steps back as padding
                right_edge = max(right_edge - padding, 0)
                break
            else: 
                width = W - abs(right_edge) - abs(left_edge)
                if width <= min_shape: break
                right_edge += 1
    
        # start left
        for i in range(W):
            mean = np.mean(slices_norm[:,W-i-1,d])
    
            if mean > threshold:
                # threshold reached, taking steps back as padding
                left_edge = min(left_edge + padding, 0)
                break
            else: 
                width = W - abs(right_edge) - abs(left_edge)
                if width <= min_shape: break
                left_edge -= 1
    
        # start right
        for i in range(H):
            mean = np.mean(slices_norm[i,:,d])
            if mean > threshold:
                # threshold reached, taking steps back as padding
                upper_edge = max(upper_edge - padding, 0)
                break
            else:
                height = W - abs(upper_edge) - abs(lower_edge)
                if height <= min_shape: break
                upper_edge += 1
    
        # start left
        for i in range(H):
            mean = np.mean(slices_norm[H-i-1,:,d])
    
            if mean > threshold:
                # threshold reached, taking steps back as padding
                lower_edge = min(lower_edge + padding, 0)
                break
            else: 
                height = W - abs(upper_edge) - abs(lower_edge)
                if height <= min_shape: break
                lower_edge -= 1

        # take into account slicing is exclusive at end
        lower_edge = lower_edge if lower_edge != 0 else None
        left_edge = left_edge if left_edge != 0 else None
                
        cropped_lr.append(lr_scan[upper_edge:lower_edge,right_edge:left_edge,d])
        cropped_hr.append(hr_scan[upper_edge:lower_edge,right_edge:left_edge,d])

    return cropped_lr, cropped_hr

def preprocess_scan(scan_lr, scan_hr):
    # cropping
    processed_lr, processed_hr = cropping(scan_lr, scan_hr, threshold=CROPPING_THRESHOLD, padding=CROPPING_PADDING, min_shape=MIN_CROPPING_SHAPE)

    return processed_lr, processed_hr

# init slicemanager -> puts current slice at beginning, i.e.: 01HC01
slicemanager = Slice_manager()

last_mouse_id = 0 # ATTENTION: change to 22 on hpc

while slicemanager.current_mouse_ID <= last_mouse_id:
    print( f'{bcolors.OKCYAN}Processing mouse: {slicemanager.mouse_list[slicemanager.current_mouse_ID]}, loc: {slicemanager.current_loc}, plane: {slicemanager.current_plane}{bcolors.ENDC}')

    try: 
        processed_lr, processed_hr = preprocess_scan(*slicemanager.remove_blacklisted())
        slicemanager.store_scan(processed_lr, 'LOW RES')
        print(f'{bcolors.OKGREEN}Succesfully preprocessed scan: {slicemanager.get_slice_ID()[:-2]}{bcolors.ENDC}')
    except Exception as e:
        print(f'{bcolors.WARNING}Encountered an error while preprocessing scan: {slicemanager.get_slice_ID()[:-2]}{bcolors.ENDC}')
        print(e)

    try:
        slicemanager.store_scan(processed_hr, 'HIGH RES')
        print(f'{bcolors.OKGREEN}Succesfully stored scan {slicemanager.get_slice_ID()[:-2]}{bcolors.ENDC}')
    except Exception as e:
        print(f'{bcolors.WARNING}Encountered an error while saving scan: {slicemanager.get_slice_ID()[:-2]}{bcolors.ENDC}')
        print(e)

    print('')

    # go to next scan
    slicemanager.next_scan()

print(f'{bcolors.OKBLUE}Preprocessing script finished. {bcolors.ENDC}')