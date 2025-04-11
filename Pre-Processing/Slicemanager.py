import numpy as np
import pydicom
import json
import pathlib
import zarr

## Used file structure:
# |_ Data
#     |_ DICOM
#         |_ HIGH RES
#             |_ MOUSE[i]
#                 |_ HEAD-THORAX
#                     |_ Coronal
#                     |_ Sagittal
#                     |_ Transax
#                 |_ THORAX-ABDOMEN
#                     |_ Coronal
#                     |_ Sagittal
#                     |_ Transax
#         |_ LOW RES
#             |_ MOUSE[i]
#                 |_ HEAD-THORAX
#                     |_ Coronal
#                     |_ Sagittal
#                     |_ Transax
#                 |_ THORAX-ABDOMEN
#                     |_ Coronal
#                     |_ Sagittal
#                     |_ Transax
#     |_ ZARR_PREPROCESSED
#         |_ ... (same as DICOM)
#     |_ BLACKLIST.json
# |_ PRE-PROCESSING
#    |_ visualizer.py
#    |_ Slicemanager.py
#    |_ preprocessing.py

## Use Visualizer.py to inspect and remove slices

## Used slice ID format:
# ID: MMLPXXR
# -> MM: Mouse ID
# -> L: Location     H = HEAD-THORAX; T = THORAX-ABDOMEN
# -> P: Plane        C = Coronal; S = Sagittal; T = Transax
# -> XX: Slice ID
# -> R: Resoulution  H = HIGH RES; L = LOW RES

# Example: 01HC01L = 1st low-resolution slice of coronal Head-Thorax image of mouse 1

# BLACKLIST.json contains IDs in format: MMLPXX (where resolution parameter is dropped (because both get blacklisted))

## Zarr array file structure:
# Resolution -> location -> mouse -> plane -> index (same as DICOM structure)

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

class Slice_manager:
    ## Slice manager is a class to navigate through the files containing the MRI images according to the filestructure specified above. 
    ## The class keeps track to which location, plane and mouse the current loaded slices belong. 
    ## After loading in the current slice, this class provides a way to identify slices with above specified ID syntax 
    ## and provides functions that keep track of which slices have been blacklisted. 
    ## The class can also manage loading and storing of zarr arrays, and load dicom files. 
    
    def __init__(self, DATA_PATH=pathlib.Path('../Data'), BLACKLIST_PATH=pathlib.Path('../Data') / 'BLACKLIST.json'):
        self.DATA_PATH = DATA_PATH
        self.BLACKLIST_PATH = BLACKLIST_PATH

        # get all mice ID by looking in data directory:
        path_mouse_list = self.DATA_PATH / 'DICOM' / 'HIGH RES' / 'HEAD-THORAX'
        self.mouse_list = sorted([f.name for f in path_mouse_list.iterdir() if f.is_dir()])
        
        self.current_mouse_ID = 0
        self.current_loc = 'HEAD-THORAX'
        self.current_plane = 'Coronal'
        self.current_slice = 0
        
        # initialize first slice
        self.current_slice_list = None
        self.hr_metadata = None
        self.lr_metadata = None
        self.hr_3D = None
        self.lr_3D = None
        
        # load in first scan
        self.get_scan_dicom()
        
        self.load_blacklisted()

        # variable that determines wether the slicemanager will search through DICOM or ZARR files (default: DICOM)
        self.manage_zarr = False
    
    def get_scan_dicom(self):
        # open and load current slice into class variables
        
        # create path object for directory to look in
        path_hr = self.DATA_PATH / 'DICOM' / 'HIGH RES' / self.current_loc / self.mouse_list[self.current_mouse_ID] / self.current_plane
        path_lr = self.DATA_PATH / 'DICOM' / 'LOW RES'  / self.current_loc / self.mouse_list[self.current_mouse_ID] / self.current_plane

        def failed_to_load():
            self.hr_3D = np.zeros((1,1,1))
            self.lr_3D = np.zeros((1,1,1))
            self.current_slice_list = [0]
            self.hr_metadata = { "ps": 'N/A', "st": 'N/A', "shape": ('N/A','N/A','N/A'), "mouse": 'N/A', "loc": 'N/A', "plane": 'N/A'}
            self.lr_metadata = { "ps": 'N/A', "st": 'N/A', "shape": ('N/A','N/A','N/A'), "mouse": 'N/A', "loc": 'N/A', "plane": 'N/A'}
            return

        # check if directory exists
        if not(path_hr.exists() and path_lr.exists()):
            print(f'{bcolors.WARNING}DICOM path doesn\'t exist, id: {self.get_slice_ID()}{bcolors.ENDC}')
            failed_to_load()
            return False

        # get a list of all slices in directory
        hr_slices = [f.name for f in path_hr.iterdir() if f.is_file()]
        lr_slices = [f.name for f in path_lr.iterdir() if f.is_file()]

        if not (len(np.array(hr_slices)) == len(np.array(lr_slices))):
            print(f'{bcolors.FAIL}Error: LR and HR directory do not correspond{bcolors.ENDC}')
            failed_to_load()
            return False
        
        if not (np.array(hr_slices)==np.array(lr_slices)).all():
            print(f'{bcolors.FAIL}Error: LR and HR directory do not correspond{bcolors.ENDC}')
            failed_to_load()
            return False
        
        self.current_slice_list = hr_slices
        self.current_slice = 0

        if len(hr_slices) == 0 or len(lr_slices) == 0:
            print(f'{bcolors.FAIL}Error: No slices found in directory, id: {self.get_slice_ID()[:-2]}{bcolors.ENDC}')
            failed_to_load()
            return False

        # get some metadata
        slice_hr_path = path_hr / hr_slices[0]
        slice_hr = pydicom.dcmread(slice_hr_path, force=True)
        shape_hr = slice_hr.pixel_array.shape
        shape_3D_hr = shape_hr + (len(hr_slices),)
        self.hr_metadata = { "ps": slice_hr.PixelSpacing, "st": slice_hr.SliceThickness, "shape": shape_3D_hr, "mouse": self.mouse_list[self.current_mouse_ID], "loc": self.current_loc, "plane": self.current_plane}

        slice_lr_path = path_lr / lr_slices[0]
        slice_lr = pydicom.dcmread(slice_lr_path, force=True)
        shape_lr = slice_lr.pixel_array.shape
        shape_3D_lr = shape_lr + (len(lr_slices),)
        self.lr_metadata = { "ps": slice_lr.PixelSpacing, "st": slice_lr.SliceThickness, "shape": shape_3D_lr, "mouse": self.mouse_list[self.current_mouse_ID], "loc": self.current_loc, "plane": self.current_plane}

        # create 3D image arrays
        self.hr_3D = np.zeros(shape_3D_hr)
        for i, slice_name in enumerate(hr_slices):
            slice_path = path_hr / slice_name
            self.hr_3D[:,:,i] = pydicom.dcmread(slice_path, force=True).pixel_array

        self.lr_3D = np.zeros(shape_3D_lr)
        for i, slice_name in enumerate(lr_slices):
            slice_path = path_lr / slice_name
            self.lr_3D[:,:,i] = pydicom.dcmread(slice_path, force=True).pixel_array

        return True
    
    def toggle_zarr_mode(self):
        if self.manage_zarr:
            self.manage_zarr = False
            self.get_scan_dicom()
        else:
            self.manage_zarr = True
            self.get_scan_zarr()

        return
    
    def get_scan_zarr(self):
        # open and load current slice into class variables
        # ! not the same as load_scan(), which just returns it, but doesn't load into active class variables self.lr and self.hr !

        # create path object for directory to look in
        path_hr = self.DATA_PATH / 'ZARR_PREPROCESSED' / 'HIGH RES' / self.current_loc / self.mouse_list[self.current_mouse_ID] / self.current_plane
        path_lr = self.DATA_PATH / 'ZARR_PREPROCESSED' / 'LOW RES'  / self.current_loc / self.mouse_list[self.current_mouse_ID] / self.current_plane

        def failed_to_load():
            self.hr_3D = np.zeros((1,1,1))
            self.lr_3D = np.zeros((1,1,1))
            self.current_slice_list = [0]
            return

        # check if directory exists
        if not(path_hr.exists() and path_lr.exists()):
            print(f'{bcolors.WARNING}Zarr path doesn\'t exist, id: {self.get_slice_ID()}{bcolors.ENDC}')
            failed_to_load()
            return False

        # get a list of all slices in directory
        hr_slices = sorted([f.name for f in path_hr.iterdir() if f.is_dir()])
        lr_slices = sorted([f.name for f in path_lr.iterdir() if f.is_dir()])

        if not (len(np.array(hr_slices)) == len(np.array(lr_slices))):
            print(f'{bcolors.FAIL}Error: LR and HR directory do not correspond{bcolors.ENDC}')
            failed_to_load()
            return False

        if not (np.array(hr_slices).shape==np.array(lr_slices).shape):
            print(f'{bcolors.FAIL}Error: LR and HR directory do not correspond{bcolors.ENDC}')
            failed_to_load()
            return False
        
        if not (np.array(hr_slices)==np.array(lr_slices)).all():
            print(f'{bcolors.FAIL}Error: LR and HR directory do not correspond{bcolors.ENDC}')
            failed_to_load()
            return False
        self.current_slice_list = hr_slices
        self.current_slice = 0

        if len(hr_slices) == 0 or len(lr_slices) == 0:
            print(f'{bcolors.FAIL}Error: No slices found in directory, id: {self.get_slice_ID()}{bcolors.ENDC}')
            failed_to_load()
            return False
        
        # create 3D image arrays
        self.lr_3D = np.zeros((self.lr_metadata.get('shape')[0], self.lr_metadata.get('shape')[1], len(lr_slices)))
        self.hr_3D = np.zeros((self.hr_metadata.get('shape')[0], self.hr_metadata.get('shape')[1], len(hr_slices)))

        # get scan arrays
        for i in range(len(lr_slices)):
            s = self.load_scan('LOW RES', i=i)
            w = self.lr_metadata.get('shape')[0] - s.shape[0]
            h = self.lr_metadata.get('shape')[1] - s.shape[1]
            self.lr_3D[:,:,i] = np.pad(s, ((w - w//2, w//2), (h - h//2, h//2)))

        for i in range(len(hr_slices)):
            s = self.load_scan('HIGH RES', i=i)
            w = self.hr_metadata.get('shape')[0] - s.shape[0]
            h = self.hr_metadata.get('shape')[1] - s.shape[1]

            self.hr_3D[:,:,i] = np.pad(s, ((w - w//2, w//2), (h - h//2, h//2)))

        return True
    
    def get_slice(self, i=None):
        # get current slice array
        i = self.current_slice if i == None else i
        return self.lr_3D[:,:,i], self.hr_3D[:,:,i]
    
    def next_slice(self):
        # go to next mouse and load in slices
        if self.current_slice == len(self.current_slice_list) - 1:
            #rollover to next plane
            if self.current_plane == 'Coronal': 
                self.current_plane = 'Sagittal'
            elif self.current_plane == 'Sagittal': self.current_plane = 'Transax'
            else:
                self.current_plane = 'Coronal'
                # rollover to next loc
                if self.current_loc == 'HEAD-THORAX': self.current_loc = 'THORAX-ABDOMEN'
                else:
                    self.current_loc = 'HEAD-THORAX'
                    # rollover to next mouse
                    if self.current_mouse_ID == len(self.mouse_list) - 1:
                        self.current_mouse_ID = 0
                    else: 
                        self.current_mouse_ID = self.current_mouse_ID + 1
                    
            # update image
            if self.manage_zarr: self.get_scan_zarr()
            else: self.get_scan_dicom()

            self.current_slice = 0
        else:
            self.current_slice = self.current_slice + 1
            
        return
        
    def previous_slice(self):
        # go to previous slice and load in slices
        if self.current_slice == 0:
            # rollover to previous plane
            if self.current_plane == 'Transax': 
                self.current_plane = 'Sagittal'
            elif self.current_plane == 'Sagittal': self.current_plane = 'Coronal'
            else:
                self.current_plane = 'Transax'
                # rollover to previous loc
                if self.current_loc == 'THORAX-ABDOMEN': self.current_loc = 'HEAD-THORAX'
                else:
                    self.current_loc = 'THORAX-ABDOMEN'
                    # rollover to previous mouse
                    if self.current_mouse_ID == 0:
                        self.current_mouse_ID = len(self.mouse_list) - 1
                    else: 
                        self.current_mouse_ID = self.current_mouse_ID - 1
                    
            # update image
            if self.manage_zarr: self.get_scan_zarr()
            else: self.get_scan_dicom()

            self.current_slice = len(self.current_slice_list) - 1
        else:
            self.current_slice = self.current_slice - 1
            
        return
        
    def set_loc(self, loc):
        # set current location and load in slices
        # loc is either 'HEAD-THORAX' or 'THORAX-ABDOMEN'
        if not (loc == 'HEAD-THORAX' or loc == 'THORAX-ABDOMEN'):
            print(f'{bcolors.FAIL}Error: invalid location given{bcolors.ENDC}')
            return
        
        self.current_loc = loc
        
        # update image
        if self.manage_zarr: self.get_scan_zarr()
        else: self.get_scan_dicom()
        
        return
    
    def set_plane(self, plane):
        # set current plane and load in slices
        # plane is either 'Coronal', 'Sagittal' or 'Transax'
        if not (plane == 'Coronal' or plane == 'Sagittal' or plane == 'Transax'):
            print(f'{bcolors.FAIL}Error: invalid plane given{bcolors.ENDC}')
            return
        
        self.current_plane = plane
        
        # update image
        if self.manage_zarr: self.get_scan_zarr()
        else: self.get_scan_dicom()
        
        return
    
    def set_slice(self, slice_i):
        # set current slice number and load in slices
        if slice_i >= len(self.current_slice_list) - 1:
            print(f'{bcolors.FAIL}Error: invalid slice id given{bcolors.ENDC}')
            return
        
        self.current_slice = slice_i
        
        return
    
    def set_mouse_id(self, mouse_i):
        if mouse_i >= len(self.mouse_list) - 1:
            print(f'{bcolors.FAIL}Error: invalid slice id given{bcolors.ENDC}')
            return
        
        self.current_mouse_ID = mouse_i
        
        # update image
        if self.manage_zarr: self.get_scan_zarr()
        else: self.get_scan_dicom()
        
        return
    
    def next_mouse(self):
        # move to next mouse and load in slices
        if self.current_mouse_ID == len(self.mouse_list) - 1:
            self.current_mouse_ID = 0
        else: 
            self.current_mouse_ID = self.current_mouse_ID + 1

        # update image
        if self.manage_zarr: self.get_scan_zarr()
        else: self.get_scan_dicom()
        
        return
    
    def previous_mouse(self):
        # move to previous mouse and load in slices
        if self.current_mouse_ID == 0:
            self.current_mouse_ID = len(self.mouse_list) - 1
        else: 
            self.current_mouse_ID = self.current_mouse_ID - 1
    
        # update image
        if self.manage_zarr: self.get_scan_zarr()
        else: self.get_scan_dicom()
        
        return
    
    def get_slice_ID(self):
        # get the ID of current slice
        mouse_id = f"{self.current_mouse_ID + 1:02}"

        L = ""
        if self.current_loc == "HEAD-THORAX": L = "H"
        else: L = "T"

        P = ""
        if self.current_plane == "Coronal": P = "C"
        elif self.current_plane == "Sagittal": P = "S"
        else: P = "T"

        slice_id = f"{self.current_slice + 1:02}"

        id = "".join([mouse_id, L, P, slice_id])

        return id
    
    def check_blacklisted(self):
        # read json file to get latest blacklist
        self.load_blacklisted()

        if self.get_slice_ID() in self.blacklist:
            self.slice_is_blacklisted = True
        else:
            self.slice_is_blacklisted = False
        return

    def load_blacklisted(self):
        # more efficient way of checking which slices are blacklisted (instead of 1 at a time) (used for preprocessing, for visualizer.py -> use check_blacklisted!)
        # read json file
        with open(self.BLACKLIST_PATH, 'r') as f:
            json_object = json.load(f)

        self.blacklist = json_object.get("blacklist")

        return

    def next_scan(self):
        # go to next scan and load in slices

        #rollover to next plane
        if self.current_plane == 'Coronal': 
            self.current_plane = 'Sagittal'
        elif self.current_plane == 'Sagittal': self.current_plane = 'Transax'
        else:
            self.current_plane = 'Coronal'
            # rollover to next loc
            if self.current_loc == 'HEAD-THORAX': self.current_loc = 'THORAX-ABDOMEN'
            else:
                self.current_loc = 'HEAD-THORAX'
                # rollover to next mouse
                if self.current_mouse_ID == len(self.mouse_list) - 1:
                    self.current_mouse_ID = 0
                else: 
                    self.current_mouse_ID = self.current_mouse_ID + 1
                
        # update image
        if self.manage_zarr: self.get_scan_zarr()
        else: self.get_scan_dicom()
        
        self.current_slice = 0
            
        return

    def previous_scan(self):
        # go to previous scan and load in slices

        # rollover to previous plane
        if self.current_plane == 'Transax': 
            self.current_plane = 'Sagittal'
        elif self.current_plane == 'Sagittal': self.current_plane = 'Coronal'
        else:
            self.current_plane = 'Transax'
            # rollover to previous loc
            if self.current_loc == 'THORAX-ABDOMEN': self.current_loc = 'HEAD-THORAX'
            else:
                self.current_loc = 'THORAX-ABDOMEN'
                # rollover to previous mouse
                if self.current_mouse_ID == 0:
                    self.current_mouse_ID = len(self.mouse_list) - 1
                else: 
                    self.current_mouse_ID = self.current_mouse_ID - 1
                
        # update image
        if self.manage_zarr: self.get_scan_zarr()
        else: self.get_scan_dicom()

        self.current_slice = len(self.current_slice_list) - 1
            
        return

    def remove_blacklisted(self):
        # cleanes 3D scan from removed slices, returns: lr, hr 3D array
        # note remove_blacklisted() doesn't use a life copy of BLACKLIST.json, check_blacklisted does (but is less efficient)

        self.current_slice = 0
        removed_arr_lr = []
        removed_arr_hr = []

        for i in range(self.lr_3D.shape[2]):
            if self.get_slice_ID() in self.blacklist:
                pass
            else:
                removed_arr_lr.append(self.lr_3D[:,:,i])
                removed_arr_hr.append(self.hr_3D[:,:,i])

            if self.current_slice >= len(self.current_slice_list) - 1: break
            else: self.current_slice += 1
                
        return  np.transpose(np.array(removed_arr_lr), (1, 2, 0)), np.transpose(np.array(removed_arr_hr), (1, 2, 0))
    
    def store_scan(self, scan, resolution):
        # stores current scan at appropriate location as zarr array

        loc = self.current_loc
        mouse = self.mouse_list[self.current_mouse_ID]
        plane = self.current_plane
        
        # open root store as a group
        root = zarr.open_group(str(self.DATA_PATH / 'ZARR_PREPROCESSED'), mode='a')

        # ensure all groups exist
        res_group = root.require_group(resolution)
        loc_group = res_group.require_group(loc)
        mouse_group = loc_group.require_group(mouse)
        plane_group = mouse_group.require_group(plane, overwrite=True)

        for i, s in enumerate(scan):
            name = str(i)
            if name in plane_group:
                # overwrite if exists
                del plane_group[name]
                
            arr = plane_group.create_array(name, shape=s.shape ,chunks=s.shape, dtype='uint16')
            arr[:] = s
        
        return
    
    def load_scan(self, resolution, loc=None, mouse=None, plane=None, i=None):
        # returns current scan from preprocessed zarr array

        loc = loc if loc else self.current_loc 
        mouse = mouse if mouse else self.mouse_list[self.current_mouse_ID]
        plane = plane if plane else self.current_plane
        i = i if i else self.current_slice
        
        # open root store as a group
        root = zarr.open_group(str(self.DATA_PATH / 'ZARR_PREPROCESSED'), mode='r')

        arr = root[resolution][loc][mouse][plane][str(i)]

        return arr