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
#    |_ Visualizer.py
#    |_ Slicemanager.py
#    |_ Preprocessing.py

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
        self.mouse_list = [f.name for f in path_mouse_list.iterdir() if f.is_dir()]
        
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
        
        self.get_scan_dicom()
        
        self.load_blacklisted()
    
    def get_scan_dicom(self):
        # open and load current slice
        
        # create path object for directory to look in
        path_hr = self.DATA_PATH / 'DICOM' / 'HIGH RES' / self.current_loc / self.mouse_list[self.current_mouse_ID]  / self.current_plane
        path_lr = self.DATA_PATH / 'DICOM' / 'LOW RES'  / self.current_loc / self.mouse_list[self.current_mouse_ID] / self.current_plane

        # get a list of all slices in directory
        hr_slices = [f.name for f in path_hr.iterdir() if f.is_file()]
        lr_slices = [f.name for f in path_lr.iterdir() if f.is_file()]
        
        if not (np.array(hr_slices)==np.array(lr_slices)).all():
            print('Error: LR and HR directory do not correspond')
            return False
        self.current_slice_list = hr_slices
        self.current_slice = 0

        if len(hr_slices) == 0 or len(lr_slices) == 0:
            print('Error: No slices found in directory')
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
    
    def get_slice(self):
        # get current slice array
        return self.lr_3D[:,:,self.current_slice], self.hr_3D[:,:,self.current_slice]
    
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
            self.get_scan_dicom()
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
            self.get_scan_dicom()
            self.current_slice = len(self.current_slice_list) - 1
        else:
            self.current_slice = self.current_slice - 1
            
        return
        
    def set_loc(self, loc):
        # set current location and load in slices
        # loc is either 'HEAD-THORAX' or 'THORAX-ABDOMEN'
        if not (loc == 'HEAD-THORAX' or loc == 'THORAX-ABDOMEN'):
            print('Error: invalid location given')
            return
        
        self.current_loc = loc
        
        # update image
        self.get_scan_dicom()
        
        return
    
    def set_plane(self, plane):
        # set current plane and load in slices
        # plane is either 'Coronal', 'Sagittal' or 'Transax'
        if not (plane == 'Coronal' or plane == 'Sagittal' or plane == 'Transax'):
            print('Error: invalid plane given')
            return
        
        self.current_plane = plane
        
        # update image
        self.get_scan_dicom()
        
        return
    
    def set_slice(self, slice_i):
        # set current slice number and load in slices
        if slice_i >= len(self.current_slice_list) - 1:
            print('Error: invalid slice id given')
            return
        
        self.current_slice = slice_i
        
        # update image
        self.get_scan_dicom()
        
        return
    
    def set_mouse_id(self, mouse_i):
        if mouse_i >= len(self.mouse_list) - 1:
            print('Error: invalid slice id given')
            return
        
        self.current_mouse_ID = mouse_i
        
        # update image
        self.get_scan_dicom()
        
        return
    
    def next_mouse(self):
        # move to next mouse and load in slices
        if self.current_mouse_ID == len(self.mouse_list) - 1:
            self.current_mouse_ID = 0
        else: 
            self.current_mouse_ID = self.current_mouse_ID + 1

        # update image
        self.get_scan_dicom()
        
        return
    
    def previous_mouse(self):
        # move to previous mouse and load in slices
        if self.current_mouse_ID == 0:
            self.current_mouse_ID = len(self.mouse_list) - 1
        else: 
            self.current_mouse_ID = self.current_mouse_ID - 1
    
        # update image
        self.get_scan_dicom()
        
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
        self.get_scan_dicom()
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
        self.get_scan_dicom()
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
        plane_group = mouse_group.require_group(plane)

        for i, s in enumerate(scan):
            name = str(i)
            if name in plane_group:
                # overwrite if exists
                del plane_group[name]
                
            arr = plane_group.create_array(name, shape=s.shape ,chunks=s.shape, dtype='uint16')
            arr[:] = s
        
        return
    
    def load_scan(self, resolution, loc=None, mouse=None, plane=None, i=None):
        # loads current scan from preprocessed zarr array

        loc = loc if loc else self.current_loc 
        mouse = mouse if mouse else self.mouse_list[self.current_mouse_ID]
        plane = plane if plane else self.current_plane
        i = i if i else self.current_slice

        print(loc)
        
        # open root store as a group
        root = zarr.open_group(str(self.DATA_PATH / 'ZARR_PREPROCESSED'), mode='r')

        arr = root[resolution][loc][mouse][plane][str(i)]

        return arr