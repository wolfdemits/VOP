## Import Packages

import random
import numpy as np
import torch # type: ignore
import zarr # type: ignore
from scipy.ndimage import zoom
from pathlib import Path


### NORMALISATION FUNCTION

def Mean_Normalisation(ImgArray, Norm_Factor=None, inverse=False):

    if not inverse:
        
        if Norm_Factor is None:     ## for LR
            Norm_Factor = np.mean(ImgArray)
            Norm_ImgArray = ImgArray / np.mean(ImgArray)
        
        else:   ## for HR
            Norm_ImgArray = ImgArray / Norm_Factor

    if inverse: 
        Norm_ImgArray = ImgArray * Norm_Factor
        
    return Norm_ImgArray, Norm_Factor


### DATASET CLASS 


class Dataset2D_MicroMRI(torch.utils.data.Dataset):

    """ Class function for preparing/preprocessing the data that we want to feed to the neural network
        * init     --> initialise all arguments, and calculate all possible image pairs that can be fed into the model
        * len      --> gives the length of the self.search list
        * getitem  --> defines a dictionary with all info about a certain pair
    """

    def __init__(self, PathZarr, SubjList, num_input_slices, RandomFlip=False, Planes = ["Coronal", "Sagittal", "Transax"], Regions=["THORAX-ABDOMEN", "HEAD-THORAX"]):
        
        # Define the "self" variables so that they can be called in the "getitem"
        self.PathZarr = PathZarr
        self.SubjList = SubjList
        self.Planes = Planes
        self.Regions = Regions
        self.RandomFlip = RandomFlip 
        self.num_input_slices = num_input_slices

        
        self.search = []
        
        # Iterate over all mice and all planes of the scan to read into the model
        
        for Subj in SubjList:
            
            for Plane in Planes:

                for Region in Regions:

                    # Now you are in a folder with maps for each slice --> need to enter each slice map and get first file
                    
                    # Paths to HR images
                    
                    # region = (Subj.split('_')[1] + '-' + Subj.split('_')[2]).upper()
                    # id = Subj.split('_')[0]
                    
                    self.path2slices_HR = PathZarr / 'HIGH RES' / Region / Subj / Plane
                    self.path2slices_HR = Path(self.path2slices_HR).absolute()
                    slicesFolder_HR = [path for path in self.path2slices_HR.iterdir() if not path.name.startswith('.')] 

                    # Paths to LR images
                    self.path2slices_LR = PathZarr / 'LOW RES' / Region / Subj / Plane
                    self.path2slices_LR = Path(self.path2slices_LR)

                    for SliceIndex, _ in enumerate(slicesFolder_HR): 
                        self.search.append((Subj, Plane, SliceIndex))

        


    def __len__(self):
        LengthLoader = len(self.search)
        return LengthLoader


    def __getitem__(self, index): 
        
        Subject, Plane, SliceIndex  = self.search[index] 
        if SliceIndex != 0:
            SliceIndex -=1 #cheap fix
        
        zf_LR = zarr.open(str(self.path2slices_LR))
        zf_HR = zarr.open(str(self.path2slices_HR))

        
        LR_Img = zf_LR[str(SliceIndex)][:]
        HR_Img = zf_HR[str(SliceIndex)][:]

        # Interpolation of the LR Image for Image Size to Match 
        #   Use: scipy.ndimage.zoom -> zoom = 2 (incr resolution with factor 2) & order = 0 (nearest neighbor interpolation is used)
        LR_Img = zoom(LR_Img, zoom=2, order=0)  




        ## Normalization
        LR_Img, LR_NormFactor = Mean_Normalisation(LR_Img)
        HR_Img, _ = Mean_Normalisation(HR_Img, LR_NormFactor) 
        
        # Expand
        LR_Img = np.expand_dims(LR_Img, axis=0)
        HR_Img = np.expand_dims(HR_Img, axis=0)
        
        # Convert into FloatTensor and Add Channel Dimension
        LR_Img = torch.FloatTensor(LR_Img)
        HR_Img = torch.FloatTensor(HR_Img) 

        if self.RandomFlip:
            if random.random() > 0.5: # vertical flip
                LR_Img = torch.flip(LR_Img, dims=[0])    
                HR_Img = torch.flip(HR_Img, dims=[0])  
            if random.random() > 0.5: # horizontal flip
                LR_Img = torch.flip(LR_Img, dims=[1])
                HR_Img = torch.flip(HR_Img, dims=[1])  
        

        # Define the item to feed to the net as a dictionary

        item = {'LR_Img': LR_Img, 
                'HR_Img': HR_Img,
                'Subject': Subject, 
                'Plane': Plane, 
                'SliceIndex': SliceIndex, 
                'LR_NormFactor': LR_NormFactor}

        return item


class CollateFn2D_MicroMRI():

    """ Upon loading the individual data (i.e., slices of different sizes, if cropping was used) into one batch
        -> Input data may not necessarily be of the same matrix size

        of the same matrix size, but all slices in one batch need to be 
    of the same size, so collate function is used to pad the slices to equal size """

    def __call__(self, item_list):

        batch = {} # Initialise an empty dictionary to store a certain batch
        

        # Iterate over all keys of an item (so over 'LR_Img', 'HR_Img', 'Subject', 'Plane' and 'SliceName'

        for key in item_list[0].keys():

            # If the key is 'LR_Img' or 'HR_Img' (= these contain the data arrays) -> They need to be padded
            
            if key in ['LR_Img', 'HR_Img']: 
                
                tensor_list = [item[key] for item in item_list]
                shape_list  = [tensor.shape for tensor in tensor_list]

                height_max  = max(shape[-2] for shape in shape_list) # Determines the max height of all tensors --> we have to padd al the other tensors to this height
                width_max   = max(shape[-1] for shape in shape_list) # Determines the max width of all tensors --> we have to padd all the other tensors to this width
                
                pad_list    = [((width_max - shape[-1]) // 2,      # left padding
                                -(-(width_max - shape[-1]) // 2),  # right padding
                                (height_max - shape[-2]) // 2,     # top padding
                                -(-(height_max - shape[-2]) // 2)) # bottom padding   
                               for shape in shape_list]
                
                tensor = torch.stack([
                            torch.nn.functional.pad(tensor, padding, value=-1) 
                            for tensor, padding in zip(tensor_list, pad_list)])
                
                # Add stacked & padded tensor data to batch dictionary using the exsisting key
                batch[key] = tensor
            
            else:
                # If key is something other than 'LR_Img' or 'HR_Img' --> Merge data of all items in a list using stack command and add to batch
                batch[key] = np.stack([item[key] for item in item_list])

        return batch
    

##################
### Dataloader ###
##################

def Get_DataLoaders(SubjTrain, SubjVal, PathZarr, Planes, Regions, batch_size, num_in_channels):
    
    # Load training data
    
    TrainSet = Dataset2D_MicroMRI(
                PathZarr = PathZarr,
                SubjList = SubjTrain,
                Planes = Planes,
                Regions=Regions,
                num_input_slices = num_in_channels,
                RandomFlip = True)
    
    TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=batch_size, collate_fn=CollateFn2D_MicroMRI(), shuffle=True)

    
    # Load validation data

    ValSet = Dataset2D_MicroMRI(
                PathZarr = PathZarr,
                SubjList = SubjVal,
                Planes = Planes,
                Regions=Regions,
                num_input_slices = num_in_channels,
                RandomFlip = True)
    
    ValLoader = torch.utils.data.DataLoader(ValSet, batch_size=batch_size, collate_fn=CollateFn2D_MicroMRI(), shuffle=True)

    return TrainLoader, ValLoader

