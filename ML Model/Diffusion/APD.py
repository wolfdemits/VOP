import pathlib
import zarr
import numpy as np
import torch
import scipy
import random

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

class APD:
    def __init__(self, DATAPATH=pathlib.Path('../../Data')):
        self.DATAPATH = DATAPATH
        return
    
    class ResidualSet(torch.utils.data.Dataset):
        def __init__(self, DATAPATH=pathlib.Path('../../Data')):
            self.RESIDUAL_PATH = DATAPATH / 'ZARR_RESIDUALS'

            root = zarr.open_group(str(self.RESIDUAL_PATH), mode='r')
            self.members = list(root.keys())
            self.length = len(self.members)
            return
        
        def __getitem__(self, index):
            root = zarr.open_group(str(self.RESIDUAL_PATH), mode='r')
            arr = root[self.members[index]][:]
            return torch.FloatTensor(arr)

        def __len__(self):
            return self.length
        
    class Dataset(torch.utils.data.Dataset):
        """ Class function for preparing/preprocessing the data that we want to feed to the neural network
            * init     --> initialise all arguments, and calculate all possible image pairs that can be fed into the model
            * len      --> gives the length of the self.search list
            * getitem  --> defines a dictionary with all info about a certain pair
        """

        def __init__(self, SubjList, DATAPATH=pathlib.Path('../../Data'), RandomFlip=False, Planes = ["Coronal", "Sagittal", "Transax"], Regions=["THORAX-ABDOMEN", "HEAD-THORAX"]):
            self.PREPROCESSED_PATH = DATAPATH / 'ZARR_PREPROCESSED_DIFFUSION'
            self.RandomFlip = RandomFlip

            root = zarr.open_group(str(self.PREPROCESSED_PATH), mode='r')

            self.search = []

            for loc in Regions:
                for subj in SubjList:
                    for plane in Planes:
                        slices = root['LOW RES'][loc][subj][plane].keys()
                        for i in slices:
                            self.search.append((loc, subj, plane, i))

            return
        
        def normalise(self, x):
            return (x - torch.mean(x)) / torch.std(x)
        
        def __getitem__(self, index):
            loc, subj, plane, i = self.search[index]

            root = zarr.open_group(str(self.PREPROCESSED_PATH), mode='r')
            lr = root['LOW RES'][loc][subj][plane][str(i)][:]
            hr = root['HIGH RES'][loc][subj][plane][str(i)][:]

            # images should be normalised when fed to network
            hr = hr.astype(np.float32)
            lr = lr.astype(np.float32)

            LR_Img = self.normalise(torch.FloatTensor(lr))
            HR_Img = self.normalise(torch.FloatTensor(hr))

            # Expand
            LR_Img = np.expand_dims(LR_Img, axis=0)
            HR_Img = np.expand_dims(HR_Img, axis=0)

            if self.RandomFlip:
                if random.random() > 0.5: # vertical flip
                    LR_Img = torch.flip(LR_Img, dims=[0])    
                    HR_Img = torch.flip(HR_Img, dims=[0])  
                if random.random() > 0.5: # horizontal flip
                    LR_Img = torch.flip(LR_Img, dims=[1])
                    HR_Img = torch.flip(HR_Img, dims=[1])  
        

            # Define the item to feed to the net as a dictionary
            # TODO: add diffusion parameters to returned dictionary
            item = {'LR_Img': LR_Img, 'HR_Img': HR_Img, 'Subject': subj, 'Plane': plane, 'Loc': loc, 'SliceIndex': i}

            return item

        def __len__(self):
            return len(self.search)
        
    def diffuse(self, t, T, x0, xT, beta, convergence_verbose=False):
        # Note: x0 and xT should always be normalised!! 
        # t = time tensor: (B,)
        # x0, xT = images: (B, 300, 300)
        B = x0.shape[0]

        sigma =  np.sqrt(15/8 * (T**3)/(T**4 - 1)) * beta

        xT = torch.FloatTensor(scipy.ndimage.zoom(xT, zoom=(1,2,2), order=0))

        residual = xT - x0

        f = lambda t: 4* t/T * (1-t/T)

        if convergence_verbose:
            critical_value_beta = np.sqrt(8/15 * (T**4 - 1)/(T**5))
            print(f'{bcolors.WARNING}Critical beta value: {critical_value_beta} {bcolors.ENDC}')
            print(f'{bcolors.WARNING}Beta value: {beta} {bcolors.ENDC}')
            if (beta > critical_value_beta*10):
                print(f'{bcolors.FAIL}Will not converge!{bcolors.ENDC}')
            elif (beta > critical_value_beta):
                print(f'{bcolors.WARNING}WARN: in range of beta_crit, might not converge! (but is possible){bcolors.ENDC}')
        
        # img: (T, B, 300, 300)
        images = torch.zeros((T+1, B, x0.shape[1], x0.shape[2]))
        images[0,:,:,:] = x0

        # sample epsilon: (T, B, 300, 300) (T=time) -> N samples = T*B
        # Note epsilons are already variance-normalized
        residualSet = APD.ResidualSet(self.DATAPATH)
        sampler = torch.utils.data.RandomSampler(residualSet, replacement=True)
        resLoader = torch.utils.data.DataLoader(residualSet, sampler=sampler, batch_size=B)

        def step(xt_1, t):
            epsilon = next(iter(resLoader))
            xt = xt_1 + 1/T * residual + sigma * f(t) * epsilon
            return xt

        for i in range(T):
            images[i+1,:,:,:] = step(xt_1=images[i,:,:,:], t=i+1)

        x_t = torch.zeros((B, x0.shape[1], x0.shape[2]))
        x_t_1 = torch.zeros((B, x0.shape[1], x0.shape[2]))

        for b in range(B):
            x_t[b] = images[t[b].item(), b,:,:].squeeze()
            x_t_1[b] = images[t[b].item()-1, b,:,:].squeeze()

        return x_t, x_t_1
    

## TODO: add convergence warning --> add functionality to residual-preprocessing to assess max residual MS

## Collate function

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

def Get_DataLoaders(SubjTrain, SubjVal, DATAPATH, Planes, Regions, batch_size):
    
    # Load training data

    TrainSet = APD.Dataset(
                SubjList = SubjTrain,
                DATAPATH = DATAPATH,
                Planes = Planes,
                Regions = Regions,
                RandomFlip = True)
    
    TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=batch_size, collate_fn=CollateFn2D_MicroMRI(), shuffle=True)

    
    # Load validation data

    ValSet = APD.Dataset(
                SubjList = SubjVal,
                DATAPATH = DATAPATH,
                Planes = Planes,
                Regions = Regions,
                RandomFlip = True)
    
    ValLoader = torch.utils.data.DataLoader(ValSet, batch_size=batch_size, collate_fn=CollateFn2D_MicroMRI(), shuffle=True)

    return TrainLoader, ValLoader

def Get_Test_DataLoader(SubjTest, PathZarr, Planes, Regions, batch_size, num_in_channels):

    TestSet = APD.Dataset(
                PathZarr = PathZarr,
                SubjList = SubjTest,
                Planes = Planes,
                Regions = Regions,
                num_input_slices = num_in_channels,
                RandomFlip = False)
    
    TestLoader = torch.utils.data.DataLoader(TestSet, batch_size=batch_size, collate_fn=CollateFn2D_MicroMRI(), shuffle=True)

    return TestLoader