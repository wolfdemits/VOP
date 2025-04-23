import shutil
import torch
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_msssim import ssim
import scipy
import torch.nn.functional as F

from DatasetClasses_MicroMRI import Mean_Normalisation


#################
## TENSORBOARD ##
#################

def Tensorboard_Initialization(tensorboard_pathname):
    shutil.rmtree(tensorboard_pathname, ignore_errors=True)
    writers = {
        'train': SummaryWriter(tensorboard_pathname / 'train'),
        'val': SummaryWriter(tensorboard_pathname / 'val')}
    return writers


#############
## LOGBOOK ##
#############

def Logbook_Initialization(dim, logbook_pathname, logbook_params, model_params):

    logbook_pathname.mkdir(exist_ok=True)

    if (dim == '2d'):

        logbook_runparams = {

            ## Run params
            'NameRun': logbook_params[0], 
            'PlanesData': logbook_params[1],
            'RegionsData': logbook_params[2],
            
            ## Training Params
            'num_in_channels': logbook_params[3], 
            'batch_size': logbook_params[4],
            'learn_rate': logbook_params[5],         
            'LR_decay': logbook_params[6],
            
            ## Model UNet
            'dimension': model_params[0],
            'num_in_channels': model_params[1],
            'features_main': model_params[2],
            'features_skip': model_params[3],
            'conv_kernel_size': model_params[4], 
            'down_mode': model_params[5],
            'up_mode': model_params[6],
            'activation': model_params[7],
            'residual_connection': model_params[8]}

    torch.save(logbook_runparams, logbook_pathname / '{}.pt'.format(logbook_params[0]))

    return 


#######################
## PLOTTING FUNCTION ##
#######################

def PlotComparison(LR_Img, DL_Img, HR_Img, Norm_Factor, FigTitle):
    
    matplotlib.rcParams.update({'font.size': 16})
    
    LR_Img = LR_Img.cpu().detach().numpy() 
    DL_Img = DL_Img.cpu().detach().numpy()
    HR_Img = HR_Img.cpu().detach().numpy()

    # LR_Img, _ = Mean_Normalisation(LR_Img, Norm_Factor, inverse=True)
    # DL_Img, _ = Mean_Normalisation(DL_Img, Norm_Factor, inverse=True)
    # HR_Img, _ = Mean_Normalisation(HR_Img, Norm_Factor, inverse=True)

    maxValue_LR, meanValue_LR = np.round(np.max(LR_Img)), np.round(np.mean(LR_Img))
    maxValue_DL, meanValue_DL = np.round(np.max(DL_Img)), np.round(np.mean(DL_Img))
    maxValue_HR, meanValue_HR = np.round(np.max(HR_Img)), np.round(np.mean(HR_Img))

    
    ## Create Figure 

    fig, ax = plt.subplots(ncols=3, figsize=(24,10))
    
    im0 = ax[0].imshow(LR_Img, vmin=0, vmax=1, cmap='gray')
    ax[0].set_title('Input: Low Resolution')
    ax[0].grid(False)
    ax[0].invert_xaxis()
    ax[0].invert_yaxis()
    ax[0].set_xlabel('Max = {} \n Mean = {}'.format(maxValue_LR, meanValue_LR))
    ax[0].get_yaxis().set_visible(False)
    fig.colorbar(im0, ax=ax[0])
            
    im1 = ax[1].imshow(DL_Img, vmin=0, vmax=1, cmap='gray')
    ax[1].set_title('DL Image')
    ax[1].grid(False)
    ax[1].invert_xaxis()
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Max = {} \n Mean = {}'.format(maxValue_DL, meanValue_DL))
    ax[1].get_yaxis().set_visible(False)
    fig.colorbar(im1, ax=ax[1])
            
    im2 = ax[2].imshow(HR_Img, vmin=0, vmax=1, cmap='gray')
    ax[2].set_title('Target: High Resolution')
    ax[2].grid(False)
    ax[2].invert_xaxis()
    ax[2].invert_yaxis()
    ax[2].set_xlabel('Max = {} \n Mean = {}'.format(maxValue_HR, meanValue_HR))
    ax[2].get_yaxis().set_visible(False)
    fig.colorbar(im2, ax=ax[2])
    
    fig.suptitle(FigTitle) 
    #plt.show()
    
    return fig


def Intermediate_Visualization(batch, LR_Img, DL_Img, HR_Img, EpochNumber, num_in_channels, ShowSlices, TensorboardWriter, ShowSubject, Path_FigSave):

    for idx, (Subject, SliceIndex, Plane, Norm_Factor) in enumerate(zip(batch['Subject'], batch['SliceIndex'], batch['Plane'], batch['LR_NormFactor'])):

        if (idx in ShowSlices) and (Subject in ShowSubject):
                
            name = '{}_{}_{}_epoch_{}'.format(Subject, Plane, SliceIndex, EpochNumber)
                
            fig = PlotComparison(
                        LR_Img = LR_Img[idx][num_in_channels//2], 
                        DL_Img = DL_Img[idx][0], 
                        HR_Img = HR_Img[idx][0],
                        Norm_Factor = Norm_Factor,
                        FigTitle = name) 
                
            TensorboardWriter.add_figure(name, fig, EpochNumber)

            if (Path_FigSave != ''):
                Epoch_Path_FigSave = '{}/epoch_{}/{}'.format(Path_FigSave, str(EpochNumber), name)
                Epoch_Path_FigSave = Path(Epoch_Path_FigSave)
                Epoch_Path_FigSave.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(Epoch_Path_FigSave, bbox_inches='tight', dpi=300)
            
            plt.close(fig)

    return

def ssim_loss(img1, img2):
    return 1 - ssim(img1, img2)

class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.8):
        """
        Hybrid loss function combining MSE and SSIM.
        Args:
            alpha (float): Weight for MSE loss (default = 0.8)
        """
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.mse = torch.nn.MSELoss()

    def forward(self, img1, img2):
        mse_loss = self.mse(img1, img2)
        ssim_value = ssim_loss(img1, img2)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_value  # (1 - SSIM)
    
def Power(x):
    s = np.sum(x**2)
    return s/x.size

def compute_SNR2(LR, HR):
    # this function is only applicable for groundtruth or model output highres images! other inputs will give incorrect outputs!
    #normalise = lambda x: (x - np.mean(x)) / np.std(x)
    normalise = lambda x: (x - x.mean()) / x.std()
    HR_norm = normalise(HR)
    LR_norm = normalise(LR)
    LR_int = scipy.ndimage.zoom(LR_norm, zoom=2, order=0)
    return Power(HR_norm) /  Power(LR_int - HR_norm)

def compute_SNR(LR, HR):
    # Ensure both tensors are on the same device
    device = LR.device
    HR = HR.to(device)

    # Resize HR to match LR if needed
    if HR.shape != LR.shape:
        HR = F.interpolate(HR, size=LR.shape[2:], mode='bilinear', align_corners=False)

    # Normalize both to [0, 1]
    HR_min, HR_max = HR.amin(dim=(1, 2, 3), keepdim=True), HR.amax(dim=(1, 2, 3), keepdim=True)
    LR_min, LR_max = LR.amin(dim=(1, 2, 3), keepdim=True), LR.amax(dim=(1, 2, 3), keepdim=True)

    HR_norm = (HR - HR_min) / (HR_max - HR_min + 1e-8)
    LR_norm = (LR - LR_min) / (LR_max - LR_min + 1e-8)

    # Compute noise and SNR
    noise = HR_norm - LR_norm
    signal_power = torch.mean(HR_norm ** 2, dim=(1, 2, 3))
    noise_power = torch.mean(noise ** 2, dim=(1, 2, 3))
    
    snr_batch = 10 * torch.log10(signal_power / (noise_power + 1e-8))

    # Return average SNR over batch
    return snr_batch.mean().item()