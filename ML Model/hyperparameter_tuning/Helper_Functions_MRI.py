import shutil
import torch
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
import torch.nn.functional as F

import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime

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
            #'LR_decay': logbook_params[6],
            
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
    
    matplotlib.rcParams.update({'font.size': 14})
    
    LR_Img = LR_Img.cpu().detach().numpy()
    DL_Img = DL_Img.cpu().detach().numpy()
    HR_Img = HR_Img.cpu().detach().numpy()

    # Calculate max and mean for the images
    maxValue_LR, meanValue_LR = np.round(np.max(LR_Img)), np.round(np.mean(LR_Img))
    maxValue_DL, meanValue_DL = np.round(np.max(DL_Img)), np.round(np.mean(DL_Img))
    maxValue_HR, meanValue_HR = np.round(np.max(HR_Img)), np.round(np.mean(HR_Img))

    # Compute the residuals
    residual_lr = HR_Img - LR_Img
    residual_dl = DL_Img - HR_Img

    # Calculate max and mean for residuals
    maxValue_residual_lr, meanValue_residual_lr = np.round(np.max(residual_lr)), np.round(np.mean(residual_lr))
    maxValue_residual_dl, meanValue_residual_dl = np.round(np.max(residual_dl)), np.round(np.mean(residual_dl))

    # Create the figure and plot
    fig, ax = plt.subplots(ncols=5, figsize=(25, 8))
    titles = ['Low Resolution', 'DL Output', 'High Resolution',
              'HR - LR', 'DL - HR']
    images = [LR_Img, DL_Img, HR_Img, residual_lr, residual_dl]

    # Plot the images and add the statistics as x-axis labels
    for i in range(5):
        im = ax[i].imshow(images[i], cmap='gray')
        #ax[i].axis('off')
        ax[i].tick_params(left=False, bottom=False, labelleft=False)  # Hide ticks and y-labels only

        # Set the title for each image
        ax[i].set_title(titles[i])

        # Set the x-axis label with max and mean values
        if i == 0:  # Low Resolution Image
            ax[i].set_xlabel(f'Max = {maxValue_LR} \nMean = {meanValue_LR}')
        elif i == 1:  # DL Image
            ax[i].set_xlabel(f'Max = {maxValue_DL} \nMean = {meanValue_DL}')
        elif i == 2:  # High Resolution Image
            ax[i].set_xlabel(f'Max = {maxValue_HR} \nMean = {meanValue_HR}')
        elif i == 3:  # Residual: HR - LR
            ax[i].set_xlabel(f'Max = {maxValue_residual_lr} \nMean = {meanValue_residual_lr}')
        elif i == 4:  # Residual: DL - HR
            ax[i].set_xlabel(f'Max = {maxValue_residual_dl} \nMean = {meanValue_residual_dl}')

        # Add colorbar to each image
        fig.colorbar(im, ax=ax[i], shrink=0.6)

    # Set the super title for the figure
    fig.suptitle(FigTitle)

    # Adjust layout for better spacing
    plt.tight_layout()

    return fig, DL_Img  # return DL_Img for saving as DICOM

def Intermediate_Visualization(batch, LR_Img, DL_Img, HR_Img, EpochNumber, num_in_channels, ShowSlices, TensorboardWriter, ShowSubject, Path_FigSave):

    for idx, (Subject, SliceIndex, Plane, Norm_Factor) in enumerate(zip(batch['Subject'], batch['SliceIndex'], batch['Plane'], batch['LR_NormFactor'])):

        if (idx in ShowSlices) and (Subject in ShowSubject):
                
            name = '{}_{}_{}_epoch_{}'.format(Subject, Plane, SliceIndex, EpochNumber)
                
            fig, dl_img_np = PlotComparison(LR_Img=LR_Img[idx][num_in_channels//2],DL_Img=DL_Img[idx][0],HR_Img=HR_Img[idx][0],Norm_Factor=Norm_Factor,FigTitle=name)
                        
            # Create save path
            dicom_save_dir = Path(Path_FigSave) / 'DICOM_DL' / f'epoch_{EpochNumber}'
            dicom_save_dir.mkdir(parents=True, exist_ok=True)

            # DICOM metadata
            file_meta = pydicom.Dataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

            dt = datetime.datetime.now()
            dicom_filename = f"{Subject}_{Plane}_{SliceIndex}_DL.dcm"
            filepath = dicom_save_dir / dicom_filename

            ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)
            ds.Modality = 'OT'
            ds.ContentDate = dt.strftime('%Y%m%d')
            ds.ContentTime = dt.strftime('%H%M%S.%f')

            # Add some minimal required DICOM attributes
            ds.PatientName = str(Subject)
            ds.PatientID = str(Subject)
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

            # DL_Img is normalized [0,1] -> convert to 8-bit for DICOM
            DL_Img_uint8 = (dl_img_np * 255).astype(np.uint8)
            ds.Rows, ds.Columns = DL_Img_uint8.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.PixelData = DL_Img_uint8.tobytes()

            ds.save_as(filepath)
            
            # Residuals for DICOM saving
            residual_hr_lr = HR_Img[idx][0].cpu().numpy() - LR_Img[idx][num_in_channels // 2].cpu().numpy()
            #residual_hr_dl = HR_Img[idx][0].cpu().numpy() - DL_Img[idx][0].cpu().numpy()
            #residual_hr_dl = HR_Img[idx][0].detach().cpu().numpy() - DL_Img[idx][0].detach().cpu().numpy()
            residual_hr_dl = DL_Img[idx][0].detach().cpu().numpy() - HR_Img[idx][0].detach().cpu().numpy()

            residual_diff = residual_hr_lr - residual_hr_dl
            # Save residuals
            save_residual_as_dicom(residual_hr_lr, Subject, Plane, SliceIndex, EpochNumber, 'hr_lr', dicom_save_dir)
            save_residual_as_dicom(residual_hr_dl, Subject, Plane, SliceIndex, EpochNumber, 'hr_dl', dicom_save_dir)
            save_residual_as_dicom(residual_diff, Subject, Plane, SliceIndex, EpochNumber, 'diff', dicom_save_dir)

                
            TensorboardWriter.add_figure(name, fig, EpochNumber)

            if (Path_FigSave != ''):
                Epoch_Path_FigSave = '{}/epoch_{}/{}'.format(Path_FigSave, str(EpochNumber), name)
                Epoch_Path_FigSave = Path(Epoch_Path_FigSave)
                Epoch_Path_FigSave.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(Epoch_Path_FigSave, bbox_inches='tight', dpi=300)
            
            plt.close(fig)

    return
    
def save_residual_as_dicom(image_array, subject, plane, slice_index, epoch_number, residual_type, save_root):
    image_array = np.clip(image_array, 0, 1)
    image_uint8 = (image_array * 255).astype(np.uint8)

    # Setup DICOM metadata
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    dt = datetime.datetime.now()
    dicom_filename = f"{subject}_{plane}_{slice_index}_{residual_type}.dcm"
    save_dir = Path(save_root) / f"residual_{residual_type}" / f"epoch_{epoch_number}"
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / dicom_filename

    ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = 'OT'
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S.%f')

    ds.PatientName = str(subject)
    ds.PatientID = str(subject)
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.InstanceNumber = int(slice_index)
    ds.Rows, ds.Columns = image_uint8.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = image_uint8.tobytes()

    ds.save_as(filepath)
    
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

# Define SSIM function
def _gaussian_window(channels, kernel_size=11, sigma=1.5, device='cpu'):
    coords = torch.arange(kernel_size, device=device) - kernel_size//2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1) @ (g / g.sum()).unsqueeze(0)
    return g.expand(channels, 1, kernel_size, kernel_size)

def ssim_index(x, y, data_range=1.0, window_size=11, sigma=1.5, K=(0.01, 0.03)):
    """
    x, y: [B, C, H, W], float tensors, in [0, data_range]
    returns: scalar SSIM index averaged over batch and channels
    """
    C = x.shape[1]
    win = _gaussian_window(C, window_size, sigma, device=x.device)

    # means
    mu_x = F.conv2d(x, win, padding=window_size//2, groups=C)
    mu_y = F.conv2d(y, win, padding=window_size//2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # variances / covariances
    sigma_x2 = F.conv2d(x * x, win, padding=window_size//2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=window_size//2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=window_size//2, groups=C) - mu_xy

    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den

    return ssim_map.mean()

def ssim_loss(img1, img2):
    return 1- ssim_index(img1,img2)

# Define Hybrid Loss Function
class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.3):
        """
        Hybrid loss function combining MSE and SSIM.
        Args:
            alpha (float): Weight for MSE loss (default = 0.😎
        """
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.mse = torch.nn.MSELoss()

    def forward(self, img1, img2):
        mse_loss = self.mse(img1, img2)
        ssim_value = ssim_loss(img1, img2)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_value  # (1 - SSIM)