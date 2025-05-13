import torch # type: ignore
from pathlib import Path
import scipy.ndimage
import torch.nn.functional as F
import pathlib
import pydicom
from pydicom.uid import ImplicitVRLittleEndian
from Without_SR_Module.Slicemanager import Slice_manager
import re
import random
from pathlib import Path
import numpy as np
import os




def _gaussian_window(channels: int,
                     kernel_size: int = 11,
                     sigma: float = 1.5,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Create a 2D Gaussian kernel of shape [channels, 1, kernel_size, kernel_size].
    Built in float32 so we can cast it to x.dtype later.
    """
    # 1D coordinates centered at zero
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size // 2)
    # 1D Gaussian
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    # outer product => 2D Gaussian
    g2d = (g1d / g1d.sum()).unsqueeze(1) @ (g1d / g1d.sum()).unsqueeze(0)
    # expand to [channels, 1, k, k]
    return g2d.expand(channels, 1, kernel_size, kernel_size)

def ssim_index(x: torch.Tensor,
               y: torch.Tensor,
               data_range: float = 1.0,
               window_size: int = 11,
               sigma: float = 1.5,
               K: tuple = (0.01, 0.03),
               eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the mean SSIM index between x and y.
    x, y: [B, C, H, W], float tensors in [0, data_range].
    Returns a scalar tensor SSIM.
    """
    # If images are constant, SSIM=1
    if data_range == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)

    B, C, H, W = x.shape
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    # Build Gaussian window in float32, then cast to x.dtype
    win = _gaussian_window(C, window_size, sigma, device=x.device).to(x.dtype)
    pad = window_size // 2

    # Local means
    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # Local variances and covariance
    sigma_x2 = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy

    # Clamp tiny negatives from numerical error
    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    # SSIM formula
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    # Mean over all pixels, channels, batch
    return (num / (den + eps)).mean()




#lr1_t = torch.from_numpy(lr1).unsqueeze(0).unsqueeze(0).float()
#hr1_t = torch.from_numpy(hr1).unsqueeze(0).unsqueeze(0).float()

#lr1_up = F.interpolate(
 #   lr1_t,
 #   size=hr1_t.shape[-2:],    
  #  mode='bilinear',          # zijn ook andere opties ma heb nu effe deze gekozen
   # align_corners=False
#)


#ssim_val = ssim_index(lr1_up, hr1_t, data_range=hr1_t.max() - hr1_t.min())
#print(f"SSIM_test = {ssim_val:.4f}")



# ── Validation driver ──────────────────────────────────────────────────────────

def validate(dl_root: Path, dicom_data_path: Path, seed: int = 42):
    # 1) Pick latest epoch folder (by highest number in name)
    epochs = [d for d in dl_root.iterdir() if d.is_dir()]
    def epoch_num(d):
        m = re.search(r'(\d+)', d.name)
        return int(m.group(1)) if m else -1
    latest = max(epochs, key=epoch_num)
    print(f"Evaluating epoch: {latest.name}")

    # 2) Collect DL files recursively
    dl_files = list(latest.glob('**/*_diff.dcm'))

    # 3) Process files
    records = []
    for f in dl_files:
        m = re.match(r'Mouse(\d+)_([A-Za-z]+)_(\w+)-(\w+)_(\d+)_diff\.dcm', f.name)
        if not m: 
            continue
        mouse_id = int(m.group(1))
        if mouse_id not in [1, 6, 10, 23]:  # 👈 filter here
            continue
        records.append({
            'path': f,
            'mouse_id': mouse_id,
            'loc': f'{m.group(3)}-{m.group(4)}',
            'plane': m.group(2),
            'slice': int(m.group(5)),
        })

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(records)

    # 4) Init slicemanager
    slicer = Slice_manager(dicom_data_path)

    # 5) Evaluate all matching records
    results = []
    for rec in records:
        print(f"Evaluating Mouse {rec['mouse_id']} {rec['plane']} slice {rec['slice']}")
        ds = pydicom.dcmread(str(rec['path']), force=True)
        if not hasattr(ds, 'file_meta') or 'TransferSyntaxUID' not in ds.file_meta:
            ds.file_meta = getattr(ds, 'file_meta', pydicom.dataset.FileMetaDataset())
            ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            ds.is_implicit_VR = True
            ds.is_little_endian = True
        dl = ds.pixel_array.astype(np.float32)


        slicer.set_mouse_id(rec['mouse_id'])
        slicer.set_loc(rec['loc'])
        slicer.set_plane(rec['plane'])
        slicer.set_slice(rec['slice'])
        lr, hr = slicer.get_slice()

        # Torch and normalize
        dl = torch.from_numpy(dl).unsqueeze(0).unsqueeze(0).float()
        hr = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0).float()

        # Normalize both
        dl = (dl - dl.min()) / (dl.max() - dl.min() + 1e-8)
        hr = (hr - hr.min()) / (hr.max() - hr.min() + 1e-8)



        if dl.shape != hr.shape:
            dl = F.interpolate(dl, size=hr.shape[-2:], mode='bilinear', align_corners=False)

        ssim = ssim_index(dl, hr, data_range=1.0).item()
        mse  = F.mse_loss(dl, hr, reduction='mean').item()

        results.append({
            'mouse': rec['mouse_id'],
            'plane': rec['plane'],
            'slice': rec['slice'],
            'ssim': ssim,
            'mse': mse
        })

    # 6) Print results
    print("\nValidation Results:")
    for r in results:
        print(f"Mouse {r['mouse']} {r['plane']} slice {r['slice']}: "
              f"SSIM={r['ssim']:.4f}, MSE={r['mse']:.6f}")

    return results


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    #DL_ROOT    = Path('/kyukon/data/gent/vo/000/gvo00006/WalkThroughPET/2425_VOP/Project/Without_SR_Module_Results/IntermediateFigures/DICOM_DL')
    DL_ROOT    = Path('C:\\Users\\daanv\\Ugent\\vop\\DICOM_DL_epoch_0') #change this to own paths
    #DICOM_DATA = Path('/kyukon/data/gent/vo/000/gvo00006/WalkThroughPET/2425_VOP/Project/Data')
    DICOM_DATA = Path('C:\\Users\daanv\\Ugent\\vop\\VOP\Data')
    validate(DL_ROOT, DICOM_DATA)
