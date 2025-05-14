import torch
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from Without_SR_Module.Slicemanager import Slice_manager
import re
import numpy as np


def _gaussian_window(channels: int, kernel_size: int = 11, sigma: float = 1.5, device: str = 'cpu') -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size // 2)
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g2d = (g1d / g1d.sum()).unsqueeze(1) @ (g1d / g1d.sum()).unsqueeze(0)
    return g2d.expand(channels, 1, kernel_size, kernel_size)


def ssim_index(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
               window_size: int = 11, sigma: float = 1.5,
               K: tuple = (0.01, 0.03), eps: float = 1e-6) -> torch.Tensor:
    if data_range == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)

    B, C, H, W = x.shape
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    win = _gaussian_window(C, window_size, sigma, device=x.device).to(x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy

    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return (num / (den + eps)).mean()


def validate(dl_root: Path, dicom_data_path: Path):
    print(f"DL_ROOT Path: {dl_root}")

    png_files = [f for f in dl_root.iterdir() if f.is_file() and f.suffix.lower() == '.png']
    print(f"Found {len(png_files)} PNG files:")

    records = []
    for f in png_files:
        m = re.match(r'Mouse(\d{2})_([A-Za-z]+)_(\d+)_DL\.png', f.name)
        if not m:
            print(f"Skipping file {f.name} due to unmatched regex")
            continue
        mouse_id = int(m.group(1))
        if mouse_id not in [1, 6, 10, 23]:
            continue
        records.append({
            'path': f,
            'mouse_id': mouse_id,
            'loc': 'THORAX-ABDOMEN',
            'plane': m.group(2),
            'slice': int(m.group(3)),
        })

    slicer = Slice_manager(dicom_data_path)

    results = []
    for rec in records:
        print(f"Evaluating Mouse {rec['mouse_id']} {rec['plane']} slice {rec['slice']}")

        # Load DL image (PNG)
        img = Image.open(rec['path']).convert('L')
        dl = np.array(img).astype(np.float32)

        slicer.set_mouse_id(rec['mouse_id'])
        slicer.set_loc(rec['loc'])
        slicer.set_plane(rec['plane'])
        slicer.set_slice(rec['slice'])

        _, hr = slicer.get_slice()

        dl = torch.from_numpy(dl).unsqueeze(0).unsqueeze(0).float()
        hr = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0).float()

        dl = (dl - dl.min()) / (dl.max() - dl.min() + 1e-8)
        hr = (hr - hr.min()) / (hr.max() - hr.min() + 1e-8)

        if dl.shape != hr.shape:
            dl = F.interpolate(dl, size=hr.shape[-2:], mode='bilinear', align_corners=False)

        ssim = ssim_index(dl, hr, data_range=1.0).item()
        mse = F.mse_loss(dl, hr, reduction='mean').item()

        results.append({
            'mouse': rec['mouse_id'],
            'plane': rec['plane'],
            'slice': rec['slice'],
            'ssim': ssim,
            'mse': mse
        })

    print("\nValidation Results:")
    for r in results:
        print(f"Mouse {r['mouse']} {r['plane']} slice {r['slice']}: SSIM={r['ssim']:.4f}, MSE={r['mse']:.6f}")

    return results


if __name__ == "__main__":
    DL_ROOT = Path('C:\\Users\\daanv\\Ugent\\vop\\PNG_DL_epoch_0')  # Replace with PNG folder
    DICOM_DATA = Path('C:\\Users\\daanv\\Ugent\\vop\\VOP\\Data')
    validate(DL_ROOT, DICOM_DATA)
