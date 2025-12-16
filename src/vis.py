import os
import torch
import numpy as np
import torch.nn as nn
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from monai.visualize.occlusion_sensitivity import OcclusionSensitivity
from monai.visualize.class_activation_maps import GradCAM


def run_visualization(model, loader, device, output_path, vis_name="gradcam",
                      vis_kwargs=None, affine_fn=None): # optional: get affine per subject
    """
    Generic runner for per-case + group-level visualization maps.
    """
    VIS_REGISTRY = {
    "gradcam": gradcam_vis_fn,
    "occlusion": occlusion_vis_fn,
    }
    vis_fn = VIS_REGISTRY[vis_name]
    vis_kwargs = vis_kwargs or {}

    vis_dir = os.path.join(output_path, 'visualization', vis_name)
    os.makedirs(vis_dir, exist_ok=True)

    vis_sum = None
    n = 0
    for i, (x, _, _, extra, pid) in enumerate(loader):
        x = x.to(device)
        extra = extra.to(device)
        
        B = x.shape[0]

        for b in range(B):
            xb = x[b:b+1]                          # [1, C, D, H, W]
            extrab = extra[b:b+1] if extra is not None else None
            pid_b = pid[b]

            heat = vis_fn(model=model, x=xb, extra=extrab, **vis_kwargs)  # [D,H,W]
            assert heat.ndim == 3
            mask = (xb != 0).any(dim=1)[0].float().cpu().numpy()
            heat = heat * mask  # mask out background

            affine = affine_fn(pid_b) if affine_fn else np.eye(4)

            # per-case outputs
            nii_path = os.path.join(vis_dir, f"{pid_b}_{vis_name}.nii.gz")
            png_path = os.path.join(vis_dir, f"{pid_b}_{vis_name}.png")

            nib.save(nib.Nifti1Image(heat.astype(np.float32), affine), nii_path)
            save_png_slices(heat, png_path)

            vis_sum = heat if vis_sum is None else vis_sum + heat
            n += 1

    # group-level average
    vis_avg = vis_sum / n
    nib.save(nib.Nifti1Image(vis_avg.astype(np.float32), np.eye(4)),
             os.path.join(vis_dir, f"group_average_{vis_name}.nii.gz"))
    save_png_slices(vis_avg,os.path.join(vis_dir, f"group_average_{vis_name}.png"))


def gradcam_vis_fn(model, x, extra, target_layer=None, normalize=True, upsample_mode="trilinear"):
    model.eval()
    
    def fwd(x_):
        return model(x_, extra=extra)
    
    # pick last Conv3d
    if target_layer is None:
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv3d):
                target_layer = name
    if target_layer is None: raise ValueError("No Conv3d layer found")

    cam = GradCAM(model, target_layers=[target_layer])

    with torch.amp.autocast("cuda", enabled=False):
        cam_map = cam(x, extra=extra)  # calls fwd(x) # class_idx – Default to None (computing class_idx from argmax); layer_idx – index of the target layer if there are multiple target layers

    # post-proc
    if cam_map.shape[-3:] != x.shape[-3:]: cam_map = F.interpolate(cam_map, x.shape[-3:], mode=upsample_mode) # upsample
    cam_np = cam_map[0, 0].cpu().numpy()
    if normalize: # normalize
        vmin, vmax = cam_np.min(), cam_np.max()
        cam_np = (cam_np - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(cam_np)

    return cam_np


@torch.no_grad()
def occlusion_vis_fn(model, x, extra, **kwargs):
    model.eval()

    def fwd(x_):
        return model(x_, extra=extra)

    occ = OcclusionSensitivity(nn_module=model, mask_size=kwargs.get("mask_size", (16,16,16)),
                               n_batch=kwargs.get("n_batch", 16), overlap=kwargs.get("overlap", 0.6),
                               activate=kwargs.get("activate", True),)
    occ_map, _ = occ(x, extra=extra)  # [1,1,D,H,W,1] # calls fwd(x)
    heat = (-occ_map).float().squeeze().cpu().numpy()
    return heat



def save_png_slices(vis_np, img_np, out_path, k=5, cmap="hot", alpha=0.4):
    """
    Save a grid of k slices for axial / coronal / sagittal views.
    """
    assert vis_np.shape == img_np.shape
    assert k % 2 == 1, "k must be odd to include center slice"

    D, H, W = vis_np.shape
    z_idxs = _sample_slices(D, k)
    y_idxs = _sample_slices(H, k)
    x_idxs = _sample_slices(W, k)

    fig, axes = plt.subplots(k, 3, figsize=(9, 3 * k))

    for i in range(k):
        axes[i, 0].imshow(img_np[z_idxs[i], :, :], cmap="gray")
        axes[i, 0].imshow(vis_np[z_idxs[i], :, :], cmap=cmap, alpha=alpha)
        axes[i, 0].set_ylabel(f"z={z_idxs[i]}")
        axes[i, 0].set_title("Axial" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_np[:, y_idxs[i], :], cmap="gray")
        axes[i, 1].imshow(vis_np[:, y_idxs[i], :], cmap=cmap, alpha=alpha)
        axes[i, 1].set_title("Coronal" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_np[:, :, x_idxs[i]], cmap="gray")
        axes[i, 2].imshow(vis_np[:, :, x_idxs[i]], cmap=cmap, alpha=alpha)
        axes[i, 2].set_title("Sagittal" if i == 0 else "")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _sample_slices(n, k=5):
    """
    Sample k slice indices from range [0, n-1], always including center.
    """
    assert k % 2 == 1, "k must be odd to include center"
    center = n // 2
    half = k // 2

    # evenly spaced offsets
    max_offset = min(center, n - center - 1)
    offsets = np.linspace(-max_offset, max_offset, k).astype(int)

    idxs = np.clip(center + offsets, 0, n - 1)
    return idxs