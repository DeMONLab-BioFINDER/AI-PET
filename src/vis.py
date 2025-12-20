import os
import torch
import numpy as np
import torch.nn as nn
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage import gaussian_gradient_magnitude
from monai.visualize.class_activation_maps import GradCAM
from monai.visualize.occlusion_sensitivity import OcclusionSensitivity


def run_visualization(model, loader, device, output_path, vis_name="gradcam",
                      vis_kwargs=None, affine_fn=None, # optional: get affine per subject
                      vis_norm="zscore", skull_suppress_pct=20,      # PET intensity percentile
                      use_pseudo_surface=False, pseudo_surface_sigma=1.0, pseudo_surface_pct=90):
    """
    Run volumetric explainability visualizations for 3D PET models.

    For each subject, this function computes a voxel-wise attribution map
    (e.g., Grad-CAM or occlusion) in model input space, suppresses background
    and skull-driven activations using PET intensity thresholds, and saves
    subject-level and group-average results.

    Outputs include:
      - Volumetric NIfTI attribution maps
      - Maximum-intensity-projection (MIP) overlays on PET
      - Optional PET-derived pseudo-surface visualizations (approximate,
        gradient-based; not anatomical surfaces)

    All visualizations are performed in the model input (preprocessed PET)
    space without registration to standard templates.
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
    for x, _, _, extra, pid in loader:
        x = x.to(device)
        extra = extra.to(device)
        
        for b in range(x.shape[0]):
            xb = x[b:b+1] # [1, C, D, H, W]
            extrab = extra[b:b+1] if extra is not None else None
            pid_b = pid[b]

            # visualize
            heat = vis_fn(model=model, x=xb, extra=extrab, **vis_kwargs)  # [D,H,W]
            assert heat.ndim == 3
            img_np = xb[0, 0].detach().cpu().numpy()  # (D,H,W)

            # PET-based brain & skull suppression
            brain_mask = img_np > np.percentile(img_np[img_np > 0], skull_suppress_pct)
            heat = heat * brain_mask

            if vis_norm is not None: heat = normalize_cam(heat, vis_norm)

            # save to NIFTI
            affine = affine_fn(pid_b) if affine_fn else np.eye(4)
            nib.save(nib.Nifti1Image(heat.astype(np.float32), affine),
                     os.path.join(vis_dir, f"{pid_b}_{vis_name}.nii.gz"))

            # MIP visualization
            save_mip_png(img_np, heat, os.path.join(vis_dir, f"{pid_b}_{vis_name}_mip.png"))
            
            # Pseudo-surface visualization (no T1w available, no affine)
            if use_pseudo_surface:
                shell = pet_pseudo_surface(img_np, sigma=pseudo_surface_sigma, pct=pseudo_surface_pct)
                shell_cam = heat * shell
                save_mip_png(img_np, shell_cam, os.path.join(vis_dir, f"{pid_b}_{vis_name}_pseudo_surface.png"))

            vis_sum = heat if vis_sum is None else vis_sum + heat
            n += 1

    # group-level average
    vis_avg = vis_sum / n
    nib.save(nib.Nifti1Image(vis_avg.astype(np.float32), np.eye(4)),
             os.path.join(vis_dir, f"group_average_{vis_name}.nii.gz"))
    save_mip_png(img_np, vis_avg, os.path.join(vis_dir, f"group_average_{vis_name}_mip.png"))

    if use_pseudo_surface:
        shell = pet_pseudo_surface(img_np, pseudo_surface_sigma, pseudo_surface_pct)
        save_mip_png(img_np, vis_avg * shell, os.path.join(vis_dir, f"group_average_{vis_name}_pseudo_surface.png"))


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


def normalize_cam(cam, mode="zscore", eps=1e-6):
    mask = cam > 0
    if not np.any(mask):
        return cam

    vals = cam[mask]

    if mode == "zscore":
        cam = (cam - vals.mean()) / (vals.std() + eps)
        cam[cam < 0] = 0
    elif mode == "percentile":
        lo, hi = np.percentile(vals, [80, 99])
        cam = np.clip((cam - lo) / (hi - lo + eps), 0, 1)

    return cam


def save_mip_png(img, cam, out_png, cmap="hot"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    views = [(img.max(0), cam.max(0), "Axial"),
             (img.max(1), cam.max(1), "Coronal"),
             (img.max(2), cam.max(2), "Sagittal")]

    for ax, (im, hm, title) in zip(axes, views):
        ax.imshow(im, cmap="gray")
        ax.imshow(hm, cmap=cmap, alpha=0.6)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def pet_pseudo_surface(img, sigma=1.0, pct=90):
    """
    PET-derived cortical shell proxy using gradient magnitude.
    """
    grad = gaussian_gradient_magnitude(img, sigma=sigma)
    thr = np.percentile(grad[grad > 0], pct)
    shell = grad >= thr
    return shell.astype(np.float32)
