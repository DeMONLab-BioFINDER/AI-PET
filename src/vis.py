import os
import torch
import numpy as np
import torch.nn as nn
import nibabel as nib
import torch.nn.functional as F
from monai.visualize.occlusion_sensitivity import OcclusionSensitivity
from monai.visualize.class_activation_maps import GradCAM

from src.data import get_transforms

'''
Only run the file if has images under the path
'''
@torch.no_grad()
def explain_one_occlusion(model, nii_path, out_nifti_path=None, return_affine=False,
                          mask_size=(16,16,16), overlap=0.6, activate=True, device="cuda"):
    model.eval().to(device)
    vol = get_transforms()(nii_path) ## !!!remains to be solved
    vol = torch.as_tensor(vol, dtype=torch.float32, device=device).unsqueeze(0)  # [1,1,D,H,W]
    if return_affine or out_nifti_path: affine = nib.load(nii_path).affine

    occ = OcclusionSensitivity(nn_module=model, mask_size=mask_size, n_batch=16, overlap=overlap, activate=activate)
    occ_map, _ = occ(x=vol)                     # [1,1,D,H,W,1] typically
    # heat = heat.clamp(min=0)              # optional: keep positive importance
    heat = (-occ_map).float().squeeze().cpu().numpy()  # [D,H,W], larger = more important

    if out_nifti_path: nib.save(nib.Nifti1Image(heat, affine), out_nifti_path)

    if return_affine:
        return heat, affine
    else:
        return heat


@torch.no_grad()
def group_occlusion_maps(model, paths, out_dir: str, *,                     
                         mask_size=(16,16,16), overlap=0.6, activate=True, device="cuda"):  # True for classification, False for pure regression
    """
    Saves:
      - group_mean.nii.gz (mean heat across all cases)
      - group_std.nii.gz  (std  heat across all cases)
    If binary classification (y_cls present with two values):
      - group_pos_mean.nii.gz
      - group_neg_mean.nii.gz
      - group_contrast_pos_minus_neg.nii.gz
    """
    os.makedirs(out_dir, exist_ok=True)

    all_sums, all_sqsum, n_total = None, None, 0
    for nii_path in paths:
        if n_total== 0:
            heat, ref_affine = explain_one_occlusion(model, nii_path, return_affine=True, mask_size=mask_size, overlap=overlap, activate=activate, device=device) # [D,H,W]
        else:
            heat, _ = explain_one_occlusion(model, nii_path, return_affine=False, mask_size=mask_size, overlap=overlap, activate=activate, device=device) # [D,H,W]

        # accumulate global mean/std
        if all_sums is None: # h # [B,D,H,W]
            shp = heat.shape                 # [D,H,W]
            all_sums  = np.zeros(shp, np.float64)
            all_sqsum = np.zeros(shp, np.float64)
        all_sums  += heat #h.sum(axis=0)
        all_sqsum += heat**2 #(h**2).sum(axis=0)
        n_total   += 1 #h.shape[0]

    # finalize
    if n_total == 0: raise RuntimeError("No samples seen in in path.")

    mean_map = (all_sums / n_total).astype(np.float32)
    var_map  = (all_sqsum / n_total - mean_map**2).clip(min=0.0)
    std_map  = np.sqrt(var_map).astype(np.float32)

    nib.save(nib.Nifti1Image(mean_map, ref_affine), os.path.join(out_dir, "group_mean.nii.gz"))
    nib.save(nib.Nifti1Image(std_map,  ref_affine), os.path.join(out_dir, "group_std.nii.gz"))


def grad_cam(model, x, # [1, C, D, H, W]
             target_layer: str | None = None,      # auto-pick last Conv3d if None
             class_idx: int | None = None,         # auto from model output if None
             upsample_mode: str = "trilinear", align_corners: bool = False, 
             normalize: bool = True) -> np.ndarray: # min-max to [0,1]
    """
    Compute Grad-CAM (or Grad-CAM++) for a 3D model that outputs logits.
    Returns a NumPy array of shape (D, H, W).
    """
    assert x.ndim == 5 and x.size(0) == 1, "x must be a single 3D volume: [1, C, D, H, W]"
    model.eval()

    # --- pick target layer: last Conv3d if not provided
    if target_layer is None:
        last = None
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv3d):
                last = name
        if last is None: raise ValueError("No Conv3d layer found; specify target_layer explicitly.")
        target_layer = last
    #print(f"Using target layer: {target_layer}")

    # --- forward once to get logits and decide class_idx if needed
    if class_idx is None:
        with torch.inference_mode():
            logits = model(x)  # expected shapes: [B,1] or [B,2] (or [B,C])
        if logits.ndim == 1: logits = logits.unsqueeze(1)    # [B] -> make it [B,1]
        n_classes = logits.shape[1]
        class_idx = 0 if n_classes == 1 else int(torch.argmin(logits, dim=1).item())  # predicted class # argmin because the pos and neg flag is revert in trainnig
    else: # sanity check
        if not (0 <= int(class_idx) < n_classes):
            raise ValueError(f"class_idx {class_idx} out of range 0..{n_classes-1}")
    
    cam = GradCAM(nn_module=model, target_layers=target_layer)
    # --- compute CAM (disable AMP for stable grads)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        cam_map = cam(x, class_idx=int(class_idx))   # [B, 1, d, h, w]
    cam_map = cam_map.detach()

    # --- upsample to input spatial size if needed
    in_spatial = x.shape[-3:]
    cam_spatial = cam_map.shape[-3:]
    if cam_spatial != in_spatial:
        print('Upsample to input space')
        cam_map = F.interpolate(cam_map, size=in_spatial, mode=upsample_mode, align_corners=align_corners)

    cam_np = cam_map[0, 0].cpu().float().numpy()  # (D, H, W)

    # --- normalize to [0,1] if requested
    if normalize:
        vmin, vmax = np.nanmin(cam_np), np.nanmax(cam_np)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            cam_np = (cam_np - vmin) / (vmax - vmin)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

    return cam_np, class_idx, logits.detach().cpu().numpy() # (D, H, W)
