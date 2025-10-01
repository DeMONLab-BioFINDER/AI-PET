import torch, nibabel as nib
import numpy as np
from monai.visualize.occlusion_sensitivity import OcclusionSensitivity

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
        if n_total== 0
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