import torch, nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Orientation, Resize, ScaleIntensityRangePercentiles
from monai.visualize.occlusion_sensitivity import OcclusionSensitivity


@torch.no_grad()
def explain_one_occlusion(model, nii_path, out_nifti_path,
                          mask_size=(16,16,16), overlap=0.6, activate=True, device="cuda"):
    model.eval().to(device)
    vol = get_transforms()(nii_path) ## !!!remains to be solved
    vol = torch.as_tensor(vol, dtype=torch.float32, device=device).unsqueeze(0)  # [1,1,D,H,W]
    affine = nib.load(nii_path).affine

    occ = OcclusionSensitivity(nn_module=model, mask_size=mask_size, n_batch=16, overlap=overlap, activate=activate)
    occ_map, _ = occ(x=vol)                     # [1,1,D,H,W,1] typically
    heat = (-occ_map).float().squeeze().cpu().numpy()  # [D,H,W], larger = more important
    nib.save(nib.Nifti1Image(heat, affine), out_nifti_path)


@torch.no_grad()
def group_occlusion_maps(model, loader, out_dir: str, *,                     # loader yields (x, y_cls, y_reg, ID) or (x, y_cls, y_reg, ID, pet_path)
                         mask_size=(16,16,16), overlap=0.6, activate=True,   # True for classification, False for pure regression
                         device="cuda", assume_model_space_affine="auto"):   # "auto" -> use affine from first sample’s original NIfTI if available, else identity
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
    model.eval().to(device)

    occ = OcclusionSensitivity(
        nn_module=model, mask_size=mask_size, n_batch=16, overlap=overlap, activate=activate
    )

    all_sums   = None
    all_sqsum  = None
    n_total    = 0

    pos_sums, pos_n = None, 0
    neg_sums, neg_n = None, 0

    ref_affine = None

    for batch in tqdm(loader, desc="Group occlusion"):
        # support both signatures
        if len(batch) == 4:
            x, y_cls, y_reg, pid = batch
            pet_paths = [None] * x.shape[0]
        else:
            x, y_cls, y_reg, pid, pet_paths = batch

        x = x.to(device, non_blocking=True)   # [B,1,D,H,W]

        occ_map, _ = occ(x=x)                 # [B,1,D,H,W,1] (typically)
        heat = (-occ_map).float().squeeze(4).squeeze(1)  # [B,D,H,W]
        heat = heat.clamp(min=0)              # optional: keep positive importance
        h = heat.cpu().numpy()                # [B,D,H,W]

        # set reference affine once
        if ref_affine is None:
            if assume_model_space_affine == "auto" and pet_paths and pet_paths[0]:
                try:
                    ref_affine = nib.load(pet_paths[0]).affine
                except Exception:
                    ref_affine = np.eye(4, dtype=np.float32)
            else:
                ref_affine = np.eye(4, dtype=np.float32)

        # accumulate global mean/std
        if all_sums is None:
            shp = h.shape[1:]                 # [D,H,W]
            all_sums  = np.zeros(shp, np.float64)
            all_sqsum = np.zeros(shp, np.float64)
        all_sums  += h.sum(axis=0)
        all_sqsum += (h**2).sum(axis=0)
        n_total   += h.shape[0]

        # if binary classification and y_cls provided, split by class
        if y_cls is not None and not torch.isnan(y_cls).all():
            y = y_cls.squeeze(1).cpu().numpy().astype(int)
            if np.unique(y).size <= 2:  # treat as binary
                pos_mask = (y == y.max())
                neg_mask = ~pos_mask
                if pos_mask.any():
                    pos_sums = h[pos_mask].sum(axis=0) if pos_sums is None else pos_sums + h[pos_mask].sum(axis=0)
                    pos_n   += int(pos_mask.sum())
                if neg_mask.any():
                    neg_sums = h[neg_mask].sum(axis=0) if neg_sums is None else neg_sums + h[neg_mask].sum(axis=0)
                    neg_n   += int(neg_mask.sum())

    # finalize
    if n_total == 0:
        raise RuntimeError("No samples seen in loader.")

    mean_map = (all_sums / n_total).astype(np.float32)
    var_map  = (all_sqsum / n_total - mean_map**2).clip(min=0.0)
    std_map  = np.sqrt(var_map).astype(np.float32)

    nib.save(nib.Nifti1Image(mean_map, ref_affine), os.path.join(out_dir, "group_mean.nii.gz"))
    nib.save(nib.Nifti1Image(std_map,  ref_affine), os.path.join(out_dir, "group_std.nii.gz"))

    # class conditionals (if available)
    if pos_sums is not None and pos_n > 0:
        pos_mean = (pos_sums / pos_n).astype(np.float32)
        nib.save(nib.Nifti1Image(pos_mean, ref_affine), os.path.join(out_dir, "group_pos_mean.nii.gz"))
    if neg_sums is not None and neg_n > 0:
        neg_mean = (neg_sums / neg_n).astype(np.float32)
        nib.save(nib.Nifti1Image(neg_mean, ref_affine), os.path.join(out_dir, "group_neg_mean.nii.gz"))
    if pos_sums is not None and neg_sums is not None and pos_n > 0 and neg_n > 0:
        contrast = (pos_sums/pos_n - neg_sums/neg_n).astype(np.float32)
        nib.save(nib.Nifti1Image(contrast, ref_affine), os.path.join(out_dir, "group_contrast_pos_minus_neg.nii.gz"))