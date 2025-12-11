# src/data.py
import os
import re
import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from pathlib import Path
from typing import List, Optional, Union
from torch.utils.data import DataLoader, Dataset
from monai.data import MetaTensor
from monai.transforms import (LoadImage, EnsureChannelFirst, Orientation, Resize,
        ScaleIntensityRangePercentiles, Compose, CropForeground, GaussianSmooth,
        Lambda)

from src.utils import seed_worker
# ------------------------------
# Master table
# ------------------------------
def build_master_table(input_path: str, preproce_method: str, targets: List[str], dataset: str, subjects: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build the table required by the training code, given the custom folder layout.
    Normal mode: discover images + join demographics.
    Cached mode: if preproce_method is empty/None OR cache files exist in cache_dir,
                 load demo.csv only (no disk scans), and return that table.
    In cached mode, adds 'ID' to preserve the order matching data.pt.
    """
    # Detect cached mode
    use_cache = (not preproce_method) or (str(preproce_method).strip() == "")
    if use_cache:
        df = pd.read_csv(Path(input_path) / "demo.csv", index_col=0) # Must have 'ID' column from 0 to len(df)
        print(f"[cache] Loaded demo.csv with {len(df)} rows (no filesystem scan).")
    else:
        pets = find_pet_files(input_path=input_path, preproc_method=preproce_method, allow=subjects)
        if pets.empty:
            raise FileNotFoundError(f"No NIfTI files found under '{input_path}' with preproc suffix  '{preproce_method}' for dataset '{dataset}'.")
        else:
            print(f'Found {pets.shape[0]} scans')
            print(pets.columns, pets.head(3))

        labels = load_participants_labels(input_path, dataset=dataset)
        labels["ID"] = labels["ID"].astype(str).str.strip()
        #df = pd.merge(pets, labels, on="ID", how="inner")
        has_date = "ScanDate" in labels.columns
        if has_date:
            # ensure YYYY-MM-DD strings
            labels["ScanDate"] = pd.to_datetime(labels["ScanDate"]).dt.strftime("%Y-%m-%d")
            pets["ScanDate"] = pets["ScanDate"].astype(str).str.slice(0,10)
            keys = ["ID", "ScanDate"]
        else:
            keys = ["ID"]

        df = pd.merge(pets, labels, on=keys, how="inner", suffixes=("", "_selected"))
        df = df.sort_values(keys + [c for c in ["pet_path"] if c in df.columns]).reset_index(drop=True)

        # Only scans with targets value
        targets_list = [t.strip() for t in targets.split(",") if t.strip()]
        df = df[~df[targets_list].isna().values].reset_index(drop=True)
        print(f'Found {df.shape[0]} scans with demographics for {targets}')

    return df


def find_pet_files(input_path: str, preproc_method: str, allow: Optional[Union[pd.DataFrame, str, Path]] = None) -> pd.DataFrame:
    """
    Discover PET NIfTI files via:
      (A) input_path/PET/**/*_{preproc_method}/*/*/*.nii*
      (B) input_path/<ID>/PET_<ScanDate>_<tracer>/SCANS/*.nii[.gz], tracer in {FBB, FBP}

    If `allow` (DataFrame or CSV) is provided:
      - must contain 'ID'; optional 'ScanDate'
      - if only 'ID': filter by ID (keep all columns combined)
      - if 'ID' + 'ScanDate': inner-join on both (keep all columns combined)
    """
    ipath = Path(input_path)
    rows = []

    # Pattern A
    root_a = ipath / "PET"
    pattern_a = f"**/*{preproc_method}/*/*/*.nii*"
    if root_a.exists():
        for nii in root_a.glob(pattern_a):
            try:
                sid = nii.relative_to(root_a).parts[0]
            except Exception:
                continue
            rows.append({
                "ID": str(sid),
                "pet_path": str(nii).replace('/._','/'),
                "imagefile": nii.name,
                "ScanDate": None,
                "tracer": None,
            })

    # Pattern B -- ADNI data on Berkeley cluster
    # e.g. /116-S-6550/PET_2018-08-29_FTP/SCANS/116-S-6550_AV1451_2018-08-29_P4-6mm_I1600375.nii
    if '.nii' in preproc_method:
        print(f'finding ADNI data /analysis/{preproc_method}')
        method_pat = re.escape(preproc_method) + r'(?:\.gz)?'
        regex_b = re.compile(
            rf'^(?P<ID>[^/]+)/PET_(?P<ScanDate>\d{{4}}-\d{{2}}-\d{{2}})_(?P<tracer>FBB|FBP)/analysis/{re.escape(preproc_method)}$')
        #regex_b = re.compile(
        #    r'^(?P<ID>[^/]+)/PET_(?P<ScanDate>\d{4}-\d{2}-\d{2})_(?P<tracer>FBB|FBP)/analysis/'r'wsuvr_cere[^/]*\.nii(?:\.gz)?$')
        glob_pattern = "*analysis/*.nii*"
    else: # step 4 ADNI data
        print(f'finding ADNI data /SCANS/')
        regex_b = re.compile(
            r'^(?P<ID>[^/]+)/PET_(?P<ScanDate>\d{4}-\d{2}-\d{2})_(?P<tracer>FBB|FBP)/SCANS/[^/]+\.nii(\.gz)?$')
        glob_pattern = "*SCANS/*.nii*"
    print("PATTERN:", regex_b.pattern)
    
    for nii in ipath.rglob(glob_pattern):
        m = regex_b.match(nii.relative_to(ipath).as_posix())
        if not m: 
            continue
        rows.append({
            "ID": m.group("ID").replace('-','_'),
            "pet_path": str(nii),
            "imagefile": nii.name,
            "ScanDate": m.group("ScanDate"),
            "tracer": m.group("tracer"),
        })

    df = pd.DataFrame(rows, columns=["ID", "pet_path", "imagefile", "ScanDate", "tracer"])
    if df.empty:
        return df

    # Deterministic dedupe: keep first path per ID
    df = (df.sort_values(["ID", "pet_path"])
            .groupby("ID", as_index=False, group_keys=False)
            .head(1).reset_index(drop=True))

    # --- Unified filtering / intersection with `allow` ---
    if allow is not None:
        if isinstance(allow, (str, Path)):
            allow_df = pd.read_csv(input_path + allow, index_col=0)
        else:
            allow_df = allow.copy()

        if "ID" not in allow_df.columns:
            raise ValueError("`allow` must contain column 'ID'.")

        # Normalize types
        df["ID"] = df["ID"].astype(str)
        allow_df["ID"] = allow_df["ID"].astype(str)

        has_date = "ScanDate" in allow_df.columns
        if has_date:
            # ensure YYYY-MM-DD strings
            allow_df["ScanDate"] = pd.to_datetime(allow_df["ScanDate"]).dt.strftime("%Y-%m-%d")
            df["ScanDate"] = df["ScanDate"].astype(str).str.slice(0,10)
            keys = ["ID", "ScanDate"]
        else:
            keys = ["ID"]

        df = pd.merge(df, allow_df, on=keys, how="inner", suffixes=("", "_allow"))
        df = df.sort_values(keys + [c for c in ["pet_path"] if c in df.columns]).reset_index(drop=True)

    return df


def load_participants_labels(input_path: str, dataset: Optional[str] = None) -> pd.DataFrame: #cache: Optional[bool] = False
    """
    Load demographics.csv from input_path and return:
    ID, site, visual_read, CL, age, gender
    """
    csv = Path(input_path) / "demographics.csv"
    if not csv.exists(): # try alternative path
        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        csv = Path(os.path.join(proj_path, "data")) / f"demographics_{dataset}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Missing {csv}. Provide columns: ID, site, visual_read, CL, age, gender, ...")
    df = pd.read_csv(csv, index_col=0)
    
    print('loaded participants:', df.shape, df.columns, df.head(3))
    # Ensure required columns are present in the dataframe.
    required = {"ID", "site", "visual_read", "CL", "age", "gender"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    return df

# ------------------------------
# Transforms & Dataset
# ------------------------------
def get_train_val_loaders(train_df, val_df, args):
    # Detect cached mode
    use_cache = (not args.data_suffix) or (str(args.data_suffix).strip() == "")
    if use_cache:
        p = Path(args.input_path) / "data.pt"
        if p.exists():
            tfm = torch.load(p, map_location="cpu", weights_only=True) # torch tensor with shape [S, D, H, W]
            print(f"Reconstructed loaders from data.pt with shape [S, D, H, W].")
        else:
            tfm = args.input_path
            print(f"Reconstructed loaders from data_idx.pt for each scan.")
    else:
        tfm = get_transforms(tuple(args.image_shape))


    dl_tr = get_loader(train_df, tfm, args, batch_size=args.batch_size, augment=True, shuffle=True)
    dl_va = get_loader(val_df, tfm, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False)
    
    return dl_tr, dl_va


def get_loader(df, tfm, args, batch_size, augment=False, shuffle=False):
    g = torch.Generator()
    g.manual_seed(args.seed)

    dataset = PETDataset(df, tfm, args.targets, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        worker_init_fn=seed_worker, generator=g,
                        num_workers=args.num_workers, pin_memory=False)

    return loader


def brain_outer_mask(x):
    """
    Returns mask of same shape: 1 inside brain, 0 outside.
    """
    vol = x.squeeze().cpu().numpy()

    # Simple foreground threshold (very safe)
    thr = vol.mean() * 0.2
    init_mask = vol > thr

    # Largest connected component (outer boundary)
    lbl, n = ndi.label(init_mask) # label connected components, lbl=labels, n=number of components, by default 4-connectivity for 2D, 6-connectivity for 3D
    if n < 1: return torch.ones_like(x)  # fallback: no masking

    largest = (lbl == np.argmax(np.bincount(lbl.flat)[1:]) + 1) # boolean array [D, H, W], largest connected component

    # Fill interior holes (this is the fix!)
    filled = ndi.binary_fill_holes(largest) # fill holes in binary object, 6-connectivity for 3D
    # erode the mask → removes skull!
    # eroded = ndi.binary_erosion(filled, iterations=2)  # 1–2 voxels is ideal, shrinks the forefround region (1s) by one layer of voxels for each iteration

    mask = torch.from_numpy(filled).float().unsqueeze(0)
    return mask.to(x.device)

def get_transforms(target_shape=(128, 128, 128), pct_lo: float = 1.0, pct_hi: float = 99.0,
    crop_foreground: bool = True, ras: bool = True, interp: str = "trilinear", out_range: tuple = (0.0, 1.0),
    smooth_sigma_vox: tuple | None = None, apply_brain_mask: bool = True,):
    """
    PET-optimized preprocessing pipeline using MONAI.

    Parameters
    ----------
    target_shape : tuple of int
        Desired (D, H, W) volume size.
    pct_lo, pct_hi : float
        Percentiles for intensity scaling.
    crop_foreground : bool
        Whether to crop to nonzero foreground.
    ras : bool
        Reorient to RAS anatomical orientation.
    interp : str
        Interpolation mode for resizing (e.g., 'trilinear').
    out_range : tuple
        Output intensity range (min, max).
    smooth_sigma_vox: tuple
        mm-based smoothing → computed for *original* voxels)
    apply_brain_mask : bool
        Whether to apply a brain mask to zero out non-brain regions.
    Returns
    -------
    monai.transforms.Compose
        A composed MONAI transform pipeline.
    """
    steps = [LoadImage(image_only=True),
             EnsureChannelFirst()] # adds a channel dimension: (D, H, W) → (C=1, D, H, W)
    if ras: steps.append(Orientation(axcodes="RAS", labels=None)) # Reorients the image to a standard anatomical orientation: Right–Anterior–Superior
    
    if smooth_sigma_vox is not None: # Smooth BEFORE crop/resize (improves SNR for bounding box & interpolation)
        steps.append(GaussianSmooth(sigma=smooth_sigma_vox))

    if crop_foreground: steps.append(CropForeground()) # Crop out huge empty regions
    steps.append(Resize(spatial_size=target_shape, mode=interp)) # Resamples to the target size
    # Intensity normalization: maps voxel values between the 1st–99th percentile to [0,1] (clipping outliers)
    steps.append(ScaleIntensityRangePercentiles(lower=pct_lo, upper=pct_hi, b_min=float(out_range[0]), b_max=float(out_range[1]), clip=True))

    # ---- Force background = 0 (critical for model to focus on brain) ----
    if apply_brain_mask:
        steps.append(Lambda(lambda x: x * brain_outer_mask(x))) # apply brain mask by threshold
    
    return Compose(steps)


class PETDataset(Dataset):
    """
    Expects:
      - table columns: ["ID", "pet_path", "visual_read", "CL", ...]
      - transforms: a MONAI Compose returning a (1, D, H, W) Tensor/MetaTensor (float-like)
    """
    def __init__(self, table: pd.DataFrame, transforms, targets, augment: bool = False, dtype=torch.float32):
        self.table = table.reset_index(drop=True)
        self.transforms = transforms
        self.targets = [t.strip() for t in targets.split(",") if t.strip()]
        self.augment = augment
        self.dtype = dtype

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]

        if isinstance(self.transforms, Compose): # expect MAINAI Compose class
            # MONAI pipeline -> Tensor/MetaTensor with shape [C=1, D, H, W]
            path = row["pet_path"]
            x = self.transforms(path)
        elif torch.is_tensor(self.transforms): # torch tensor with shape  [S, D, H, W]
            fid = str(int(row["ID"]))
            x = self.transforms[fid]
            x = x.unsqueeze(0) # ➜ becomes [1, D, H, W]
        elif isinstance(self.transforms, (str, bytes, os.PathLike)):
            fid = str(int(row["ID"]))
            path = Path(self.transforms) / "data" / "data_{}.pt".format(fid)
            x = torch.load(path, map_location="cpu", weights_only=True)
        else:
            raise ValueError('PETDataset(), transforms data loading wrong. Should be either MONAI Compose(), or full torch tensor, or string Path')

        if isinstance(x, MetaTensor): x = x.as_tensor()
        x = x.to(dtype=self.dtype)
        ndim = x.ndim
        
        # Lightweight augmentation: random flips along spatial dims (D, H, W)
        if self.augment:
            r = torch.rand(3, device=x.device)  # one draw per spatial dim
            if r[0] < 0.5: x = torch.flip(x, dims=[ndim-3])  # D
            if r[1] < 0.5: x = torch.flip(x, dims=[ndim-2])  # H
            if r[2] < 0.5: x = torch.flip(x, dims=[ndim-1])  # W

        # Targets
        y_cls, y_reg = torch.tensor([float('nan')]), torch.tensor([float('nan')])
        if 'visual_read' in self.targets:
            y_cls = torch.tensor([row["visual_read"]], dtype=torch.float32)
        if 'CL' in self.targets:
            y_reg = torch.tensor([row["CL"]], dtype=torch.float32)

        return x, y_cls, y_reg, row["ID"]
