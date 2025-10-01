# src/data.py
import os
import torch
import pandas as pd
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset, Subset
from monai.data import MetaTensor
from monai.transforms import (LoadImage, EnsureChannelFirst, Orientation, Resize,
        ScaleIntensityRangePercentiles, Compose, CropForeground)

from src.utils import seed_worker
# ------------------------------
# Master table
# ------------------------------
def build_master_table(input_path: str, preproc_method: str, targets: List[str], subjects: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build the table required by the training code, given the custom folder layout.
    Normal mode: discover images + join demographics.
    Cached mode: if preproc_method is empty/None OR cache files exist in cache_dir,
                 load demo.csv only (no disk scans), and return that table.
    In cached mode, adds 'ID' to preserve the order matching data.pt.
    """
    # Detect cached mode
    use_cache = (not preproc_method) or (str(preproc_method).strip() == "")
    if use_cache:
        df = pd.read_csv(Path(input_path) / "demo.csv", index_col=0) # Must have 'ID' column from 0 to len(df)
        print(f"[cache] Loaded demo.csv with {len(df)} rows (no filesystem scan).")
    else:
        pets = find_pet_files(input_path=input_path, preproc_method=preproc_method, subjects_list=subjects)
        if pets.empty:
            raise FileNotFoundError(f"No NIfTI files found under '{input_path}' with preproc suffix '_{preproc_method}'.")
        else:
            print(f'Found {pets.shape[0]} scans')

        labels = load_participants_labels(input_path)
        labels["ID"] = labels["ID"].astype(str).str.strip()
        df = pd.merge(pets, labels, on="ID", how="inner")

        # Only scans with targets value
        targets_list = [t.strip() for t in targets.split(",") if t.strip()]
        df = df[~df[targets_list].isna().values].reset_index(drop=True)
        print(f'Found {df.shape[0]} scans with demographics for {targets}')

    return df


def find_pet_files(input_path: str, preproc_method: str, subjects_list: Optional[str] = None) -> pd.DataFrame:
    """
    Glob-only discovery of PET NIfTI files under:
        input_path/**/*_{preproc_method}/*/*/*.nii*
    Picks exactly one file per subject (first by sorted path).

    Returns columns: ID, pet_path, imagefile, site_from_path(None).
    """
    root = Path(input_path + '/PET/')
    suffix = f"{preproc_method}"
    pattern = f"**/*{suffix}/*/*/*.nii*"

    rows = []
    for nii in root.glob(pattern):
        # subjectID assumed as the top-level folder under input_path
        try:
            sid = nii.relative_to(root).parts[0]
        except Exception:
            continue
        rows.append({"ID": sid,
                     "pet_path": str(nii).replace('/._','/'), # !!! make this better later
                     "imagefile": nii.name,
                     #"site_from_path": None,
        })

    df = pd.DataFrame(rows)
    if df.empty: return df
    # Deterministic: sort only keep the first scan per subject
    df = (df.sort_values(["ID", "pet_path"])
            .groupby("ID", as_index=False, group_keys=False)
            .head(1).reset_index(drop=True))
    
    # Optional filter to provided subject list
    if subjects_list:
        subjects_path = input_path / subjects_list
        if not subjects_path.exists(): raise ValueError(f'Input {subjects_list} not exists')
        allow = pd.read_csv(subjects_path)
        subj_set = {str(s).strip() for s in allow['ID'] if str(s).strip()} # .strip() remove white spaces
        df = df[df["ID"].astype(str).isin(subj_set)].reset_index(drop=True)

    return df


def load_participants_labels(input_path: str) -> pd.DataFrame: #cache: Optional[bool] = False
    """
    Load demographics.csv from input_path and return:
    ID, site, visual_read, CL, age, gender
    """
    csv = Path(input_path) / "demographics.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing {csv}. Provide columns: ID, site, visual_read, CL, age, gender, ...")
    df = pd.read_csv(csv, index_col=0)
    
    # Ensure required columns are present in the dataframe.
    required = {"ID", "site", "visual_read", "CL", "age", "gender"}
    #if cache: required.discard("ID")
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
    dl_va = get_loader(train_df, tfm, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False)
    
    return dl_tr, dl_va


def get_loader(df, tfm, args, batch_size, augment=False, shuffle=False):
    g = torch.Generator()
    g.manual_seed(args.seed)

    dataset = PETDataset(df, tfm, args.targets, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       worker_init_fn=seed_worker, generator=g,
                       num_workers=args.num_workers, pin_memory=True)

    return loader


def get_transforms(target_shape=(128, 128, 128), pct_lo: float = 1.0, pct_hi: float = 99.0,
    crop_foreground: bool = True, ras: bool = True, interp: str = "trilinear", out_range: tuple = (0.0, 1.0)):
    """
    Build a preprocessing pipeline for PET volumes using MONAI.
    
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
    
    Returns
    -------
    monai.transforms.Compose
        A composed MONAI transform pipeline.
    """
    steps = [LoadImage(image_only=True),
             EnsureChannelFirst()] # adds a channel dimension: (D, H, W) → (C=1, D, H, W)
    if ras:
        steps.append(Orientation(axcodes="RAS", labels=None)) # Reorients the image to a standard anatomical orientation: Right–Anterior–Superior
    if crop_foreground: # Removes empty background by finding the smallest bounding box around the “non-zero” region
        steps.append(CropForeground())
    steps.append(Resize(spatial_size=target_shape, mode=interp)) # Resamples to the target size
    # Intensity normalization: maps voxel values between the 1st–99th percentile to [0,1] (clipping outliers)
    # Produces stable input scale across subjects
    steps.append(ScaleIntensityRangePercentiles(
        lower=pct_lo, upper=pct_hi,
        b_min=float(out_range[0]), b_max=float(out_range[1]),
        clip=True,
        ))
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
