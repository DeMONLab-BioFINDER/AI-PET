# abpet/utils.py
import os, csv, json, time, math
import torch, random, inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold

from src.models import *


def set_seed(seed: int = 42, deterministic: bool = False, set_pythonhashseed: bool = True):
    if set_pythonhashseed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # Seeds CPU and (per PyTorch docs) CUDA RNG; no separate call for MPS exists.
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    # Global determinism (may error if an op lacks a deterministic variant)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[warn] deterministic_algorithms not fully supported: {e}")

    #  MONAI convenience:
    from monai.utils import set_determinism as monai_set_det
    monai_set_det(seed=seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_checkpoint(model, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)

def get_device(prefer_cuda=True, force_cpu=False):
    """
    Returns a torch.device among: cuda, mps, cpu.
    prefer_cuda: if True, choose CUDA over MPS when both are present (e.g., eGPU on Mac).
    """
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None \
              and torch.backends.mps.is_built() and torch.backends.mps.is_available()
    if prefer_cuda and has_cuda and not force_cpu: return torch.device("cuda")
    if not prefer_cuda and has_mps: return torch.device("mps")
    if not force_cpu and has_cuda: return torch.device("cuda")
    if not force_cpu and has_mps: return torch.device("mps")

    torch.backends.cudnn.benchmark = True  # 3D convs benefit
    return torch.device("cpu")

def load_best_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    Load model weights from checkpoint path into the given model.
    """
    if os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(ckpt_path, map_location=device)
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(state_dict, strict=False)
    return model

def compute_smooth_sigma_vox(voxel_sizes_mm: tuple[float, float, float], fwhm_current_mm: float, fwhm_target_mm: float) -> tuple[float, float, float] | None:
    """
    Returns per-axis sigma in *voxels* for MONAI.GaussianSmooth to top-up smoothing
    from fwhm_current_mm to fwhm_target_mm. If no extra smoothing needed, returns (0,0,0).
    """
    if fwhm_target_mm <= fwhm_current_mm:
        return (0.0, 0.0, 0.0)

    fwhm_extra_mm = np.sqrt(fwhm_target_mm**2 - fwhm_current_mm**2)  # quadrature
    sigma_mm = fwhm_extra_mm / 2.354820045  # mm → σ
    vx, vy, vz = voxel_sizes_mm
    return (sigma_mm / vx, sigma_mm / vy, sigma_mm / vz)


def append_metrics_csv(csv_path: str, data: dict, mode: str = "row"):
    """
    Append metrics to a CSV file using pandas.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    data : dict
        Metrics to append.
    mode : {"row", "column"}
        - "row": append one experiment per row (recommended)
        - "column": append one experiment per column
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if mode == "row":
        df_new = pd.DataFrame([data])  # one row
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
    elif mode == "column":
        df_new = pd.DataFrame.from_dict(data, orient="index", columns=["value"])
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path, index_col=0)
            df = pd.concat([df_old, df_new], axis=1)
    else:
        raise ValueError("mode must be 'row' or 'column'")
    if not os.path.exists(csv_path): df = df_new
    df.to_csv(csv_path)


def combine_metrics_for_minimize(m: dict) -> float:
    """
    Turn your fold metrics into a scalar to *minimize*.
    Adjust weights if you prefer.
    """
    auc  = m.get("auc")
    mae  = m.get("mae")
    rmse = m.get("rmse")
    r2   = m.get("r2")

    parts = []
    if auc  is not None and auc  == auc: parts.append(1.0 - float(auc))  # 1 - AUC
    if mae  is not None and mae  == mae: parts.append(float(mae))
    if rmse is not None and rmse == rmse: parts.append(float(rmse))
    if r2   is not None and r2   == r2:   parts.append(1.0 - float(r2))  # 1 - R2

    return sum(parts) if parts else 1e9  # big penalty if missing


def clone_args(args, **overrides):
    """Create a shallow, mutable copy of args with some fields overridden."""
    d = vars(args).copy()
    d.update(overrides)
    return SimpleNamespace(**d)


def make_splits(df, labels, n_splits, seed):
    """Freeze splits once so objective() is deterministic across trials."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df, labels))


def hold_out_set(df, labels, test_size: float = 0.2, seed: int = 42,) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a single held-out lockbox using StratifiedKFold via make_splits().
    K is chosen as round(1/test_size). Returns (train_idx, test_idx).

    Note: exact size will be ~1/K of data (not arbitrary percentages).
    """
    # ensure sane K
    #k = max(2, int(round(1.0 / float(test_size))))
    k = 5
    splits = make_splits(df, labels, n_splits=k, seed=seed)
    train_idx, test_idx = splits[-1]   # pick the last fold as test to reduce accidental bias
    return np.asarray(train_idx), np.asarray(test_idx)


def _resolve_model_class(name: str):
    """Return a model class by name from models.py; raise a helpful error if missing."""
    try:
        return globals()[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model '{name}'. Ensure it's defined in models.py "
            f"and the name matches exactly (case-sensitive)."
        ) from e

def save_train_test_subjects(df_train, df_test, output_path, savename):
    df_train.to_csv(os.path.join(output_path, f'{savename}_training-set.csv'))
    df_test.to_csv(os.path.join(output_path, f'{savename}_testing-set.csv'))


def build_model_from_args(args, device=None, n_classes: int | None = None):
    """
    Dynamically instantiate a model by name from models.py using args.model.
    - Filters kwargs to what the class __init__ accepts.
    - Merges defaults from args with optional --model_kwargs (JSON or dict).
    - If args.resume is set, loads weights with strict=False.
    """
    ModelCls = _resolve_model_class(args.model)

    # Defaults from args (common across your models)
    defaults = {"in_channels": getattr(args, "in_channels", 1),
                "widths": tuple(getattr(args, "widths", (32, 64, 128, 256))),
                "dropout": getattr(args, "dropout", 0.3)}
    
    extra_dim = 0
    # scalar CL input
    if getattr(args, "input_cl", None) is not None:
        extra_dim += 1
        print('add extra input CL to the last FC layer')
    # global image-derived features
    if getattr(args, "extra_global_feats", None):
        # expect comma-separated string: "p95,std,frac_hi"
        feats = [f for f in args.extra_global_feats.split(",") if f.strip()]
        extra_dim += len(feats)
        print(f'add extra global input {args.extra_global_feats} to the last FC layer')
    defaults["extra_dim"] = extra_dim

    # JSON-only model kwargs
    extra = {}
    if hasattr(args, "model_kwargs") and args.model_kwargs:
        try:
            extra = json.loads(args.model_kwargs)
        except json.JSONDecodeError as e:
            raise ValueError(f"--model_kwargs must be valid JSON: {e}")
        if "widths" in extra and isinstance(extra["widths"], list):
            extra["widths"] = tuple(extra["widths"])

    # keep only params accepted by __init__
    allowed = set(inspect.signature(ModelCls.__init__).parameters) - {"self", "*args", "**kwargs"}
    params = {k: v for k, v in {**defaults, **extra}.items() if k in allowed}

    # auto-wire class count if not provided
    if n_classes is not None:
        if "num_classes" in allowed and "num_classes" not in params:
            params["num_classes"] = n_classes

    # Instantiate:  Build + move
    model = ModelCls(**params)
    if device is not None:
        model = model.to(device)

    # resume weights (Optional)
    if getattr(args, "resume", ""):
        state = torch.load(args.resume, map_location=device or "cpu")
        model.load_state_dict(state, strict=False)

    print(model)
    return model


def plot_metrics_from_csv(csv_path: str, out_png: str):
    """
    Plot per-epoch validation metrics (AUC/ACC and/or MAE/RMSE/R2).
    Uses a twin y-axis only if both families exist.
    """
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns: return

    # Define metric families and keep only those present
    defs = [("eval_metric", "Evaluation_metric"), ("train_loss", "Training loss")]

    plt.figure(figsize=(10, 6), dpi=300)
    ax1 = plt.gca()

    x = df["epoch"]
    #if cls and reg:
    #    _plot_lines(ax1, x, df, cls, "Classification")
    #    ax2 = ax1.twinx()
    #    _plot_lines(ax2, x, df, reg, "Regression")
    #else:
    #    _plot_lines(ax1, x, df, cls or reg, "Classification" if cls else "Regression")
    _plot_lines(ax1, x, df, defs)

    ax1.set_xlabel("Epoch")
    plt.title("Validation metrics vs training loss per epoch")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

def _plot_lines(ax, x, df, metrics, ylabel):
    for k, lbl in metrics:
        ax.plot(x, df[k], label=lbl)
    ax.set_ylabel(ylabel)
    if metrics:
        ax.legend(loc="best")