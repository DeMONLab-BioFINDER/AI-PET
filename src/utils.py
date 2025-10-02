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

def append_metrics_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

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
                "dropout": getattr(args, "dropout", 0.3),
            }

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
        if "out_channels" in allowed and "out_channels" not in params:
            params["out_channels"] = n_classes

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


def append_epoch_metrics_csv(csv_path: str, epoch: int, metrics: dict):
    """
    Append one row per epoch to a CSV using pandas.
    If file does not exist, create it with a header.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    row = {"epoch": epoch, **metrics}
    df_row = pd.DataFrame([row])

    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)


def plot_metrics_from_csv(csv_path: str, out_png: str):
    """
    Plot per-epoch validation metrics (AUC/ACC and/or MAE/RMSE/R2).
    Uses a twin y-axis only if both families exist.
    """
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns: return

    # Define metric families and keep only those present
    cls_defs = [("auc", "AUC"), ("acc", "ACC")]
    reg_defs = [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")]
    cls = [(k, lbl) for k, lbl in cls_defs if k in df.columns]
    reg = [(k, lbl) for k, lbl in reg_defs if k in df.columns]

    if not cls and not reg: return  # nothing to plot

    plt.figure(figsize=(10, 6), dpi=300)
    ax1 = plt.gca()

    x = df["epoch"]
    #if cls and reg:
    #    _plot_lines(ax1, x, df, cls, "Classification")
    #    ax2 = ax1.twinx()
    #    _plot_lines(ax2, x, df, reg, "Regression")
    #else:
    #    _plot_lines(ax1, x, df, cls or reg, "Classification" if cls else "Regression")
    _plot_lines(ax1, x, df, cls or reg, "Classification" if cls else "Regression")

    ax1.set_xlabel("Epoch")
    plt.title("Validation metrics per epoch")
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


### Runtime estimation
def estimate_runtime(model, loader, device, *,
        N: int, K: int, B: int, E: int, T: int = 0, # number of training samples (after hold-out), n_splits, batch size (for training), full epochs for final runs, Optuna trials (0 => skip tuning estimate)
        E_proxy: int = 8, measure_batches: int = 32):  #proxy epochs during tuning   

    """
    Benchmarks sec/step on your machine and projects total runtime.

    Returns a dict with seconds for:
      - sec_per_step
      - per_epoch
      - cv_total
      - tune_total (0 if T==0)
      - final_train_total
    """
    # 1) micro-benchmark (forward-only; includes I/O + transforms)
    sec_per_step = _benchmark_seconds_per_step(model, loader, device, warmup_batches=8, measure_batches=measure_batches)

    # 2) steps per epoch for one fold (train portion size ~ N*(K-1)/K)
    train_samples_per_fold = N * (K - 1) / K
    steps_per_epoch_fold = math.ceil(train_samples_per_fold / max(B, 1))
    time_per_epoch = steps_per_epoch_fold * sec_per_step

    # 3) projections
    total_time_cv = K * E * time_per_epoch                         # K-fold full training (no tuning)
    total_time_tune = T * K * E_proxy * time_per_epoch if T > 0 else 0.0

    steps_full = math.ceil(N / max(B, 1))                          # full train pool (no CV) for lockbox run
    time_final_train = E * steps_full * sec_per_step

    # 4) pretty print
    print("\n=== Runtime estimate (based on live benchmark) ===")
    print(f"sec/step           : {sec_per_step:.4f} s")
    print(f"steps/epoch (fold) : {steps_per_epoch_fold}")
    print(f"time/epoch (fold)  : {_fmt_hms(time_per_epoch)}")
    print(f"K-fold (K={K}, E={E}): {_fmt_hms(total_time_cv)}")
    if T > 0:
        print(f"Tuning (T={T}, proxy_epochs={E_proxy}, K={K}): {_fmt_hms(total_time_tune)}")
    print(f"Final train on N={N} (E={E}): {_fmt_hms(time_final_train)}")
    print("==================================================\n")


def _benchmark_seconds_per_step(model, loader, device, warmup_batches=8, measure_batches=32):
    model.eval()
    it = iter(loader)
    # warmup (no grad)
    with torch.no_grad():
        for _ in range(min(warmup_batches, len(loader))):
            x, *_ = next(it)
            x = x.to(device, non_blocking=True)
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    # measure
    times = []
    with torch.no_grad():
        for _ in range(min(measure_batches, len(loader))):
            t0 = time.perf_counter()
            try:
                x, *_ = next(it)
            except StopIteration:
                it = iter(loader); x, *_ = next(it)
            x = x.to(device, non_blocking=True)
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return (sum(times) / len(times)) if times else float("inf")


def _fmt_hms(seconds: float) -> str:
    if not math.isfinite(seconds): return "n/a"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:  return f"{d}d {h}h {m}m"
    if h:  return f"{h}h {m}m {s}s"
    if m:  return f"{m}m {s}s"
    return f"{s}s"