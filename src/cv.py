# src/cv.py
import os, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data import get_train_val_loaders
from src.early_stopping import EarlyStopper
from src.train import train_one_epoch, eval_epoch
from src.cam import find_last_conv3d, GradCAM3D, save_gradcam_overlays
from src.utils import append_metrics_row, save_checkpoint, build_model_from_args, append_epoch_metrics_csv, plot_metrics_from_csv, save_train_test_subjects


def kfold_cv(df_clean, stratify_labels, args):
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    metrics_path = os.path.join(args.output_path, "metrics.csv")

    pbar = tqdm(enumerate(skf.split(df_clean, stratify_labels), start=1),
                total=args.n_splits, desc="Stratified K-Fold", position=0, leave=True)

    for i, (tr_idx, va_idx) in pbar:
        fold_name = f"kfold-{i}"
        train_df = df_clean.iloc[tr_idx].reset_index(drop=True)
        val_df   = df_clean.iloc[va_idx].reset_index(drop=True)
        pbar.set_postfix(train=len(train_df), val=len(val_df))

        m = run_fold(train_df, val_df, args, fold_name=fold_name)
        
        # Log metrics
        row = {"fold": i, **m}
        append_metrics_row(metrics_path, row)

        tqdm.write(f"[{fold_name}] AUC={m.get('auc'):.3f} ACC={m.get('acc'):.3f} "
                   f"MAE={m.get('mae'):.2f} RMSE={m.get('rmse'):.2f} R2={m.get('r2'):.3f}"
                   f"val_metric={m.get('val_metric'):.2f}")

    print(f"\nDone. Metrics saved to: {metrics_path}")


def run_fold(train_df, val_df, args, fold_name: str, *, 
             use_early_stop: bool = True, es_patience=10, es_min_delta=1e-3,
             use_scheduler: bool = True, final_retrain: bool = False, on_epoch_end=None):
    """
    Train and return metrics dict.
    Modes:
      - Normal (CV): use_early_stop=True, use_scheduler=True, final_retrain=False
      - Final retrain (Option A): final_retrain=True  => fixed epochs, no early stop/scheduler;
        then evaluate once on val_df (outer test). No leakage.
    """
    fold_dir, ckpt_dir, viz_dir = _make_outfolder_fold(args.output_path, fold_name)
    save_train_test_subjects(train_df, val_df, fold_dir, fold_name)

    dl_tr, dl_va = get_train_val_loaders(train_df, val_df, args)

    # Determine classification/regression
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    n_classes = int(train_df["visual_read"].dropna().nunique()) if 'visual_read' in targets_list else None

    model = build_model_from_args(args, device=args.device, n_classes=n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    if use_scheduler and not final_retrain: # Scheduler only for CV mode (not in final retrain)
        sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-6) # scheduler (maximize AUC+ACC)
    else:
        sched = None

    # Early stopper only for CV mode
    es = EarlyStopper(patience=es_patience, min_delta=es_min_delta) if use_early_stop and not final_retrain else None

    csv_path = os.path.join(fold_dir, "metrics_per_epoch.csv")
    csv_loss_path = os.path.join(fold_dir, "trainning_loss_per_epoch.csv")

    # --- Training loop ---
    best_epoch = 0  # track for CV
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"{fold_name} epochs", position=1, leave=False, dynamic_ncols=True)
    for epoch in epoch_bar:
        tr_loss, tr_loss_all = train_one_epoch(model, dl_tr, opt, scaler, args.device, args.loss_w_cls, args.loss_w_reg)
        
        # In final retrain mode: do NOT evaluate on dl_va each epoch (no leakage)
        if final_retrain:
            # Log training loss only
            epoch_bar.set_postfix(train_loss=f"{tr_loss:.4f}")
            append_epoch_metrics_csv(csv_loss_path, epoch, tr_loss_all)
            continue

        # CV mode: evaluate each epoch on validation    
        metrics = eval_epoch(model, dl_va, args.device, sched)

        # Keep the tqdm bar neat
        epoch_bar.set_postfix(
            train_loss=f"{tr_loss:.4f}",
            AUC=f"{metrics.get('auc', float('nan')):.3f}",
            ACC=f"{metrics.get('acc', float('nan')):.3f}",
            MAE=f"{metrics.get('mae', float('nan')):.2f}",
            RMSE=f"{metrics.get('rmse', float('nan')):.2f}",
            R2=f"{metrics.get('r2', float('nan')):.3f}",
            val_metric=f"{metrics.get('val_metric', float('nan')):.3f}"
        )

        # scheduler on val_metric (CV mode only)
        val_metric = metrics.get('val_metric', float('nan'))
        if sched is not None and np.isfinite(val_metric): sched.step(val_metric)

        # report to Optuna if callback provided
        if on_epoch_end is not None: on_epoch_end(int(fold_name.split('-k')[-1]) if 'trial' in fold_name else 0, epoch, val_metric)

        # checkpoint best
        #if epoch == 1 and not os.path.exists(os.path.join(ckpt_dir, "best.pt")):
        #    save_checkpoint(model, os.path.join(ckpt_dir, "best.pt"))
        # append epoch metrics to CSV
        append_epoch_metrics_csv(csv_path, epoch, {**metrics, "train_loss": tr_loss})
        append_epoch_metrics_csv(csv_loss_path, epoch, tr_loss_all)

        # early stopping / checkpoint ONLY on improvement
        if es is not None:
            stop, improved = es.step(val_metric, epoch)
            if improved:
                best_epoch = es.best_epoch
                save_checkpoint(model, os.path.join(ckpt_dir, "best.pt"))
            if stop:
                tqdm.write(f"[{fold_name}] Early stopping at epoch {epoch} "
                           f"(no improvement > {es_min_delta} for {es_patience} epochs).")
                break

    # ===== End of epoch loop =====
    # Finalize weights to evaluate:
    if final_retrain: # In final retrain, there is no val-based checkpoint; use LAST-EPOCH weights
        pass
    else: # In CV mode, load the best val checkpoint if it exists; else use last weights
        best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(best_path):
            try:
                sd = torch.load(best_path, map_location=args.device, weights_only=True)
            except TypeError:
                sd = torch.load(best_path, map_location=args.device)
            state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
            model.load_state_dict(state_dict, strict=False)
    
    # Visuals: only in CV mode and only using validation loader
    if not final_retrain:
        # Grad-CAM snapshots for the best model on validation set
        target_layer = find_last_conv3d(model)  # generic for CNN3D, UNet3D, etc.
        last_metrics = eval_epoch(model, dl_va, args.device)
        task = "cls" if not np.isnan(metrics['auc']) else ("reg" if not np.isnan(metrics['r2']) else "auto")
        cam = GradCAM3D(model, target_layer=target_layer, task=task)
        save_gradcam_overlays(model, cam, dl_va, viz_dir, args.device, max_items=8)

        # metrics to return (CV)
        final_metrics = last_metrics
        final_metrics["best_epoch"] = int(best_epoch)
        plot_metrics_from_csv(csv_path, os.path.join(fold_dir, "metrics.png"))
        tqdm.write(f"[{fold_name}] AUC={final_metrics.get('auc', float('nan')):.3f} "
                f"ACC={final_metrics.get('acc', float('nan')):.3f} "
                f"MAE={final_metrics.get('mae', float('nan')):.2f} "
                f"RMSE={final_metrics.get('rmse', float('nan')):.2f} "
                f"R2={final_metrics.get('r2', float('nan')):.3f} "
                f"val_metric={final_metrics.get('val_metric', float('nan')):.3f}")
        return final_metrics

    # Final retrain mode: evaluate ONCE on dl_va (which you pass as OUTER TEST)
    test_metrics = eval_epoch(model, dl_va, args.device)
    tqdm.write(f"[{fold_name}] TEST AUC={test_metrics.get('auc', float('nan')):.3f} "
               f"ACC={test_metrics.get('acc', float('nan')):.3f} "
               f"MAE={test_metrics.get('mae', float('nan')):.2f} "
               f"RMSE={test_metrics.get('rmse', float('nan')):.2f} "
               f"R2={test_metrics.get('r2', float('nan')):.3f}")
    return test_metrics



def _make_outfolder_fold(output_path, fold_name):
    fold_dir = os.path.join(output_path, fold_name)
    ckpt_dir = os.path.join(fold_dir, "checkpoints")
    viz_dir  = os.path.join(fold_dir, "viz")
    os.makedirs(ckpt_dir, exist_ok=True)

    return fold_dir, ckpt_dir, viz_dir


def get_stratify_labels(df: pd.DataFrame, cols):
    """
    Use the exact columns in `cols` for stratification.
    - Drops rows with NaNs in ANY of these columns.
    - Returns (df_clean, y) where y is a 1D Series used by StratifiedKFold.
    """
    labels = [t.strip() for t in cols.split(",") if t.strip()]
    for c in labels:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in dataframe for stratification.")

    df_clean = df.dropna(subset=labels).copy()
    # make a single label by concatenating the values as strings
    stratify_labels = df_clean[labels].astype(str).agg("|".join, axis=1)
    return df_clean, stratify_labels


def cv_median_best_epoch(df_train, stratify_labels_train, args) -> int:
    """Run CV on df_train to collect best_epoch per fold; return median."""
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    best_epochs = []

    for i, (tr_idx, va_idx) in enumerate(skf.split(df_train, stratify_labels_train), start=1):
        fold_name = f"kfold-{i}"
        tr_df = df_train.iloc[tr_idx].reset_index(drop=True)
        va_df = df_train.iloc[va_idx].reset_index(drop=True)
        m = run_fold(tr_df, va_df, args, fold_name=fold_name,
                     use_early_stop=True, use_scheduler=True, final_retrain=False)
        be = int(m.get("best_epoch", 0))
        if be > 0:
            best_epochs.append(be)

    if not best_epochs:
        # fallback to args.epochs if something went wrong
        return int(args.epochs)

    return int(np.median(best_epochs))
