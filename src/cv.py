# src/cv.py
import os, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from src.data import get_train_val_loaders
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
                   f"MAE={m.get('mae'):.2f} RMSE={m.get('rmse'):.2f} R2={m.get('r2'):.3f}")

    print(f"\nDone. Metrics saved to: {metrics_path}")


def run_fold(train_df, val_df, args, fold_name: str):
    fold_dir, ckpt_dir, viz_dir = _make_outfolder_fold(args.output_path, fold_name)
    save_train_test_subjects(train_df, val_df, fold_dir, fold_name)

    dl_tr, dl_va = get_train_val_loaders(train_df, val_df, args)

    # get the number of class if running classification
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    n_classes = int(train_df["visual_read"].dropna().nunique()) if 'visual_read' in targets_list else None

    model = build_model_from_args(args, device=args.device, n_classes=n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    best_score = float("inf")
    csv_path = os.path.join(fold_dir, "metrics_per_epoch.csv")

    # epoch-level progress bar (doesn't clash with dataloader bars)
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"{fold_name} epochs", position=1, leave=False, dynamic_ncols=True)
    for epoch in epoch_bar:
        tr_loss = train_one_epoch(model, dl_tr, opt, scaler, args.device, args.loss_w_cls, args.loss_w_reg)
        metrics = eval_epoch(model, dl_va, args.device)

        # Keep the tqdm bar neat
        epoch_bar.set_postfix(
            train_loss=f"{tr_loss:.4f}",
            AUC=f"{metrics.get('auc', float('nan')):.3f}",
            ACC=f"{metrics.get('acc', float('nan')):.3f}",
            MAE=f"{metrics.get('mae', float('nan')):.2f}",
            RMSE=f"{metrics.get('rmse', float('nan')):.2f}",
            R2=f"{metrics.get('r2', float('nan')):.3f}"
        )

        # checkpoint best
        if epoch == 1 and not os.path.exists(os.path.join(ckpt_dir, "best.pt")):
            save_checkpoint(model, os.path.join(ckpt_dir, "best.pt"))
        if metrics["val_loss"] < best_score:
            best_score = metrics["val_loss"]
            save_checkpoint(model, os.path.join(ckpt_dir, "best.pt"))

        # append epoch metrics to CSV
        append_epoch_metrics_csv(csv_path, epoch, {**metrics, "train_loss": tr_loss})

        # Grad-CAM snapshots on a schedule
        target_layer = find_last_conv3d(model)  # generic for CNN3D, UNet3D, etc.
        task = "cls" if not np.isnan(metrics['auc']) else ("reg" if not np.isnan(metrics['r2']) else "auto")
        if epoch in {1, args.epochs} or (epoch % max(1, args.epochs // 3) == 0):
            cam = GradCAM3D(model, target_layer=target_layer, task=task)
            save_gradcam_overlays(model, cam, dl_va, viz_dir, args.device, max_items=8)

    # Load best and final 
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        try:
            sd = torch.load(best_path, map_location=args.device)
        except TypeError:
            sd = torch.load(best_path, map_location=args.device)
        state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(state_dict, strict=False)
    else:
        tqdm.write("[warn] No best.pt found; using last-epoch weights.")

    final_metrics = eval_epoch(model, dl_va, args.device)

    # One final plot after training this fold
    plot_metrics_from_csv(csv_path, os.path.join(fold_dir, "metrics.png"))

    # Clean, single line after the bar
    tqdm.write(f"[{fold_name}] AUC={final_metrics.get('auc', float('nan')):.3f} "
               f"ACC={final_metrics.get('acc', float('nan')):.3f} "
               f"MAE={final_metrics.get('mae', float('nan')):.2f} "
               f"RMSE={final_metrics.get('rmse', float('nan')):.2f} "
               f"R2={final_metrics.get('r2', float('nan')):.3f}")

    return final_metrics


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
