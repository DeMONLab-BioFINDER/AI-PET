# src/cv.py
import os, torch, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data import get_train_val_loaders
from src.early_stopping import EarlyStopper
from src.train import train_one_epoch, inference
from src.vis import run_visualization
from src.utils import append_metrics_csv, save_checkpoint, build_model_from_args, load_best_checkpoint, plot_metrics_from_csv, save_train_test_subjects


def kfold_cv(df_clean, stratify_labels, args):
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    metrics_path = os.path.join(args.output_path, "metrics.csv")
    results_path = os.path.join(args.output_path, "results.csv")

    pbar = tqdm(enumerate(skf.split(df_clean, stratify_labels), start=1),
                total=args.n_splits, desc="Stratified K-Fold", position=0, leave=True)

    for i, (tr_idx, va_idx) in pbar:
        fold_name = f"kfold-{i}"
        train_df = df_clean.iloc[tr_idx].reset_index(drop=True)
        val_df   = df_clean.iloc[va_idx].reset_index(drop=True)
        pbar.set_postfix(train=len(train_df), val=len(val_df))

        m, r = run_fold(train_df, val_df, args, fold_name=fold_name)
        
        # Log results
        append_metrics_csv(metrics_path, {"fold": i, **m}, mode='row')
        append_metrics_csv(results_path, {"fold": i, **r}, mode='row')

        tqdm.write(f"[{fold_name}] AUC={m.get('auc'):.3f} ACC={m.get('acc'):.3f} "
                   f"MAE={m.get('mae'):.2f} RMSE={m.get('rmse'):.2f} R2={m.get('r2'):.3f}"
                   f"val_metric={m.get('val_metric'):.2f}")

    print(f"\nDone. Metrics saved to: {metrics_path}")



def run_fold(train_df, val_df, args, fold_name: str, *, optuna_report=None):
    """
    Train and inferecen on test set return metrics dict.
    Modes:
      - Normal CV: train_only=False, args.tune=False -> do early_stop and scheduler on same training set; inference on test set (val_df) loading the best trained model
      - Hyperparameter tuning inner fold: train_only=False, args.tune=True -> do early_stop and scheduler, stopping on the test set; inference on test set (val_df) loading the best trained model
      - Hyperparameter tuning outer fold: train_only=True, args.tune=True -> no early_stop/scheduler; inference on test set (val_df) using the trained model
    """
    train_only = (fold_name == "nestedcv-outer-test")
    output_fold_dir, path_list = _make_outfolder_fold(args.output_path, fold_name) # path_list: csv_path, csv_loss_path, ckpt_path
    save_train_test_subjects(train_df, val_df, output_fold_dir, fold_name)

    dl_tr, dl_va = get_train_val_loaders(train_df, val_df, args)
    _, dl_eval = get_train_val_loaders(train_df, train_df, args)

    # Determine classification/regression
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    n_classes = int(train_df["visual_read"].dropna().nunique()) if 'visual_read' in targets_list else None

    model = build_model_from_args(args, device=args.device, n_classes=n_classes)

    # ---- Train ----
    if args.tune:
        model, best_epoch = train_model(model, dl_tr, dl_va, args=args, fold_name=fold_name,
                                        path_list=path_list, optuna_report=optuna_report)
    else:
        print('Train (early stop and scheduler using training set, if not "train_only")')
        model, best_epoch = train_model(model, dl_tr, dl_eval, args=args, fold_name=fold_name, path_list=path_list)
    
    plot_metrics_from_csv(path_list['train_eval_csv'], path_list['train_eval_png'])

    # ---- Test (inference) ----
    # Load best checkpoint
    if not train_only: model = load_best_checkpoint(model, ckpt_path=path_list['ckpt'], device=args.device) # In final retrain, there is no val-based checkpoint; use LAST-EPOCH weights
    
    # inference and save resutls
    metrics_tr, df_result_tr = inference(model, dl_va, args.device)
    metrics_te, df_result_te = inference(model, dl_va, args.device)
    metrics_te["best_epoch"] = int(best_epoch)
    pickle.dump({'train':{'metric': metrics_tr, 'preds': df_result_tr},
                 'test':{'metric': metrics_te, 'preds': df_result_te}}, open(path_list['train-test_eval_pkl'],'wb'))

    # ---- Interpretation: grad-CAM or ... ----
    run_visualization(model, dl_va, args.device, args.output_path, vis_name=args.visualization_name) # the best model on validation set, save .png and .nii

    return metrics_te, df_result_te


def train_model(model, dl_tr, dl_va, *, args, fold_name, path_list, optuna_report=None):
    """
    Unified training loop.
    dl_tr and dl_va should be the same if not doing hyperparameter tunning CV training
    when doing fine-tuning few shots, do early stop, not scheduler <-- dl_tr and dl_va should all be the few-shots images, don't touch test set
    """
    train_only = (fold_name == "nestedcv-outer-test") #!!! train only, no early stop

    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6) # NOT using Scheduler in final retrain
    es = EarlyStopper(patience=args.es_patience, min_delta=args.es_min_delta)

    best_epoch = 0
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"{fold_name} epochs", position=1, leave=False, dynamic_ncols=True)
    for epoch in epoch_bar:
        tr_loss_mean, tr_loss_all = train_one_epoch(model=model, loader=dl_tr, opt=optimizer, scaler=scaler,
                                                    device=args.device, loss_w_cls=args.loss_w_cls, loss_w_reg=args.loss_w_reg,
                                                    reg_loss=args.reg_loss, smoothl1_beta=args.smoothl1_betes)

        # ---- Inference on training (or inner test of validation) ----
        metrics, _ = inference(model, dl_va, args.device)
        eval_metric = metrics.get("eval_metric", float("nan"))

        if not train_only and 'few-shot' not in fold_name and np.isfinite(eval_metric): scheduler.step(eval_metric) # no scheduler when test only and fine-tunning few shots

        # ---- Optuna report, if callback provided ----
        if optuna_report is not None: optuna_report(int(fold_name.split('-k')[-1]) if 'trial' in fold_name else 0, epoch, eval_metric)

        # ---- save training loss and evaluation metrics (used to early stop) ----
        append_metrics_csv(path_list['train_loss_csv'], {"epoch": epoch, **tr_loss_all}, mode='row')
        append_metrics_csv(path_list['train_eval_csv'], {"epoch": epoch, "train_loss": tr_loss_mean, **metrics}, mode='row')

        epoch_bar.set_postfix(train_loss=f"{tr_loss_mean:.4f}", eval_metric=f"{eval_metric:.3f}")
        
        # ---- Early stopping + checkpoint ----
        if not train_only:
            stop, improved = es.step(eval_metric, epoch)
            if improved:
                best_epoch = es.best_epoch
                save_checkpoint(model, path_list['ckpt'])
            if stop:
                tqdm.write(
                    f"[{fold_name}] Early stopping at epoch {epoch} "
                    f"(no improvement > {args.es_min_delta} for {args.es_patience} epochs)."
                )
                break

    return model, best_epoch


def _make_outfolder_fold(output_path, fold_name):
    output_fold_dir = os.path.join(output_path, fold_name)
    ckpt_dir = os.path.join(output_fold_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_eval_csv_path = os.path.join(output_fold_dir, "trainning_metrics_per_epoch.csv")
    train_loss_csv_path = os.path.join(output_fold_dir, "trainning_loss_allsubjects_per_epoch.csv")
    train_eval_png_path = os.path.join(output_fold_dir, "trainning_metrics_per_epoch.csv")
    test_eval_pkl_path = os.path.join(output_fold_dir, "train-test_preds-metrics_thisfold.csv")

    path_list = {'train_eval_csv': train_eval_csv_path, 'train_loss_csv': train_loss_csv_path, 
                 'train_eval_png': train_eval_png_path, 'train-test_eval_pkl': test_eval_pkl_path,
                 'ckpt': os.path.join(ckpt_dir, f"{fold_name}_best.pt")}

    return output_fold_dir, path_list


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
        m = run_fold(tr_df, va_df, args, fold_name=fold_name)
        be = int(m.get("best_epoch", 0))
        if be > 0:
            best_epochs.append(be)

    if not best_epochs:
        # fallback to args.epochs if something went wrong
        return int(args.epochs)

    return int(np.median(best_epochs))
