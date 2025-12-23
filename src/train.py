# src/train.py
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error, root_mean_squared_error, roc_curve, balanced_accuracy_score, f1_score, matthews_corrcoef


def train_one_epoch(model, loader, opt, scaler, device, loss_w_cls, loss_w_reg, reg_loss, smoothl1_beta):
    model.train()
    losses = []

    for x, y_cls, y_reg, extra, _ in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        extra = extra.to(device)
        y_cls = y_cls.to(device) if y_cls is not None else None
        y_reg = y_reg.to(device) if y_reg is not None else None

        opt.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, _ = compute_total_loss(model, x, y_cls, y_reg, extra=extra,
                    loss_w_cls=loss_w_cls, loss_w_reg=loss_w_reg, reg_loss=reg_loss, smoothl1_beta=smoothl1_beta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss, _ = compute_total_loss(model, x, y_cls, y_reg, extra=extra,
                loss_w_cls=loss_w_cls, loss_w_reg=loss_w_reg, reg_loss=reg_loss, smoothl1_beta=smoothl1_beta)
            loss.backward()
            opt.step()

        losses.append(float(loss.detach().item()))
    losses_dict = {i: v for i, v in enumerate(losses)}

    return (float(np.mean(losses)), losses_dict) if losses else (0.0, {})


@torch.no_grad()
def inference(model, loader, device):
    probs, ycls, preds, any_cls, cents, yreg, any_reg = evals(model, loader, device)

    metrics = compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg)
    # es_metric <- val_metric

    if any_cls: 
        df_result = pd.DataFrame({'y': np.concatenate(ycls, axis=0), 
                                  'pred':np.concatenate(preds, axis=0),
                                  'prob':np.concatenate(probs, axis=0)})
    if any_reg:
        df_result = pd.DataFrame({'y': np.concatenate(yreg, axis=0),
                                  'pred':np.concatenate(cents, axis=0)})

    return metrics, df_result


@torch.no_grad()
def evals(model, loader, device):
    model.eval()

    probs, ycls, preds = [], [], []
    cents, yreg = [], []
    any_cls, any_reg = False, False
    for x, y_cls, y_reg, extra, _ in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        extra = extra.to(device)
        out = model(x, extra=extra)
        logit, cent, _ = unpack_model_outputs(out, y_reg)

        if (not y_cls.isnan().all()) and (logit is not None):
            any_cls = True
            y_cls = y_cls.to(device)
            if logit.ndim == 2 and logit.shape[1] == 1:
                p = torch.sigmoid(logit).detach().cpu().numpy().ravel()
                preds.append((p > 0.5).astype(int)) 
                probs.append(p)                    # 1D
                ycls.append(y_cls.cpu().numpy().ravel().astype(int))
            elif logit.ndim == 2 and logit.shape[1] == 2:
                # binary: two logits -> softmax, take prob of class 1
                sm = F.softmax(logit, dim=1).detach().cpu().numpy()
                p1 = sm[:, 1]
                preds.append((p1 > 0.5).astype(int))
                probs.append(p1)                   # 1D
                ycls.append(y_cls.cpu().numpy().ravel().astype(int))
            else:
                # multiclass C>2: keep full prob matrix
                sm = F.softmax(logit, dim=1).detach().cpu().numpy()
                preds.append(sm.argmax(axis=1))
                probs.append(sm)                    # 2D
                ycls.append(y_cls.cpu().numpy().ravel())

        if (not y_reg.isnan().all()) and (cent is not None):
            any_reg = True
            y_reg = y_reg.to(device)
            cents.append(cent.cpu().numpy().ravel())
            yreg.append(y_reg.cpu().numpy().ravel())

    return probs, ycls, preds, any_cls, cents, yreg, any_reg


def compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg):
    
    metrics = {"auc": np.nan, "acc": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "eval_metric": 0.0} #"val_loss": 0.0, 
    if any_cls:
        ycls  = np.concatenate(ycls)
        preds = np.concatenate(preds)
        probs = np.concatenate(probs, axis=0)   # [N,out_channel]

        # ignore nans
        pr_mask = np.isfinite(probs) if probs.ndim == 1 else np.isfinite(probs).all(axis=1)
        mask_acc = np.isfinite(preds) & pr_mask # use preds to mask, as y has masked out nans in input
        ycls = ycls[mask_acc]
        preds = preds[mask_acc]
        probs = probs[mask_acc] if probs.ndim == 1 else probs[mask_acc, :]

        labels = np.unique(ycls).astype(int)
        if probs.ndim == 1:
            metrics["auc"] = float(roc_auc_score(ycls, probs)) 
        else:
            if labels.size < 2:
                metrics["auc"] = float("nan")
            elif labels.size == 2:
                # binary AUC: take the column for the chosen positive class
                pos = int(labels.max())
                metrics["auc"] = roc_auc_score((ycls == pos).astype(int), probs[:, pos])
                metrics["acc_opt"], metrics["bacc"], metrics["f1"], metrics["mcc"], metrics["best_thr"] = opt_threshold(ycls, probs, pos)

            else:
                metrics["auc"] = float(roc_auc_score(ycls, probs, multi_class="ovr", average="macro", labels=labels))
        metrics["acc"] = float(accuracy_score(ycls, preds))
        metrics['eval_metric'] = np.nansum([metrics["auc"], metrics["acc"]])
    if any_reg:
        cents = np.concatenate(cents)
        yreg  = np.concatenate(yreg)

        # ignore nans
        mask_reg = np.isfinite(cents) # use cents to mask, as y has masked out nans in input
        yreg = yreg[mask_reg]
        cents = cents[mask_reg]

        metrics["mae"]  = float(mean_absolute_error(yreg, cents))
        metrics["rmse"] = float(root_mean_squared_error(yreg, cents))
        metrics["r2"]   = float(r2_score(yreg, cents))

        mae_ref = np.median(np.abs(yreg - np.median(yreg))) # Reference scale for MAE (robust to outliers), Median Absolute Deviation
        if not np.isfinite(mae_ref) or mae_ref <= 1e-6: # Fallback if targets are constant or numerically unstable
            mae_ref = max(np.std(yreg), 1e-6)
        mae_good = 1.0 - np.clip(metrics["mae"] / mae_ref, 0.0, 1.0) if np.isfinite(metrics["mae"]) else np.nan # Convert MAE into a "goodness" score in [0, 1]
        metrics['eval_metric'] = np.nansum([mae_good, metrics["r2"]])

    return metrics


def compute_total_loss(model: torch.nn.Module, x: torch.Tensor, y_cls: torch.Tensor,
                       y_reg: torch.Tensor, extra: torch.Tensor, loss_w_cls: float, loss_w_reg: float,
                       reg_loss: str, smoothl1_beta: float): # "mse" or "smoothl1" # CL units
    """
    Forward + weighted loss.
    Returns: (loss, (logit, cent))
    """
    # ---- regression loss selector ----
    if reg_loss == "mse":
        reg_criterion = nn.MSELoss()
    elif reg_loss == "smoothl1":
        reg_criterion = nn.SmoothL1Loss(beta=smoothl1_beta)
    else:
        raise ValueError(f"Unknown reg_loss: {reg_loss}")

    out = model(x, extra=extra)
    logit, cent, _ = unpack_model_outputs(out, y_reg)

    total = 0.0
    used_head = False
    # ---- classification loss ----
    if (not y_cls.isnan().all()) and (loss_w_cls > 0) and (logit is not None):
        if logit.ndim == 2 and logit.shape[1] == 1:
            assert set(torch.unique(y_cls).tolist()) <= {0.0, 1.0}, "BCE requires binary {0,1} targets" # BCE path: targets must be float 0/1
            loss_cls = nn.BCEWithLogitsLoss()(logit.squeeze(1), y_cls.squeeze(1).float())
        else:
            yl = y_cls.squeeze(1).long()
            assert yl.min() >= 0 and yl.max() < logit.shape[1], "CE: target out of range" # CE path: targets must be long in [0..C-1]
            loss_cls = F.cross_entropy(logit, yl)
        total = total + loss_w_cls * loss_cls
        used_head = True
    # ---- regression loss ----
    if (not y_reg.isnan().all()) and (loss_w_reg > 0) and (cent is not None):
        loss_reg = reg_criterion(cent, y_reg)
        total += loss_w_reg * loss_reg
        used_head = True

    if not used_head: raise ValueError("No usable heads/targets: check model outputs and loss weights.")

    return total, (logit, cent)


def opt_threshold(ycls, probs, pos):
    ybin = (ycls == pos).astype(int)             # binary ground-truth 0/1
    prob1 = probs if probs.ndim == 1 else probs[:, pos]
    
    # ROC-derived best threshold (Youden's J)
    fpr, tpr, thr = roc_curve(ybin, prob1)
    # roc_curve returns thresholds aligned with tpr/fpr; choose max(tpr - fpr)
    j = tpr - fpr
    best_ix = int(np.argmax(j))
    best_thr = float(thr[best_ix])

    yhat_opt = (prob1 >= best_thr).astype(int)

    acc_opt = float((yhat_opt == ybin).mean())
    bacc    = float(balanced_accuracy_score(ybin, yhat_opt))
    f1      = float(f1_score(ybin, yhat_opt))
    mcc     = float(matthews_corrcoef(ybin, yhat_opt))
    best_thr= best_thr

    return acc_opt, bacc, f1, mcc, best_thr


def unpack_model_outputs(out: Any, y_reg) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize model outputs to (logit, cent, feats), where any can be None.

    Accepted patterns:
    - dict-like: keys 'logit'/'cls', 'cent'/'reg', 'feats' optional
    - tuple/list of length 3: (logit, cent, feats)
    - tuple/list of length 2: (logit, cent)  [feats=None]
    - single tensor: assumed to be 'logit' (classification) by default

    NOTE: If your model only returns regression, return a dict {'cent': tensor}
          or a single tensor AND set loss_w_cls=0 so the training path uses it as regression only.
    """
    # dict-like (preferred)
    if hasattr(out, "get"):
        logit = out.get("logit", out.get("cls", None))
        cent  = out.get("cent",  out.get("reg", None))
        feats = out.get("feats", None)
        return logit, cent, feats

    # tuple/list
    if isinstance(out, (tuple, list)):
        if len(out) == 3:
            return out[0], out[1], out[2]
        if len(out) == 2:
            return out[0], out[1], None
        if len(out) == 1:
            t = out[0]
            return t, None, None  # treat as single-head (classification by default)

    # single tensor
    if isinstance(out, torch.Tensor):
        if not y_reg.isnan().all():
            return None, out, None
        else:
            return out, None, None

    # unknown shape
    return None, None, None

