# src/train.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error, root_mean_squared_error, roc_curve, balanced_accuracy_score, f1_score, matthews_corrcoef


def train_one_epoch(model, loader, opt, scaler, device, output_loss=None, loss_w_cls=1.0, loss_w_reg=1.0):
    model.train()
    mse = nn.MSELoss()
    losses = []

    for x, y_cls, y_reg, _ in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y_cls = y_cls.to(device) if y_cls is not None else None
        y_reg = y_reg.to(device) if y_reg is not None else None

        opt.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, _ = compute_total_loss(model, x, y_cls, y_reg,
                    loss_w_cls=loss_w_cls, loss_w_reg=loss_w_reg)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss, _ = compute_total_loss(model, x, y_cls, y_reg,
                loss_w_cls=loss_w_cls, loss_w_reg=loss_w_reg,
                mse=mse)
            loss.backward()
            opt.step()

        losses.append(float(loss.detach().item()))
    losses_dict = {i: v for i, v in enumerate(losses)}

    return (float(np.mean(losses)), losses_dict) if losses else (0.0, {})


@torch.no_grad()
def eval_epoch(model, loader, device, loss_w_cls=1.0, loss_w_reg=1.0):
    probs, ycls, preds, any_cls, cents, yreg, any_reg = evals(model, loader, device)

    metrics = compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg)

    # robust combined score proxy (only include what exists)
    parts = []
    if any_cls: 
        if np.isfinite(metrics["auc"]): parts.append(metrics["auc"])
        if np.isfinite(metrics["acc"]): parts.append(metrics["acc"])
    if any_reg:
        #yreg = yreg.detach().cpu().numpy() if isinstance(yreg, torch.Tensor) else np.asarray(yreg)
        yreg = np.concatenate(yreg)
        mae_ref = np.median(np.abs(yreg - np.median(yreg)))
        if not np.isfinite(mae_ref) or mae_ref <= 1e-6:
            mae_ref = max(np.std(yreg), 1e-6)
        mae_good = 1.0 - np.clip(metrics["mae"] / mae_ref, 0.0, 1.0) if np.isfinite(metrics["mae"]) else np.nan
        r2_good  = metrics["r2"] if np.isfinite(metrics["r2"]) else np.nan
        parts.append(mae_good)
        parts.append(r2_good)

    metrics["val_metric"] = float(np.sum(parts)) if parts else 0.0  # never NaN

    return metrics


@torch.no_grad()
def evals(model, loader, device):
    model.eval()

    probs, ycls, preds = [], [], []
    cents, yreg = [], []
    any_cls, any_reg = False, False
    for x, y_cls, y_reg, _ in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        out = model(x)
        logit, cent, _ = unpack_model_outputs(out, y_cls, y_reg)

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
    
    metrics = {"auc": np.nan, "acc": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "val_metric": 0.0} #"val_loss": 0.0, 

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

    return metrics


def compute_total_loss(model: torch.nn.Module, x: torch.Tensor, y_cls: Optional[torch.Tensor], y_reg: Optional[torch.Tensor],
                       *, loss_w_cls: float = 1.0, loss_w_reg: float = 1.0,
                       mse: Optional[nn.Module] = None):
    """
    Forward + weighted loss. Handles missing heads/targets.
    Returns: (loss, (logit, cent))
    """
    if mse is None: mse = nn.MSELoss()

    out = model(x)
    logit, cent, _ = unpack_model_outputs(out, y_cls, y_reg)

    total = 0.0
    used_head = False

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
    if (not y_reg.isnan().all()) and (loss_w_reg > 0) and (cent is not None):
        total = total + loss_w_reg * mse(cent, y_reg)
        used_head = True

    if not used_head: raise ValueError("No usable heads/targets: ensure your model returns the required outputs "
            "(logit and/or cent) and corresponding loss weights/targets are set.")

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


def unpack_model_outputs(out: Any, y_cls, y_reg) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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

