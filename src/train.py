# src/train.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error, mean_squared_error


def train_one_epoch(model, loader, opt, scaler, device, loss_w_cls=1.0, loss_w_reg=1.0):
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
        print(float(loss.detach().item()), end='\t')
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_epoch(model, loader, device, loss_w_cls=1.0, loss_w_reg=1.0):
    model.eval()
    probs, ycls, preds = [], [], []
    cents, yreg = [], []

    any_cls = False
    any_reg = False

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

    metrics = {"val_loss": 0.0, "auc": np.nan, "acc": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan}

    if any_cls:
        ycls  = np.concatenate(ycls)
        preds = np.concatenate(preds)
        probs = np.concatenate(probs, axis=0)   # [N,out_channel]

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
            else:
                metrics["auc"] = float(roc_auc_score(ycls, probs, multi_class="ovr", average="macro", labels=labels))
        metrics["acc"] = float(accuracy_score(ycls, preds))
    if any_reg:
        cents = np.concatenate(cents)
        yreg  = np.concatenate(yreg)
        metrics["mae"]  = float(mean_absolute_error(yreg, cents))
        metrics["rmse"] = float(mean_squared_error(yreg, cents))
        metrics["r2"]   = float(r2_score(yreg, cents))

    # robust combined score proxy (only include what exists)
    parts = []
    if any_cls and np.isfinite(metrics["auc"]): parts.append(1 - metrics["auc"])
    if any_reg:
        if np.isfinite(metrics["mae"]):  parts.append(metrics["mae"])
        if np.isfinite(metrics["rmse"]): parts.append(metrics["rmse"])
        if np.isfinite(metrics["r2"]):   parts.append(1 - metrics["r2"])

    metrics["val_loss"] = float(np.sum(parts)) if parts else 0.0  # never NaN

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
            bce = nn.BCEWithLogitsLoss()
            loss_cls = bce(logit.squeeze(1), y_cls.squeeze(1).float())
        else:
            loss_cls = F.cross_entropy(logit, y_cls.squeeze(1).long())
        total = total + loss_w_cls * loss_cls
        used_head = True
    if (not y_reg.isnan().all()) and (loss_w_reg > 0) and (cent is not None):
        total = total + loss_w_reg * mse(cent, y_reg)
        used_head = True

    if not used_head: raise ValueError("No usable heads/targets: ensure your model returns the required outputs "
            "(logit and/or cent) and corresponding loss weights/targets are set.")

    return total, (logit, cent)

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

