# src/hyperparam_spaces.py
import json
from typing import Dict, Tuple, Optional

# ---- COMMON (shared across models) ----
def suggest_common(trial, base_args) -> Dict:
    """Common hyperparams (LR, WD, BS, dropout)."""
    return {
        "lr":           trial.suggest_float("lr", 1e-5, 3e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 5e-4),
        "batch_size":   trial.suggest_categorical("batch_size", [2, 4, 8]),
        "dropout":      trial.suggest_float("dropout", 0.0, 0.5),
    }

# ---- MODEL-SPECIFIC SUGGESTORS ----
def suggest_CNN3D(trial, base_args, common: Dict) -> Tuple[str, str]:
    """Return (model_name, model_kwargs_json) for CNN3D."""
    opts = [(16,32,64), (16,32,64,128), (32,64,128,256)]
    key = trial.suggest_categorical("cnn_widths", [json.dumps(o) for o in opts])
    widths = tuple(json.loads(key))
    norm       = trial.suggest_categorical("cnn_norm", ["batch", "instance"])
    pool_every = trial.suggest_categorical("cnn_pool_every", [1, 2])

    kwargs = {
        "in_channels": getattr(base_args, "in_channels", 1),
        "widths": list(widths),
        "pool_every": pool_every,
        "norm": norm,
        "dropout": common["dropout"],
    }
    return "CNN3D", json.dumps(kwargs)

def suggest_UNet3D(trial, base_args, common: Dict) -> Tuple[str, str]:
    """Return (model_name, model_kwargs_json) for MONAI UNet3D/BasicUNet."""
    use_basic   = trial.suggest_categorical("unet_use_basic", [True, False])
    opts = [(16,16,32,64,128,32), (16,16,32,64,128,16), (32,32,64,96,128,32), (32,32,64,128,256,32),] # basic UNET
    key = trial.suggest_categorical("unet_channels", [json.dumps(o) for o in opts])
    channels = tuple(json.loads(key))
    num_res     = trial.suggest_categorical("unet_num_res_units", [1, 2])

    kwargs = {
        "in_channels": getattr(base_args, "in_channels", 1),
        "out_channels": 1,
        "channels": list(channels),
        "strides": [2,2,2,2],
        "num_res_units": num_res,
        "norm": "instance",
        "dropout": min(common["dropout"], 0.3),  # U-Net usually needs less dropout
        "use_basic": use_basic,
    }
    return "UNet3D", json.dumps(kwargs)


# ---- REGISTRY ----
SUGGESTORS = {
    "CNN3D":  suggest_CNN3D,
    "UNet3D": suggest_UNet3D,
}

def suggest_model(trial, base_args, common: Dict, model_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Return (model_name, model_kwargs_json) for a specific model.
    No model selection is performed.

    model_name resolution order:
    1) explicit model_name arg
    2) base_args.model_name or base_args.model or base_args.arch
    """
    name = (model_name
        or getattr(base_args, "model_name", None)
        or getattr(base_args, "model", None)
        or getattr(base_args, "arch", None))
    if name is None:
        if len(SUGGESTORS) == 1:
            name = next(iter(SUGGESTORS))
        else:
            raise ValueError(
                "Please specify the model to tune via model_name=... "
                "or base_args.model_name/model/arch."
            )
    if name not in SUGGESTORS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(SUGGESTORS.keys())}")

    # (Optional) keep the chosen model in the study for bookkeeping
    try:
        trial.set_user_attr("arch", name)
    except Exception:
        pass

    return SUGGESTORS[name](trial, base_args, common)

#def suggest_model(trial, base_args, common: Dict) -> Tuple[str, str]:
#    """
#    Pick an architecture and return (model_name, model_kwargs_json).
#    If you want a fixed model, skip the 'arch' suggestion and call its suggestor directly.
#    """
#    arch = trial.suggest_categorical("arch", list(SUGGESTORS.keys()))
#    return SUGGESTORS[arch](trial, base_args, common)
