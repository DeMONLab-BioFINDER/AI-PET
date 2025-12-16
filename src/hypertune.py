# src/hypertune.py
import os
import optuna
from copy import deepcopy
from types import SimpleNamespace

from src.cv import run_fold
from src.hyperparam_spaces import suggest_common, suggest_model
from src.utils import clone_args, combine_metrics_for_minimize
from src.hypertune_plot import optuna_plot

def objective(trial, base_args, df_clean, splits, model_name):
    """
    One Optuna trial: suggest hyperparams -> run proxy K-fold -> return mean score (lower=better).
    """
    # --- hyperparams to be tuned ---
    common = suggest_common(trial, base_args)
    model_name, model_kwargs_json = suggest_model(trial, base_args, common, model_name)

    # proxy settings (shorter training)
    proxy_epochs = getattr(base_args, "proxy_epochs", 8)
    proxy_folds  = getattr(base_args, "proxy_folds", len(splits))

    # patch args for this trial
    targs = clone_args(
        base_args,
        model=model_name,
        model_kwargs=model_kwargs_json,       # JSON string only
        lr=common["lr"],
        weight_decay=common["weight_decay"],
        batch_size=common["batch_size"] if model_name != "UNet3D" else min(common["batch_size"], 4), # UNet often needs smaller batch sizes
        dropout=common["dropout"],            # in case your code reads it elsewhere
        epochs=min(base_args.epochs, proxy_epochs),
        output_path=os.path.join(base_args.output_path, f"optuna_trial_{trial.number}"),
        es_patience=getattr(base_args, "es_patience", 10),
        es_min_delta=getattr(base_args, "es_min_delta", 1e-3),
    )

    # run proxy folds
    scores = []
    reporter = PruningReporter(trial)  # <— stateful callback
    for i, (tr_idx, va_idx) in enumerate(splits[:proxy_folds], start=1):
        fold_name = f"trial{trial.number}-k{i}"
        train_df = df_clean.iloc[tr_idx].reset_index(drop=True)
        val_df   = df_clean.iloc[va_idx].reset_index(drop=True)

        # run one fold (your run_fold already logs/plots inside its own folder)
        m = run_fold(train_df, val_df, targs, fold_name=fold_name, on_epoch_end=reporter)
        val = combine_metrics_for_minimize(m)
        scores.append(val)

        # pruning support (report intermediate)
        trial.report(sum(scores)/len(scores), step=reporter.step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    # plot and log
    #if trial.number % 5 == 0:
    try:
        optuna_plot(targs.output_path, trial.study)
    except Exception: # never break the trial because of plotting I/O
        pass
    return sum(scores)/len(scores)

class PruningReporter:
    """Keeps a monotonic step counter and reports to Optuna per epoch."""
    def __init__(self, trial: optuna.trial.Trial):
        self.trial = trial
        self.step = 0

    def __call__(self, fold_idx: int, epoch: int, val_loss: float):
        self.step += 1
        self.trial.report(val_loss, step=self.step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

def create_study_from_args(args) -> optuna.study.Study:
    """Create an Optuna study using CLI args (direction, storage, pruner)."""
    study_kwargs = dict(direction="minimize")

    if getattr(args, "study_name", ""):
        study_kwargs["study_name"] = args.study_name
    if getattr(args, "storage", ""):
        # e.g., "sqlite:///optuna.db"
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True

    # Sampler: explicit TPE with good defaults
    study_kwargs["sampler"] = optuna.samplers.TPESampler(
        seed=getattr(args, "seed", None), n_startup_trials=10,
        multivariate=True, group=True)
    
    # Default pruner: ASHA
    study_kwargs["pruner"] = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,# start pruning from very early epochs
        reduction_factor=3,     # 1/3 kept each rung
        min_early_stopping_rate=0)

    return optuna.create_study(**study_kwargs)


def run_optuna(study, objective_fn, args, df_clean, splits, model_name):
    """
    Optimize the given objective function with the study and return it (for chaining).
    """
    study.optimize(
        lambda trial: objective_fn(trial, args, df_clean, splits, model_name),
        n_trials=getattr(args, "n_trials", 30),
        timeout=getattr(args, "tune_timeout", None),  # seconds or None
        gc_after_trial=True,
    )
    return study


def get_best_args(args, study, out_subdir: str = "best_params") -> SimpleNamespace:
    """
    Merge best trial params into args, fix types, and set output_path/epochs for retraining.
    """
    best = deepcopy(vars(args))
    best.update(study.best_params)

    # widths may come back as tuple; keep consistency with your code
    if "widths" in best and isinstance(best["widths"], tuple):
        best["widths"] = list(best["widths"])

    best_args = SimpleNamespace(**best)
    best_args.epochs = args.epochs  # full epochs for retrain
    best_args.output_path = os.path.join(args.output_path, out_subdir)
    os.makedirs(best_args.output_path, exist_ok=True)
    return best_args


def print_best(study):
    print("\nBest trial:")
    print("  value:", study.best_value)
    print("  params:", study.best_params)