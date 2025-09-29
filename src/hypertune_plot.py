#src/hypertune_plot.py 
import os
import json
import math
import pandas as pd
import matplotlib.pyplot as plt


def optuna_plot(output_path, study):
    # Persist and visualize tuning
    tune_dir = os.path.join(output_path, "tuning")
    save_study_csv(study, tune_dir)
    save_best_params_json(study, tune_dir)
    plot_optimization_history(study, os.path.join(tune_dir, "history.png"))
    plot_param_scatter(study, os.path.join(tune_dir, "param_scatter.png"))
    plot_optuna_mpl(study, tune_dir)  # parallel coordinate / slice / importances (if available)
    log_top_trials(study, tune_dir, k=10)


def save_study_csv(study, out_dir: str) -> str:
    """
    Save all trials as a CSV for later analysis.
    Returns the csv path.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "system_attrs", "duration"))
    csv_path = os.path.join(out_dir, "optuna_trials.csv")
    df.to_csv(csv_path, index=False)

    print(f"[tuning] Wrote: {csv_path} and plots under {out_dir}")


def save_best_params_json(study, out_dir: str) -> str:
    """
    Save best params and value to a JSON file.
    """
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
    }
    path = os.path.join(out_dir, "best_params.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_optimization_history(study, out_png: str):
    """
    Simple optimization history (best value over trials).
    Uses matplotlib (no Plotly dependency).
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    values = []
    best_so_far = []
    cur_best = math.inf if study.direction.name == "MINIMIZE" else -math.inf

    for t in study.trials:
        if t.value is None or not math.isfinite(t.value):
            values.append(float("nan"))
            best_so_far.append(cur_best if math.isfinite(cur_best) else float("nan"))
            continue
        v = float(t.value)
        values.append(v)
        if study.direction.name == "MINIMIZE":
            cur_best = v if v < cur_best else cur_best
        else:
            cur_best = v if v > cur_best else cur_best
        best_so_far.append(cur_best)

    x = list(range(len(values)))
    plt.figure(figsize=(9, 5))
    plt.plot(x, values, marker="o", label="trial value")
    plt.plot(x, best_so_far, linestyle="--", label="best so far")
    plt.xlabel("Trial")
    plt.ylabel("Objective")
    plt.title("Optuna Optimization History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_param_scatter(study, out_png: str):
    """
    For each numeric/categorical parameter, scatter trial value vs parameter.
    Matplotlib-only, robust when optuna.visualization is unavailable.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    trials = [t for t in study.trials if t.value is not None and math.isfinite(t.value)]
    if not trials:
        return

    # Collect params
    all_keys = set()
    for t in trials:
        all_keys.update(t.params.keys())
    all_keys = sorted(all_keys)
    if not all_keys:
        return

    n = len(all_keys)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(6 * cols, 4.5 * rows))

    for i, k in enumerate(all_keys, start=1):
        xs, ys = [], []
        for t in trials:
            if k in t.params:
                xs.append(t.params[k])
                ys.append(float(t.value))
        if not xs:
            continue
        ax = plt.subplot(rows, cols, i)
        # Handle categorical: map to integers
        if not isinstance(xs[0], (int, float)):
            cats = {v: j for j, v in enumerate(sorted(set(xs)))}
            ax.scatter([cats[v] for v in xs], ys, alpha=0.8)
            ax.set_xticks(list(cats.values()))
            ax.set_xticklabels(list(cats.keys()), rotation=30, ha="right")
        else:
            ax.scatter(xs, ys, alpha=0.8)
        ax.set_xlabel(k)
        ax.set_ylabel("Objective")
        ax.set_title(f"value vs {k}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_optuna_mpl(study, out_dir: str):
    """
    If available, use Optuna's matplotlib visualizers:
      - parallel coordinate
      - slice
      - param importance (requires importance module)
    Silently no-op if modules are missing.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        from optuna.visualization.matplotlib import plot_parallel_coordinate, plot_slice
        fig = plot_parallel_coordinate(study)
        fig.set_size_inches(9, 6)
        fig.savefig(os.path.join(out_dir, "parallel_coordinate.png"), dpi=150)
        plt.close(fig)

        fig = plot_slice(study)
        fig.set_size_inches(9, 6)
        fig.savefig(os.path.join(out_dir, "slice.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # Importance (optional)
    try:
        from optuna.importance import get_param_importances
        imp = get_param_importances(study)
        if imp:
            keys = list(imp.keys())
            vals = [imp[k] for k in keys]
            plt.figure(figsize=(8, 5))
            plt.barh(keys, vals)
            plt.xlabel("Importance")
            plt.title("Hyperparameter Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "param_importances.png"), dpi=150)
            plt.close()
    except Exception:
        pass


def log_top_trials(study, out_dir: str, k: int = 10) -> str:
    """
    Save a small CSV of the top-k trials with params and values.
    """
    os.makedirs(out_dir, exist_ok=True)
    trials = [t for t in study.trials if t.value is not None and math.isfinite(t.value)]
    if not trials:
        path = os.path.join(out_dir, "top_trials.csv")
        pd.DataFrame(columns=["number", "value", "params"]).to_csv(path, index=False)
        return path

    trials = sorted(trials, key=lambda t: t.value, reverse=(study.direction.name == "MAXIMIZE"))
    top = trials[:k]
    rows = []
    for t in top:
        rows.append({
            "number": t.number,
            "value": float(t.value),
            "params": json.dumps(t.params),
            "duration_sec": getattr(t, "duration", None).total_seconds() if getattr(t, "duration", None) else None,
            "state": str(t.state),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "top_trials.csv")
    df.to_csv(path, index=False)
    return path
