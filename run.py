from src.warnings import ignore_warnings
ignore_warnings()

from src.params import parse_arguments
from src.utils import set_seed, make_splits, hold_out_set, save_train_test_subjects, clone_args
from src.data import build_master_table
from src.cv import get_stratify_labels, run_fold, cv_median_best_epoch
from src.hypertune import create_study_from_args, run_optuna, objective, print_best, get_best_args


def main(args):
    # 1) data
    df = build_master_table(args.input_path, args.data_suffix, args.targets, args.dataset)
    df_clean, stratify_labels = get_stratify_labels(df, args.stratifycvby)

    # held-out set (never touch during hyperparameter tunning
    tr_idx, te_idx = hold_out_set(df_clean, stratify_labels, test_size=0.2, seed=args.seed)
    df_train = df_clean.iloc[tr_idx].reset_index(drop=True)
    df_test = df_clean.iloc[te_idx].reset_index(drop=True)
    save_train_test_subjects(df_train, df_test, args.output_path, 'Hold-out')
    _, stratify_labels_train = get_stratify_labels(df_train, args.stratifycvby)
    
    # 2) tuning or direct CV
    if args.tune: # default is tuninig    
        # --- Tuning ---
        print('Runing Hyperparameter tuning...')
        splits = make_splits(df_train, stratify_labels_train, args.n_splits, args.seed)
        # to save the splits
        study  = create_study_from_args(args)
        study  = run_optuna(study, objective, args, df_train, splits, args.model)
        print_best(study)

        # Retrain with best params (full epochs) on df_train and evaluate ONCE on df_test
        best_args = get_best_args(args, study, out_subdir="best_params")
        E_final = cv_median_best_epoch(df_train, stratify_labels_train, best_args)
        print(f"Final retrain epochs (median best_epoch across folds): {E_final}")
        best_args_fixed = clone_args(best_args, epochs=E_final)
        print("\nRetraining with best params on full training set…")
        print(best_args_fixed)

        print("\nRetraining on FULL TRAIN pool with fixed epochs (no early stop), then one-shot TEST eval…")
        final_metrics = run_fold(df_train, df_test, best_args_fixed, fold_name="outer-test",
            use_early_stop=False, use_scheduler=False, final_retrain=True)

        print(f"\nOUTER TEST: AUC={final_metrics.get('auc'):.3f} "
              f"ACC={final_metrics.get('acc'):.3f} MAE={final_metrics.get('mae'):.2f} "
              f"RMSE={final_metrics.get('rmse'):.2f} R2={final_metrics.get('r2'):.3f}")

    else: # this is just to test the model, not valid for publication
        print('NOOO tuning...')
        # --- Direct CV (no tuning) on train_pool to gauge stability ---
        # kfold_cv(df_train, stratify_labels_train, args)
        # 2 split only: single final train vs test
        final_metrics = run_fold(df_train, df_test, args, fold_name="outer-test")
        print(f"\nOUTER TEST: AUC={final_metrics.get('auc'):.3f} "
              f"ACC={final_metrics.get('acc'):.3f} MAE={final_metrics.get('mae'):.2f} "
              f"RMSE={final_metrics.get('rmse'):.2f} R2={final_metrics.get('r2'):.3f}")

    print('DONE!')

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    #args.device = get_device(force_cpu=True)
    print("Using device:", args.device)

    set_seed(args.seed)

    main(args)