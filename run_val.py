from src.warnings import ignore_warnings
ignore_warnings()

import os
import numpy as np
import pandas as pd

from src.params import parse_arguments
from src.utils import get_device, set_seed
from src.data import  get_loader
from src.train import evals, compute_metrics
from src.validation import load_validation_data, load_preatrained_model, finetune, freeze_all_but_last_k

import torch.multiprocessing as mp
os.environ["NIBABEL_KEEP_FILE_OPEN"] = "0"
mp.set_sharing_strategy("file_system")

def main(args):
    # Load Validation Dataset
    tfm, dl_va, df = load_validation_data(args)
    # Load Pretrained Model
    model, targets_list = load_preatrained_model(args, df)
    
    # FEW-SHOT FINETUNING 
    if args.few_shot_csv is not None:
        print("\n========== FEW-SHOT FINETUNING MODE ==========\n")
        # Load few-shot dataset
        df_fs = pd.read_csv(args.few_shot_csv)
        dl_fs_tr = get_loader(df_fs, tfm, args, augment=True, shuffle=True)
        dl_fs_va = get_loader(df_fs, tfm, args, augment=False, shuffle=False)

        # Freeze except last K layers
        model = freeze_all_but_last_k(model, args.unfreeze_layers)
        # Fine-tune
        model = finetune(model, dl_fs_tr, dl_fs_va, args, epochs=args.finetune_epochs)
        print("\n========== FEW-SHOT FINETUNING COMPLETE ==========\n")

    # FINAL VALIDATION
    probs, ycls, preds, any_cls, cents, yreg, any_reg = evals(model, dl_va, args.device)
    if 'visual_read' in targets_list:
        df_result = pd.DataFrame({'y': np.concatenate(ycls, axis=0), 'pred':np.concatenate(preds, axis=0), 'prob':np.concatenate(probs, axis=0)})
    elif 'CL' in targets_list:
        df_result = pd.DataFrame({'y': np.concatenate(yreg, axis=0), 'pred':np.concatenate(cents, axis=0)})
    
    out_csv = os.path.join(args.output_path, f'External_validation_{args.dataset}_{args.data_suffix}_{args.targets}.csv')
    df_result.to_csv(out_csv)
    print(f"Saved predictions → {out_csv}")

    metrics = compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg)
    print("\nFINAL METRICS:", metrics)
    print('DONE!')


if __name__ == "__main__":
    args = parse_arguments()
    #args.device = get_device()
    args.device = get_device(force_cpu=True)
    print(args)
    set_seed(args.seed)

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"best_model_folder : {args.best_model_folder}")
    print(f"voxel_sizes       : {args.voxel_sizes}")
    print(f"few_shot_csv      : {args.few_shot_csv}")
    print(f"unfreeze_layers   : {args.unfreeze_layers}")
    print(f"finetune_epochs   : {args.finetune_epochs}")
    print(f"device            : {args.device}")
    print("===========================================\n")

    main(args)
