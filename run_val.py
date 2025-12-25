from src.warnings import ignore_warnings
ignore_warnings()

import os
import numpy as np
import pandas as pd

from src.params import parse_arguments
from src.utils import get_device, set_seed
from src.train import inference
from src.validation import load_validation_data, load_preatrained_model, run_few_shots

import torch.multiprocessing as mp
os.environ["NIBABEL_KEEP_FILE_OPEN"] = "0"
mp.set_sharing_strategy("file_system")

def main(args):
    # Load the full external validation Dataset
    tfm, dl, df = load_validation_data(args)
    # Load Pretrained Model
    model, targets_list = load_preatrained_model(args, df)
    
    # FEW-SHOT FINETUNING 
    if args.few_shot > 0:
        print("\n========== FEW-SHOT FINETUNING MODE ==========\n")
        print('set finetune epochs', args.epochs, '(orginal) to', args.finetune_epochs)
        args.epochs = args.finetune_epochs
        df_metrics, df_results, df_ids = run_few_shots(args, df, tfm, model, targets_list)
        print("metrics:")
        print(df_metrics.describe())
        out_csv_prefix = os.path.join(args.output_path, f'External_validation_{args.dataset}_{args.data_suffix}_{args.targets}_fewshot-{args.few_shot}_iter-{args.few_shot_iterations}')
        df_ids.to_csv(f"{out_csv_prefix}_subject_ids.csv", index=False)
        print("\n========== FEW-SHOT FINETUNING COMPLETE ==========\n")
    else:
        print("\n========== INFERENCE DIRECTLY ==========\n")
        metrics, df_results = inference(model, dl, args.device)
        print("metrics:", metrics)
        df_metrics = pd.DataFrame([metrics])
        out_csv_prefix = os.path.join(args.output_path, f'External_validation_{args.dataset}_{args.data_suffix}_{args.targets}_zeroshot')
        print("\n========== ZERO-SHOT COMPLETE ==========\n")

    # Save Results
    df_results.to_csv(f'{out_csv_prefix}_results.csv', index=False)
    df_metrics.to_csv(f'{out_csv_prefix}_metrics.csv', index=False)
    print(f"Saved predictions -> {out_csv_prefix}")

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
    print(f"few_shot          : {args.few_shot}")
    print(f"unfreeze_layers   : {args.unfreeze_layers}")
    print(f"finetune_epochs   : {args.finetune_epochs}")
    print(f"device            : {args.device}")
    print("===========================================\n")

    main(args)
