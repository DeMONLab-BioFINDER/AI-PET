from src.warnings import ignore_warnings
ignore_warnings()

import os
import torch
import numpy as np
import pandas as pd

from src.params import parse_arguments
from src.utils import build_model_from_args, get_device, set_seed
from src.data import build_master_table, get_transforms, get_loader
from src.train import evals, compute_metrics


def main(args):
    if args.dataset == 'ADNI': # Berkeley server, load NIfTI files
        test_set = os.path.join(args.input_path, 'ADNI_found_scans.csv')
        if os.path.exists(test_set):
            print('loading validation dataframe')
            df = pd.read_csv(test_set, index_col=0)
        else:
            print('finding scans from folder')
            df = build_master_table(args.input_path, args.data_suffix, args.targets, subjects='demographics_adni.csv')
            df.to_csv(os.path.join(args.input_path, 'ADNI_found_scans.csv'))
        tfm = get_transforms()
    elif args.dataset == 'IDEAS': # Berzelius, load torch tensors
        print('Validate on IDEAS test set...')
        test_set = os.path.join(args.best_model_folder,'Hold-out_testing-set.csv')
        print(test_set)
        df = pd.read_csv(test_set, index_col=0)
        tfm = args.input_path
    
    dl_va = get_loader(df, tfm, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False)
    
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    n_classes = int(df["visual_read"].dropna().nunique()) if 'visual_read' in targets_list else None
    model = build_model_from_args(args, device=args.device, n_classes=n_classes)
    sd = torch.load(os.path.join(args.best_model_folder, 'outer-test/checkpoints', 'best.pt'), map_location=args.device, weights_only=True)
    state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)
    
    probs, ycls, preds, any_cls, cents, yreg, any_reg = evals(model, dl_va, args.device)
    if 'visual_read' in targets_list:
        df_result = pd.DataFrame({'y': np.concatenate(ycls, axis=0), 'pred':np.concatenate(preds, axis=0), 'prob':np.concatenate(probs, axis=0)})
    elif 'CL' in targets_list:
        df_result = pd.DataFrame({'y': np.concatenate(yreg, axis=0), 'pred':np.concatenate(cents, axis=0)})
    df_result.to_csv(os.path.join(args.output_path, f'External_validation_{args.dataset}_{args.targets}.csv'))

    metrics = compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg)
    print(metrics)

    print('DONE!')

    return 


if __name__ == "__main__":
    args = parse_arguments()
    #args.device = get_device()
    args.device = get_device(force_cpu=True)
    print("Using device:", args.device)
    print(args)

    set_seed(args.seed)

    main(args)