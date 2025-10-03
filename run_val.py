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
    if os.path.exists(args.input_path + 'ADNI_found_scans.csv'):
        print('loading validation dataframe')
        df = pd.read_csv(args.input_path + 'ADNI_found_scans.csv', index_col=0)
    else:
        print('finding scans from folder')
        df = build_master_table(args.input_path, args.data_suffix, args.targets, subjects='demographics_adni.csv')
        df.to_csv(args.input_path + 'ADNI_found_scans.csv')

    tfm = get_transforms()
    dl_va = get_loader(df, tfm, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False)
    
    model = build_model_from_args(args, device=args.device, n_classes=2)
    sd = torch.load(args.input_path + 'model_VR_2split.pt', map_location=args.device, weights_only=True)
    state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)
    
    probs, ycls, preds, any_cls, cents, yreg, any_reg = evals(model, dl_va, args.device)
    df_result = pd.DataFrame({'y': np.concatenate(ycls, axis=0), 'pred':np.concatenate(preds, axis=0), 'prob':np.concatenate(probs, axis=0)})
    df_result.to_csv(args.output_path + 'External_validation.csv')

    metrics = compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg)
    print(metrics)

    print('DONE!')


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    args.device = get_device()
    #args.device = get_device(force_cpu=True)
    print("Using device:", args.device)

    set_seed(args.seed)

    main(args)