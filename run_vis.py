from src.warnings import ignore_warnings
ignore_warnings()

import os
import torch
import pandas as pd

from src.params import parse_arguments
from src.utils import build_model_from_args, get_device, set_seed
from src.data import get_loader
from src.vis import run_visualization

import torch.multiprocessing as mp
os.environ["NIBABEL_KEEP_FILE_OPEN"] = "0"
mp.set_sharing_strategy("file_system")

def main(args):
    df_train = pd.read_csv(args.best_model_folder + '/train-test-split/train-test-split_training-set.csv', index_col=0)
    df_test = pd.read_csv(args.best_model_folder + '/train-test-split/train-test-split_testing-set.csv', index_col=0)
    df = pd.concat([df_train, df_test], ignore_index=True)
    if args.vis_img_list is not None:
        subject_id_list = [int(t.strip()) for t in args.vis_img_list.split(',') if t.strip()]
        df_select = df[df['ID'].isin(subject_id_list)]
    print('df_select:',df_select)
    dl_va = get_loader(df_select, args.input_path, args, batch_size=1, augment=False, shuffle=False, train_test='test')
    
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    n_classes = int(df["visual_read"].dropna().nunique()) if 'visual_read' in targets_list else None
    model = build_model_from_args(args, device=args.device, n_classes=n_classes)
    sd = torch.load(os.path.join(args.best_model_folder, 'train-test-split/checkpoints', 'train-test-split_best.pt'), map_location=args.device, weights_only=True)
    state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)

    run_visualization(model, dl_va, args.device, args.output_path, vis_name=args.visualization_name, vis_norm=None) # the best model on validation set, save .png and .nii

    print('DONE!')

    return 


if __name__ == "__main__":
    args = parse_arguments()
    args.device = get_device()
    #args.device = get_device(force_cpu=True)
    print("Using device:", args.device)
    print(args)

    set_seed(args.seed)

    main(args)
