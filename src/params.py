# src/params.py
import os
import argparse
from datetime import datetime

def parse_arguments():
    # Load essential settings from external file if available
    script_dir, proj_path = get_proj_path()

    parser = argparse.ArgumentParser(description="Aβ-PET → visual read (binary) / Centiloid (regression) with Optuna hyperparameter tuning.")

    parser.add_argument("--model", type=str, default="CNN3D",
                help="Class name in models.py (e.g., CNN3D, UNet3D, ResNet50_3D, DenseNet121_3D...)")
    parser.add_argument('--model_name_extra', type=str, default="IDEAS_Inten_Norm", help='Extra name to be used as the result folder name. E.g. parameters or others tests names')
    parser.add_argument("--input_path", type=str, default='', help='images save in BIDS format. If not input, will set as <proj_path>/data')
    parser.add_argument("--data_suffix", type=str, default='', help='images finding pattern **/*<suffix>/*/*/*.nii* for find_pet_images function, specifically to IDEAS data. e.g._Inten_Norm')
    parser.add_argument("--targets", type=str, default="visual_read", help="Predict variables name, corresponds to column names in demographics.csv, seperate by ,")

    # Model args
    parser.add_argument("--model_kwargs", type=str, default="",
                    help='JSON string of extra kwargs for the selected model (e.g., \'{"features": 32}\')')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Training
    parser.add_argument("--epochs", type=int, default=200) # true model should start with 30 
    parser.add_argument("--loss_w_cls", type=float, default=1.0)
    parser.add_argument("--loss_w_reg", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8) # 8 on the cluster, 2 on mac
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to load (optional)")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision if CUDA is available.")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for StratifiedKFold.")
    parser.add_argument("--stratifycvby", default="visual_read,site", help=",site,tracer, List of column names to stratify by (e.g., visual_read CL age gender).")
    parser.add_argument("--image_shape", nargs=3, type=int, default=[128,128,128])

    # Hypertune - Optuna
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_trials", type=int, default=60) #60
    parser.add_argument("--proxy_epochs", type=int, default=60, help="Epochs per trial (proxy).")
    parser.add_argument("--proxy_folds", type=int, default=5, help="Folds per trial (proxy).") #5
    parser.add_argument("--study_name", type=str, default="optuna")
    parser.add_argument("--storage", type=str, default="", help='Optuna storage, e.g. "sqlite:///optuna.db"')
    parser.add_argument("--tune_timeout", type=int, default=None, help="Seconds to stop tuning (optional).")
    
    # Validation / Testing
    parser.add_argument("--dataset", type=str, default="IDEAS", choices=['ADNI', 'IDEAS'], help="Dataset name, ADNI or IDEAS")
    parser.add_argument("--best_model_folder", type=str, default="CNN3D_CL_2split80-20_stratify-visual_read,site_IDEAS_Inten_Norm_20251004_02221", help="Path to the folder that contains the best model checkpoint for external validation.")
    
    # Parse arguments and set up the output directory
    args, unknown = parser.parse_known_args()
    args = make_output_dir(args, proj_path, script_dir)

    return args


def get_proj_path():
    """
    Dynamically determine the project path based on the current script's location.
    
    Returns:
        list: A list containing the script directory and the project path.
    """
    # Print a message indicating that the project path is being set up
    print("Setting project path based on the current script directory")
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.abspath(os.path.join(current_file_dir, os.pardir))

    # Set the project path to the grandparent folder (uncomment if needed)
    proj_path = os.path.abspath(os.path.join(script_dir, os.pardir)) # set to grandparent folder
    print("Project path set to:", proj_path)
    return [script_dir, proj_path]


def make_output_dir(args, proj_path, script_path):
    """
    Create the output directory based on model arguments.

    Args:
        args (Namespace): Parsed arguments.
        proj_path (str): Project path.

    Returns:
        Namespace: Updated arguments with output path.
    """
    # Generate output date and time
    args.output_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Program starts at {}".format(datetime.strptime( args.output_date_time, "%Y%m%d_%H%M%S").strftime("%d-%m-%Y %H:%M:%S")))

    # Define paths
    args.proj_path = proj_path
    args.script_path = script_path
    if not args.input_path: args.input_path = os.path.join(proj_path, "data") # set input path to <proj_path>/data is not stated

    # Construct validation path
    if args.best_model_folder and not os.path.isabs(args.best_model_folder):
        args.best_model_folder = os.path.join(args.proj_path, "results", args.best_model_folder)
        args.output_path = os.path.join(args.best_model_folder, 'validation')
    else:
        # Construct output path'
        tune = f'hypertune-optuna-{args.n_trials}trials' if args.tune else '2split80-20'
        args.output_name = "_".join([args.model, args.targets, tune, f'stratify-{args.stratifycvby}', args.model_name_extra, args.output_date_time])
        args.output_path = os.path.join(proj_path, "results", args.output_name)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output directory created at: {args.output_path}")

    return args