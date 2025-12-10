# finetune.py
from src.warnings import ignore_warnings
ignore_warnings()

import os
import torch
import pandas as pd

from src.data import build_master_table, get_transforms, get_loader
from src.train import train_one_epoch, evals, compute_metrics
from src.utils import compute_smooth_sigma_vox, build_model_from_args

import torch.multiprocessing as mp
os.environ["NIBABEL_KEEP_FILE_OPEN"] = "0"
mp.set_sharing_strategy("file_system")


def load_validation_data(args):
    """
    Load the validation dataset based on the specified dataset type in args.

    Returns:
        tuple: A tuple containing the transformations and the validation dataloader. 
    Notes
    -----
    - For 'ADNI' dataset, NIfTI files are loaded and smoothed if voxel sizes are provided.
    - For 'IDEAS' dataset, torch tensors are loaded directly from the specified input path. 
    
    """

    if 'ADNI' in args.dataset: # Berkeley server, load NIfTI files
        print('Validate on ADNI test set...')
        test_set = os.path.join(args.proj_path, "data", f'{args.dataset}_found_scans_{args.data_suffix}_{args.targets}.csv')
        if os.path.exists(test_set):
            print('loading validation dataframe')
            df = pd.read_csv(test_set, index_col=0)
        else:
            print('finding scans from folder')
            df = build_master_table(args.input_path, args.data_suffix, args.targets, args.dataset)
            df.to_csv(os.path.join(args.proj_path, "data", f'{args.dataset}_found_scans_{args.data_suffix}_{args.targets}.csv'))
        
        sigma_vox = compute_smooth_sigma_vox(args.voxel_sizes, fwhm_current_mm=6.0, fwhm_target_mm=10.0) if args.voxel_sizes else None # for ADNI SCANS
        tfm = get_transforms(smooth_sigma_vox = sigma_vox)
    elif 'IDEAS' in args.dataset: # Berzelius, load torch tensors
        print('Validate on IDEAS test set...')
        test_set = os.path.join(args.best_model_folder,'Hold-out_testing-set.csv')
        print(test_set)
        df = pd.read_csv(test_set, index_col=0)
        tfm = args.input_path
    
    dl_va = get_loader(df, tfm, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False)

    return tfm, dl_va, df

def load_preatrained_model(args, df) -> torch.nn.Module:
    """
    Load a pretrained model checkpoint into the given model architecture.
    Parameters
    ----------
    args : argparse.Namespace in params.py
    df : DataFrame containing the dataset information (used to determine n_classes)

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    n_classes = int(df["visual_read"].dropna().nunique()) if 'visual_read' in targets_list else None

    model = build_model_from_args(args, device=args.device, n_classes=n_classes)

    ckpt = os.path.join(args.best_model_folder, "outer-test/checkpoints/best.pt")
    print(f"Loading pretrained model: {ckpt}")
    sd = torch.load(ckpt, map_location=args.device, weights_only=True)

    state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)

    return model, targets_list


def freeze_all_but_last_k(model, k: int):
    """
    Freezes all layers except the last K layers of the model.head (or Conv stack).
    K = 1 means: only the final linear layer is trainable.
    """

    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the last K layers of the Sequential head
    head_layers = list(model.head.children())
    if k > len(head_layers): 
        k = len(head_layers)

    unfreeze = head_layers[-k:]

    for layer in unfreeze:
        for p in layer.parameters():
            p.requires_grad = True
    
    print(f"✓ Unfroze last {k} layers of the model head.")
    return model


def finetune(model: torch.nn.Module, dl_tr: torch.utils.data.DataLoader, dl_va: torch.utils.data.DataLoader, args, epochs: int = 10) -> torch.nn.Module:
    """
    Fine-tune the *last few layers* of a pretrained model using a small labeled dataset.

    Parameters
    ----------
    model : The pretrained model with selected layers unfrozen (typically via `freeze_all_but_last_k()` before calling this function).
    dl_tr : Few-shot training dataloader. Should contain very small number of subjects (e.g., 5–20).
    dl_va : Few-shot validation dataloader (usually same subjects or a small held-out subset).
    args : argparse.Namespace in params.py
    epochs : Number of finetuning epochs. Defaults to 10. Note: In few-shot learning, 5–20 epochs is often sufficient.

    Returns
    -------
        The finetuned model with weights updated using few-shot data.
        The returned model has best-validation weights loaded into memory.

    Notes
    -----
    - Only parameters with `requires_grad=True` are updated.
    - Best checkpoint is selected by lowest `val_loss`.
    - Uses the existing train_one_epoch(), evals(), and compute_metrics() utilities
      from your training framework for consistency.

    """
    # ---------------------------
    # Optimizer only updates trainable (unfrozen) layers
    # ---------------------------
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr,
                            weight_decay=args.weight_decay,)

    # Mixed precision scaler (only used if CUDA available)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Track best validation score
    best_val = float("inf")
    ckpt_path = os.path.join(args.output_path, "finetuned.pt")

    for epoch in range(1, epochs + 1):
        print(f"\n---- Fine-tuning Epoch {epoch}/{epochs} ----")

        # TRAIN on few-shot data
        tr_loss, _ = train_one_epoch(model, dl_tr, opt, scaler, args.device, args.loss_w_cls, args.loss_w_reg)
        print(f"Train loss = {tr_loss:.4f}")

        # VALIDATE
        probs, ycls, preds, any_cls, cents, yreg, any_reg = evals(model, dl_va, args.device)
        metrics = compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg)
        print(f"Validation metrics = {metrics}")

        # SAVE BEST MODEL CHECKPOINT
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"✓ Saved finetuned checkpoint → {ckpt_path}")

    # ---------------------------
    # Load best validation checkpoint
    # ---------------------------
    sd = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(sd["model"], strict=False)

    return model


def inference_one_scan(model_path, pet_path, args, id=None):
    model = torch.load(model_path, map_location=args.device) # using device of current 

    if id is None: id = pet_path.split('/')[-1].split('_')[1] # specific to IDEAS dataset
    df = pd.DataFrame({'ID': id, 'pet_path': pet_path})

    tfm = get_transforms(tuple(args.image_shape))
    loader = get_loader(df, tfm, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False)

    # evaluations
    probs, ycls, preds, any_cls, cents, yreg, any_reg = evals(model, loader, args.device)
    if not torch.isnan(ycls).all() or not torch.isnan(yreg).all():
        metrics = compute_metrics(ycls, preds, probs, any_cls, yreg, cents, any_reg)
    else:
        metrics = None

    # attention map
    _ = explain_one_occlusion(model, pet_path, out_nifti_path=None, return_affine=False,
                                 mask_size=(16,16,16), overlap=0.6, activate=True, device="cuda")