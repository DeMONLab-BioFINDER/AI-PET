# src.inference.py
# load trained model and run inference for one subejct
import torch
from torch.utils.data import DataLoader

from src.data import get_transforms, get_loader
from src.train import evals, compute_metrics


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

                                 