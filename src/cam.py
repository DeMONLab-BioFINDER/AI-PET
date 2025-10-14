# src/cam.py
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional


class GradCAM3D:
    """
    Generic 3D Grad-CAM:
    - Works with scalar heads (regression/classification) and 3D seg logits.
    - You provide a target layer (ideally the last Conv3d). If None, we auto-pick it.

    Args
    ----
    model : nn.Module
    target_layer : nn.Module or None (will auto-find last Conv3d if None)
    task : {"auto","cls","reg","seg"}  (auto guesses from output shape)
    target_class : int or None  (for [B,C] outputs)
    seg_channel  : int          (for [B,C,D,H,W] outputs)
    seg_reduction : {"mean","max"} reduction over voxels for seg target
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, task: str = "auto",
        target_class: Optional[int] = None, seg_channel: int = 0, seg_reduction: str = "mean",):
        self.model = model
        self.model.eval()
        self.layer = target_layer or find_last_conv3d(model)
        if self.layer is None:
            raise ValueError("GradCAM3D: No Conv3d layer found; please pass target_layer explicitly.")
        self.task = task
        self.target_class = target_class
        self.seg_channel = seg_channel
        assert seg_reduction in {"mean","max"}
        self.seg_reduction = seg_reduction

        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            # grad_out is a tuple; first element is gradient w.r.t. outputs
            self.gradients = grad_out[0].detach()

        self.h1 = self.layer.register_forward_hook(fwd_hook)
        self.h2 = self.layer.register_full_backward_hook(bwd_hook)

    def __del__(self):
        try:
            self.h1.remove(); self.h2.remove()
        except Exception:
            pass

    def _select_target_scalar(self, y: torch.Tensor) -> torch.Tensor:
        """
        Turn model output y into a scalar per sample to backprop from.
        Shapes handled:
          - [B, 1] or [B]      -> use as-is (sum over batch for CAM)
          - [B, C]             -> pick target_class (default 0 if None)
          - [B, C, D, H, W]    -> pick seg_channel then reduce over voxels
        """
        if y.ndim == 1:
            return y
        if y.ndim == 2:
            # classification logits or regression with extra dim
            idx = self.target_class if self.target_class is not None else 0
            idx = max(0, min(idx, y.size(1)-1))
            return y[:, idx]
        if y.ndim == 5:
            c = max(0, min(self.seg_channel, y.size(1)-1))
            v = y[:, c]  # [B, D, H, W]
            if self.seg_reduction == "mean":
                return v.mean(dim=(1,2,3))  # [B]
            else:
                return v.amax(dim=(1,2,3))  # [B]
        # Fallback: try to flatten to [B] by averaging
        return y.view(y.size(0), -1).mean(dim=1)

    def _infer_task(self, y: torch.Tensor) -> str:
        if y.ndim == 5:
            return "seg"
        if y.ndim in (1,2):
            # could be cls or reg; doesn't matter for CAM
            return "cls"
        return "reg"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns CAM volume normalized to [0,1]: shape [B, 1, D, H, W]
        """
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_()
        y = self.model(x)

        task = self.task if self.task != "auto" else self._infer_task(y)
        # Select scalar target per sample, then sum over batch for a single backward
        target_scalar = self._select_target_scalar(y)
        target_scalar.sum().backward()

        A = self.activations        # [B,C,D,H,W]
        G = self.gradients          # [B,C,D,H,W]
        if A is None or G is None:
            raise RuntimeError("GradCAM3D: hooks did not capture activations/gradients.")

        weights = G.mean(dim=(2,3,4), keepdim=True)         # [B,C,1,1,1]
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))  # [B,1,D,H,W]

        # normalize 0..1 per-sample
        cam_min = cam.amin(dim=(2,3,4), keepdim=True)
        cam = cam - cam_min
        cam_max = cam.amax(dim=(2,3,4), keepdim=True) + 1e-6
        cam = cam / cam_max
        return cam


def save_gradcam_overlays(model, cam, loader, out_dir, device, max_items=8):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for x, y_cls, y_reg, pid in loader:
        if count >= max_items: break

        x = x.to(device, non_blocking=True)

        with torch.enable_grad(): heat = cam(x)

        # --- Resize CAM to input volume size for clean overlays ---
        if heat.shape[2:] != x.shape[2:]:
            heat = F.interpolate(heat, size=x.shape[2:], mode="trilinear", align_corners=False) # [B,1,D,H,W]

        x_np = x.detach().cpu().numpy()          # detach before numpy
        h_np = heat.detach().cpu().numpy()
        for b in range(x_np.shape[0]):
            if count >= max_items: return

            vol = x_np[b,0]
            hm = h_np[b,0]

            # safe central indices (now same shapes)
            Dz, Hy, Wx = vol.shape
            z, y, xidx = Dz // 2, Hy // 2, Wx // 2

            fig = plt.figure(figsize=(10,6))

            ax1 = fig.add_subplot(1,3,1); ax1.imshow(vol[z,:,:], cmap="gray"); ax1.imshow(hm[z,:,:], alpha=0.4); ax1.set_title("Axial"); ax1.axis("off")
            ax2 = fig.add_subplot(1,3,2); ax2.imshow(vol[:,y,:], cmap="gray"); ax2.imshow(hm[:,y,:], alpha=0.4); ax2.set_title("Coronal"); ax2.axis("off")
            ax3 = fig.add_subplot(1,3,3); ax3.imshow(vol[:,:,xidx], cmap="gray"); ax3.imshow(hm[:,:,xidx], alpha=0.4); ax3.set_title("Sagittal"); ax3.axis("off")
            
            fig.suptitle(f"Grad-CAM – {pid[b]}")
            out_path = os.path.join(out_dir, f"{pid[b]}_gradcam.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            
            count += 1


def save_gradcam_overlays_avg_by_slice(model, cam, loader, out_dir, device, max_items=None,
    slice_fracs=(0.25, 0.5, 0.75), overlay_cam=True,
    clip_percentile=0.5, cmap_img="gray", cam_alpha=0.4):
    """
    Computes the average volume (and average CAM) across subjects, then
    saves ONE figure per slice position. Each figure has 3 subplots:
    Axial / Coronal / Sagittal for that slice position.

    Args
    ----
    model, cam, loader, device : as usual
    out_dir : str
    max_items : int | None  cap subjects used (None = all)
    slice_fracs : tuple[float]  positions in [0,1] along each axis
    overlay_cam : bool  overlay mean CAM if True
    clip_percentile : float  robust intensity scaling (0–2 recommended)
    cmap_img : str  matplotlib cmap for base image
    cam_alpha : float  overlay transparency
    """
    os.makedirs(out_dir, exist_ok=True)

    sum_vol, sum_cam, n_seen = None, None, 0

    for x, y_cls, y_reg, pid in loader:
        if max_items is not None and n_seen >= max_items:
            break

        b = x.shape[0]
        if max_items is not None:
            b = min(b, max_items - n_seen)
            x = x[:b]

        x = x.to(device, non_blocking=True)

        with torch.enable_grad():
            heat = cam(x)  # [B,1,D,H,W] (will resize)

        if heat.shape[2:] != x.shape[2:]:
            heat = F.interpolate(heat, size=x.shape[2:], mode="trilinear", align_corners=False)

        x_np = x.detach().cpu().numpy()[:, 0]      # [B, D, H, W]
        h_np = heat.detach().cpu().numpy()[:, 0]   # [B, D, H, W]

        if sum_vol is None:
            sum_vol = np.zeros_like(x_np[0], dtype=np.float64)
            if overlay_cam:
                sum_cam = np.zeros_like(h_np[0], dtype=np.float64)

        sum_vol += x_np[:b].sum(axis=0)  # true sum
        if overlay_cam:
            sum_cam += h_np[:b].sum(axis=0)

        n_seen += b
    print('finish loading')
    if n_seen == 0:
        print("No data encountered — nothing saved.")
        return

    mean_vol = sum_vol / n_seen
    mean_cam = (sum_cam / n_seen) if overlay_cam else None

    D, H, W = mean_vol.shape

    def frac_to_idx(f, L):
        f = float(np.clip(f, 0.0, 1.0))
        return int(round(f * (L - 1)))

    # robust display range for the mean image
    if clip_percentile is not None:
        lo = np.percentile(mean_vol, max(0, 50 - clip_percentile * 50))
        hi = np.percentile(mean_vol, min(100, 50 + clip_percentile * 50))
        lo, hi = float(lo), float(hi)
    else:
        lo, hi = float(mean_vol.min()), float(mean_vol.max())

    # optional CAM normalization per-view for nicer overlays
    def norm_cam(arr):
        vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
        vmax = vmax if vmax > vmin else vmin + 1e-6
        return np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)

    # ---- Save one figure per slice fraction ----
    for f in slice_fracs:
        z = frac_to_idx(f, D)
        y = frac_to_idx(f, H)
        xidx = frac_to_idx(f, W)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # Axial
        axes[0].imshow(mean_vol[z, :, :], cmap=cmap_img, vmin=lo, vmax=hi)
        if overlay_cam:
            axes[0].imshow(norm_cam(mean_cam[z, :, :]), alpha=cam_alpha)
        axes[0].set_title(f"Axial (z={z}, f={f:.2f})")
        axes[0].axis("off")

        # Coronal
        axes[1].imshow(mean_vol[:, y, :], cmap=cmap_img, vmin=lo, vmax=hi)
        if overlay_cam:
            axes[1].imshow(norm_cam(mean_cam[:, y, :]), alpha=cam_alpha)
        axes[1].set_title(f"Coronal (y={y}, f={f:.2f})")
        axes[1].axis("off")

        # Sagittal
        axes[2].imshow(mean_vol[:, :, xidx], cmap=cmap_img, vmin=lo, vmax=hi)
        if overlay_cam:
            axes[2].imshow(norm_cam(mean_cam[:, :, xidx]), alpha=cam_alpha)
        axes[2].set_title(f"Sagittal (x={xidx}, f={f:.2f})")
        axes[2].axis("off")

        plt.tight_layout()
        fname = os.path.join(out_dir, f"avg_axcorSag_f{f:.2f}.png")
        plt.savefig(fname, dpi=150)
        plt.close(fig)

    print(f"Saved {len(slice_fracs)} figures (n={n_seen}) to: {out_dir}")


def save_gradcam_nifti(model, cam, loader, out_dir, device, max_items=999, use_ref_affine=True):
    """
    Save Grad-CAM volumes as NIfTI in the model input space (post-preprocessing).
    If `use_ref_affine` is True and a per-sample PET path is provided by the loader,
    we will borrow the source affine (shape may differ; that's OK—it's only an orientation/pose hint).
    """
    os.makedirs(out_dir, exist_ok=True)
    count = 0

    for batch in loader:
        # Support both old (x,y_cls,y_reg,ID) and new (x,y_cls,y_reg,ID,pet_path) signatures
        if len(batch) == 4:
            x, y_cls, y_reg, pid = batch
            pet_path = [None] * x.shape[0]
        else:
            x, y_cls, y_reg, pid, pet_path = batch

        if count >= max_items:
            break

        x = x.to(device, non_blocking=True)

        with torch.enable_grad():
            heat = cam(x)  # [B,1,D,H,W] normalized 0..1

        # Ensure CAM is same spatial size as network input (defensive)
        if heat.shape[2:] != x.shape[2:]:
            heat = F.interpolate(heat, size=x.shape[2:], mode="trilinear", align_corners=False)

        h_np = heat.detach().cpu().numpy()  # [B,1,D,H,W]
        B = h_np.shape[0]

        for b in range(B):
            if count >= max_items:
                return
            vol = h_np[b, 0].astype(np.float32)  # [D,H,W], already 0..1

            # Choose an affine:
            affine = np.eye(4, dtype=np.float32)
            if use_ref_affine and pet_path[b]:
                try:
                    affine = nib.load(pet_path[b]).affine.astype(np.float32)
                except Exception:
                    pass  # fall back to identity if loading fails

            # Write NIfTI (.nii.gz)
            out_path = os.path.join(out_dir, f"{pid[b]}_gradcam.nii.gz")
            nib.save(nib.Nifti1Image(vol, affine), out_path)
            count += 1


def find_last_conv3d(module: nn.Module) -> Optional[nn.Module]:
    """
    Return the last nn.Conv3d found by DFS over modules.
    If none exists, returns None.
    """
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            last = m
    return last