# src/cam.py
import os
import torch
import torch.nn as nn
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