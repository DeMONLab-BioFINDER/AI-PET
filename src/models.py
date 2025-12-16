# src/models.py (append these)
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet, BasicUNet, DenseNet121, resnet

class CNN3D(nn.Module):
    """
    A small 3D CNN for scalar output.
    - Stack of Conv3d -> BN -> ReLU -> (optional MaxPool)
    - Global average pool
    - Linear head -> [B, 1]

    Args
    ----
    in_channels : int
        Input channels (default 1 for PET).
    widths : tuple[int, ...]
        Feature widths per conv stage, e.g. (16, 32, 64, 128).
    pool_every : int
        Apply MaxPool3d(2) after every N conv stages (set 1 to pool after each).
    dropout : float
        Dropout before the final linear head.
    norm : str
        "batch" or "instance" normalization.
    """
    def __init__(self, in_channels: int = 1, widths=(16, 32, 64, 128), pool_every: int = 1,
                 dropout: float = 0.2, norm: str = "batch", num_classes=1, extra_dim=0):
        super().__init__()
        assert pool_every >= 1
        assert norm in {"batch", "instance"}
        print(f'in_channels: {in_channels}, widths: {widths}, pool_every: {pool_every}, dropout: {dropout}, norm: {norm}, num_classes: {num_classes}')
        def make_norm(c):
            if norm == "batch":
                return nn.BatchNorm3d(c)  # ok if batch_size >= ~8
            else:
                return nn.InstanceNorm3d(c, affine=True, track_running_stats=False)

        layers = []
        c_in = in_channels
        for i, c_out in enumerate(widths, start=1):
            layers += [
                nn.Conv3d(c_in, c_out, 3, 1, 1, bias=False),
                make_norm(c_out),
                nn.ReLU(inplace=True),
                nn.Conv3d(c_out, c_out, 3, 1, 1, bias=False),
                make_norm(c_out),
                nn.ReLU(inplace=True),
            ]
            if (i % pool_every) == 0:
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            c_in = c_out

        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool3d(1)
        in_fc = widths[-1] + extra_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_fc, num_classes))

    def forward(self, x, extra):
        x = self.features(x)       # [B, C, D, H, W]
        x = self.gap(x).flatten(1) # [B, C]

        if extra is not None and torch.isfinite(extra).any():
            if extra.ndim == 1: extra = extra.unsqueeze(0)
            x = torch.cat([x, extra], dim=1)

        y = self.head(x)           # [B, num_classes]
        return y
    

class UNet3D(nn.Module):
    """
    MONAI 3D U-Net for volumetric prediction (e.g., segmentation).
    Returns logits of shape [B, out_channels, D, H, W].
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2), num_res_units: int = 2, norm: str = "instance",
                 dropout: float = 0.0, up_kernel_size: int = 2, use_basic: bool = False):  # use_basic: True -> BasicUNet, False -> UNet
    
        super().__init__()
        if use_basic:
            # BasicUNet has a simplified API
            self.net = BasicUNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                features=list(channels),  dropout=dropout)  # BasicUNet uses "features"
        else:
            self.net = UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                            channels=list(channels), strides=list(strides), num_res_units=num_res_units,
                            norm=norm, dropout=dropout, up_kernel_size=up_kernel_size)

    def forward(self, x):
        return self.net(x)  # [B, out_channels, D, H, W]


class DenseNet121_3D(nn.Module):
    """
    MONAI 3D DenseNet121 for scalar prediction (classification or regression).
    Returns logits of shape [B, 1].
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,   # keep =1 for single-output head
                 dropout_prob: float = 0.0):
        super().__init__()
        self.backbone = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,  # classifier head -> [B, out_channels]
            dropout_prob=dropout_prob)

    def forward(self, x):
        # DenseNet121 already includes GAP + Linear -> [B, out_channels]
        y = self.backbone(x)
        # Ensure shape [B, 1] for BCEWithLogitsLoss/MSE downstream
        if y.ndim == 2 and y.size(1) == 1:
            return y
        return y.view(y.size(0), -1)[:, :1]


class ResNet50_3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.backbone = resnet.ResNet(block=resnet.Bottleneck, layers=[3,4,6,3],
                                      block_inplanes=resnet.get_inplanes(),
                                      n_input_channels=in_channels,
                                      conv1_t_size=7, conv1_t_stride=2,
                                      no_max_pool=False, shortcut_type='B',
                                      widen_factor=1.0, num_classes=out_channels,    # -> [B, out_channels]
                                      spatial_dims=3)
    def forward(self, x):
        y = self.backbone(x)
        return y if y.ndim == 2 else y.view(y.size(0), -1)[:, :1]