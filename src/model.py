"""
model.py — definicija modela.

Nudimo dva izbora:
  - UNet3D   : klasican 3D U-Net, jednostavniji, manji VRAM
  - SegResNet: rezidualna mreza, jaca, treba vise VRAM

Oba su iz MONAI biblioteke — testirani su na medicinskim 3D volumenima.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, UNet

from brats_config.config import IN_CHANNELS, MODEL_NAME, NUM_CLASSES, OUT_CHANNELS


def build_model(name: str = MODEL_NAME) -> nn.Module:
    """
    Vrati model prema imenu iz konfiguracije.
    name: 'unet3d' ili 'segresnet'
    """
    if name == "unet3d":
        model = UNet(
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            channels=(16, 32, 64, 128, 256),  # filteri po razini
            strides=(2, 2, 2, 2),             # downsampling koraci
            num_res_units=2,                  # rezidualni blokovi po razini
            dropout=0.1,
        )

    elif name == "segresnet":
        model = SegResNet(
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            init_filters=32,
        )

    else:
        raise ValueError(f"Nepoznat model: '{name}'. Koristi 'unet3d' ili 'segresnet'.")

    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    """Vrati (ukupno parametara, trainable parametara)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for name in ["unet3d", "segresnet"]:
        model = build_model(name)
        total, trainable = count_params(model)
        print(f"{name:12s}  ukupno params: {total:>10,}  trainable: {trainable:>10,}")

        # Test forward pass s dummy inputom
        x = torch.zeros(1, IN_CHANNELS, 64, 64, 64)
        out = model(x)
        print(f"{'':12s}  input: {tuple(x.shape)}  output: {tuple(out.shape)}\n")
