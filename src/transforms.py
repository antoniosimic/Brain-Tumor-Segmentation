"""
transforms.py — MONAI pipeline za preprocessing i augmentaciju.

Train pipeline:
  1. Ucitaj sve 5 NIfTI datoteka
  2. Dodaj channel dimenziju
  3. Remap labela: 0,1,2,4  ->  0,1,2,3
  4. Spoji 4 modaliteta u jedan 4-kanalni volumen
  5. Z-score normalizacija po kanalu (samo brain vokseli)
  6. Nasumicni 3D patch (128x128x128), balansiran tumor/ne-tumor
  7. Augmentacije: flip, rotacija, promjena intenziteta

Val pipeline:
  1-5. Isto kao train (deterministicki)
  6.   Nema augmentacija, nema patcheva (cijeli volumen)
"""

import numpy as np
from monai.transforms import (
    ConcatItemsd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambda,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
)

from brats_config.config import MODALITIES, PATCH_SIZE, PATCHES_PER_VOL

# Sve kljucne rijeci koje ucitavamo
_ALL_KEYS   = MODALITIES + ["seg"]
_IMAGE_KEYS = MODALITIES


def _remap_labels(seg):
    """
    BraTS originalne labele:  0, 1, 2, 4
    Remap na uzastopne:       0, 1, 2, 3

    Moramo ovo jer PyTorch loss funkcije ocekuju uzastopne klase
    (0, 1, 2, 3) — indeks 4 bi napravio out-of-range gresku.
    Radi i s numpy arrayom i s MONAI MetaTensorom.
    """
    out = seg.clone() if hasattr(seg, "clone") else seg.copy()
    out[seg == 4] = 3
    return out


def get_train_transforms() -> Compose:
    return Compose([
        # ── Ucitavanje ────────────────────────────────────────────────
        LoadImaged(keys=_ALL_KEYS),
        EnsureChannelFirstd(keys=_ALL_KEYS),

        # ── Remap labela PRIJE spajanja ───────────────────────────────
        Lambdad(keys="seg", func=_remap_labels),

        # ── Spoji 4 modaliteta u jedan tensor (4, H, W, D) ───────────
        ConcatItemsd(keys=_IMAGE_KEYS, name="image", dim=0),

        # ── Normalizacija: Z-score, po kanalu, samo brain vokseli ─────
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        EnsureTyped(keys=["image", "seg"]),

        # ── Patch sampling (pozitivno=tumor, negativno=ne-tumor 1:1) ─
        RandCropByPosNegLabeld(
            keys=["image", "seg"],
            label_key="seg",
            spatial_size=PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=PATCHES_PER_VOL,
        ),

        # ── Augmentacije ──────────────────────────────────────────────
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),

        ToTensord(keys=["image", "seg"]),
    ])


def get_val_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=_ALL_KEYS),
        EnsureChannelFirstd(keys=_ALL_KEYS),
        Lambdad(keys="seg", func=_remap_labels),
        ConcatItemsd(keys=_IMAGE_KEYS, name="image", dim=0),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "seg"]),
        ToTensord(keys=["image", "seg"]),
    ])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from src.dataset import get_patient_dicts
    from monai.data import Dataset

    dicts = get_patient_dicts()
    if not dicts:
        print("Nema pacijenata u data/. Provjeri putanju.")
        exit()

    print(f"Testiranje transforms na {dicts[0]['flair']}")

    ds = Dataset(data=[dicts[0]], transform=get_val_transforms())
    sample = ds[0]

    print(f"  image shape : {sample['image'].shape}")   # (4, 240, 240, 155)
    print(f"  seg   shape : {sample['seg'].shape}")     # (1, 240, 240, 155)
    print(f"  image dtype : {sample['image'].dtype}")
    print(f"  seg   unique: {sample['seg'].unique()}")  # [0, 1, 2, 3]

    ds_train = Dataset(data=[dicts[0]], transform=get_train_transforms())
    patches = ds_train[0]
    print(f"\nTrain patch primjer (lista od {len(patches)} patcha):")
    for p in patches:
        print(f"  image: {p['image'].shape}  seg: {p['seg'].shape}")
