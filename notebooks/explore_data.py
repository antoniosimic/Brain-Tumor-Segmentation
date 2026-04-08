"""
Data exploration script for BraTS 2021 Task 1.
Run with: python notebooks/explore_data.py
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
MODALITIES = ["flair", "t1", "t1ce", "t2"]

# ── 1. Load one patient ──────────────────────────────────────────────────────

patient_id = "BraTS2021_00495"
patient_dir = DATA_DIR / patient_id

print(f"=== Patient: {patient_id} ===\n")

volumes = {}
for mod in MODALITIES:
    path = patient_dir / f"{patient_id}_{mod}.nii.gz"
    img = nib.load(str(path))
    vol = img.get_fdata()
    volumes[mod] = vol
    print(f"  {mod:6s}  shape={vol.shape}  dtype={vol.dtype}  "
          f"min={vol.min():.1f}  max={vol.max():.1f}  mean={vol.mean():.1f}")

seg_path = patient_dir / f"{patient_id}_seg.nii.gz"
seg = nib.load(str(seg_path)).get_fdata().astype(np.int32)
print(f"\n  {'seg':6s}  shape={seg.shape}  unique labels={np.unique(seg).tolist()}")

# ── 2. Class distribution ────────────────────────────────────────────────────

print("\n=== Class distribution ===\n")
total = seg.size
for label, name in [(0, "Background (healthy)"), (1, "Necrotic core (NCR)"),
                    (2, "Edema (ED)"), (4, "Enhancing tumor (ET)")]:
    count = (seg == label).sum()
    print(f"  Label {label}  {name:<25s}  {count:>8d} voxels  ({100*count/total:.2f}%)")

# ── 3. BraTS evaluation regions ─────────────────────────────────────────────

print("\n=== BraTS evaluation regions ===\n")
WT = (seg > 0)                              # whole tumor: 1+2+4
TC = ((seg == 1) | (seg == 4))              # tumor core:  1+4
ET = (seg == 4)                             # enhancing:   4 only

for name, mask in [("WT (Whole Tumor)", WT), ("TC (Tumor Core)", TC), ("ET (Enhancing)", ET)]:
    print(f"  {name:<22s}  {mask.sum():>7d} voxels  ({100*mask.sum()/total:.2f}%)")

# ── 4. Visualize one axial slice ────────────────────────────────────────────

mid_slice = seg.shape[2] // 2  # midpoint along z-axis

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle(f"{patient_id}  —  axial slice z={mid_slice}", fontsize=13)

cmap_seg = plt.cm.get_cmap("tab10", 5)

for i, mod in enumerate(MODALITIES):
    axes[i].imshow(volumes[mod][:, :, mid_slice].T, cmap="gray", origin="lower")
    axes[i].set_title(mod.upper())
    axes[i].axis("off")

axes[4].imshow(seg[:, :, mid_slice].T, cmap=cmap_seg, vmin=0, vmax=4, origin="lower")
axes[4].set_title("Segmentation")
axes[4].axis("off")

plt.tight_layout()
out_path = "outputs/sample_slice.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSlice visualization saved → {out_path}")
plt.show()

# ── 5. Second patient sanity check ──────────────────────────────────────────

print("\n=== Patient BraTS2021_00621 (sanity check) ===\n")
p2 = "BraTS2021_00621"
for mod in MODALITIES + ["seg"]:
    path = DATA_DIR / p2 / f"{p2}_{mod}.nii.gz"
    vol = nib.load(str(path)).get_fdata()
    print(f"  {mod:6s}  shape={vol.shape}")
