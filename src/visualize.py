"""
visualize.py — generira detaljne vizualizacije predikcija modela.

Za svakog pacijenta generira vise aksijalni presjeka gdje je tumor vidljiv,
s usporedbom ground truth vs predikcija i oznacenim BraTS regijama.

Pokretanje:
  python src/visualize.py
  python src/visualize.py --checkpoint outputs/best_model.pth --n_slices 6
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).parent.parent))

from brats_config.config import OUTPUT_DIR, PATCH_SIZE
from src.dataset import get_patient_dicts, train_val_split
from src.model import build_model
from src.postprocess import postprocess
from src.transforms import get_val_transforms
from src.evaluate import pred_to_regions, compute_dice

# ── Boje ─────────────────────────────────────────────────────────────────────
SEG_CMAP = ListedColormap(["#000000", "#3399FF", "#FF9800", "#FF3333"])
REGION_COLORS = {"WT": "#4CAF50", "TC": "#FF9800", "ET": "#F44336"}


def find_best_slices(seg: np.ndarray, n: int = 6) -> list[int]:
    """Vrati n aksijalni z-indeksa s najvise tumora — ravnomjerno rasporedenih."""
    counts = (seg > 0).sum(axis=(0, 1))
    threshold = counts.max() * 0.08
    z_tumor = np.where(counts > threshold)[0]

    if len(z_tumor) == 0:
        mid = seg.shape[2] // 2
        return [mid]

    # Ravnomjerno uzorkovani z-indeksi duž tumora
    indices = np.linspace(0, len(z_tumor) - 1, n, dtype=int)
    return [int(z_tumor[i]) for i in indices]


def save_comparison(flair, t1ce, gt, pred, patient_id, dice_scores, n_slices=6):
    """
    Spremi usporedbu GT vs Predikcija za n_slices presjeka.
    Svaki presjek prikazuje: FLAIR | T1ce | GT | Predikcija
    """
    zs = find_best_slices(gt, n=n_slices)

    fig = plt.figure(figsize=(18, 3.2 * len(zs)))
    gs = gridspec.GridSpec(len(zs), 4, hspace=0.05, wspace=0.03)

    col_titles = ["FLAIR", "T1ce (kontrast)", "Ground Truth", "Predikcija modela"]

    for row, z in enumerate(zs):
        flair_sl = flair[:, :, z].T
        t1ce_sl  = t1ce[:, :, z].T
        gt_sl    = gt[:, :, z].T
        pred_sl  = pred[:, :, z].T

        for col, (data, cmap, vmin, vmax) in enumerate([
            (flair_sl, "gray",    None, None),
            (t1ce_sl,  "gray",    None, None),
            (gt_sl,    SEG_CMAP,  0,    3),
            (pred_sl,  SEG_CMAP,  0,    3),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                      origin="lower", interpolation="nearest")
            ax.axis("off")

            # Oznaci z-indeks
            ax.text(0.02, 0.97, f"z={z}", transform=ax.transAxes,
                    color="white", fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, pad=6)

    # Legenda klasa
    patches = [
        mpatches.Patch(color="#3399FF", label="NCR — Nekroza"),
        mpatches.Patch(color="#FF9800", label="ED — Edem"),
        mpatches.Patch(color="#FF3333", label="ET — Aktivni tumor"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    # Naslov s Dice scoreovima
    d = dice_scores
    fig.suptitle(
        f"{patient_id}\n"
        f"Dice:  WT = {d['WT']:.3f}   TC = {d['TC']:.3f}   ET = {d['ET']:.3f}",
        fontsize=13, y=1.005
    )

    out = OUTPUT_DIR / f"viz_{patient_id}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Spremljeno: {out}")
    return out


def save_overlay(flair, gt, pred, patient_id, z):
    """
    Spremi overlay: FLAIR s GT konturama i predikcijom za jedan presjek.
    Prikazuje 3 panela: samo FLAIR | FLAIR + GT overlay | FLAIR + Pred overlay
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{patient_id}  |  z = {z}  — Tumor overlay", fontsize=12)

    flair_sl = flair[:, :, z].T
    gt_sl    = gt[:, :, z].T
    pred_sl  = pred[:, :, z].T

    axes[0].imshow(flair_sl, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR (original)", fontsize=11)

    axes[1].imshow(flair_sl, cmap="gray", origin="lower")
    for label, color in zip([1, 2, 3], ["#3399FF", "#FF9800", "#FF3333"]):
        mask = (gt_sl == label).astype(float)
        masked = np.ma.masked_where(mask == 0, mask)
        axes[1].imshow(masked, cmap=ListedColormap([color]),
                       origin="lower", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth overlay", fontsize=11)

    axes[2].imshow(flair_sl, cmap="gray", origin="lower")
    for label, color in zip([1, 2, 3], ["#3399FF", "#FF9800", "#FF3333"]):
        mask = (pred_sl == label).astype(float)
        masked = np.ma.masked_where(mask == 0, mask)
        axes[2].imshow(masked, cmap=ListedColormap([color]),
                       origin="lower", alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title("Predikcija modela overlay", fontsize=11)

    patches = [
        mpatches.Patch(color="#3399FF", alpha=0.7, label="NCR"),
        mpatches.Patch(color="#FF9800", alpha=0.7, label="Edem"),
        mpatches.Patch(color="#FF3333", alpha=0.7, label="ET"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / f"overlay_{patient_id}_z{z}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return out


def visualize(checkpoint_path, n_patients=None, n_slices=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uredaj: {device}")

    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model ucitan: {checkpoint_path}")

    all_dicts = get_patient_dicts()
    _, val_dicts = train_val_split(all_dicts)
    if n_patients:
        val_dicts = val_dicts[:n_patients]

    print(f"\nGeneriranje vizualizacija za {len(val_dicts)} pacijenata...\n")

    for i, patient_dict in enumerate(val_dicts):
        pid = Path(patient_dict["flair"]).parts[-2]
        print(f"[{i+1}/{len(val_dicts)}] {pid}")

        ds     = Dataset(data=[patient_dict], transform=get_val_transforms())
        loader = DataLoader(ds, batch_size=1, num_workers=0)
        batch  = next(iter(loader))

        inputs = batch["image"].to(device)
        labels = batch["seg"].squeeze().cpu().numpy().astype(np.int32)

        with torch.no_grad():
            outputs = sliding_window_inference(
                inputs=inputs, roi_size=PATCH_SIZE,
                sw_batch_size=1, predictor=model, overlap=0.25,
            )

        pred = outputs.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)
        pred = postprocess(pred, min_voxels=50)

        flair = inputs[0, 0].cpu().numpy()
        t1ce  = inputs[0, 2].cpu().numpy()  # t1ce je 3. kanal (flair,t1,t1ce,t2)

        # Dice po regijama
        pred_reg = pred_to_regions(torch.from_numpy(pred))
        gt_reg   = pred_to_regions(torch.from_numpy(labels))
        dice = {
            r: compute_dice(
                pred_reg[r].numpy().astype(bool),
                gt_reg[r].numpy().astype(bool)
            )
            for r in ["WT", "TC", "ET"]
        }
        print(f"  Dice  WT={dice['WT']:.3f}  TC={dice['TC']:.3f}  ET={dice['ET']:.3f}")

        # Glavna usporedba (n_slices presjeka)
        save_comparison(flair, t1ce, labels, pred, pid, dice, n_slices=n_slices)

        # Overlay za najdebljí presjek tumora
        z_best = find_best_slices(labels, n=1)[0]
        save_overlay(flair, labels, pred, pid, z_best)

    print(f"\nSve slike spremljene u: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--n_patients", type=int, default=None)
    parser.add_argument("--n_slices",   type=int, default=6)
    args = parser.parse_args()

    visualize(
        checkpoint_path=args.checkpoint,
        n_patients=args.n_patients,
        n_slices=args.n_slices,
    )
