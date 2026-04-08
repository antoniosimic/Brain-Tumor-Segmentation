"""
evaluate.py — evaluacija modela na validacijskom skupu.

Sto radi:
  1. Ucita checkpoint
  2. Za svakog val pacijenta pokrene sliding window inference
  3. Izracuna Dice i HD95 po BraTS regijama (WT, TC, ET)
  4. Ispise tablicu rezultata i spremi grafove

BraTS regije (iz remapanih labela 0,1,2,3):
  WT = 1 + 2 + 3  (cijeli tumor)
  TC = 1 + 3      (tumor core)
  ET = 3          (enhancing tumor)

Pokretanje:
  python src/evaluate.py
  python src/evaluate.py --checkpoint outputs/best_model.pth --n_patients 20
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from brats_config.config import NUM_CLASSES, OUTPUT_DIR, PATCH_SIZE
from src.dataset import get_patient_dicts, train_val_split
from src.model import build_model
from src.postprocess import postprocess
from src.transforms import get_val_transforms


def pred_to_regions(pred: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Pretvori 4-klasnu predikciju u 3 BraTS binarne regije.
    pred: shape (1, H, W, D) ili (H, W, D), vrijednosti 0-3
    """
    if pred.dim() == 4:
        pred = pred.squeeze(0)

    WT = (pred > 0).long()
    TC = ((pred == 1) | (pred == 3)).long()
    ET = (pred == 3).long()

    return {"WT": WT, "TC": TC, "ET": ET}


def compute_dice(pred_bin: np.ndarray, gt_bin: np.ndarray, smooth: float = 1e-5) -> float:
    """Dice koeficijent za jednu binarnu regiju."""
    intersection = (pred_bin & gt_bin).sum()
    return (2.0 * intersection + smooth) / (pred_bin.sum() + gt_bin.sum() + smooth)


def compute_hd95(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """
    Hausdorff Distance 95% percentil.
    Mjeri maksimalnu udaljenost izmedju granica predikcije i ground trutha.
    Manji = bolji.
    """
    from scipy.ndimage import distance_transform_edt

    if not pred_bin.any() and not gt_bin.any():
        return 0.0
    if not pred_bin.any() or not gt_bin.any():
        return np.inf

    # Povrsinski vokseli
    pred_border = pred_bin ^ __import__("scipy").ndimage.binary_erosion(pred_bin)
    gt_border   = gt_bin   ^ __import__("scipy").ndimage.binary_erosion(gt_bin)

    # Udaljenosti od svake povrsine do druge
    dist_pred_to_gt = distance_transform_edt(~gt_border)[pred_border]
    dist_gt_to_pred = distance_transform_edt(~pred_border)[gt_border]

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def evaluate(
    checkpoint_path: str | Path,
    n_patients: int | None = None,
    use_postprocess: bool = True,
    save_plots: bool = True,
) -> pd.DataFrame:
    """
    Evaluiraj model na validacijskom skupu.

    Args:
        checkpoint_path: putanja do .pth filea
        n_patients:      koliko pacijenata evaluirati (None = svi)
        use_postprocess: da li primijeniti postprocessing
        save_plots:      da li spremiti vizualizacije

    Returns:
        DataFrame s rezultatima po pacijentu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uredaj: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint ucitan: {checkpoint_path}")

    # ── Validacijski pacijenti ────────────────────────────────────────────────
    all_dicts = get_patient_dicts()
    _, val_dicts = train_val_split(all_dicts)

    if n_patients:
        val_dicts = val_dicts[:n_patients]

    print(f"Evaluacija na {len(val_dicts)} pacijenata...\n")

    val_ds     = Dataset(data=val_dicts, transform=get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # ── Petlja evaluacije ─────────────────────────────────────────────────────
    results = []

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluacija")):
        patient_id = Path(val_dicts[i]["flair"]).parts[-2]

        inputs = batch["image"].to(device)
        labels = batch["seg"].squeeze().cpu().numpy().astype(np.int32)

        with torch.no_grad():
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=PATCH_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )

        # Argmax → klasa po vokselu
        pred = outputs.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)

        # Postprocessing
        if use_postprocess:
            pred = postprocess(pred, min_voxels=50)

        # BraTS regije
        pred_regions = pred_to_regions(torch.from_numpy(pred))
        gt_regions   = pred_to_regions(torch.from_numpy(labels))

        row = {"patient_id": patient_id}
        for region in ["WT", "TC", "ET"]:
            p = pred_regions[region].numpy().astype(bool)
            g = gt_regions[region].numpy().astype(bool)

            row[f"Dice_{region}"] = compute_dice(p, g)
            row[f"HD95_{region}"] = compute_hd95(p, g)

        results.append(row)

        # Vizualizacija prvih 3 pacijenta
        if save_plots and i < 3:
            _save_slice_plot(inputs, labels, pred, patient_id)

    df = pd.DataFrame(results)

    # ── Ispis tablice ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("REZULTATI EVALUACIJE")
    print("="*70)

    metric_cols = [c for c in df.columns if c != "patient_id"]
    print(df[metric_cols].describe().round(3).to_string())

    print("\nProsjek po metrikama:")
    for col in metric_cols:
        finite = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {col:12s}: {finite.mean():.3f} ± {finite.std():.3f}")

    # ── Spremi CSV ────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRezultati spremljeni: {csv_path}")

    # ── Graf Dice po regijama ─────────────────────────────────────────────────
    if save_plots:
        _save_dice_plot(df)

    return df


def _save_slice_plot(inputs, labels, pred, patient_id):
    """Spremi usporedbu GT vs predikcija za jedan pacijent."""
    flair = inputs[0, 0].cpu().numpy()   # (H, W, D)
    z = flair.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{patient_id}  |  axial z={z}", fontsize=11)

    axes[0].imshow(flair[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR")

    axes[1].imshow(labels[:, :, z].T, cmap="tab10", vmin=0, vmax=3, origin="lower")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred[:, :, z].T, cmap="tab10", vmin=0, vmax=3, origin="lower")
    axes[2].set_title("Predikcija")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / f"pred_{patient_id}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def _save_dice_plot(df: pd.DataFrame):
    """Boxplot Dice scoreova po regijama."""
    dice_cols = ["Dice_WT", "Dice_TC", "Dice_ET"]
    data = [df[c].dropna().values for c in dice_cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=["WT", "TC", "ET"], patch_artist=True)

    colors = ["#4CAF50", "#FF9800", "#F44336"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Dice Score")
    ax.set_title("Dice Score po BraTS regijama")
    ax.set_ylim(0, 1)
    ax.axhline(y=df[dice_cols].mean().mean(), color="navy", linestyle="--",
               alpha=0.5, label=f"prosjek={df[dice_cols].mean().mean():.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "dice_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graf spremljen: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="outputs/best_model.pth")
    parser.add_argument("--n_patients",   type=int, default=None)
    parser.add_argument("--no_postproc",  action="store_true")
    parser.add_argument("--no_plots",     action="store_true")
    args = parser.parse_args()

    df = evaluate(
        checkpoint_path=args.checkpoint,
        n_patients=args.n_patients,
        use_postprocess=not args.no_postproc,
        save_plots=not args.no_plots,
    )
