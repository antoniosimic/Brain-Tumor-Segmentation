"""
dataset.py — gradi listu pacijenata i dijeli je na train/val.

Svaki element liste je dict:
  {
    "flair": Path,
    "t1":    Path,
    "t1ce":  Path,
    "t2":    Path,
    "seg":   Path,
  }
"""

import random
from pathlib import Path

from configs.config import DATA_DIR, MODALITIES, VAL_SPLIT, RANDOM_SEED


def get_patient_dicts(data_dir: Path = DATA_DIR) -> list[dict]:
    """
    Skeniraj data_dir i vrati listu dict-ova, jedan po pacijentu.
    Preskoci mape koje nemaju svih 5 ocekivanih datoteka.
    """
    patient_dicts = []

    for patient_dir in sorted(data_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        pid = patient_dir.name  # npr. "BraTS2021_00495"

        entry = {}
        valid = True

        for mod in MODALITIES:
            path = patient_dir / f"{pid}_{mod}.nii.gz"
            if not path.exists():
                print(f"  [UPOZORENJE] {pid}: nedostaje {mod}, preskacemo.")
                valid = False
                break
            entry[mod] = str(path)

        seg_path = patient_dir / f"{pid}_seg.nii.gz"
        if not seg_path.exists():
            print(f"  [UPOZORENJE] {pid}: nedostaje seg, preskacemo.")
            valid = False

        if valid:
            entry["seg"] = str(seg_path)
            patient_dicts.append(entry)

    return patient_dicts


def train_val_split(
    patient_dicts: list[dict],
    val_fraction: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """
    Patient-level split — ne mijesamo sliceove razlicitih pacijenata.
    To je kljucno da nema 'data leakage' izmedju traina i vala.
    """
    random.seed(seed)
    shuffled = patient_dicts.copy()
    random.shuffle(shuffled)

    n_val   = max(1, int(len(shuffled) * val_fraction))
    val     = shuffled[:n_val]
    train   = shuffled[n_val:]

    return train, val


if __name__ == "__main__":
    dicts = get_patient_dicts()
    print(f"Ukupno pacijenata: {len(dicts)}")

    train_dicts, val_dicts = train_val_split(dicts)
    print(f"Train: {len(train_dicts)}  |  Val: {len(val_dicts)}")

    if dicts:
        print(f"\nPrimjer jednog unosa:")
        for k, v in dicts[0].items():
            print(f"  {k:6s}: {v}")
