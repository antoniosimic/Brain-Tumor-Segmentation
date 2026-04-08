"""
postprocess.py — ciscenje predikcija modela.

Ulaz:  segmentacijska maska kao numpy array, shape (H, W, D), vrijednosti 0-3
Izlaz: ista maska ali ociscena

Sto radimo:
  1. Uklonimo male izolirane komponente (sitni noise vokseli)
  2. Popunimo male rupe unutar maski

Koristimo scipy.ndimage za connected component analizu.
"""

import numpy as np
from scipy import ndimage


def remove_small_components(mask: np.ndarray, min_voxels: int = 50) -> np.ndarray:
    """
    Ukloni izolirane komponente manje od min_voxels voksela.

    Primjer: model predvidi 10 voksela nekroze daleko od glavnog tumora
             — to je vjerojatno noise, uklanjamo ga.
    """
    out = np.zeros_like(mask)

    for label in [1, 2, 3]:
        binary = (mask == label)
        if not binary.any():
            continue

        # Pronadi sve izolirane "otoke" iste klase
        labeled, num_features = ndimage.label(binary)

        for component_id in range(1, num_features + 1):
            component = (labeled == component_id)
            if component.sum() >= min_voxels:
                out[component] = label

    # Zdravo tkivo = sve sto nije tumor
    out[mask == 0] = 0
    background = (out == 0) & (mask == 0)
    out[background] = 0

    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Popuni male rupe unutar svake klase.

    Primjer: model predvidi prsten nekroze ali ostavi rupu u sredini
             — popunimo tu rupu.
    """
    out = mask.copy()

    for label in [1, 2, 3]:
        binary = (mask == label)
        if not binary.any():
            continue

        filled = ndimage.binary_fill_holes(binary)
        # Dodaj samo voksele koji su bili 0 (pozadina) a sad su popunjeni
        newly_filled = filled & (mask == 0)
        out[newly_filled] = label

    return out


def postprocess(
    pred: np.ndarray,
    min_voxels: int = 50,
    do_fill_holes: bool = True,
) -> np.ndarray:
    """
    Glavni postprocessing pipeline.

    Args:
        pred:          predikcija modela, shape (H, W, D), vrijednosti 0-3
        min_voxels:    minimalna velicina komponente da ostane
        do_fill_holes: da li popunjavamo rupe

    Returns:
        ociscena maska istog shapea
    """
    pred = remove_small_components(pred, min_voxels=min_voxels)

    if do_fill_holes:
        pred = fill_holes(pred)

    return pred


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import nibabel as nib
    from src.dataset import get_patient_dicts

    # Test na ground truth kao "fake predikciji"
    dicts = get_patient_dicts()
    if not dicts:
        print("Nema pacijenata u data/")
        exit()

    seg = nib.load(dicts[0]["seg"]).get_fdata().astype(np.int32)
    seg[seg == 4] = 3  # remap

    print(f"Originalna maska — unique: {np.unique(seg)}")
    print(f"Vokseli po klasi: {[(int(l), int((seg==l).sum())) for l in np.unique(seg)]}")

    # Simuliraj noise — dodaj 30 izoliranih voksela klase 1 na random poziciju
    noisy = seg.copy()
    noisy[10:12, 10:12, 10:12] = 1

    cleaned = postprocess(noisy, min_voxels=50)

    noise_removed = int((noisy == 1).sum()) - int((cleaned == 1).sum())
    print(f"\nDodano noise voksela: {4}")
    print(f"Uklonjeno postprocessingom: {noise_removed}")
    print(f"Postprocess OK.")
