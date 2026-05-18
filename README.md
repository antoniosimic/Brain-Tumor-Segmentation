# Brain Tumor Segmentation — BraTS 2021

Semanticka segmentacija mozdanih tumora na MRI snimkama pomocu dubokog ucenja.
Dataset: [MICCAI BraTS 2021 Task 1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data)

> Puna tehnicka dokumentacija (arhitektura, preprocessing, loss, metrike, augmentacije, postprocessing): **[`outputs/projekt_dokumentacija.md`](outputs/projekt_dokumentacija.md)**

---

## Sto radi ovaj projekt

Za svakog pacijenta postoje 4 MRI volumena (modaliteta). Model ih uzima kao 4-kanalni 3D ulaz i za svaki voksel predvida jednu od 4 klase:

| Label | Klasa |
|-------|-------|
| 0 | Zdravo tkivo (pozadina) |
| 1 | Nekroticna jezgra (NCR) |
| 2 | Edem (ED) |
| 3 | Aktivni tumor (ET) — originalno label 4 u BraTS datasetu |

Evaluacija se radi po 3 izvedene BraTS regije: **WT** (cijeli tumor = 1+2+3), **TC** (jezgra tumora = 1+3), **ET** (aktivni dio = 3).

---

## Rezultati — najbolji model (v4)

**Arhitektura:** SegResNet (MONAI), ~18.8M parametara, ~72 MB
**Trening:** 200 epoha, patch 96×96×96, 150 pacijenata, Kaggle T4 GPU

| Klasa | Val Dice |
|-------|----------|
| NCR (Nekroza) | 0.700 |
| ED (Edem) | 0.817 |
| ET (Aktivni tumor) | 0.821 |
| **Mean** | **0.7795** |

### Tijek razvoja modela

| Verzija | Arhitektura | Epohe | Patch | Pacijenata | Val Dice |
|---------|-------------|-------|-------|------------|----------|
| v1 | 3D UNet | 50 | 64³ | 80 | 0.606 |
| v2 | 3D UNet | 100 | 64³ | 150 | 0.643 |
| v3 | 3D UNet | 100 | 96³ | 150 | 0.695 |
| **v4** | **SegResNet** | **200** | **96³** | **150** | **0.7795** |

> Datoteka `outputs/evaluation_results.csv` u repozitoriju sadrzi samo jedan demo redak (pacijent BraTS2021_00621) — pune validacijske brojke (Dice + HD95 po svim ~250 pacijenata) dobiju se ponovnim pokretanjem `src/evaluate.py` na Kaggleu s GPU-om.

---

## Struktura repozitorija

```
Brain-Tumor-Segmentation/
├── brats_config/
│   └── config.py             # centralna konfiguracija (hyperparametri, putanje)
├── src/
│   ├── dataset.py            # ucitavanje pacijenata, train/val split
│   ├── transforms.py         # MONAI preprocessing (normalizacija, patch sampling, augmentacije)
│   ├── model.py              # SegResNet i 3D UNet (oba iz MONAI)
│   ├── train.py              # petlja za treniranje (AMP, SmartCache, sliding window val)
│   ├── postprocess.py        # connected components + fill holes
│   ├── evaluate.py           # Dice + HD95 po WT/TC/ET regijama
│   └── visualize.py          # detaljne vizualizacije GT vs predikcija
├── notebooks/
│   ├── explore_data.ipynb    # vizualna eksploracija BraTS podataka
│   └── evaluate_model.ipynb  # prikaz rezultata istreniranog modela
├── outputs/
│   ├── best_model.pth                       # istreniran SegResNet checkpoint (~72 MB)
│   ├── evaluation_results.csv               # demo (jedan pacijent — vidi gore)
│   ├── viz_BraTS2021_00621.png              # primjer detaljne vizualizacije
│   ├── overlay_BraTS2021_00621_z37.png      # primjer overlay-a na FLAIR-u
│   └── projekt_dokumentacija.md             # puna tehnicka dokumentacija
├── data/                     # pacijenti — NIJE na GitHubu (previse veliki, ~13 GB)
├── requirements.txt
└── README.md
```

---

## Postavljanje okruzenja (lokalno)

```bash
git clone https://github.com/antoniosimic/Brain-Tumor-Segmentation.git
cd Brain-Tumor-Segmentation

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Provjera da sve radi (bez stvarnog dataseta):

```bash
python src/model.py          # ispise broj parametara za UNet3D i SegResNet
```

---

## Podaci

Dataset je prevelik za Git (~13 GB). Preuzmi ga na jedan od dva nacina:

**Opcija A — Kaggle CLI:**

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json

mkdir -p data && cd data
kaggle datasets download dschettler8845/brats-2021-task1 --file BraTS2021_Training_Data.tar
tar -xf BraTS2021_Training_Data.tar
```

**Opcija B — Kaggle notebook (preporuceno za trening):**
Dataset je vec dostupan na Kaggleu bez skidanja — vidi sekciju "Trening na Kaggleu" ispod.

Ocekivana struktura `data/` mape:

```
data/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_flair.nii.gz
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
├── BraTS2021_00002/
│   └── ...
```

---

## Kako pokrenuti model

### 1) Inferencija + evaluacija na vec istreniranom checkpointu

Najjednostavniji nacin da reproduciras rezultate — koristi `outputs/best_model.pth` koji je vec u repozitoriju:

```bash
# Evaluacija na cijelom val skupu (~250 pacijenata) — pise Dice + HD95 po WT/TC/ET
python src/evaluate.py --checkpoint outputs/best_model.pth

# Brzo na samo 10 pacijenata
python src/evaluate.py --checkpoint outputs/best_model.pth --n_patients 10

# Bez postprocessinga
python src/evaluate.py --checkpoint outputs/best_model.pth --no_postproc
```

Rezultat: `outputs/evaluation_results.csv` (Dice/HD95 po pacijentu) + `outputs/dice_boxplot.png`.

### 2) Detaljne vizualizacije predikcija

```bash
# Generira viz_<pid>.png (4 stupca × 6 presjeka) + overlay_<pid>_z<z>.png za svakog pacijenta
python src/visualize.py --checkpoint outputs/best_model.pth --n_patients 5 --n_slices 6
```

### 3) Trening od nule (lokalno)

Sporo bez GPU-a — koristi samo za debug:

```bash
python src/train.py
```

Najbolji checkpoint se sprema u `outputs/best_model.pth` po `mean_dice` na val skupu.

### 4) Trening na Kaggleu (preporuceno — besplatan T4 GPU)

1. Idi na [kaggle.com](https://kaggle.com) → **Create** → **New Notebook**
2. Dodaj dataset: desno → **Add data** → potrazi i dodaj `brats-2021-task1` (dschettler8845)
3. Ukljuci GPU: **Session options** → **Accelerator: GPU T4 x1**
4. U notebook celije:

```python
# Setup
!git clone https://github.com/antoniosimic/Brain-Tumor-Segmentation.git
%cd Brain-Tumor-Segmentation
!pip install -r requirements.txt -q

# Konfiguracija putanje podataka
import sys
sys.path.insert(0, '/kaggle/working/Brain-Tumor-Segmentation')
from brats_config import config
from pathlib import Path
config.DATA_DIR = Path('/kaggle/input/brats-2021-task1/BraTS2021_Training_Data')

# Trening (~6-8h za 200 epoha na T4)
from src.train import main
main()

# Evaluacija
from src.evaluate import evaluate
evaluate(checkpoint_path='outputs/best_model.pth')

# Vizualizacije
from src.visualize import visualize
visualize(checkpoint_path='outputs/best_model.pth', n_patients=5, n_slices=6)
```

### 5) Programatski iz Pythona

```python
import torch
from src.model import build_model
from src.dataset import get_patient_dicts, train_val_split
from src.transforms import get_val_transforms
from src.postprocess import postprocess
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader

# Ucitaj model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
model.eval()

# Inferencija na jednom pacijentu
patient = get_patient_dicts()[0]
ds = Dataset(data=[patient], transform=get_val_transforms())
batch = next(iter(DataLoader(ds, batch_size=1)))

with torch.no_grad():
    out = sliding_window_inference(
        inputs=batch["image"].to(device),
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,
    )

pred = out.argmax(dim=1).squeeze().cpu().numpy()
pred = postprocess(pred, min_voxels=50)
# pred shape: (240, 240, 155), vrijednosti 0-3
```

---

## Konfiguracija

Sve postavke su u [`brats_config/config.py`](brats_config/config.py):

| Parametar | Vrijednost | Opis |
|-----------|------------|------|
| `MODEL_NAME` | `"segresnet"` | Mozes prebaciti na `"unet3d"` za laksu arhitekturu |
| `PATCH_SIZE` | `(96, 96, 96)` | Velicina 3D isjecka za trening |
| `PATCHES_PER_VOL` | `1` | Broj patcha po pacijentu po epohi |
| `BATCH_SIZE` | `1` | MRI volumeni su veliki |
| `NUM_EPOCHS` | `200` | Broj epoha treniranja |
| `LEARNING_RATE` | `1e-4` | AdamW + CosineAnnealingLR scheduler |
| `VAL_INTERVAL` | `10` | Validacija svaku 10. epohu |
| `MAX_TRAIN_PATIENTS` | `150` | Ogranicava trening set (Kaggle GPU limit) |
| `VAL_SPLIT` | `0.2` | 20% pacijenata za validaciju |
| `RANDOM_SEED` | `42` | Deterministicki split |
| `FEATURE_SIZE` | `48` | Init filteri za SegResNet (ignorira se za UNet) |

Detaljnije obrazlozenje hyperparametara: vidi sekciju 5 u [`outputs/projekt_dokumentacija.md`](outputs/projekt_dokumentacija.md).

---

## Grane

| Grana | Opis |
|-------|------|
| `main` | Stabilan kod + dokumentacija + checkpoint (v4 SegResNet) |
| `baseline-unet3d` | Razvojna grana — sadrzi eksperimentalne konfiguracije (drukciji `PATCHES_PER_VOL`, `NUM_EPOCHS`, `MAX_TRAIN_PATIENTS`) |

---

## Tehnicki stack

- **Python 3.10+**, PyTorch 2.x, MONAI 1.x
- **NumPy, SciPy, Pandas** — numericki alati, postprocessing, CSV
- **Matplotlib, Seaborn** — vizualizacije
- **nibabel** — citanje NIfTI (.nii.gz) MRI volumena
- **kaggle** — preuzimanje dataseta
- GPU: **Kaggle T4 (16 GB VRAM, ~13 GB RAM)** — sve postavke (SmartCacheDataset, AMP, patch size) optimizirane su za taj limit

---

## Tim

| Clan | Zadatak |
|------|---------|
| Luka Bobic, Mirko Busic | Analiza podataka, Preprocessing |
| Jakov Malic, Lovre Jurjevic | Arhitektura, Treniranje |
| Antonio Simic, Lovro Travica | Postprocessing, Evaluacija |
