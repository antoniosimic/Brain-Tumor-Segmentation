# Brain Tumor Segmentation — BraTS 2021

Semanticka segmentacija mozdanih tumora na MRI snimkama pomocu dubokog ucenja.  
Dataset: [MICCAI BraTS 2021 Task 1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data)

---

## Sto radi ovaj projekt

Za svakog pacijenta postoje 4 MRI volumena (modaliteta). Model ih uzima kao 4-kanalni 3D ulaz i za svaki voksel predvida jednu od 4 klase:

| Label | Klasa |
|-------|-------|
| 0 | Zdravo tkivo (pozadina) |
| 1 | Nekroticna jezgra (NCR) |
| 2 | Edem (ED) |
| 3 | Aktivni tumor (ET) — originalno label 4 u BraTS datasetu |

Evaluacija se radi po 3 izvedene BraTS regije: **WT** (cijeli tumor), **TC** (jezgra tumora), **ET** (aktivni dio).

---

## Trenutni rezultati

Model: 3D UNet, 4.8M parametara, treniran 50 epoha na 80 pacijenata (Kaggle T4 GPU)

| Regija | Dice |
|--------|------|
| WT (Whole Tumor) | 0.606 (val, best checkpoint) |
| TC (Tumor Core) | — |
| ET (Enhancing Tumor) | — |

> Puni val rezultati (Dice + HD95 po svim 250 pacijenata) dolaze pokretanjem `src/evaluate.py` na Kaggleu s GPU-om.

---

## Struktura repozitorija

```
Brain-Tumor-Segmentation/
├── brats_config/
│   └── config.py           # sve postavke (patch size, epohe, LR...)
├── src/
│   ├── dataset.py          # ucitavanje pacijenata, train/val split
│   ├── transforms.py       # preprocessing (normalizacija, patch sampling, augmentacije)
│   ├── model.py            # 3D UNet i SegResNet
│   ├── train.py            # petlja za treniranje
│   ├── postprocess.py      # ciscenje predikcija (connected components, fill holes)
│   └── evaluate.py         # Dice, HD95 po WT/TC/ET, vizualizacije
├── notebooks/
│   ├── explore_data.ipynb  # vizualna eksploracija BraTS podataka
│   └── evaluate_model.ipynb # prikaz rezultata istreniranog modela
├── outputs/
│   ├── best_model.pth      # istreniran checkpoint (19MB)
│   ├── evaluation_results.csv
│   └── *.png               # grafovi i vizualizacije
├── data/                   # pacijenti — NIJE na GitHubu (previse veliki)
└── requirements.txt
```

---

## Postavljanje okruzenja

```bash
git clone https://github.com/antoniosimic/Brain-Tumor-Segmentation.git
cd Brain-Tumor-Segmentation
git checkout baseline-unet3d

pip install -r requirements.txt
```

---

## Podaci

Dataset je prevelik za Git (~13 GB). Preuzmi ga na jedan od dva nacina:

**Opcija A — Kaggle CLI:**
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json

cd data/
kaggle datasets download dschettler8845/brats-2021-task1 --file BraTS2021_Training_Data.tar
tar -xf BraTS2021_Training_Data.tar
```

**Opcija B — Kaggle notebook (preporuceno za trening):**  
Dataset je vec dostupan na Kaggleu bez skidanja — vidi sekciju ispod.

Ocekivana struktura `data/` mape:
```
data/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_flair.nii.gz
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
└── ...
```

---

## Lokalno pokretanje

```bash
# Provjera da sve radi
python src/dataset.py
python src/transforms.py
python src/model.py

# Trening (lokalno — sporo, samo za debug)
python src/train.py

# Evaluacija na validacijskom skupu
python src/evaluate.py --checkpoint outputs/best_model.pth

# Prikaz rezultata u notebooku
jupyter notebook notebooks/evaluate_model.ipynb
```

---

## Trening na Kaggleu (preporuceno — besplatan GPU T4)

1. Idi na [kaggle.com](https://kaggle.com) → **Create** → **New Notebook**
2. Dodaj dataset: desno → **Add data** → `brats-2021-task1`
3. Ukljuci GPU: **Session options** → **Accelerator: GPU T4 x1**
4. U notebook celije upisi:

```python
# Postavljanje
!git clone https://github.com/antoniosimic/Brain-Tumor-Segmentation.git
%cd Brain-Tumor-Segmentation
!git checkout baseline-unet3d
!pip install -r requirements.txt -q

# Konfiguracija putanje podataka
import sys
sys.path.insert(0, '/kaggle/working/Brain-Tumor-Segmentation')
from brats_config import config
from pathlib import Path
config.DATA_DIR = Path('/kaggle/input/brats-2021-task1/BraTS2021_Training_Data')

# Trening
from src.train import main
main()

# Evaluacija
from src.evaluate import evaluate
evaluate(checkpoint_path='outputs/best_model.pth')
```

---

## Konfiguracija

Sve postavke su u `brats_config/config.py`:

| Parametar | Vrijednost | Opis |
|-----------|-----------|------|
| `MODEL_NAME` | `"unet3d"` | Promijeni u `"segresnet"` za drugaciju arhitekturu |
| `PATCH_SIZE` | `(64, 64, 64)` | Velicina 3D isjecka za trening |
| `BATCH_SIZE` | `1` | MRI volumeni su veliki |
| `NUM_EPOCHS` | `100` | Broj epoha treniranja |
| `LEARNING_RATE` | `1e-4` | |
| `MAX_TRAIN_PATIENTS` | `150` | Maksimalan broj pacijenata za trening |
| `VAL_SPLIT` | `0.2` | 20% pacijenata za validaciju |

---

## Grane

| Grana | Opis |
|-------|------|
| `main` | Stabilan kod, dokumentacija |
| `baseline-unet3d` | Aktivni razvoj — 3D UNet pipeline |

---

## Tim

| Clan | Zadatak |
|------|---------|
| Luka Bobic, Mirko Busic | Analiza podataka, Preprocessing |
| Jakov Malic, Lovre Jurjevic | Arhitektura, Treniranje |
| Antonio Simic, Lovro Travica | Postprocessing, Evaluacija |
