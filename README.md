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

Evaluacija se radi po 3 izvedene BraTS regije: **WT** (cijeli tumor), **TC** (jezgra), **ET** (aktivni dio).

---

## Struktura repozitorija

```
Brain-Tumor-Segmentation/
├── configs/
│   └── config.py          # sve postavke na jednom mjestu (patch size, epohe, LR...)
├── src/
│   ├── dataset.py         # ucitavanje pacijenata, train/val split
│   ├── transforms.py      # preprocessing pipeline (normalizacija, patch sampling, augmentacije)
│   ├── model.py           # 3D UNet i SegResNet (biramo u config.py)
│   ├── train.py           # petlja za treniranje
│   ├── postprocess.py     # (TODO) ciscenje predikcija
│   └── evaluate.py        # (TODO) Dice, HD95, vizualizacije
├── notebooks/
│   └── explore_data.ipynb # vizualna eksploracija BraTS podataka
├── data/                  # ovdje idu pacijenti — NIJE na GitHubu (.gitignore)
├── outputs/               # checkpointi i slike — NIJE na GitHubu
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
# Postavi Kaggle API token (kaggle.com -> Settings -> API -> Create New Token)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json

# Preuzmi i raspakiraj
cd data/
kaggle datasets download dschettler8845/brats-2021-task1 --file BraTS2021_Training_Data.tar
tar -xf BraTS2021_Training_Data.tar
```

**Opcija B — Kaggle notebook (preporuceno za trening):**  
Dataset je vec dostupan na Kaggleu bez downloadanja — vidi sekciju "Trening na Kaggleu" ispod.

Ocekivana struktura `data/` mape nakon raspakiravanja:
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

## Lokalno pokretanje (za razvoj i testiranje)

### Eksploracija podataka
```bash
jupyter notebook notebooks/explore_data.ipynb
```

### Provjera da sve radi (s 2 sample pacijenta)
```bash
python src/dataset.py
python src/transforms.py
python src/model.py
```

### Pokretanje treninga (lokalno, CPU — sporo, samo za debug)
```bash
python src/train.py
```

---

## Trening na Kaggleu (preporuceno — besplatan GPU T4)

1. Idi na [kaggle.com](https://kaggle.com) → **Create** → **New Notebook**
2. Dodaj dataset: desno → **Add data** → trazi `brats-2021-task1`
3. Dodaj GitHub repo:
```python
# U prvoj celiji notebooka:
!git clone https://github.com/antoniosimic/Brain-Tumor-Segmentation.git
%cd Brain-Tumor-Segmentation
!git checkout baseline-unet3d
!pip install -r requirements.txt -q
```
4. Postavi putanju podataka i pokreni trening:
```python
# Kaggle dataset je na /kaggle/input/brats-2021-task1/
import sys
sys.path.insert(0, '/kaggle/working/Brain-Tumor-Segmentation')

from configs import config
config.DATA_DIR = __import__('pathlib').Path('/kaggle/input/brats-2021-task1')

from src.train import main
main()
```
5. Ukljuci GPU: desno → **Session options** → **Accelerator: GPU T4 x1**

---

## Konfiguracija

Sve postavke su u `configs/config.py`. Najvaznije:

| Parametar | Vrijednost | Opis |
|-----------|-----------|------|
| `MODEL_NAME` | `"unet3d"` | Promijeni u `"segresnet"` za jaci model |
| `PATCH_SIZE` | `(128,128,128)` | Velicina 3D isjecka za trening |
| `BATCH_SIZE` | `1` | MRI volumeni su ogromni |
| `NUM_EPOCHS` | `100` | |
| `LEARNING_RATE` | `1e-4` | |
| `VAL_SPLIT` | `0.2` | 20% pacijenata za validaciju |

---

## Grane

| Grana | Opis |
|-------|------|
| `main` | Stabilan kod, samo provjerene stvari |
| `baseline-unet3d` | Trenutni baseline — 3D UNet pipeline |
| `experiment/segresnet` | (planirano) SegResNet arhitektura |

---

## Tim

| Clan | Zadatak |
|------|---------|
| Luka Bobic, Mirko Busic | Analiza podataka, Preprocessing |
| Jakov Malic, Lovre Jurjevic | Arhitektura, Treniranje |
| Antonio Simic, Lovro Travica | Postprocessing, Evaluacija |
