# Segmentacija mozdanih tumora — Tehnička dokumentacija projekta

**Kolegij:** Digitalna obrada i analiza slike  
**Dataset:** MICCAI BraTS 2021 Task 1  
**Repozitorij:** https://github.com/antoniosimic/Brain-Tumor-Segmentation  

---

## 1. Pregled projekta

Cilj projekta je automatska **semanticka segmentacija mozdanih tumora** na MRI snimkama pomocu dubokog ucenja. Za svakog pacijenta model prima 4 MRI modaliteta kao ulaz i za svaki voksel (3D piksel) predvida kojoj klasi pripada.

Projekt implementira kompletan pipeline:
```
Sirovi MRI podaci
      ↓
Preprocessing (normalizacija, remap labela)
      ↓
Treniranje modela (3D neuronska mreza)
      ↓
Postprocessing (uklanjanje suma)
      ↓
Evaluacija (Dice, HD95 po BraTS regijama)
      ↓
Vizualizacija rezultata
```

---

## 2. Dataset — BraTS 2021 Task 1

### 2.1 Opis podataka

**BraTS** (Brain Tumor Segmentation) je standardni benchmark dataset za segmentaciju mozdanih tumora. Koristimo **Task 1 — 2021 verziju** s 1251 pacijentom.

Svaki pacijent ima **4 MRI modaliteta**:

| Modalitet | Puni naziv | Sto prikazuje |
|-----------|-----------|---------------|
| **FLAIR** | Fluid Attenuated Inversion Recovery | Edem (oteklina) oko tumora |
| **T1** | T1-weighted | Anatomska struktura mozga |
| **T1ce** | T1 s kontrastom (gadolinium) | Aktivni (enhancing) tumor |
| **T2** | T2-weighted | Edem, likvorne prostore |

Svaki volumen ima dimenzije **240 × 240 × 155 voksela** (~145 MB po pacijentu).

### 2.2 Klase segmentacije

BraTS originalno ima labele 0, 1, 2, 4. Mi ih remapiramo na uzastopne vrijednosti zbog PyTorch loss funkcija:

| Originalna labela | Remapirana | Klasa | Opis |
|-------------------|-----------|-------|------|
| 0 | 0 | Pozadina | Zdravo tkivo mozga |
| 1 | 1 | NCR | Nekroticna jezgra tumora |
| 2 | 2 | ED | Edem (oteklina oko tumora) |
| 4 | **3** | ET | Aktivni (enhancing) tumor |

**Distribucija voksela** (zasto je problem tezak):
- Pozadina (klasa 0): ~98.9% voksela
- Tumor ukupno: ~1.1% voksela

Ovo je ekstremna **klasna neravnoteza** — model mora nauciti prepoznati rijetke voksele tumora u moru zdravog tkiva.

### 2.3 BraTS evaluacijske regije

BraTS evaluacija se ne mjeri po klasama direktno, vec po **3 izvedene binarne regije**:

| Regija | Naziv | Koje klase ukljucuje | Klinicki znacaj |
|--------|-------|---------------------|-----------------|
| **WT** | Whole Tumor | 1 + 2 + 3 | Cijeli tumor — za planiranje operacije |
| **TC** | Tumor Core | 1 + 3 | Jezgra — za radioterapiju |
| **ET** | Enhancing Tumor | 3 | Aktivni dio — za pracenje terapije |

---

## 3. Preprocessing pipeline

### 3.1 Ucitavanje i priprema

```
1. Ucitaj 5 NIfTI datoteka (4 modaliteta + segmentacija)
2. Dodaj channel dimenziju: (H,W,D) → (1,H,W,D)
3. Remap labela: 4 → 3
4. Spoji 4 modaliteta: (1,H,W,D) × 4 → (4,H,W,D)
5. Z-score normalizacija po kanalu (samo brain vokseli, nonzero=True)
```

### 3.2 Z-score normalizacija

Za svaki od 4 kanala racunamo:

```
x_norm = (x - mean(x)) / std(x)
```

gdje `mean` i `std` racunamo samo na nenultim vokselima (nonzero=True) jer je pozadina MRI snimke 0. Ovo osigurava da svaki modalitet ima priblizno jednaku skalu neovisno o aparatu kojim je sniman.

### 3.3 Patch-based trening

MRI volumeni su preveliki za direktno treniranje (4 × 240 × 240 × 155 × float32 ≈ 220 MB po pacijentu). Koristimo **3D patch sampling**:

- Velicina patcha: **96 × 96 × 96 voksela**
- Strategija: `RandCropByPosNegLabeld` s omjerom 1:1 (tumor:ne-tumor)
- Ovo osigurava da ~50% patchy sadrzavaju tumor — bori se s klasnom neravnoteom

### 3.4 Augmentacije (samo za trening)

| Augmentacija | Vjerojatnost | Opis |
|-------------|-------------|------|
| Random Flip | 50% po osi | Zrcaljenje po x, y, z osi |
| Random Rotate90 | 50% | Rotacija za 90/180/270 stupnjeva |
| Scale Intensity | 50% | Mnozi intenzitete s faktorom ±10% |
| Shift Intensity | 50% | Pomice intenzitete za ±10% |

Augmentacije povecavaju raznolikost podataka i sprecavaju overfitting.

### 3.5 Train/Val split

- **80% train** (~1001 pacijenata), **20% val** (~250 pacijenata)
- Split na razini pacijenta — nema "data leakage" (isti pacijent ne moze biti u oba skupa)
- Deterministicki shuffle (seed=42) za reproducibilnost

---

## 4. Arhitektura modela — SegResNet

Koristimo **SegResNet** iz MONAI biblioteke — rezidualna enkoder-dekoder arhitektura specificno dizajnirana za medicinsku 3D segmentaciju.

### 4.1 Osnovne karakteristike

| Parametar | Vrijednost |
|-----------|-----------|
| Ulazni kanali | 4 (FLAIR, T1, T1ce, T2) |
| Izlazni kanali | 4 (klase 0-3) |
| Prostorne dimenzije | 3D |
| Broj parametara | ~4.7M |
| Velicina modela | ~72 MB |

### 4.2 Principi rada

**SegResNet** kombinira:
1. **Encoder** — sekvenca 3D konvolucija s downsampling koracima. Svaki blok smanjuje prostornu rezoluciju i povecava broj filtera (uce apstraktne reprezentacije)
2. **Rezidualni blokovi** — skip connections unutar blokova sprecavaju nestajanje gradijenata (vanishing gradient) kod dubokih mreza
3. **Decoder** — upsample + konvolucija, postupno vraca prostornu rezoluciju
4. **Skip connections** — direktne veze iz encodera u dekoder cuvaju fine prostorne detalje

```
Input (4,96,96,96)
      ↓
   Encoder
   [Conv → ResBlock → Downsample] × N
      ↓
   Bottleneck (visoko-apstraktne znacajke)
      ↓
   Decoder
   [Upsample → Skip connection + ResBlock] × N
      ↓
Output (4,96,96,96)  ← logiti po klasi za svaki voksel
```

### 4.3 Zasto SegResNet umjesto klasicnog UNet-a?

| | 3D UNet | SegResNet |
|--|---------|-----------|
| Rezidualni blokovi | Da (ograniceno) | Da (jace) |
| BraTS val Dice | 0.695 | **0.7795** |
| Velicina modela | 19 MB | 72 MB |
| Brzina treniranja | Brze | Sporije |

---

## 5. Treniranje

### 5.1 Konfiguracija

| Hyperparametar | Vrijednost | Obrazlozenje |
|----------------|-----------|--------------|
| Batch size | 1 | MRI volumeni su veliki |
| Patch size | 96 × 96 × 96 | Kompromis: kontekst vs. GPU RAM |
| Epohe | 200 | Model konvergira postupno |
| Learning rate | 1e-4 | Standardno za medicinsku segmentaciju |
| Optimizer | AdamW | Adam + L2 regularizacija |
| LR scheduler | CosineAnnealingLR | Postupno smanjuje LR do nule |
| Train pacijenata | 150 | Ograniceno Kaggle GPU vremenom |
| Val pacijenata | ~250 | Cijeli val skup |
| Val interval | svake 10 epohe | Balans: tocnost vs. brzina |

### 5.2 Loss funkcija — DiceCELoss

Koristimo kombinaciju dva lossa:

**Dice Loss:**
```
Dice Loss = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
```
gdje je P = predikcija, G = ground truth, ε = smoothing faktor.

- Direktno optimizira Dice koeficijent (nasa evaluacijska metrika)
- Otporan na klasnu neravnotezu (ne kaze "sve je pozadina")

**Cross-Entropy Loss:**
```
CE Loss = -Σ y_i * log(p_i)
```
- Stabilizira trening, posebno u ranim epohama
- Daje gradijente i za negativne primjere

**DiceCELoss = Dice Loss + Cross-Entropy Loss**

Kombinacija je empirijski pokazala bolje rezultate od svakog lossa zasebno.

### 5.3 Mixed Precision Training

Koristimo `torch.amp.autocast` i `GradScaler`:
- Forward pass i loss racunaju se u **float16** (pola memorije)
- Backward pass (gradijenti) u **float32** (preciznost)
- Ubrzava trening ~1.5-2x na GPU-u bez gubitka tocnosti

### 5.4 SmartCacheDataset

Kaggle T4 GPU ima ~13 GB RAM-a. Klasicno cachiranje 150 pacijenata × 135 MB = 20 GB → OOM.

`SmartCacheDataset` drzi **60 pacijenata u RAM-u** odjednom i rotira ih svaku epohu — nema OOM-a, nema pisanja na disk.

### 5.5 Sliding Window Inference (validacija)

Za validaciju ne mozemo cijeli volumen (240×240×155) staviti u GPU odjednom. Koristimo **sliding window inference**:

```
Prolazi 96×96×96 prozorom po cijelom volumenu s preklapanjem 25%
Usrednji se predikcije na mjestima gdje se prozori preklapaju
Rezultat: predikcija za cijeli 240×240×155 volumen
```

---

## 6. Postprocessing

### 6.1 Uklanjanje malih komponenti

Model ponekad predvidi izolirane "otocice" tumora koji su zapravo sum:

```python
remove_small_components(mask, min_voxels=50)
```

Za svaku klasu posebno:
1. Pronadi sve izolirane grupe voksela (connected components)
2. Ukloni sve grupe s manje od 50 voksela
3. Zadrzaj samo statisticki znacajne regije

### 6.2 Popunjavanje rupa

```python
fill_holes(mask)
```

Popunjava male rupe unutar segmentiranih regija — npr. model predvidi prsten nekroze ali ostavi prazninu u sredini.

---

## 7. Evaluacijske metrike

### 7.1 Dice Score (DSC)

**Najvaznija metrika u medicinskoj segmentaciji.**

```
Dice = (2 * |P ∩ G|) / (|P| + |G|)
```

- **P** = skup voksela koje model predvidi kao tumor
- **G** = skup voksela koji su stvarno tumor (ground truth)
- Vrijednost: **0 = nema poklapanja, 1 = savrsena predikcija**

**Intuicija:** Mjeri omjer preklapanja predikcije i stvarnog tumora. Dice = 0.8 znaci da se 80% predvidenog i stvarnog tumora poklapa.

**Primjer:**
```
Ground truth:  ████████░░░░    (8 voksela tumora)
Predikcija:    ██████░░░░░░    (6 voksela predvideno)
Preklapanje:   ██████          (6 voksela tocno)

Dice = (2 × 6) / (8 + 6) = 12/14 = 0.857
```

### 7.2 Hausdorff Distance 95% (HD95)

**Mjeri maksimalnu udaljenost granica tumora.**

```
HD95 = 95. percentil { max_dist(P→G), max_dist(G→P) }
```

- Racuna se na rubnim vokselima (granici) predikcije i ground trutha
- Mjeri koliko je "daleko" najgora predikcija granice
- **Manji = bolji** (0 = savrsena granica)
- 95. percentil umjesto maksimuma — ignorirace se mali broj outlier voksela

**Intuicija:** Dok Dice mjeri koliko volumena se poklapa, HD95 mjeri koliko su granice tocne. Model moze imati dobar Dice ali lose HD95 ako predvidi neke voksele daleko od stvarnog tumora.

### 7.3 Metrike po BraTS regijama

Obje metrike racunamo za svaku od 3 BraTS regije:

| Metrika | WT | TC | ET |
|---------|----|----|-----|
| Dice | Dice_WT | Dice_TC | Dice_ET |
| HD95 | HD95_WT | HD95_TC | HD95_ET |

Ovo daje ukupno **6 metrika** po pacijentu, sto je standardni BraTS evaluacijski protokol.

---

## 8. Rezultati

### 8.1 Tijek razvoja modela

| Verzija | Arhitektura | Epohe | Patch | Pacijenata | Val Dice |
|---------|------------|-------|-------|-----------|----------|
| v1 | 3D UNet | 50 | 64³ | 80 | 0.606 |
| v2 | 3D UNet | 100 | 64³ | 150 | 0.643 |
| v3 | 3D UNet | 100 | 96³ | 150 | 0.695 |
| **v4** | **SegResNet** | **200** | **96³** | **150** | **0.7795** |

### 8.2 Finalni rezultati po klasama (v4 — best model)

Iz training loga (validacijski skup, ~250 pacijenata):

| Klasa | Val Dice |
|-------|---------|
| NCR (Nekroza) | 0.700 |
| ED (Edem) | 0.817 |
| ET (Activni tumor) | 0.821 |
| **Mean** | **0.7795** |

### 8.3 Kontekst rezultata

| Razina | Mean Dice | Opis |
|--------|-----------|------|
| Slucajno pogadanje | ~0.01 | Referentna tocka |
| Nas model (v4) | **0.780** | Studentski projekt, 150/1251 pacijenata |
| Solidni akademski | ~0.85 | Puni dataset, dulje treniranje |
| BraTS pobjednici | ~0.90+ | State-of-the-art, kompleksni ansambli |

Nasa razlika od SOTA-e je uglavnom zbog:
1. Koristimo samo 150 od 1251 dostupnih pacijenata (Kaggle GPU limit)
2. Jednostavnija arhitektura bez ansambla modela
3. Bez test-time augmentacije

---

## 9. Tehnicki stack

| Komponenta | Tehnologija | Verzija |
|------------|------------|---------|
| Jezik | Python | 3.10+ |
| Deep Learning | PyTorch | 2.x |
| Medicinska AI | MONAI | 1.x |
| Numericki alati | NumPy, SciPy | — |
| Vizualizacija | Matplotlib | — |
| Analiza podataka | Pandas | — |
| GPU trening | Kaggle T4 GPU | 16 GB VRAM |
| MRI format | NIfTI (.nii.gz) | nibabel |

---

## 10. Struktura koda

```
Brain-Tumor-Segmentation/
├── brats_config/
│   └── config.py         # Centralna konfiguracija (hyperparametri, putanje)
├── src/
│   ├── dataset.py        # Ucitavanje pacijenata, train/val split
│   ├── transforms.py     # MONAI preprocessing pipeline
│   ├── model.py          # SegResNet i UNet3D arhitekture
│   ├── train.py          # Petlja za treniranje
│   ├── postprocess.py    # Connected components, fill holes
│   ├── evaluate.py       # Dice, HD95, CSV, grafovi
│   └── visualize.py      # Detaljne vizualizacije GT vs predikcija
├── notebooks/
│   ├── explore_data.ipynb    # Eksploracija BraTS podataka
│   └── evaluate_model.ipynb  # Interaktivni prikaz rezultata
├── outputs/
│   ├── best_model.pth        # Istrenirani model (72 MB)
│   ├── evaluation_results.csv
│   └── *.png                 # Vizualizacije
└── requirements.txt
```

---

## 11. Zakljucak

Implementirali smo kompletan pipeline za segmentaciju mozdanih tumora od ucitavanja sirovih MRI podataka do kvantitativne evaluacije. Kljucni doprinosi:

1. **Robustan preprocessing** — Z-score normalizacija, label remap, patch sampling s balansiranjem klasa
2. **SegResNet arhitektura** — rezidualna 3D mreza optimizirana za medicinske volumene
3. **Stabilan trening** — mixed precision, SmartCacheDataset za Kaggle GPU ogranicenja, cosine LR scheduler
4. **BraTS-kompatibilna evaluacija** — Dice i HD95 po WT/TC/ET regijama
5. **Postprocessing** — connected component filtering i hole filling

Finalni model postize **val Dice = 0.7795** (NCR=0.700, ED=0.817, ET=0.821) sto je solidan rezultat za studentski projekt s ogranicenim GPU resursima.
