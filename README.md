# Brain Tumor Segmentation

## Opis projekta

Projekt se bavi semantičkom segmentacijom moždanih tumora na multispektralnim MRI snimkama koristeći metode dubokog učenja. Kao izvor podataka koristi se **MICCAI BraTS 2021 Task 1** skup podataka, koji za svakog pacijenta sadrži četiri MRI volumena snimljena u zajedničkom prostoru.

Cilj je razviti model koji za svaki voksel određuje pripada li:

- zdravom tkivu
- nekrotičnoj jezgri tumora
- edemu
- aktivnom dijelu tumora

Projekt je zamišljen kao potpuni pipeline koji uključuje analizu podataka, preprocessing, odabir i treniranje modela, postprocessing te evaluaciju i usporedbu rezultata.

## Podaci

Za svakog pacijenta koriste se četiri MRI modaliteta:

- **FLAIR**: naglašava edem
- **T1**: prikazuje anatomske granice zdravog tkiva
- **T1ce**: ističe aktivnu jezgru tumora nakon kontrasta
- **T2**: pomaže u detekciji tekućine i promjena u tkivu

Ova četiri volumena tretiraju se kao četiri ulazna kanala jednog 3D uzorka.

BraTS 2021 oznake segmentacije su:

- `0` pozadina / zdravo tkivo
- `1` nekrotična jezgra tumora (`NCR`)
- `2` edem (`ED`)
- `4` aktivni dio tumora (`ET`)

U praksi je preporučeno tijekom treniranja napraviti remap oznaka `0, 1, 2, 4 -> 0, 1, 2, 3`, kako bi model radio s uzastopnim indeksima klasa.

Važna prednost BraTS skupa podataka je što su volumeni već:

- registrirani u isti anatomski prostor
- resamplirani na istu rezoluciju
- skull-stripped

To znači da se može fokusirati na modeliranje i kvalitetan preprocessing bez dodatne registracije između modaliteta.

## Preporučene tehnologije

Za ovaj projekt preporučuje se sljedeći tehnološki stack:

- **Python** kao glavni programski jezik
- **PyTorch** kao osnovni framework za duboko učenje
- **MONAI** za medicinsku segmentaciju i 3D imaging workflow
- **NiBabel** za učitavanje i rad s NIfTI (`.nii.gz`) datotekama
- **NumPy** i **Pandas** za obradu podataka
- **Matplotlib** i **Seaborn** za analizu i vizualizaciju
- **scikit-learn** za validaciju, podjelu skupova i pomoćne metrike

### Zašto PyTorch + MONAI?

Najpraktičniji put za ovakav projekt je **PyTorch + MONAI**.

Razlozi:

- MONAI je razvijen specifično za medicinske slike i 3D segmentaciju
- sadrži gotove transformacije za volumene, patch-based trening i inferenciju
- uključuje gotove mreže, loss funkcije i metrike prilagođene segmentaciji
- dobro se nadovezuje na PyTorch i standardni istraživački workflow

TensorFlow je moguć izbor, ali za MRI segmentaciju i studentski projekt ove vrste PyTorch i MONAI u pravilu daju brži i čišći razvoj.

## Preporučeni smjer rada

Projekt je najbolje raditi postupno, od stabilnog baseline rješenja prema složenijim poboljšanjima.

### 1. Analiza podataka

Prvi korak je detaljno razumjeti ulazne podatke:

- provjeriti dimenzije svih volumena
- potvrditi da svi modaliteti jednog pacijenta imaju kompatibilan shape
- analizirati raspodjelu intenziteta po modalitetima
- provjeriti omjer tumorskih i netumorskih voksela
- procijeniti neuravnoteženost među klasama

Posebno je važno napraviti **patient-level split**, a ne split po sliceovima, kako bi se izbjeglo curenje informacija između skupa za treniranje i validacijskog skupa.

### 2. Preprocessing

Preprocessing bi trebao biti jednostavan, ali konzistentan:

- remap oznaka na uzastopne klase
- normalizacija intenziteta po volumenu ili po modalitetu
- cropanje na brain region ili regiju interesa
- treniranje na 3D patchovima umjesto na cijelim volumenima
- osnovne augmentacije poput flipova, malih rotacija i promjena intenziteta

Zbog veličine MRI volumena patch-based trening je najrealniji pristup, posebno ako je dostupna ograničena GPU memorija.

### 3. Arhitektura modela

Preporučeni početni model je neki stabilan **3D convolutional baseline**, npr.:

- `3D UNet`
- `SegResNet`

To je vrlo dobar izbor za prvi radni model jer:

- podržavaju 3D ulaz
- dovoljno su jaki za ozbiljan baseline
- jednostavniji su za treniranje i debugiranje od transformera

Naprednije opcije koje se mogu isprobati kasnije:

- `SwinUNETR`
- `nnU-Net` kao snažan baseline za usporedbu

Preporuka je **ne krenuti odmah s transformerima**, nego prvo dobiti pouzdan rezultat s 3D U-Net stilom modela.

### 4. Treniranje

Za treniranje je preporučeno:

- koristiti kombinirani loss poput `Dice + Cross Entropy`
- pratiti validacijsku metriku nakon svake epohe
- spremati najbolje checkpointove
- koristiti early stopping ako trening postane nestabilan
- koristiti sliding-window inferenciju za validaciju i testiranje

Klase tumora zauzimaju mali dio volumena, pa sama cross-entropy funkcija često nije dovoljna. Dice komponenta je zato vrlo korisna zbog neuravnoteženosti klasa.

### 5. Postprocessing

Nakon predikcije moguće je dodatno poboljšati rezultate jednostavnim postprocessing koracima:

- uklanjanje vrlo malih izoliranih komponenti
- popunjavanje sitnih rupa u maski
- prilagodba pragova ako se koriste probabilističke mape

Postprocessing ne treba biti prekompliciran. Često i mali zahvati daju vidljivo bolje i čišće segmentacije.

### 6. Evaluacija i usporedba

Evaluaciju treba raditi nad BraTS regijama:

- **ET**: enhancing tumor
- **TC**: tumor core
- **WT**: whole tumor

Preporučene metrike:

- **Dice score**
- **Hausdorff Distance 95% (HD95)**
- po potrebi i **sensitivity** i **specificity**

Osim konačnih metrika, korisno je napraviti i vizualnu usporedbu:

- ground truth vs. predikcija
- usporedba više modela ili više preprocessing varijanti
- primjeri dobrih i loših segmentacija

## Preporučeni eksperimentalni plan

Najrealniji plan za ovakav projekt je:

1. napraviti potpuno funkcionalan baseline pipeline
2. istrenirati prvi `3D UNet` ili `SegResNet`
3. evaluirati rezultate na validacijskom skupu
4. dodati augmentacije i bolji sampling patchova
5. isprobati još jednu arhitekturu ili jedan jači baseline
6. usporediti rezultate i izvući zaključke

Na taj način projekt ostaje izvediv, a opet pokazuje cijeli istraživački proces.

## Preporučena struktura repozitorija

Kako bi projekt ostao pregledan, preporučena je ovakva struktura:

```text
.
├── README.md
├── data/
├── notebooks/
├── src/
│   ├── dataset.py
│   ├── transforms.py
│   ├── model.py
│   ├── train.py
│   ├── infer.py
│   ├── postprocess.py
│   └── evaluate.py
├── configs/
├── outputs/
└── requirements.txt
```

Ova struktura olakšava odvajanje eksperimentalnog koda, modela, evaluacije i rezultata.

## Glavne preporuke

- krenuti s jednostavnim i stabilnim baseline modelom
- koristiti `patient-level` podjelu podataka
- trenirati na 3D patchovima
- koristiti `PyTorch + MONAI` kao glavni stack
- rezultate prikazati i brojčano i vizualno
- uvoditi složenije ideje tek nakon stabilnog osnovnog pipelinea

## Zaključak

Ovaj projekt je vrlo dobar primjer primjene dubokog učenja na medicinskim slikama jer kombinira stvarni 3D medicinski podatkovni skup, višekanalni ulaz, klasnu neuravnoteženost i standardne izazove semantičke segmentacije.

Najbolji izvedbeni put je:

- **Python + PyTorch + MONAI**
- **3D UNet ili SegResNet** kao početni model
- **patch-based trening**
- **Dice-based evaluacija uz HD95**

Takav pristup je tehnički realan, dobro objašnjiv u projektnom radu i dovoljno ozbiljan da pokaže razumijevanje cijelog procesa od podataka do evaluacije.

## Korisni izvori

- [BraTS 2021 službena stranica](https://www.med.upenn.edu/cbica/brats2021/)
- [MONAI dokumentacija](https://docs.monai.io/en/stable/)
- [PyTorch dokumentacija](https://docs.pytorch.org/docs/main/)
- [nnU-Net repozitorij](https://github.com/MIC-DKFZ/nnUNet)
- [NiBabel dokumentacija](https://nipy.org/nibabel/reference/nibabel.html)
- [TorchIO dokumentacija](https://torchio.readthedocs.io/)
