from pathlib import Path

# ── Putanje ──────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Podaci ───────────────────────────────────────────────────────────────────
MODALITIES   = ["flair", "t1", "t1ce", "t2"]
NUM_CLASSES  = 4          # 0=zdravo, 1=NCR, 2=edem, 3=ET  (remap iz 0,1,2,4)
VAL_SPLIT    = 0.2        # 20% pacijenata za validaciju
RANDOM_SEED  = 42

# ── Patch sampling ───────────────────────────────────────────────────────────
PATCH_SIZE      = (64, 64, 64)      # velicina 3D patcha — sigurno za T4 16GB
PATCHES_PER_VOL = 1                 # 1 patch po pacijentu — manji efektivni batch

# ── Trening ──────────────────────────────────────────────────────────────────
BATCH_SIZE          = 1
NUM_EPOCHS          = 50
LEARNING_RATE       = 1e-4
VAL_INTERVAL        = 5    # evaluiraj svaku N-tu epohu
MAX_TRAIN_PATIENTS  = 80   # None = svi pacijenti; 80 = ~11GB RAM, sigurno za T4

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "unet3d"   # "unet3d" ili "segresnet"
IN_CHANNELS  = 4
OUT_CHANNELS = NUM_CLASSES
FEATURE_SIZE = 48         # SegResNet; ignorira se za UNet
