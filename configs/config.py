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
PATCH_SIZE      = (128, 128, 128)   # velicina 3D patcha
PATCHES_PER_VOL = 2                 # patchevi po pacijentu po epohi

# ── Trening ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 1
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-4
VAL_INTERVAL  = 2         # evaluiraj svaku N-tu epohu

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "unet3d"   # "unet3d" ili "segresnet"
IN_CHANNELS  = 4
OUT_CHANNELS = NUM_CLASSES
FEATURE_SIZE = 48         # SegResNet; ignorira se za UNet
