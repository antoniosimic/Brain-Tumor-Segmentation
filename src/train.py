"""
train.py — petlja za treniranje.

Pokretanje:
  python src/train.py

Sto radi:
  1. Ucita pacijente i napravi train/val split
  2. Kreira DataLoader s MONAI transformacijama
  3. Inicijalizira model, optimizer, loss
  4. Trenira N epoha, evaluira svake VAL_INTERVAL epohe
  5. Sprema best checkpoint prema val Dice scoreu
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism

from brats_config.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    OUTPUT_DIR,
    PATCH_SIZE,
    VAL_INTERVAL,
)
from src.dataset import get_patient_dicts, train_val_split
from src.model import build_model, count_params
from src.transforms import get_train_transforms, get_val_transforms


def main():
    set_determinism(seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uredaj: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Podaci ───────────────────────────────────────────────────────────────
    all_dicts = get_patient_dicts()
    print(f"\nUkupno pacijenata: {len(all_dicts)}")

    train_dicts, val_dicts = train_val_split(all_dicts)
    print(f"Train: {len(train_dicts)}  |  Val: {len(val_dicts)}")

    # CacheDataset ucita sve pacijente jednom u RAM — epohe 2-N su puno brze
    # cache_rate=1.0 = spremi sve; smanji ako nemas dovoljno RAM-a (T4 ima 16GB)
    # cache_rate=0.07 = ~70 pacijenata u RAM (~10GB) — stane na T4 x2
    train_ds = CacheDataset(data=train_dicts, transform=get_train_transforms(), cache_rate=0.07, num_workers=2)
    val_ds   = CacheDataset(data=val_dicts,   transform=get_val_transforms(),   cache_rate=0.07, num_workers=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    total, trainable = count_params(model)
    print(f"\nModel: {total:,} parametara ({trainable:,} trainable)")

    # ── Loss, optimizer, metrika ──────────────────────────────────────────────
    loss_fn   = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    # Post-processing za evaluaciju
    post_pred  = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])

    # ── Mixed precision — prepolovi VRAM, gotovo ista tocnost ────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Trening ───────────────────────────────────────────────────────────────
    best_val_dice = 0.0
    checkpoint_path = OUTPUT_DIR / "best_model.pth"

    print(f"\nKrecemo trenirati {NUM_EPOCHS} epoha...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            if isinstance(batch, list):
                inputs = torch.cat([b["image"] for b in batch]).to(device)
                labels = torch.cat([b["seg"]   for b in batch]).to(device)
            else:
                inputs = batch["image"].to(device)
                labels = batch["seg"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss    = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(steps, 1)
        print(f"Epoha {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}", end="")

        # ── Validacija ────────────────────────────────────────────────────────
        if epoch % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["seg"].to(device)

                    # Sliding window inference na cijelom volumenu
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs,
                        roi_size=PATCH_SIZE,
                        sw_batch_size=2,
                        predictor=model,
                        overlap=0.5,
                    )

                    val_outputs = [post_pred(i)  for i in decollate_batch(val_outputs)]
                    val_labels  = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

            # Dice po klasi (ignoriramo background — index 0)
            metric = dice_metric.aggregate()  # shape: (num_classes - 1,)
            dice_metric.reset()

            mean_dice = metric.mean().item()
            class_dice = metric.tolist()

            print(f"  |  val Dice={mean_dice:.4f}  "
                  f"[NCR={class_dice[0]:.3f}, ED={class_dice[1]:.3f}, ET={class_dice[2]:.3f}]",
                  end="")

            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                torch.save(model.state_dict(), checkpoint_path)
                print("  ← BEST", end="")

        print()  # novi red

    print(f"\nTreniranje zavrseno. Najbolji val Dice: {best_val_dice:.4f}")
    print(f"Checkpoint spremljen: {checkpoint_path}")


if __name__ == "__main__":
    main()
