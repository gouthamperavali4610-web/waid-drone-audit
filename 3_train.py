"""
3_train.py
-----------
Fine-tunes YOLOv8-nano on WAID. Run this ONCE (~20-40 min on an RTX 4070).

"Fine-tuning" means: start from a model that already learned general vision
from millions of photos (COCO), then keep training it on WAID so it learns
these six specific aerial animals. This is why you don't need a supercomputer
or a million images -- you're adjusting an existing model, not building one
from scratch.

Output: runs/waid_baseline/weights/best.pt  <-- the file every other script uses
"""

from pathlib import Path
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# SETTINGS -- these are the knobs you'd change if you wanted to experiment
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
DATA_YAML = PROJECT_DIR / "waid_local.yaml"   # written by 2_prepare_waid.py

MODEL = "yolov8n.pt"   # 'n' = nano, the smallest/fastest. Downloads automatically.
                       # Options: yolov8n (fastest) < s < m < l < x (most accurate)
                       # Start with nano. It matches the "low-cost" framing of
                       # your research question, and it trains fast.

EPOCHS = 50            # How many times the model studies the whole dataset.
                       # 50 is a solid starting point. More isn't always better --
                       # watch for the val score plateauing.

IMGSZ = 640            # Images get resized to 640x640 for training. This is the
                       # standard. Bigger = better on small objects but slower.

BATCH = 16             # How many images the GPU processes at once.
                       # 16 is safe on 8 GB VRAM. If you get an "out of memory"
                       # error, drop to 8. If it runs fine, you can try 32.

SEED = 0               # Fixes the randomness so re-running gives the SAME result.
                       # Critical for a credible audit -- it makes your work
                       # reproducible, which is the entire point of your paper.

RUN_NAME = "waid_baseline"


def main():
    # --- Safety checks before we burn 30 minutes -------------------------
    if not DATA_YAML.exists():
        raise SystemExit(f"Missing {DATA_YAML}\nRun: python 2_prepare_waid.py")

    if not torch.cuda.is_available():
        print("\n" + "!" * 62)
        print("WARNING: No GPU detected. This will take 10-30 HOURS on CPU.")
        print("Run 1_setup_check.py to fix your PyTorch install first.")
        print("!" * 62)
        if input("\nContinue anyway? (y/n): ").lower() != "y":
            raise SystemExit("Cancelled -- good call.")
        device = "cpu"
    else:
        print(f"Training on: {torch.cuda.get_device_name(0)}")
        device = 0   # 0 means "the first GPU"

    # --- Load the starting model -----------------------------------------
    # This downloads yolov8n.pt (~6 MB) the first time you run it.
    model = YOLO(MODEL)

    # --- Train -----------------------------------------------------------
    # This one call does everything: loads data, trains, validates each epoch,
    # saves checkpoints, and writes out graphs of the training progress.
    print(f"\nStarting training: {EPOCHS} epochs at {IMGSZ}px, batch {BATCH}")
    print("You'll see a table of numbers per epoch. Watch mAP50 climb.\n")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        device=device,
        project="runs",       # top-level output folder
        name=RUN_NAME,        # subfolder name
        exist_ok=True,        # allow overwriting a previous run of the same name
        patience=15,          # stop early if val score hasn't improved in 15 epochs
        plots=True,           # save training graphs + confusion matrix as images
    )

    # --- Evaluate the finished model --------------------------------------
    print("\n" + "=" * 62)
    print("FINAL EVALUATION")
    print("=" * 62)

    metrics = model.val()

    # These are the four numbers you report in your paper.
    print(f"\n  mAP@0.5      : {metrics.box.map50:.4f}   <- the headline number")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}   <- stricter; always lower")
    print(f"  Precision    : {metrics.box.mp:.4f}   <- of boxes drawn, how many were right")
    print(f"  Recall       : {metrics.box.mr:.4f}   <- of animals present, how many were found")

    print("\n  Per-class mAP@0.5:")
    for i, name in model.names.items():
        try:
            print(f"    {name:<10} {metrics.box.maps[i]:.4f}")
        except (IndexError, TypeError):
            pass

    # --- How to interpret this -------------------------------------------
    print("\n" + "-" * 62)
    print("HOW TO READ THIS (for your write-up):")
    print("  The WAID paper's SE-YOLO scored 0.983 mAP. That was THEIR custom")
    print("  model, not YOLOv8-nano. If you land in the 0.90-0.97 range, that")
    print("  is a perfectly good baseline -- you're auditing the dataset, not")
    print("  competing on leaderboard scores.")
    print("\n  IMPORTANT: this high score is on WAID's OWN test images. That is")
    print("  the in-distribution ceiling. The whole point of your study is what")
    print("  happens when you move away from it (scripts 4 and 5).")

    weights = Path("runs") / RUN_NAME / "weights" / "best.pt"
    print(f"\n[done] Your trained model: {weights}")
    print("\nNext step:  python 4_identify.py <folder_of_your_photos>")


if __name__ == "__main__":
    main()
