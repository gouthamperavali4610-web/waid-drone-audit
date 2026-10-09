"""
7_compare_models.py
--------------------
Compares TWO detectors on the SAME images:

  COCO model (yolov8n.pt)  -- trained on ~118k everyday photos, mostly taken
                              from the ground. Knows 80 classes including
                              dog, bird, sheep, cow, zebra, horse, cat.
  WAID model (best.pt)     -- your fine-tune, trained only on aerial drone
                              images of 6 animals.

Why this is a real experiment: sheep, cow, and zebra exist in BOTH datasets.
So running both models on aerial images tests one thing only -- does training
on aerial data matter? Same animals, same photos, only the training differs.

Usage:
    python 7_compare_models.py                 (uses WAID's test images)
    python 7_compare_models.py my_photos       (uses your own photos)

Output: results/model_comparison.csv
"""

import sys
from pathlib import Path

import yaml
import pandas as pd
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).parent
WAID_WEIGHTS = PROJECT_DIR / "runs" / "waid_baseline" / "weights" / "best.pt"
COCO_WEIGHTS = "yolov8n.pt"       # downloads itself if not present
DATA_YAML = PROJECT_DIR / "waid_local.yaml"
OUT_CSV = PROJECT_DIR / "results" / "model_comparison.csv"

CONFIDENCE = 0.25
MAX_IMAGES = 300      # keep it quick; raise for the final paper run

# Which COCO classes count as "an animal". COCO uses numbers for classes;
# these are the animal ones. We ignore people, cars, etc.
COCO_ANIMAL_IDS = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse",
    18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe",
}

# The three animals that exist in BOTH datasets -- these give the fairest
# head-to-head, because both models were actually taught to find them.
SHARED = {"sheep": "sheep", "cattle": "cow", "zebra": "zebra"}


def load_ground_truth(images_dir, image_names):
    """
    Read the label files so we know the RIGHT answer for each image.

    Labels live in a 'labels' folder mirroring the 'images' folder, and each
    line in a label file is one animal.
    """
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    labels_dir = Path(*parts)

    truth = {}
    for name in image_names:
        label_file = labels_dir / (Path(name).stem + ".txt")
        count = 0
        if label_file.exists():
            with open(label_file) as f:
                count = sum(1 for line in f if line.strip())
        truth[name] = count
    return truth


def run_model(model, image_paths, animal_filter=None):
    """
    Run one model over the images and count what it found in each.

    animal_filter: for the COCO model we only count animal classes, since it
    would otherwise report chairs and people we don't care about.
    """
    found = {}
    results = model.predict(source=[str(p) for p in image_paths],
                            conf=CONFIDENCE, verbose=False, stream=True)
    for result in results:
        name = Path(result.path).name
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)
                if animal_filter is not None and class_id not in animal_filter:
                    continue          # not an animal -- skip it
                detections.append({
                    "label": model.names[class_id],
                    "conf": float(box.conf),
                })
        found[name] = detections
    return found


def main():
    if not WAID_WEIGHTS.exists():
        raise SystemExit(f"No WAID model at {WAID_WEIGHTS}\nRun: python 3_train.py")

    # --- Pick which images to test on -------------------------------------
    if len(sys.argv) > 1:
        images_dir = Path(sys.argv[1])
        using_waid_test = False
    else:
        cfg = yaml.safe_load(open(DATA_YAML))
        images_dir = Path(cfg.get("test", cfg["val"]))
        using_waid_test = True

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:MAX_IMAGES]

    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    print(f"Testing on : {images_dir}")
    print(f"Images     : {len(image_paths)}\n")

    # --- Load both models --------------------------------------------------
    print("Loading COCO model (general-purpose, ground-level training)...")
    coco_model = YOLO(COCO_WEIGHTS)
    print("Loading WAID model (your aerial fine-tune)...\n")
    waid_model = YOLO(str(WAID_WEIGHTS))

    # --- Run both ----------------------------------------------------------
    print("Running COCO model...")
    coco_found = run_model(coco_model, image_paths, animal_filter=COCO_ANIMAL_IDS)
    print("Running WAID model...")
    waid_found = run_model(waid_model, image_paths, animal_filter=None)

    # --- Ground truth (only available for WAID's own images) ---------------
    truth = load_ground_truth(images_dir, [p.name for p in image_paths]) \
        if using_waid_test else {}

    # --- Build the comparison table ---------------------------------------
    rows = []
    for path in image_paths:
        name = path.name
        coco_dets = coco_found.get(name, [])
        waid_dets = waid_found.get(name, [])

        rows.append({
            "image": name,
            "actual_animals": truth.get(name, ""),
            "coco_found": len(coco_dets),
            "waid_found": len(waid_dets),
            "coco_labels": ", ".join(sorted({d["label"] for d in coco_dets})),
            "waid_labels": ", ".join(sorted({d["label"] for d in waid_dets})),
            "coco_avg_conf": round(sum(d["conf"] for d in coco_dets) / len(coco_dets), 3)
                             if coco_dets else 0,
            "waid_avg_conf": round(sum(d["conf"] for d in waid_dets) / len(waid_dets), 3)
                             if waid_dets else 0,
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 64)
    print("COMPARISON: ground-trained (COCO) vs aerial-trained (WAID)")
    print("=" * 64)

    coco_total = df["coco_found"].sum()
    waid_total = df["waid_found"].sum()

    print(f"\nAnimals detected across {len(df)} images:")
    print(f"  COCO model : {coco_total}")
    print(f"  WAID model : {waid_total}")

    if using_waid_test:
        actual = df["actual_animals"].sum()
        print(f"  Actually present : {actual}")
        print(f"\nDetection rate (found / actually there):")
        print(f"  COCO : {100*coco_total/actual:5.1f}%")
        print(f"  WAID : {100*waid_total/actual:5.1f}%")

    # How often did each model find NOTHING at all?
    coco_blank = (df["coco_found"] == 0).sum()
    waid_blank = (df["waid_found"] == 0).sum()
    print(f"\nImages where the model found nothing:")
    print(f"  COCO : {coco_blank} / {len(df)}  ({100*coco_blank/len(df):.1f}%)")
    print(f"  WAID : {waid_blank} / {len(df)}  ({100*waid_blank/len(df):.1f}%)")

    print(f"\n[done] Full table -> {OUT_CSV}")

    print("\n" + "-" * 64)
    print("WHAT THIS MEANS:")
    print("  Both models know sheep, cow/cattle and zebra. The ONLY difference")
    print("  is that COCO learned them from ground-level photos and WAID")
    print("  learned them from the air.")
    print("  If COCO does much worse on these aerial images, that is direct")
    print("  evidence that aerial-specific training data matters -- and that")
    print("  reusing general-purpose models for drone work is risky.")
    print("  That finding applies to far more people than a WAID-only audit,")
    print("  because COCO-pretrained models are everywhere.")


if __name__ == "__main__":
    main()
