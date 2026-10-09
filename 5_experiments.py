"""
5_experiments.py
-----------------
THIS IS THE RESEARCH. Everything before it was setup.

Two experiments, both run on WAID's own test images so you have exact ground
truth to score against:

  ALTITUDE PROXY  -- shrink images, then blow them back up. The animal now
                     occupies fewer real pixels, exactly like flying higher.
                     We measure how much accuracy is lost at each level.

  OCCLUSION PROXY -- paint over a growing slice of each animal, simulating
                     tree canopy or brush. We measure the same thing.

Output: results/experiments.csv  (feed to 6_plot_curves.py)

Why proxies instead of real flights? Because here you know the exact right
answer for every image, so you can measure the drop precisely. Your real drone
footage (script 4) then gets compared against these curves -- that comparison
IS your sim-to-real gap.
"""

import shutil
from pathlib import Path

import cv2
import yaml
import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
WEIGHTS = PROJECT_DIR / "runs" / "waid_baseline" / "weights" / "best.pt"
DATA_YAML = PROJECT_DIR / "waid_local.yaml"
WORK_DIR = PROJECT_DIR / "degraded"        # temp folder for modified images
RESULTS_CSV = PROJECT_DIR / "results" / "experiments.csv"

# How much to shrink by. 1 = original. 8 = eight times smaller (very high up).
SHRINK_FACTORS = [1, 2, 3, 4, 6, 8]

# What fraction of each animal to cover up. 0.0 = clear view, 0.8 = mostly hidden.
OCCLUSION_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]

MAX_IMAGES = 500   # Cap per condition to keep runtime sane. Raise for the
                   # final paper run; 500 is plenty while you're testing.


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_eval_images(config):
    """Use the test split if WAID has one, otherwise fall back to val."""
    for key in ("test", "val"):
        if key in config:
            folder = Path(config[key])
            if folder.is_dir():
                return folder
    raise SystemExit("No test or val folder found in waid_local.yaml")


def labels_dir_for(images_dir):
    """
    YOLO expects labels in a folder called 'labels' that mirrors 'images'.
    So .../images/test  ->  .../labels/test
    """
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts)


def read_label_file(path):
    """
    Read one label file into a list of animals.

    Each line is:  class_id  x_center  y_center  width  height
    All the position numbers are fractions of the image size (0 to 1), which
    is handy: they stay correct even when we resize the image.
    """
    animals = []
    if not path.exists():
        return animals
    with open(path) as f:
        for line in f:
            bits = line.split()
            if len(bits) >= 5:
                animals.append({
                    "class_id": int(bits[0]),
                    "xc": float(bits[1]), "yc": float(bits[2]),
                    "w": float(bits[3]),  "h": float(bits[4]),
                })
    return animals


def build_dataset(name, images_dir, labels_src, transform, config):
    """
    Create a modified copy of the test set and a .yaml pointing at it.

    'transform' is a function we pass in that changes each image -- either the
    shrink one or the occlusion one. The LABELS are copied unchanged, which is
    deliberate: the animal is still there and still in the same place. If the
    model stops finding it, that's a genuine miss. That's what we're measuring.
    """
    out_root = WORK_DIR / name
    out_images = out_root / "images"
    out_labels = out_root / "labels"

    # Start clean each time so old runs don't contaminate new ones
    if out_root.exists():
        shutil.rmtree(out_root)
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:MAX_IMAGES]

    bbox_widths_px = []   # track how big animals actually are, in pixels

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        label_path = labels_src / (img_path.stem + ".txt")
        animals = read_label_file(label_path)

        modified, widths = transform(image, animals)
        bbox_widths_px.extend(widths)

        cv2.imwrite(str(out_images / img_path.name), modified)
        if label_path.exists():
            shutil.copy(label_path, out_labels / label_path.name)

    # Write a config file pointing the validator at this modified copy
    cfg = {
        "path": str(out_root.resolve()),
        "train": str(out_images.resolve()),   # unused, but YOLO wants the key
        "val": str(out_images.resolve()),
        "names": config["names"],
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    median_width = sorted(bbox_widths_px)[len(bbox_widths_px) // 2] if bbox_widths_px else 0
    return yaml_path, len(image_files), median_width


# --- the two transformations ------------------------------------------------
def make_shrink_transform(factor):
    """
    Shrink the image by 'factor', then stretch it back to its original size.

    Detail that was lost in the shrink cannot come back -- so the animal now
    has genuinely less information in it, just like being photographed from
    higher up. We resize back so the model always receives the same input
    dimensions, keeping the comparison fair.
    """
    def transform(image, animals):
        h, w = image.shape[:2]

        # Record the animal's real detail-level in pixels at this altitude
        widths = [a["w"] * w / factor for a in animals]

        if factor == 1:
            return image, widths   # unchanged baseline

        # INTER_AREA is the correct choice for shrinking (averages pixels)
        small = cv2.resize(image, (max(1, w // factor), max(1, h // factor)),
                           interpolation=cv2.INTER_AREA)
        # INTER_LINEAR when blowing back up -- this is where detail is lost
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return restored, widths

    return transform


def make_occlusion_transform(fraction):
    """
    Cover the top 'fraction' of each animal's box with a neutral grey block.

    Covering from the TOP is a deliberate choice -- it mimics an animal walking
    under branches or a bush, seen from a drone above. Grey is used rather than
    black so we're testing occlusion, not introducing a stark artificial edge
    the model might key on.
    """
    def transform(image, animals):
        h, w = image.shape[:2]
        widths = [a["w"] * w for a in animals]

        if fraction <= 0:
            return image, widths

        out = image.copy()
        for a in animals:
            # Convert from fractions-of-image back to actual pixel positions
            box_w = a["w"] * w
            box_h = a["h"] * h
            x1 = int((a["xc"] * w) - box_w / 2)
            y1 = int((a["yc"] * h) - box_h / 2)

            # How tall the covering block should be
            cover_h = int(box_h * fraction)
            x2 = int(x1 + box_w)
            y2 = int(y1 + cover_h)

            # Clamp to the image edges so we don't try to draw off-canvas
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                out[y1:y2, x1:x2] = 128   # 128 = mid-grey
        return out, widths

    return transform


# ---------------------------------------------------------------------------
def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"No trained model at {WEIGHTS}\nRun: python 3_train.py")

    config = read_yaml(DATA_YAML)
    images_dir = get_eval_images(config)
    labels_src = labels_dir_for(images_dir)

    print(f"Evaluating on : {images_dir}")
    print(f"Labels from   : {labels_src}")
    print(f"Images per run: up to {MAX_IMAGES}\n")

    model = YOLO(str(WEIGHTS))
    rows = []

    # --- EXPERIMENT 1: ALTITUDE -------------------------------------------
    print("=" * 62)
    print("EXPERIMENT 1 -- ALTITUDE (image detail reduced)")
    print("=" * 62)

    for factor in SHRINK_FACTORS:
        name = f"altitude_x{factor}"
        print(f"\n  Building {name} ...", end=" ", flush=True)
        yaml_path, n_imgs, median_px = build_dataset(
            name, images_dir, labels_src, make_shrink_transform(factor), config)
        print(f"{n_imgs} images, median animal width {median_px:.1f}px")

        metrics = model.val(data=str(yaml_path), verbose=False, plots=False)

        row = {
            "experiment": "altitude",
            "level": factor,
            "level_label": f"{factor}x smaller",
            "median_animal_width_px": round(median_px, 2),
            "mAP50": round(metrics.box.map50, 4),
            "mAP50_95": round(metrics.box.map, 4),
            "precision": round(metrics.box.mp, 4),
            "recall": round(metrics.box.mr, 4),
        }
        rows.append(row)
        print(f"     mAP50={row['mAP50']:.3f}  recall={row['recall']:.3f}")

    # --- EXPERIMENT 2: OCCLUSION ------------------------------------------
    print("\n" + "=" * 62)
    print("EXPERIMENT 2 -- OCCLUSION (animals partly covered)")
    print("=" * 62)

    for fraction in OCCLUSION_LEVELS:
        pct = int(fraction * 100)
        name = f"occlusion_{pct}"
        print(f"\n  Building {name} ...", end=" ", flush=True)
        yaml_path, n_imgs, median_px = build_dataset(
            name, images_dir, labels_src, make_occlusion_transform(fraction), config)
        print(f"{n_imgs} images")

        metrics = model.val(data=str(yaml_path), verbose=False, plots=False)

        row = {
            "experiment": "occlusion",
            "level": fraction,
            "level_label": f"{pct}% covered",
            "median_animal_width_px": round(median_px, 2),
            "mAP50": round(metrics.box.map50, 4),
            "mAP50_95": round(metrics.box.map, 4),
            "precision": round(metrics.box.mp, 4),
            "recall": round(metrics.box.mr, 4),
        }
        rows.append(row)
        print(f"     mAP50={row['mAP50']:.3f}  recall={row['recall']:.3f}")

    # --- Save --------------------------------------------------------------
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 62)
    print(df.to_string(index=False))
    print(f"\n[done] Saved -> {RESULTS_CSV}")
    print("\nNext step:  python 6_plot_curves.py")
    print(f"\nTIP: the modified images are in {WORK_DIR} -- open a few and look")
    print("     at them. Include example frames in your paper so readers can")
    print("     see what '60% occluded' actually looks like.")


if __name__ == "__main__":
    main()
