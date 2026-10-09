"""
2_prepare_waid.py
------------------
Gets the WAID dataset ready to train on. Run this ONCE.

Three jobs:
  1. Download WAID (if you don't already have it).
  2. Fix the data.yaml file -- dataset configs usually contain paths from
     whoever exported them, which don't exist on your computer. We rewrite
     them to point at YOUR folders.
  3. Count how many images and animals are in each class, and save that table.
     Do NOT skip this -- class balance is a real finding you should report in
     your paper, and most student projects never bother to check it.
"""

import os
import subprocess
import shutil
from pathlib import Path
from collections import Counter

import yaml

# ---------------------------------------------------------------------------
# CONFIG -- change these if your setup differs
# ---------------------------------------------------------------------------
WAID_REPO = "https://github.com/xiaohuicui/WAID.git"
PROJECT_DIR = Path(__file__).parent          # the folder this script lives in
CLONE_DIR = PROJECT_DIR / "WAID"             # where the dataset gets downloaded
OUTPUT_YAML = PROJECT_DIR / "waid_local.yaml"  # the fixed config we generate
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1 -- download the dataset
# ---------------------------------------------------------------------------
def download_waid():
    """Clone the WAID repo from GitHub if we don't already have it."""
    if CLONE_DIR.exists():
        print(f"[skip] WAID already exists at {CLONE_DIR}")
        return

    print(f"Downloading WAID from {WAID_REPO} ...")
    print("(This is ~1.5 GB and may take 5-15 minutes.)")
    # subprocess.run just runs a terminal command from inside Python.
    # This is the same as typing `git clone <url>` yourself.
    subprocess.run(["git", "clone", WAID_REPO, str(CLONE_DIR)], check=True)
    print("[done] Download complete.")


# ---------------------------------------------------------------------------
# STEP 2 -- find the image folders
# ---------------------------------------------------------------------------
def find_split_dirs():
    """
    Locate the train / validation / test image folders.

    WAID was exported from Roboflow, so the layout is normally:
        WAID/WAID/images/train,  WAID/WAID/labels/train
    ...but we search for it rather than assume, because repo layouts change.
    """
    # rglob("*") searches every subfolder recursively.
    candidates = [p for p in CLONE_DIR.rglob("images") if p.is_dir()]
    if not candidates:
        raise SystemExit(
            "Could not find an 'images' folder inside WAID.\n"
            f"Look inside {CLONE_DIR} yourself and set the paths manually."
        )

    images_root = candidates[0]
    labels_root = images_root.parent / "labels"

    print(f"\nFound images at : {images_root}")
    print(f"Found labels at : {labels_root}")

    splits = {}
    # WAID may name the validation folder 'valid' or 'val' -- accept either.
    for split, aliases in [("train", ["train"]),
                           ("val", ["valid", "val"]),
                           ("test", ["test"])]:
        for alias in aliases:
            candidate = images_root / alias
            if candidate.is_dir():
                splits[split] = candidate
                break

    if "train" not in splits or "val" not in splits:
        raise SystemExit(f"Missing train or val folder. Found: {list(splits)}")

    return splits, labels_root


# ---------------------------------------------------------------------------
# STEP 3 -- find the class names
# ---------------------------------------------------------------------------
def find_class_names():
    """
    Read the class names out of whatever .yaml file shipped with the dataset.
    Falls back to the six classes named in the WAID paper if none is found.
    """
    for yaml_file in CLONE_DIR.rglob("*.yaml"):
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "names" in data:
                names = data["names"]
                # 'names' can be a list ['sheep','cattle'] or a dict {0:'sheep'}
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names)]
                print(f"\nClass names read from: {yaml_file.name}")
                return list(names)
        except Exception:
            continue  # not a valid yaml, keep looking

    print("\n[warn] No class list found in the repo -- using the paper's six.")
    return ["sheep", "cattle", "seal", "camel", "kiang", "zebra"]


# ---------------------------------------------------------------------------
# STEP 4 -- write a config file with paths that work on YOUR machine
# ---------------------------------------------------------------------------
def write_yaml(splits, class_names):
    """
    YOLO needs a .yaml file telling it where the data is and what the classes
    are. We write absolute paths so it works no matter where you run from.
    """
    config = {
        "path": str(CLONE_DIR.resolve()),
        "train": str(splits["train"].resolve()),
        "val": str(splits["val"].resolve()),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    if "test" in splits:
        config["test"] = str(splits["test"].resolve())

    with open(OUTPUT_YAML, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"\n[done] Wrote config -> {OUTPUT_YAML}")
    return config


# ---------------------------------------------------------------------------
# STEP 5 -- count images and animals per class
# ---------------------------------------------------------------------------
def count_dataset(splits, labels_root, class_names):
    """
    Walk the label files and tally up how many animals of each class exist.

    Each label file is one image. Each LINE inside it is one animal, and the
    first number on the line is the class id. So counting lines by class id
    gives us the number of animal instances -- which matters more than the
    number of images.
    """
    print("\n" + "=" * 62)
    print("DATASET COMPOSITION  (report this in your paper)")
    print("=" * 62)

    rows = []
    grand_total = Counter()

    for split, images_dir in splits.items():
        labels_dir = labels_root / images_dir.name
        if not labels_dir.is_dir():
            print(f"[warn] no labels folder for '{split}', skipping")
            continue

        # Count image files
        n_images = sum(1 for p in images_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

        # Count animal instances by class
        class_counts = Counter()
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue          # skip blank lines
                    class_id = int(line.split()[0])   # first number = class id
                    class_counts[class_id] += 1

        grand_total.update(class_counts)
        total_instances = sum(class_counts.values())

        print(f"\n{split.upper()}: {n_images} images, {total_instances} animals")
        for class_id, name in enumerate(class_names):
            count = class_counts.get(class_id, 0)
            pct = (100 * count / total_instances) if total_instances else 0
            # A simple text bar so imbalance is visually obvious
            bar = "#" * int(pct / 2)
            print(f"   {name:<10} {count:>7}  ({pct:5.1f}%) {bar}")
            rows.append({"split": split, "class": name,
                         "instances": count, "percent": round(pct, 2)})

    # --- The interpretation you should actually write down -----------------
    print("\n" + "-" * 62)
    total = sum(grand_total.values())
    if total:
        biggest = max(grand_total.values())
        smallest = min(grand_total.values()) or 1
        ratio = biggest / smallest
        print(f"Total animal instances: {total}")
        print(f"Imbalance ratio (most common : least common) = {ratio:.1f} : 1")
        print("\nWhat this means for your audit:")
        print("  Classes with few instances are where WAID's coverage is thin.")
        print("  If your drone footage does badly on a rare class, that is a")
        print("  DATA coverage gap, not a model failure -- which is exactly the")
        print("  distinction your research question is about. Say so explicitly.")

    # Save the table so you can paste it into the paper
    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "dataset_composition.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[done] Saved table -> {out_csv}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Check git is installed before trying to use it
    if shutil.which("git") is None and not CLONE_DIR.exists():
        raise SystemExit(
            "git is not installed. Install it from https://git-scm.com/downloads\n"
            "or download the WAID zip manually from github.com/xiaohuicui/WAID"
        )

    download_waid()
    splits, labels_root = find_split_dirs()
    class_names = find_class_names()
    print(f"Classes ({len(class_names)}): {class_names}")
    write_yaml(splits, class_names)
    count_dataset(splits, labels_root, class_names)

    print("\nNext step:  python 3_train.py")
