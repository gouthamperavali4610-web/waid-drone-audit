"""
4_identify.py
--------------
THE MAIN ONE. Point it at your photos, it tells you what animals are in them.

Usage:
    python 4_identify.py my_photos              (a whole folder)
    python 4_identify.py my_photos/sheep1.jpg   (one image)
    python 4_identify.py flight.mp4             (a video -- samples frames)

Outputs:
    results/identified/        annotated copies of your images (boxes drawn on)
    results/detections.csv     one row per animal found, for your analysis
"""

import sys
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
WEIGHTS = PROJECT_DIR / "runs" / "waid_baseline" / "weights" / "best.pt"

CONFIDENCE = 0.25   # Ignore guesses the model is less than 25% sure about.
                    # Lower  = catches more animals, more false alarms.
                    # Higher = fewer false alarms, misses more animals.
                    # 0.25 is the standard default. Worth testing 0.1 and 0.5
                    # to see how it shifts your precision/recall trade-off.

VIDEO_EVERY_N_FRAMES = 30   # For videos: analyse 1 frame in every 30
                            # (~1 per second on 30fps footage). Adjacent frames
                            # are nearly identical, so analysing all of them
                            # wastes time and inflates your counts.

OUT_DIR = PROJECT_DIR / "results" / "identified"
CSV_PATH = PROJECT_DIR / "results" / "detections.csv"

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv"}


def extract_video_frames(video_path, out_folder):
    """
    Pull still frames out of a video so we can run detection on them.

    Your action cam records video, but the detector works on images -- so we
    grab one frame every N frames and save them as .jpg files.
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))   # open the video file

    frame_index = 0
    saved = 0
    while True:
        ok, frame = cap.read()      # read the next frame
        if not ok:
            break                   # no more frames -- we're done
        if frame_index % VIDEO_EVERY_N_FRAMES == 0:
            out_path = out_folder / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_index += 1

    cap.release()
    print(f"Extracted {saved} frames from {video_path.name}")
    return out_folder


def main():
    # --- Read the folder/file the user typed after the script name --------
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit("Tell me what to look at, e.g.:  python 4_identify.py my_photos")

    target = Path(sys.argv[1])
    if not target.exists():
        raise SystemExit(f"Can't find: {target}")

    # --- Load your trained model -----------------------------------------
    if not WEIGHTS.exists():
        raise SystemExit(f"No trained model at {WEIGHTS}\nRun: python 3_train.py")

    print(f"Loading model: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    print(f"Can identify: {list(model.names.values())}\n")

    # --- Work out what we're looking at ----------------------------------
    if target.is_file() and target.suffix.lower() in VIDEO_TYPES:
        # It's a video -- turn it into frames first
        source = extract_video_frames(target, PROJECT_DIR / "results" / "frames" / target.stem)
    else:
        source = target   # a single image, or a folder of them

    # --- Run the detector -------------------------------------------------
    # save=True writes copies of your images with boxes drawn on them.
    results = model.predict(
        source=str(source),
        conf=CONFIDENCE,
        save=True,
        project=str(OUT_DIR.parent),
        name=OUT_DIR.name,
        exist_ok=True,
        verbose=False,      # we print our own cleaner summary below
    )

    # --- Report what was found -------------------------------------------
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    species_tally = {}

    print("=" * 62)
    print("RESULTS")
    print("=" * 62)

    for result in results:
        image_name = Path(result.path).name
        boxes = result.boxes   # all animals found in this one image

        if boxes is None or len(boxes) == 0:
            print(f"\n{image_name}: nothing found")
            continue

        print(f"\n{image_name}: {len(boxes)} animal(s)")

        for box in boxes:
            # box.cls is the class number; model.names turns it into a word
            species = model.names[int(box.cls)]
            # box.conf is how sure the model is, from 0 to 1
            confidence = float(box.conf)
            # box.xyxy gives the corners of the rectangle: left, top, right, bottom
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            width_px = x2 - x1
            height_px = y2 - y1
            area_px = width_px * height_px

            print(f"   {species:<10} {confidence*100:5.1f}% sure   "
                  f"size on screen: {int(width_px)}x{int(height_px)} px")

            species_tally[species] = species_tally.get(species, 0) + 1

            rows.append({
                "image": image_name,
                "species": species,
                "confidence": round(confidence, 4),
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
                "width_px": round(width_px, 1),
                "height_px": round(height_px, 1),
                "area_px": round(area_px, 1),
                # You fill this in yourself from your Pixhawk flight log --
                # it's what turns these detections into an altitude curve.
                "altitude_m": "",
            })

    # --- Summary ----------------------------------------------------------
    print("\n" + "=" * 62)
    if species_tally:
        print("TOTAL BY SPECIES:")
        for species, count in sorted(species_tally.items(), key=lambda kv: -kv[1]):
            print(f"   {species:<10} {count}")
    else:
        print("No animals detected in any image.")
        print("\nIf you expected some, try:")
        print("  - Lowering CONFIDENCE at the top of this file (try 0.1)")
        print("  - Checking the animal is one of the six WAID classes")
        print("  - Checking the photo is taken from ABOVE (WAID is all aerial --")
        print("    a ground-level photo of a sheep may well fail, and that")
        print("    itself is a coverage-gap finding worth writing down)")

    # --- Save the CSV -----------------------------------------------------
    if rows:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[done] Data table    -> {CSV_PATH}")

    print(f"[done] Marked images -> {OUT_DIR}")
    print("\nTIP: fill in the altitude_m column from your Pixhawk log, and this")
    print("     CSV becomes the raw data for your altitude envelope curve.")


if __name__ == "__main__":
    main()
