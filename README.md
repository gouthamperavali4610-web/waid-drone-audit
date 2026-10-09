# WAID Drone Audit — Model Pipeline

Complete code for the vision side of the project: train a detector on WAID,
identify animals in your own photos, run the altitude + occlusion degradation
experiments, and plot the envelope curves.

Built for: Windows/Linux laptop with an **RTX 4070**. Everything runs locally.

---

## One-time setup (do this first)

Open a terminal (Windows: PowerShell or Command Prompt) in this folder.

### 1. Make a virtual environment
Keeps this project's packages separate from the rest of your computer.

```bash
python -m venv venv
```

Activate it (you must do this EVERY time you open a new terminal):

```bash
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

You'll know it worked when your prompt starts with `(venv)`.

### 2. Install PyTorch WITH GPU SUPPORT (important — do this BEFORE ultralytics)

This is the #1 thing beginners get wrong. Installing ultralytics first can pull
a CPU-only PyTorch, which ignores your 4070 and trains ~50x slower.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

(`cu124` = CUDA 12.4. If that errors, try `cu121`.)

### 3. Install everything else

```bash
pip install ultralytics opencv-python matplotlib pandas pyyaml
```

### 4. Verify your GPU is actually being used

```bash
python 1_setup_check.py
```

It must print `CUDA available: True` and name your RTX 4070. If it says False,
re-do step 2 — don't continue, or you'll wait hours instead of minutes.

---

## Run order

| Script | What it does | Roughly how long |
|---|---|---|
| `1_setup_check.py` | Confirms GPU + packages are working | seconds |
| `2_prepare_waid.py` | Downloads/verifies WAID, fixes the data.yaml paths, prints class counts | 5–15 min (download) |
| `3_train.py` | Fine-tunes YOLOv8n on WAID → produces `best.pt` | 20–40 min on a 4070 |
| `4_identify.py` | **Point it at your photos → it labels the animals** | seconds |
| `5_experiments.py` | Altitude + occlusion degradation → writes `results/experiments.csv` | 10–30 min |
| `6_plot_curves.py` | Turns the CSV into the envelope curve figures | seconds |

You only run 2 and 3 **once**. After that, `4_identify.py` is the one you'll
use over and over.

---

## The short version

```bash
venv\Scripts\activate          # every new terminal
python 2_prepare_waid.py       # once
python 3_train.py              # once (~30 min)
python 4_identify.py my_photos # as often as you like
python 5_experiments.py        # once you want the research results
python 6_plot_curves.py
```
