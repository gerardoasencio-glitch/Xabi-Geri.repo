import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

# ---------------------------------------
# PROJECT FOLDER
# ---------------------------------------

folder = Path(__file__).resolve().parent


# ---------------------------------------
# PYTHON FILES
# ---------------------------------------

tracker = folder / "THE_Tracker.py"
animation = folder / "animate.py"


# ---------------------------------------
# FIND CSV FILE
# ---------------------------------------

csv_files = list((folder / "simulations").glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV files found!")
    sys.exit(1)

# Select the most recently modified CSV
csv_file = max(
    csv_files,
    key=lambda f: f.stat().st_mtime
)

print("Using simulation:", csv_file.name, flush=True)


# ---------------------------------------
# CHECK REQUIRED FILES
# ---------------------------------------

if not tracker.exists():
    print("ERROR: THE_Tracker.py not found!")
    sys.exit(1)

if not animation.exists():
    print("ERROR: animate.py not found!")
    sys.exit(1)


# ---------------------------------------
# START EXCEL TRACKER
# ---------------------------------------

print("Starting Excel Tracker...", flush=True)

tracker_process = subprocess.Popen(
    [
        sys.executable,
        str(tracker),
        str(csv_file)
    ],
    cwd=folder
)


# ---------------------------------------
# START ROCKET ANIMATION
# ---------------------------------------

# Ask whether to open the animation.
root = tk.Tk()
root.withdraw()

try:
    show_animation = messagebox.askyesno(
        "Flight animation",
        f"Would you like to see the flight animation?\n\n{csv_file.name}",
        parent=root,
    )
finally:
    root.destroy()

animation_process = None

if show_animation:
    print("Starting Rocket Animation...", flush=True)

    animation_process = subprocess.Popen(
        [
            sys.executable,
            str(animation),
            str(csv_file),
        ],
        cwd=folder,
    )

# Wait for the programs that were started.
processes = [tracker_process]

if animation_process is not None:
    processes.append(animation_process)

try:
    tracker_result = tracker_process.wait()

    animation_result = (
        animation_process.wait()
        if animation_process is not None
        else None
    )

except KeyboardInterrupt:
    print("\nStopping programs...")

    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        process.wait()

    sys.exit(130)

if tracker_result == 0:
    print("Excel Tracker finished successfully!")
else:
    print("Excel Tracker encountered an error.")

if animation_result is not None:
    if animation_result == 0:
        print("Rocket Animation finished successfully!")
    else:
        print("Rocket Animation encountered an error.")
else:
    print("Animation skipped.")

print("All programs finished!")