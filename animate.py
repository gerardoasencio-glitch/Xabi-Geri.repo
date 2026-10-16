import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from THE_Tracker import read_csv, convert


# ---------------------------------------
# LOAD SIMULATION DATA
# ---------------------------------------

if len(sys.argv) < 2:
    print("ERROR: No CSV file provided!")
    sys.exit(1)

csv_file = Path(sys.argv[1])

if not csv_file.is_file():
    print("ERROR: CSV file not found!")
    sys.exit(1)

print("Loading animation:", csv_file.name)


# ---------------------------------------
# READ CSV DATA
# ---------------------------------------

header, rows = read_csv(csv_file, {})

data, _ = convert(header, rows, {})

if "altitude" not in data:
    print("ERROR: Altitude data not found!")
    sys.exit(1)

# Remove measurements with missing altitude
samples = [
    (i, t, h)
    for i, (t, h) in enumerate(
        zip(data["time"], data["altitude"])
    )
    if h is not None
]

if len(samples) < 2:
    print("ERROR: Not enough altitude data!")
    sys.exit(1)

indices, time, altitude = zip(*samples)


# ---------------------------------------
# CREATE ANIMATION WINDOW
# ---------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

fig.suptitle("Rocket Flight Simulation", fontsize=16)


# ---------------------------------------
# ROCKET FLIGHT DISPLAY
# ---------------------------------------

ax1.set_title("Rocket Flight")

ax1.set_xlim(-1, 1)

minimum = min(0, min(altitude))
maximum = max(0, max(altitude))
padding = max(1, (maximum - minimum) * 0.1)

ax1.set_ylim(minimum - padding, maximum + padding)

ax1.set_ylabel("Altitude (ft)")
ax1.set_xticks([])

# Ground level
ax1.axhline(0, color="gray", linewidth=2)

# Rocket represented by a triangle
rocket, = ax1.plot([], [], "r^", markersize=18)

# Flight information
info = ax1.text(
    0.05,
    0.95,
    "",
    transform=ax1.transAxes,
    verticalalignment="top"
)


# ---------------------------------------
# ALTITUDE GRAPH
# ---------------------------------------

ax2.set_title("Altitude vs Time")

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Altitude (ft)")

ax2.set_xlim(time[0], time[-1])
ax2.set_ylim(minimum - padding, maximum + padding)

ax2.grid(True)

altitude_line, = ax2.plot([], [], "b-", linewidth=2)

current_point, = ax2.plot([], [], "ro")


# ---------------------------------------
# UPDATE ANIMATION
# ---------------------------------------

def update(frame):

    current_time = time[frame]
    current_altitude = altitude[frame]

    # Move rocket
    rocket.set_data([0], [current_altitude])

    # Update altitude graph
    altitude_line.set_data(
        time[:frame + 1],
        altitude[:frame + 1]
    )

    current_point.set_data(
        [current_time],
        [current_altitude]
    )

    # Display live measurements
    text = (
        f"Time: {current_time:.2f} s\n"
        f"Altitude: {current_altitude:.1f} ft"
    )

    # Include velocity if available
    velocity = data.get("velocity")

    if velocity is not None:
        current_velocity = velocity[indices[frame]]

        if current_velocity is not None:
            text += f"\nVelocity: {current_velocity:.1f} m/s"

    info.set_text(text)

    return rocket, altitude_line, current_point, info


# ---------------------------------------
# START ANIMATION
# ---------------------------------------

# Limit the number of displayed frames
step = max(1, len(time) // 500)

frames = list(range(0, len(time), step))

if frames[-1] != len(time) - 1:
    frames.append(len(time) - 1)

animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=40,
    repeat=False
)

plt.tight_layout()

plt.show()