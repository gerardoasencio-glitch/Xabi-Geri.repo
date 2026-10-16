import sys
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.transforms import Affine2D

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

header, rows = read_csv(csv_file, {})
data, _ = convert(header, rows, {})

if "altitude" not in data:
    print("ERROR: Altitude data not found!")
    sys.exit(1)

samples = [
    (i, t, h)
    for i, (t, h) in enumerate(
        zip(data["time"], data["altitude"])
    )
    if t is not None and h is not None
]

if len(samples) < 2:
    print("ERROR: Not enough flight data!")
    sys.exit(1)

indices, time, altitude = zip(*samples)

min_altitude = min(0, min(altitude))
max_altitude = max(altitude)
altitude_range = max(1, max_altitude - min_altitude)

# Optional thrust measurements
thrust_data = data.get("thrust")

max_thrust = max(
    (v for v in (thrust_data or []) if v is not None),
    default=0
)


# ---------------------------------------
# CREATE SIMULATION WINDOW
# ---------------------------------------

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(13, 7),
    gridspec_kw={"width_ratios": [1, 1.5]}
)

fig.suptitle(
    "ROCKET FLIGHT SIMULATION",
    fontsize=17,
    fontweight="bold"
)

# ---------------------------------------
# CREATE SKY AND GROUND
# ---------------------------------------

ax1.set_title("Rocket Flight")

ax1.set_xlim(-9, 9)
ax1.set_ylim(0, 100)

ax1.set_facecolor("#DCEFFF")
ax1.set_xticks([])
ax1.set_yticks([])

ground = Rectangle(
    (-9, 0),
    18,
    13,
    facecolor="#607D52"
)

ax1.add_patch(ground)

ax1.plot(
    [-9, 9],
    [13, 13],
    color="#344B30",
    linewidth=3
)

# Launchpad
ax1.plot(
    [-3, 3],
    [13, 13],
    color="black",
    linewidth=5
)


# ---------------------------------------
# BUILD THE ROCKET MODEL
# ---------------------------------------

rocket_parts = []

# Animated exhaust flame
flame = Polygon(
    [
        (-0.65, -8),
        (0, -14),
        (0.65, -8)
    ],
    closed=True,
    facecolor="#FF7900",
    edgecolor="none"
)

ax1.add_patch(flame)
rocket_parts.append(flame)

# Engine nozzle
engine = Rectangle(
    (-0.55, -8),
    1.1,
    1.5,
    facecolor="#444444",
    edgecolor="black"
)

ax1.add_patch(engine)
rocket_parts.append(engine)

# Left fin
left_fin = Polygon(
    [
        (-0.9, -4),
        (-2.4, -8),
        (-0.9, -7)
    ],
    closed=True,
    facecolor="#C62828",
    edgecolor="black"
)

ax1.add_patch(left_fin)
rocket_parts.append(left_fin)

# Right fin
right_fin = Polygon(
    [
        (0.9, -4),
        (2.4, -8),
        (0.9, -7)
    ],
    closed=True,
    facecolor="#C62828",
    edgecolor="black"
)

ax1.add_patch(right_fin)
rocket_parts.append(right_fin)

# Main rocket body
body = Rectangle(
    (-0.9, -7),
    1.8,
    13,
    facecolor="#F5F5F5",
    edgecolor="#333333",
    linewidth=2
)

ax1.add_patch(body)
rocket_parts.append(body)

# Nose cone
nose = Polygon(
    [
        (-0.9, 6),
        (0, 10),
        (0.9, 6)
    ],
    closed=True,
    facecolor="#C62828",
    edgecolor="black",
    linewidth=2
)

ax1.add_patch(nose)
rocket_parts.append(nose)

# Black body stripe
stripe = Rectangle(
    (-0.9, 3.5),
    1.8,
    1,
    facecolor="#222222"
)

ax1.add_patch(stripe)
rocket_parts.append(stripe)

# Rocket logo
logo = Circle(
    (0, 0),
    radius=0.6,
    facecolor="#1565C0",
    edgecolor="black"
)

ax1.add_patch(logo)
rocket_parts.append(logo)


# ---------------------------------------
# CREATE ALTITUDE GRAPH
# ---------------------------------------

ax2.set_title("Altitude vs Time")

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Altitude (ft)")

ax2.set_xlim(time[0], time[-1])

padding = max(1, altitude_range * 0.1)

ax2.set_ylim(
    min_altitude - padding,
    max_altitude + padding
)

ax2.grid(True, alpha=0.3)

# Complete trajectory in the background
ax2.plot(
    time,
    altitude,
    color="lightgray",
    linewidth=2
)

# Animated trajectory
altitude_line, = ax2.plot(
    [],
    [],
    color="red",
    linewidth=2
)

current_point, = ax2.plot(
    [],
    [],
    "ro",
    markersize=8
)

# Apogee reference
ax2.axhline(
    max_altitude,
    color="orange",
    linestyle="--",
    label=f"Apogee: {max_altitude:.0f} ft"
)

ax2.legend()


# ---------------------------------------
# LIVE FLIGHT INFORMATION
# ---------------------------------------

info = ax1.text(
    0.05,
    0.96,
    "",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontweight="bold"
)

ax1.text(
    0.03,
    0.03,
    "Rocket drawing not to scale",
    transform=ax1.transAxes,
    fontsize=8
)


# ---------------------------------------
# ANIMATION FUNCTION
# ---------------------------------------

def update(frame):

    current_time = time[frame]
    current_altitude = altitude[frame]

    # Convert altitude to display position
    fraction = (
        (current_altitude - min_altitude)
        / altitude_range
    )

    rocket_y = 22 + fraction * 55

    # Move all rocket components together
    transform = (
        Affine2D().translate(0, rocket_y)
        + ax1.transData
    )

    for part in rocket_parts:
        part.set_transform(transform)

    # -----------------------------------
    # ANIMATE ENGINE FLAME
    # -----------------------------------

    if thrust_data is not None and max_thrust > 0:

        current_thrust = thrust_data[indices[frame]]

        engine_on = (
            current_thrust is not None
            and current_thrust > 0.05 * max_thrust
        )

    else:
        # Illustrative early-launch flame
        flight_fraction = (
            (current_time - time[0])
            / (time[-1] - time[0])
        )

        engine_on = flight_fraction < 0.12

    flame.set_visible(engine_on)

    if engine_on:

        flicker = math.sin(frame * 1.5)

        flame_length = 5 + flicker

        flame.set_xy([
            (-0.65, -8),
            (0, -8 - flame_length),
            (0.65, -8)
        ])

    # -----------------------------------
    # UPDATE ALTITUDE GRAPH
    # -----------------------------------

    altitude_line.set_data(
        time[:frame + 1],
        altitude[:frame + 1]
    )

    current_point.set_data(
        [current_time],
        [current_altitude]
    )

    # -----------------------------------
    # UPDATE LIVE INFORMATION
    # -----------------------------------

    text = (
        f"TIME: {current_time:.2f} s\n"
        f"ALTITUDE: {current_altitude:.0f} ft"
    )

    velocity = data.get("velocity")

    if velocity is not None:

        current_velocity = velocity[indices[frame]]

        if current_velocity is not None:
            text += (
                f"\nVELOCITY: "
                f"{current_velocity:.1f} m/s"
            )

    acceleration = data.get("acceleration")

    if acceleration is not None:

        current_acceleration = acceleration[indices[frame]]

        if current_acceleration is not None:
            text += (
                f"\nACCELERATION: "
                f"{current_acceleration:.1f} m/s²"
            )

    info.set_text(text)

    return rocket_parts + [
        altitude_line,
        current_point,
        info
    ]


# ---------------------------------------
# START ROCKET ANIMATION
# ---------------------------------------

step = max(1, len(time) // 500)

frames = list(range(0, len(time), step))

if frames[-1] != len(time) - 1:
    frames.append(len(time) - 1)

animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=40,
    repeat=False,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()