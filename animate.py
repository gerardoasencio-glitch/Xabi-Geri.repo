"""Loxodon flight replay. Keep this file next to THE_Tracker.py.

Run from your project folder:
    /usr/bin/python3 animate.py "15mph_main1,000ft.csv"

Your existing main.py can also launch this file with a CSV argument.
This viewer reads data only; it never runs the tracker or writes Excel files.
Recovery events and orientation are illustrative estimates, not exported events.
"""

import argparse
from itertools import count
import json
import math
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Polygon, Rectangle, FancyBboxPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button


# --------------------- SETTINGS ---------------------
ROCKET_NAME = "Loxodon"
DROGUE_DELAY_SECONDS = 0.5
MAIN_DEPLOY_ALTITUDE_FT = 1000.0  # Relative to the first recorded altitude.
CHUTE_OPENING_SECONDS = 1.2
FRAME_INTERVAL_MS = 40
MAX_DISPLAY_FRAMES = 500        # Sample-based playback, NOT real-time playback.
ROCKET_SCALE = 1.5

BACKGROUND = "#080F1F"
PANEL = "#111E33"
SKY = "#0B1930"
WHITE = "#F4F8FF"
BLUE = "#347FF4"
DARK_BLUE = "#153F83"
LIGHT_BLUE = "#A8CEFF"
CYAN = "#56D9FF"
ORANGE = "#FFAE42"
MUTED = "#94A8C5"
BORDER = "#304766"


class FlightReplay:
    """Draw a flight and keep one reusable animation timer for every replay."""

    def __init__(self, data: dict, source_name: str = "Simulation") -> None:
        if "time" not in data or "altitude" not in data:
            raise ValueError("The CSV needs recognized time and altitude columns.")

        samples = [
            (i, t, h)
            for i, (t, h) in enumerate(zip(data["time"], data["altitude"]))
            if t is not None and h is not None
            and math.isfinite(t) and math.isfinite(h)
        ]
        if len(samples) < 2:
            raise ValueError("At least two valid time/altitude samples are required.")

        self.indices, self.times, self.heights = map(list, zip(*samples))
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("The recorded times must be strictly increasing.")

        self.data = data
        self.minimum = min(self.heights)
        self.maximum = max(self.heights)
        self.height_range = max(1.0, self.maximum - self.minimum)
        self.apogee_index = self.heights.index(self.maximum)
        self.apogee_time = self.times[self.apogee_index]

        # These are the same estimated recovery conditions as the earlier viewer.
        # They are not a reading of OpenRocket's actual deployment configuration.
        self.drogue_index = next((
            i for i in range(self.apogee_index + 1, len(self.times))
            if self.times[i] >= self.apogee_time + DROGUE_DELAY_SECONDS
            and self.heights[i] < self.maximum - 1.0
        ), None)
        self.main_index = None
        if self.drogue_index is not None:
            self.main_index = next((
                i for i in range(self.drogue_index, len(self.times))
                if self.heights[i] - self.heights[0] <= MAIN_DEPLOY_ALTITUDE_FT
            ), None)

        thrust = data.get("thrust", [])
        self.maximum_thrust = max((v for v in thrust if v is not None), default=0.0)

        # Always include the event frames, even when a large CSV is thinned out.
        step = max(1, math.ceil(len(self.times) / MAX_DISPLAY_FRAMES))
        frames = set(range(0, len(self.times), step))
        frames.update([self.apogee_index, len(self.times) - 1])
        for event in (self.drogue_index, self.main_index):
            if event is not None:
                frames.add(event)
        self.frames = sorted(frames)
        self.cursor = 0
        self.paused = False
        self.finished = False
        self.closed = False
        self.animation = None

        self.fig = plt.figure(figsize=(14, 8.5), facecolor=BACKGROUND)
        self.fig.canvas.manager.set_window_title("Loxodon Flight Replay")
        self.fig.text(0.06, 0.95, "LOXODON / FLIGHT REPLAY", color=WHITE,
                      fontsize=22, fontweight="bold")
        display_name = source_name if len(source_name) <= 65 else source_name[:62] + "..."
        self.fig.text(0.06, 0.915, display_name, color=MUTED, fontsize=10)

        grid = self.fig.add_gridspec(
            2, 2, width_ratios=[1, 1.25], height_ratios=[3, 1.5],
            left=0.06, right=0.96, top=0.86, bottom=0.18,
            wspace=0.22, hspace=0.42,
        )
        self.sky_ax = self.fig.add_subplot(grid[:, 0])
        self.graph_ax = self.fig.add_subplot(grid[0, 1])
        self.data_ax = self.fig.add_subplot(grid[1, 1])

        self._build_scene()
        self._build_rocket()
        self._build_parachutes()
        self._build_graph()
        self._build_telemetry()
        self._build_controls()

        # An infinite timer sequence lets us pause on the final sample instead
        # of exhausting FuncAnimation and losing its timer. Repeat simply resets
        # our sample cursor, redraws the launch frame, and resumes that same timer.
        self.animation = FuncAnimation(
            self.fig, self._tick, frames=count(), init_func=self._initialize,
            interval=FRAME_INTERVAL_MS, repeat=False, blit=False,
            cache_frame_data=False,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.draw_sample(0)

    def _build_scene(self) -> None:
        ax = self.sky_ax
        ax.set_facecolor(SKY)
        ax.set_xlim(-36, 36)
        ax.set_ylim(0, 100)
        # Equal units keep the airframe and lettering from stretching as they turn.
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("FLIGHT VIEW", color=CYAN, fontsize=11, pad=12)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        rng = random.Random(42)
        ax.scatter([rng.uniform(-34, 34) for _ in range(65)],
                   [rng.uniform(18, 97) for _ in range(65)],
                   s=[rng.uniform(1, 5) for _ in range(65)],
                   color=WHITE, alpha=0.35, linewidths=0, zorder=0)
        ax.add_patch(Rectangle((-36, 0), 72, 9, facecolor="#263D40", zorder=1))
        ax.plot([-36, 36], [9, 9], color="#698F8D", linewidth=1.5, zorder=2)
        ax.plot([-5, 5], [9, 9], color=LIGHT_BLUE, linewidth=4, zorder=3)

    def _build_rocket(self) -> None:
        self.rocket_parts = []

        def add(part):
            self.sky_ax.add_patch(part)
            self.rocket_parts.append(part)
            return part

        self.flame = add(Polygon([(-0.65, -7), (0, -12), (0.65, -7)],
                                 closed=True, facecolor=ORANGE,
                                 edgecolor="none", zorder=4))
        self.inner_flame = add(Polygon([(-0.3, -7), (0, -10), (0.3, -7)],
                                       closed=True, facecolor="#FFF3BF",
                                       edgecolor="none", zorder=4.1))
        # Blue fins and nose; white airframe with a light-blue shaded edge.
        for side in (-1, 1):
            add(Polygon([(side, -3), (side * 2.6, -7), (side, -6)],
                        closed=True, facecolor=BLUE, edgecolor=LIGHT_BLUE,
                        linewidth=1, zorder=5))
        add(Rectangle((-0.6, -7), 1.2, 1.0, facecolor=DARK_BLUE,
                      edgecolor=LIGHT_BLUE, linewidth=0.8, zorder=5))
        add(Rectangle((-1, -6), 2, 11, facecolor=WHITE,
                      edgecolor=LIGHT_BLUE, linewidth=1, zorder=6))
        add(Rectangle((0.73, -6), 0.27, 11, facecolor="#D4E5FF",
                      edgecolor="none", zorder=6.1))
        add(Polygon([(-1, 5), (0, 8), (1, 5)], closed=True,
                    facecolor=BLUE, edgecolor=LIGHT_BLUE, linewidth=1, zorder=6))
        add(Polygon([(0, 8), (0.35, 5), (1, 5)], closed=True,
                    facecolor=DARK_BLUE, edgecolor="none", zorder=6.1))
        for y in (-5.7, 4.1):
            add(Rectangle((-1, y), 2, 0.5, facecolor=BLUE,
                          edgecolor="none", zorder=6.2))

        # Draw letters as an actual rocket component, not a screen-fixed label.
        # The glyph path is centered, fitted inside the body, and turned vertically.
        letters = TextPath((0, 0), ROCKET_NAME, size=1,
                           prop=FontProperties(family="DejaVu Sans", weight="bold"))
        bounds = letters.get_extents()
        text_scale = min(8.2 / bounds.width, 1.3 / bounds.height)
        text_pose = (Affine2D()
                     .translate(-bounds.x0 - bounds.width / 2,
                                -bounds.y0 - bounds.height / 2)
                     .scale(text_scale).rotate_deg(90).translate(0, -0.5))
        self.airframe_label = add(PathPatch(
            text_pose.transform_path(letters), facecolor=DARK_BLUE,
            edgecolor="none", zorder=7,
        ))

    def _build_parachutes(self) -> None:
        def canopy(width, height, color):
            points = [
                (width * math.cos(math.pi - j * math.pi / 30),
                 height * math.sin(math.pi - j * math.pi / 30))
                for j in range(31)
            ]
            patch = Polygon(points, closed=True, facecolor=color,
                            edgecolor=WHITE, linewidth=1.2, visible=False, zorder=4)
            self.sky_ax.add_patch(patch)
            return patch

        self.drogue = canopy(3, 4, ORANGE)
        self.main_chute = canopy(6, 6, BLUE)
        self.chute_lines = [
            self.sky_ax.plot([], [], color=WHITE, linewidth=0.9,
                             visible=False, zorder=3.5)[0]
            for _ in range(3)
        ]

    def _build_graph(self) -> None:
        ax = self.graph_ax
        ax.set_facecolor(PANEL)
        ax.set_title("ALTITUDE VS TIME", color=WHITE, fontsize=11, pad=12)
        ax.set_xlabel("Time (s)", color=MUTED)
        ax.set_ylabel("Altitude (ft)", color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(True, color=WHITE, alpha=0.10)
        ax.set_xlim(self.times[0], self.times[-1])
        padding = max(1, self.height_range * 0.12)
        ax.set_ylim(self.minimum - padding, self.maximum + padding)
        ax.plot(self.times, self.heights, color=BORDER, linewidth=1.5)
        self.flight_line, = ax.plot([], [], color=CYAN, linewidth=2.2)
        self.flight_point, = ax.plot([], [], "o", color=ORANGE, markersize=6)
        for idx, label, color in (
            (self.apogee_index, "Maximum recorded altitude", ORANGE),
            (self.drogue_index, "Drogue (estimated)", CYAN),
            (self.main_index, "Main (estimated)", LIGHT_BLUE),
        ):
            if idx is not None:
                ax.axvline(self.times[idx], color=color, linestyle="--",
                           linewidth=1, alpha=0.65, label=label)
        ax.legend(loc="upper right", fontsize=7, facecolor=PANEL,
                  edgecolor=BORDER, labelcolor=WHITE)

    def _build_telemetry(self) -> None:
        ax = self.data_ax
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.02, 1.04, "SIMULATION DATA", color=CYAN,
                fontsize=11, fontweight="bold")
        self.stage_text = ax.text(0.98, 1.04, "", color=LIGHT_BLUE,
                                  fontsize=8, ha="right")
        self.telemetry = {}
        for label, x, y in (("TIME", 0.02, 0.50), ("ALTITUDE", 0.52, 0.50),
                             ("VELOCITY", 0.02, 0.02), ("ACCELERATION", 0.52, 0.02)):
            ax.add_patch(FancyBboxPatch(
                (x, y), 0.45, 0.39, boxstyle="round,pad=0.008",
                facecolor=PANEL, edgecolor=BORDER, linewidth=1,
            ))
            ax.text(x + 0.035, y + 0.26, label, color=MUTED, fontsize=8)
            self.telemetry[label] = ax.text(x + 0.035, y + 0.08, "--",
                                            color=WHITE, fontsize=16,
                                            fontweight="bold")

    def _build_controls(self) -> None:
        self.fig.text(0.06, 0.12,
                      "Recovery timing and rotation are estimated. Artwork is not to scale.",
                      color=MUTED, fontsize=9)
        self.fig.text(0.06, 0.09,
                      "Sample-based playback, not real time. Repeat only restarts this viewer.",
                      color=MUTED, fontsize=9)
        self.status_text = self.fig.text(0.06, 0.045, "PLAYING", color=CYAN,
                                         fontsize=10, fontweight="bold")
        self.fig.text(0.30, 0.045, "Space: pause / play     R: repeat",
                      color=MUTED, fontsize=9)
        pause_ax = self.fig.add_axes([0.70, 0.029, 0.105, 0.052])
        repeat_ax = self.fig.add_axes([0.825, 0.029, 0.115, 0.052])
        self.pause_button = Button(pause_ax, "Pause", color="#203D61", hovercolor="#315D89")
        self.repeat_button = Button(repeat_ax, "Repeat", color=DARK_BLUE, hovercolor=BLUE)
        for button in (self.pause_button, self.repeat_button):
            button.label.set_color(WHITE)
            button.label.set_fontweight("bold")
            for spine in button.ax.spines.values():
                spine.set_color(BORDER)
        self.pause_button.on_clicked(self.toggle_pause)
        self.repeat_button.on_clicked(self.repeat)

    def rocket_angle(self, index: int) -> float:
        """Illustrative attitude only; this does not calculate flight dynamics."""
        elapsed = self.times[index] - self.apogee_time
        if elapsed <= 0:
            return 0.0
        angle = 115 * (1 - math.exp(-elapsed / 1.5))
        if self.main_index is not None and index >= self.main_index:
            main_time = self.times[self.main_index]
            initial_angle = 115 * (1 - math.exp(-(main_time - self.apogee_time) / 1.5))
            angle = 20 + (initial_angle - 20) * math.exp(-(self.times[index] - main_time) / 2)
        return angle

    def _update_parachutes(self, index: int, rocket_y: float, pose) -> None:
        main_open = self.main_index is not None and index >= self.main_index
        drogue_open = (self.drogue_index is not None
                       and index >= self.drogue_index and not main_open)
        self.drogue.set_visible(drogue_open)
        self.main_chute.set_visible(main_open)
        for line in self.chute_lines:
            line.set_visible(drogue_open or main_open)
            if not (drogue_open or main_open):
                line.set_data([], [])
        if not (drogue_open or main_open):
            return

        event = self.main_index if main_open else self.drogue_index
        patch = self.main_chute if main_open else self.drogue
        width = 6 if main_open else 3
        chute_y = rocket_y + (19 if main_open else 16) * ROCKET_SCALE
        elapsed = self.times[index] - self.times[event]
        opening = min(1.0, max(0.05, elapsed / CHUTE_OPENING_SECONDS))
        scale = opening * ROCKET_SCALE
        patch.set_transform(Affine2D().scale(scale).translate(0, chute_y)
                            + self.sky_ax.transData)
        attach_x, attach_y = pose.transform((0, 8))
        for line, x in zip(self.chute_lines, (-width, 0, width)):
            line.set_data([x * scale, attach_x], [chute_y, attach_y])

    def draw_sample(self, index: int) -> tuple:
        t = self.times[index]
        h = self.heights[index]
        fraction = (h - self.minimum) / self.height_range
        rocket_y = 19.5 + fraction * 39.0
        pose = (Affine2D().scale(ROCKET_SCALE).rotate_deg(self.rocket_angle(index))
                .translate(0, rocket_y))
        transform = pose + self.sky_ax.transData
        for part in self.rocket_parts:
            part.set_transform(transform)

        # Show exhaust only when thrust data supports it; do not invent a burn.
        thrust_series = self.data.get("thrust")
        thrust = thrust_series[self.indices[index]] if thrust_series is not None else None
        engine_on = (thrust is not None and self.maximum_thrust > 0
                     and thrust > 0.05 * self.maximum_thrust)
        self.flame.set_visible(engine_on)
        self.inner_flame.set_visible(engine_on)
        if engine_on:
            flicker = math.sin(index * 1.5)
            self.flame.set_xy([(-0.65, -7), (0, -12 - flicker), (0.65, -7)])
            self.inner_flame.set_xy([(-0.3, -7), (0, -9.8 - flicker * 0.4), (0.3, -7)])

        self._update_parachutes(index, rocket_y, pose)
        self.flight_line.set_data(self.times[:index + 1], self.heights[:index + 1])
        self.flight_point.set_data([t], [h])
        self.telemetry["TIME"].set_text(f"{t:.2f} s")
        self.telemetry["ALTITUDE"].set_text(f"{h:,.0f} ft")
        for key, label, unit in (("velocity", "VELOCITY", "m/s"),
                                  ("acceleration", "ACCELERATION", "m/s\u00b2")):
            values = self.data.get(key)
            value = values[self.indices[index]] if values is not None else None
            self.telemetry[label].set_text("--" if value is None else f"{value:.1f} {unit}")

        if self.main_index is not None and index >= self.main_index:
            stage = "MAIN DEPLOYED (EST.)"
        elif self.drogue_index is not None and index >= self.drogue_index:
            stage = "DROGUE DEPLOYED (EST.)"
        elif index == self.apogee_index:
            stage = "MAXIMUM ALTITUDE"
        elif index > self.apogee_index:
            stage = "DESCENT"
        else:
            stage = "POWERED ASCENT" if engine_on else "ASCENT"
        self.stage_text.set_text(stage)
        return ()  # blit=False: Matplotlib redraws the entire figure.

    def _initialize(self) -> tuple:
        return self.draw_sample(self.frames[0])

    def _tick(self, _timer_frame: int) -> tuple:
        if self.paused or self.finished or self.closed:
            return ()
        self.draw_sample(self.frames[self.cursor])
        if self.cursor == len(self.frames) - 1:
            self.finished = True
            self.paused = True
            self.animation.pause()
            self.pause_button.label.set_text("Play")
            self.status_text.set_text("FINISHED - PRESS REPEAT")
        else:
            self.cursor += 1
        return ()

    def repeat(self, _event=None) -> None:
        """Restart from the first recorded sample, including after completion."""
        if self.closed:
            return
        self.animation.pause()
        self.cursor = 0
        self.paused = False
        self.finished = False
        # Reset attitude, chutes, lines, graph, flame and every numeric readout.
        self.draw_sample(self.frames[0])
        self.pause_button.label.set_text("Pause")
        self.status_text.set_text("PLAYING")
        self.animation.resume()
        self.fig.canvas.draw_idle()

    def toggle_pause(self, _event=None) -> None:
        if self.closed:
            return
        if self.finished:
            self.repeat()
            return
        self.paused = not self.paused
        if self.paused:
            self.animation.pause()
            self.pause_button.label.set_text("Play")
            self.status_text.set_text("PAUSED")
        else:
            self.animation.resume()
            self.pause_button.label.set_text("Pause")
            self.status_text.set_text("PLAYING")
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key in (" ", "space"):
            self.toggle_pause()
        elif event.key in ("r", "R"):
            self.repeat()

    def _on_close(self, _event) -> None:
        self.closed = True
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()

    def show(self) -> None:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        help="CSV to replay; otherwise use the newest CSV next to this script")
    parser.add_argument("--config", type=Path,
                        help="Optional tracker-compatible JSON column/unit configuration")
    args = parser.parse_args()

    if args.csv is None:
        candidates = [p for p in Path(__file__).resolve().parent.glob("*.csv") if p.is_file()]
        if not candidates:
            parser.error("No CSV found. Provide a CSV path or place one next to animate.py.")
        args.csv = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        csv_file = args.csv.expanduser().resolve()
        if not csv_file.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        cfg = json.loads(args.config.expanduser().read_text()) if args.config else {}
        if not isinstance(cfg, dict):
            raise ValueError("The configuration must contain a JSON object.")

        # Reuse the existing reader without changing THE_Tracker.py or calling main().
        from THE_Tracker import read_csv, convert
        header, rows = read_csv(csv_file, cfg)
        data, _ = convert(header, rows, cfg)
        dashboard = FlightReplay(data, csv_file.name)
        print(f"Opening Loxodon replay: {csv_file.name}")
        print("Repeat restarts the visualization only; no Excel entries are added.")
        print("Recovery timing and rocket orientation are estimated, not exported events.")
        if not data.get("thrust"):
            print("No thrust data: engine exhaust is hidden.")
        dashboard.show()
        return 0
    except (ValueError, KeyError, OSError, ImportError, IndexError) as exc:
        print(f"Animation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
