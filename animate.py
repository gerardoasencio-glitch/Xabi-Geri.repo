"""Loxodon-1 cinematic 3D flight replay; reads CSVs without writing tracker data.

    python animate.py simulations/flight.csv
    python animate.py --speed 4 --export flight.mp4

Controls: Space pause, R replay, C camera, H instruments, arrows seek,
+/- playback speed. Drag the timeline to scrub. MP4 needs ffmpeg; GIF uses Pillow.
The live viewer uses pyglet for hardware rendering. NumPy/Matplotlib provide data
handling and movie exports; --software selects the compatible software viewer.
The tracker reader is reused for its column configuration and units; its main()
is never called, and the viewer never writes Excel files.

Position, wind, thrust and attitude come from the export when available. OpenRocket
calls elevation "Vertical orientation (zenith)" in older exports. Recovery motion,
canopy dimensions, landscape and airframe geometry are visual approximations, not
a new dynamics simulation. In particular, recovery attitude is not resolved by
OpenRocket: the viewer adds a clearly identified hanging/swinging recovery rig.
"""

import argparse
from collections import deque
from dataclasses import dataclass
import json
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import matplotlib

# TkAgg supports partial redraws on macOS; Apple's native Matplotlib backend
# does not. Respect an explicitly requested backend and never select old Tk 8.5.
if not os.environ.get("MPLBACKEND"):
    if "--export" in sys.argv or ("--software" not in sys.argv and importlib.util.find_spec("pyglet")):
        matplotlib.use("Agg")
    elif sys.platform == "darwin":
        try:
            import tkinter
            if tkinter.TkVersion >= 8.6:
                matplotlib.use("TkAgg")
        except ImportError:
            pass

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FFMpegWriter, PillowWriter, writers
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as DrawingPath
from matplotlib.widgets import Button, Slider
from PIL import Image


ROCKET_NAME = "LOXODON-1"
FRAME_INTERVAL_MS = 1000 / 60
# Artistic defaults, in metres. Diameter uses Reference length when exported.
ROCKET_LENGTH_M = 3.8
ROCKET_DIAMETER_M = 0.14
MAIN_DEPLOY_ALTITUDE_FT = 1000.0  # Fallback only; actual CSV events take priority.
DROGUE_DELAY_SECONDS = 0.5
CHUTE_OPENING_SECONDS = 1.8
BACKGROUND = "#07121d"
WHITE = "#edf4f5"
MUTED = "#94adba"
CYAN = "#70dce0"
ORANGE = "#ffb35a"
BLUE = "#267fc2"


def unit_vector(value, fallback=(0, 0, 1)):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-9 else np.asarray(fallback, dtype=float)


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def rotation_from_axis(axis, roll=0.0):
    """Orthonormal body-to-world basis; local +Z is the nose, ENU is world."""
    z = unit_vector(axis)
    # Shortest rotation from world +Z, continuous through North/horizontal.
    # The earlier cross(North, nose) frame could flip 180 degrees there.
    if z[2] > -1 + 1e-9:
        q = 1 / (1 + z[2])
        x = np.array([1 - z[0] ** 2 * q, -z[0] * z[1] * q, -z[0]])
        y = np.array([-z[0] * z[1] * q, 1 - z[1] ** 2 * q, -z[1]])
    else:
        x, y = np.array([1., 0., 0.]), np.array([0., -1., 0.])
    c, s = np.cos(roll), np.sin(roll)
    return np.column_stack((c * x + s * y, -s * x + c * y, z))


def monotone_slopes(times, positions):
    """Shape-preserving Hermite tangents: smooth motion through every CSV sample."""
    dt = np.diff(times)
    slopes = np.diff(positions, axis=0) / dt[:, None]
    result = np.zeros_like(positions)
    result[0], result[-1] = slopes[0], slopes[-1]
    if len(times) > 2:
        a, b = slopes[:-1], slopes[1:]
        w1, w2 = (2 * dt[1:] + dt[:-1])[:, None], (dt[1:] + 2 * dt[:-1])[:, None]
        valid = (a * b > 0) & (abs(a) > 1e-12) & (abs(b) > 1e-12)
        divisor = np.divide(w1, a, out=np.zeros_like(a), where=valid)
        divisor += np.divide(w2, b, out=np.zeros_like(b), where=valid)
        result[1:-1] = np.divide(w1 + w2, divisor, out=np.zeros_like(a), where=valid)
    return result


def numeric_column(header, rows, name, units):
    """Read one optional export column, with units converted to SI."""
    for index, title in enumerate(header):
        match = re.fullmatch(re.escape(name) + r"\s*[\[(](.*?)[\])]", title, re.I)
        if match and match[1] in units:
            values = []
            for row in rows:
                try:
                    values.append(float(row[index]) * units[match[1]])
                except (ValueError, TypeError):
                    values.append(np.nan)
            return np.asarray(values)
    return None


def load_flight(path, config=None):
    """Preserve tracker mappings and enrich them locally with 3D export data."""
    from THE_Tracker import read_csv, convert
    config = config or {}
    header, rows = read_csv(path, config)
    data, _ = convert(header, rows, config)
    length = {"m": 1, "ft": 0.3048, "in": 0.0254}
    speed = {"m/s": 1, "ft/s": 0.3048, "mph": 0.44704, "km/h": 1 / 3.6}
    angle = {"°": np.pi / 180, "deg": np.pi / 180, "rad": 1}
    for key, name, units in (
        ("east", "Position East of launch", length),
        ("north", "Position North of launch", length),
        ("vertical_velocity", "Vertical velocity", speed),
        ("elevation", "Vertical orientation (zenith)", angle),
        ("azimuth", "Lateral orientation (azimuth)", angle),
        ("roll_rate", "Roll rate", {"°/s": np.pi / 180, "rad/s": 1}),
        ("wind_speed", "Wind velocity", speed),
        ("wind_direction", "Wind direction", angle),
        ("diameter", "Reference length", length),
    ):
        column = numeric_column(header, rows, name, units)
        if column is not None:
            data[key] = column
    # Newer OpenRocket versions use the clearer "elevation" label.
    if "elevation" not in data:
        column = numeric_column(header, rows, "Vertical orientation (elevation)", angle)
        if column is not None:
            data["elevation"] = column
    events = []
    for match in re.finditer(
        r"#\s*Event\s+(\w+)\s+occurred at t=([\d.eE+\-]+)\s+seconds([^\n]*)",
        path.read_text(encoding=config.get("encoding", "utf-8-sig")),
    ):
        events.append((match[1], float(match[2]), match[3]))
    data["events"] = events
    return data


class FlightData:
    """Continuous-time sampling; all rendering and seeking are deterministic."""

    def __init__(self, data, drogue_time=None, main_time=None, main_altitude=1000):
        if "time" not in data or "altitude" not in data:
            raise ValueError("The CSV needs time and altitude columns.")
        raw_t = np.asarray(data["time"], dtype=float)
        raw_h = np.asarray(data["altitude"], dtype=float)
        if len(raw_t) != len(raw_h):
            raise ValueError("Time and altitude have different sample counts.")
        mask = np.isfinite(raw_t) & np.isfinite(raw_h)
        self.times, self.heights = raw_t[mask], raw_h[mask]
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Need at least two strictly increasing valid times.")
        self.start, self.end = float(self.times[0]), float(self.times[-1])
        self.series = {}
        self._columns, self._angles = {}, {}
        for key, values in data.items():
            if key == "events":
                continue
            array = np.asarray(values, dtype=float)
            if array.ndim == 1 and len(array) == len(raw_t):
                self.series[key] = array[mask]
        self.apogee_index = int(np.argmax(self.heights))
        self.apogee_time = self.times[self.apogee_index]
        self.events = sorted(data.get("events", []), key=lambda event: event[1])
        deployments = sorted(set(t for name, t, _ in self.events
                                 if name == "RECOVERY_DEVICE_DEPLOYMENT"))
        # Device names may only appear in deployment warnings; these identify
        # main-only/drogue-only exports instead of assuming every first event is a drogue.
        named = {}
        for name, t, detail in self.events:
            if name == "SIM_WARN" and "deployment" in detail.lower():
                if "drogue" in detail.lower():
                    named["drogue"] = min(deployments, key=lambda event: abs(event - t)) if deployments else t
                elif "main" in detail.lower():
                    named["main"] = min(deployments, key=lambda event: abs(event - t)) if deployments else t
        self.recovery_source = "CSV EVENTS"
        if drogue_time is not None or main_time is not None:
            self.recovery_source = "MANUAL TIMING"
        if deployments:
            self.drogue_time = named.get("drogue")
            self.main_time = named.get("main")
            unassigned = [t for t in deployments if t not in named.values()]
            if self.drogue_time is None and unassigned:
                self.drogue_time = unassigned.pop(0)
            if self.main_time is None and unassigned:
                self.main_time = unassigned.pop(0)
        else:
            self.recovery_source = "ESTIMATED TIMING"
            after = np.flatnonzero((self.times >= self.apogee_time + DROGUE_DELAY_SECONDS)
                                  & (self.heights < max(self.heights) - 1))
            self.drogue_time = float(self.times[after[0]]) if len(after) else None
            below = np.flatnonzero((self.times >= (self.drogue_time or self.end + 1))
                                  & (self.heights - self.heights[0] <= main_altitude))
            self.main_time = float(self.times[below[0]]) if len(below) else None
        if drogue_time is not None:
            self.drogue_time = drogue_time
        if main_time is not None:
            self.main_time = main_time
        for event_time in (self.drogue_time, self.main_time):
            if event_time is not None and not self.start <= event_time <= self.end:
                raise ValueError("Deployment times must be inside the CSV time range.")
        if (self.drogue_time is not None and self.main_time is not None
                and self.main_time < self.drogue_time):
            raise ValueError("Main deployment cannot precede drogue deployment.")
        self.has_position = all(self.available(key) for key in ("east", "north"))
        self.has_attitude = all(self.available(key) for key in ("elevation", "azimuth"))
        self.has_wind = all(self.available(key) for key in ("wind_speed", "wind_direction"))
        altitude = (self.heights - self.heights[0]) * 0.3048
        self.positions = np.column_stack((self.column("east"), self.column("north"), altitude))
        self.position_slopes = monotone_slopes(self.times, self.positions)
        self.velocities = np.gradient(self.positions, self.times, axis=0)
        self.ground_time = next((t for name, t, _ in self.events if name == "GROUND_HIT"), self.end)
        bearings = np.array([self.circular("wind_direction", t) for t in self.times])
        speeds = self.column("wind_speed")
        wind = -speeds[:, None] * np.column_stack((np.sin(bearings), np.cos(bearings), np.zeros_like(speeds)))
        self.wind_displacements = np.vstack((np.zeros(3), np.cumsum(
            (wind[1:] + wind[:-1]) * .5 * np.diff(self.times)[:, None], axis=0)))
        # Missing angular-rate samples after recovery must not invent continued spin.
        roll_rate = np.nan_to_num(self.series.get("roll_rate", np.zeros(len(self.times))))
        self.roll = np.concatenate(([0], np.cumsum(
            0.5 * (roll_rate[1:] + roll_rate[:-1]) * np.diff(self.times))))
        self.maximum_thrust = max(0, np.nanmax(self.column("thrust")))
        self.diameter = float(np.clip(self.value("diameter", self.start,
                                                ROCKET_DIAMETER_M), 0.04, 1.0))

    def available(self, key):
        return key in self.series and np.any(np.isfinite(self.series[key]))

    def column(self, key, default: float = 0.0):
        cache_key = (key, default)
        if cache_key in self._columns:
            return self._columns[cache_key]
        values = self.series.get(key, np.full(len(self.times), float(default)))
        valid = np.isfinite(values)
        if not valid.any():
            return np.full(len(self.times), float(default))
        result = np.interp(self.times, self.times[valid], values[valid])
        self._columns[cache_key] = result
        return result

    def value(self, key, t, default: float = 0.0):
        if not self.available(key):
            return float(default)
        return float(np.interp(t, self.times, self.column(key)))

    def vector(self, values, t):
        return np.array([np.interp(t, self.times, values[:, i]) for i in range(3)])

    def position(self, t):
        t = float(np.clip(t, self.start, self.end))
        i = min(len(self.times) - 2, max(0, int(np.searchsorted(self.times, t)) - 1))
        dt = self.times[i + 1] - self.times[i]
        u = (t - self.times[i]) / dt
        return ((2 * u ** 3 - 3 * u ** 2 + 1) * self.positions[i]
                + (u ** 3 - 2 * u ** 2 + u) * dt * self.position_slopes[i]
                + (-2 * u ** 3 + 3 * u ** 2) * self.positions[i + 1]
                + (u ** 3 - u ** 2) * dt * self.position_slopes[i + 1])

    def wind_displacement(self, t):
        return self.vector(self.wind_displacements, t)

    def wind(self, t):
        if not self.has_wind:
            return np.zeros(3)
        # OpenRocket wind direction is the bearing FROM which wind arrives.
        # Its internal wind vector is subtracted from air-relative velocity.
        direction = self.circular("wind_direction", t)
        speed = self.value("wind_speed", t)
        return -speed * np.array([np.sin(direction), np.cos(direction), 0])

    def circular(self, key, t):
        if not self.available(key):
            return 0.0
        if key not in self._angles:
            raw = self.series[key]
            valid = np.isfinite(raw)
            # Unwrap before filling missing angles, especially at 359° -> 0°.
            self._angles[key] = (self.times[valid], np.unwrap(raw[valid]))
        times, angles = self._angles[key]
        return float(np.interp(t, times, angles))

    def attitude(self, t):
        if self.has_attitude:
            elevation = self.value("elevation", t, np.pi / 2)
            heading = self.circular("azimuth", t)
            # OpenRocket azimuth and wind bearing both run clockwise from North.
            axis = np.array([np.cos(elevation) * np.sin(heading),
                             np.cos(elevation) * np.cos(heading), np.sin(elevation)])
        else:
            axis = unit_vector(self.vector(self.velocities, t))
            if t <= self.start + 0.1:
                axis = np.array([0., 0., 1.])
        roll = float(np.interp(t, self.times, self.roll))
        return axis, roll

    def opening(self, t, event):
        return 0.0 if event is None or t < event else float(smoothstep(
            (t - event) / CHUTE_OPENING_SECONDS))

    def stage(self, t):
        ground = next((time for name, time, _ in self.events if name == "GROUND_HIT"), None)
        if ground is not None and t >= ground:
            return "TOUCHDOWN"
        if self.main_time is not None and t >= self.main_time:
            return "MAIN / FINAL DESCENT"
        if self.drogue_time is not None and t >= self.drogue_time:
            return "DROGUE / CONTROLLED DESCENT"
        if abs(t - self.apogee_time) < 0.65:
            return "APOGEE"
        if t > self.apogee_time:
            return "BALLISTIC DESCENT"
        return "POWERED ASCENT" if self.value("thrust", t) > 1 else "COAST / ASCENT"


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: list
    palette: np.ndarray

    def __post_init__(self):
        width = max(map(len, self.faces))
        self.face_indices = np.array([face + [face[-1]] * (width - len(face))
                                      for face in self.faces])
        triangles = self.vertices[self.face_indices[:, :3]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        self.normals = normals / np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)


def surface_mesh(rings, segments=24, color="#e7e9de", stripe=None):
    """Lathe a profile into a cylindrical mesh with shaded circumferential panels."""
    theta = np.arange(segments) * 2 * np.pi / segments
    vertices = np.array([[r * np.cos(a), r * np.sin(a), z]
                         for z, r in rings for a in theta])
    faces, palette = [], []
    for j in range(len(rings) - 1):
        for k in range(segments):
            faces.append([j * segments + k, j * segments + (k + 1) % segments,
                          (j + 1) * segments + (k + 1) % segments,
                          (j + 1) * segments + k])
            palette.append(colors.to_rgb(stripe(k, j) if stripe else color))
    return Mesh(vertices, faces, np.asarray(palette))


def rocket_mesh(length, radius):
    vertices, faces, palette = [], [], []

    def merge(mesh):
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        faces.extend([[index + offset for index in face] for face in mesh.faces])
        palette.extend(mesh.palette)

    merge(surface_mesh([(-length * .49, radius * .8), (-length * .46, radius),
                        (length * .29, radius)], color="#e6e9e3"))
    # Rounded ogive nose rather than a flat triangle.
    profile = [(length * .29 + length * .21 * f,
                radius * np.cos(f * np.pi / 2)) for f in np.linspace(0, 1, 10)]
    merge(surface_mesh(profile, color="#207bc3"))
    for z in (-length * .43, length * .25, -length * .02):
        merge(surface_mesh([(z, radius * 1.013), (z + length * .012, radius * 1.013)],
                           color="#183c5c"))
    merge(surface_mesh([(-length * .5, radius * .68), (-length * .535, radius * .85),
                        (-length * .535, radius * .48), (-length * .5, radius * .4)],
                       color="#292e36"))
    for angle in np.arange(4) * np.pi / 2:
        local = np.array([[radius, -.012, -length * .27],
                          [radius * 3.4, -.012, -length * .46],
                          [radius * 3.0, -.012, -length * .50],
                          [radius, -.012, -length * .46]])
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        start = len(vertices)
        vertices.extend(local @ rot.T)
        vertices.extend((local + [0, .024, 0]) @ rot.T)
        for face in ([0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3]):
            faces.append([start + i for i in face])
            palette.append(colors.to_rgb("#227abd"))
    # Small surface lettering is actual 3D geometry and rotates with the body.
    glyphs = TextPath((0, 0), ROCKET_NAME, size=1,
                     prop=FontProperties(family="DejaVu Sans", weight="bold"))
    bounds = glyphs.get_extents()
    scale = min(length * .43 / bounds.width, radius * 1.25 / bounds.height)
    for side in (-1, 1):
        for polygon in glyphs.to_polygons():
            start = len(vertices)
            for gx, gy in np.asarray(polygon, dtype=float).reshape(-1, 2):
                vertices.append([side * (gy - bounds.y0 - bounds.height / 2) * scale,
                                 -side * radius * 1.015,
                                 (gx - bounds.x0 - bounds.width / 2) * scale + length * .02])
            faces.append(list(range(start, len(vertices))))
            palette.append(colors.to_rgb("#15394d"))
    return Mesh(np.array(vertices), faces, np.array(palette))


class PerspectiveScene:
    """Small NumPy perspective renderer. One sorted collection avoids 3D axis overhead."""

    def __init__(self, ax):
        self.ax = ax
        self.aspect = 16 / 9
        self.eye = np.array([10., -10., 3.])
        self.target = np.zeros(3)
        self.polygons = PolyCollection([], linewidths=0, antialiaseds=False)
        self.lines = LineCollection([], linewidths=.6, antialiaseds=True)
        self.particles = ax.scatter([], [], s=[], edgecolors="none", zorder=4)
        ax.add_collection(self.polygons)
        ax.add_collection(self.lines)
        self.light = unit_vector((0.35, -0.6, 0.8))
        self._polygon_paths, self._line_paths = [], []
        self.clear()

    def clear(self):
        self.faces, self.shades, self.depths = [], [], []
        self.segments, self.line_colors, self.line_widths = [], [], []
        self.points, self.sizes, self.point_colors = [], [], []

    def camera(self, eye, target, aspect):
        self.eye, self.target, self.aspect = eye, target, aspect
        forward = unit_vector(target - eye)
        right = unit_vector(np.cross(forward, (0, 0, 1)), (1, 0, 0))
        up = np.cross(right, forward)
        self.view = np.array([right, up, forward])

    def project(self, vertices):
        local = (np.asarray(vertices) - self.eye) @ self.view.T
        depth = local[:, 2]
        screen = local[:, :2] * (1.75 / np.maximum(depth[:, None], .08))
        screen[:, 0] /= self.aspect
        return screen, depth

    def mesh(self, mesh, position=None, rotation=None, alpha: float = 1.0, emissive=False):
        vertices = mesh.vertices
        if rotation is not None:
            vertices = vertices @ rotation.T
        if position is not None:
            vertices = vertices + position
        projected, depth = self.project(vertices)
        indices = mesh.face_indices
        polys, depths = projected[indices], depth[indices]
        means = np.mean(depths, axis=1)
        normals = mesh.normals if rotation is None else mesh.normals @ rotation.T
        light = .50 + .50 * np.maximum(0, normals @ self.light)
        shades = mesh.palette.copy() if emissive else np.clip(mesh.palette * light[:, None], 0, 1)
        if not emissive:
            haze = np.clip((means - 80) / 6500, 0, .85)[:, None]
            shades = shades * (1 - haze) + np.array([.46, .65, .72]) * haze
        rgba = np.column_stack((shades, np.full(len(shades), alpha)))
        visible = ((np.min(depths, axis=1) > .08)
                   & (np.max(polys[:, :, 0], axis=1) >= -1.5)
                   & (np.min(polys[:, :, 0], axis=1) <= 1.5)
                   & (np.max(polys[:, :, 1], axis=1) >= -1.4)
                   & (np.min(polys[:, :, 1], axis=1) <= 1.4))
        self.faces.extend(polys[visible])
        self.shades.extend(rgba[visible])
        self.depths.extend(means[visible])
        # Clip terrain/large polygons crossing the near plane rather than
        # dropping the entire face and revealing holes beneath the camera.
        crossing = np.flatnonzero((np.min(depths, axis=1) <= .08)
                                  & (np.max(depths, axis=1) > .08))
        if len(crossing):
            local = (vertices - self.eye) @ self.view.T
            for index in crossing:
                polygon = local[mesh.faces[index]]
                clipped = []
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                    if a[2] >= .08:
                        clipped.append(a)
                    if (a[2] >= .08) != (b[2] >= .08):
                        clipped.append(a + (b - a) * ((.08 - a[2]) / (b[2] - a[2])))
                if len(clipped) >= 3:
                    clipped = np.array(clipped)
                    screen = clipped[:, :2] * 1.75 / clipped[:, 2:3]
                    screen[:, 0] /= self.aspect
                    self.faces.append(screen)
                    self.shades.append(rgba[index])
                    self.depths.append(means[index])

    def line(self, points, color=WHITE, width=.65, alpha: float = 1.0):
        projected, depths = self.project(points)
        rgba = colors.to_rgba(color, alpha)
        for i in range(len(projected) - 1):
            if min(depths[i:i + 2]) > .08:
                self.segments.append(projected[i:i + 2])
                self.line_colors.append(rgba)
                self.line_widths.append(width)

    def line_segments(self, segments, color=WHITE, width=.65, alpha: float = 1.0):
        segments = np.asarray(segments).reshape(-1, 2, 3)
        projected, depths = self.project(segments.reshape(-1, 3))
        screen = projected.reshape(-1, 2, 2)
        valid = np.min(depths.reshape(-1, 2), axis=1) > .08
        self.segments.extend(screen[valid])
        count = int(np.count_nonzero(valid))
        self.line_colors.extend([colors.to_rgba(color, alpha)] * count)
        self.line_widths.extend([width] * count)

    def particle(self, position, radius, color, alpha):
        p, depth = self.project(np.asarray(position)[None, :])
        if depth[0] > .1 and np.all(abs(p[0]) < 1.3):
            self.points.append(p[0])
            # Scatter area is points squared; scale for the actual figure height.
            points = radius * 1.75 / depth[0] * self.ax.bbox.height * .72
            self.sizes.append(max(.1, points ** 2))
            self.point_colors.append(colors.to_rgba(color, alpha))

    def finish(self):
        order = np.argsort(-np.asarray(self.depths), kind="stable")
        # Keep the Path objects alive. set_verts()/set_segments() constructed
        # thousands of new objects every frame, which dominated render time.
        self._reuse_paths(self.polygons, self._polygon_paths, [self.faces[i] for i in order])
        # Matplotlib's public color API accepts sequences of RGBA tuples.
        face_colors = [(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
                       for c in np.asarray(self.shades)[order]]
        self.polygons.set_facecolor(face_colors)
        self._reuse_paths(self.lines, self._line_paths, self.segments)
        self.lines.set_colors(self.line_colors)
        self.lines.set_linewidth(self.line_widths)
        self.particles.set_offsets(np.asarray(self.points).reshape(-1, 2))
        self.particles.set_sizes(self.sizes)
        self.particles.set_facecolor(self.point_colors)

    @staticmethod
    def _reuse_paths(collection, pool, vertices):
        while len(pool) < len(vertices):
            pool.append(DrawingPath(np.zeros((3, 2))))
        for path, points in zip(pool, vertices):
            path.vertices = np.asarray(points)
        collection.get_paths()[:] = pool[:len(vertices)]
        collection.stale = True


def build_landscape():
    """Low-poly coastal range, deterministic terrain, launch apron and access road."""
    rng = np.random.default_rng(17)
    vertices, faces, palette = [], [], []
    grid = np.linspace(-16000, 16000, 49)
    for y in grid:
        for x in grid:
            mountain = max(0, y - 3300) * .16
            z = mountain * (.6 + .4 * np.sin(x / 1000) * np.cos(y / 1400)) - 1
            vertices.append((x, y, z))
    for j in range(len(grid) - 1):
        for k in range(len(grid) - 1):
            a = j * len(grid) + k
            for face in ([a, a + 1, a + len(grid)],
                         [a + 1, a + len(grid) + 1, a + len(grid)]):
                faces.append(face)
                ocean = grid[k] > 2400
                base = np.array([.15, .37, .46]) if ocean else np.array([.27, .36, .24])
                palette.append(np.clip(base + rng.uniform(-.035, .035), 0, 1))
    for size, z, color in ((30, -.04, "#7c8175"), (8, .0, "#abb0a6")):
        offset = len(vertices)
        vertices.extend([(-size, -size, z), (size, -size, z),
                         (size, size, z), (-size, size, z)])
        faces.append([offset + i for i in range(4)])
        palette.append(colors.to_rgb(color))
    offset = len(vertices)
    vertices.extend([(-3, -30, -.02), (3, -30, -.02), (3, -2000, -.02), (-3, -2000, -.02)])
    faces.append([offset + i for i in range(4)])
    palette.append(colors.to_rgb("#6b7269"))
    return Mesh(np.asarray(vertices), faces, np.asarray(palette))


def canopy_mesh(radius, opening, color, phase):
    """Vented hemispherical canopy with alternating gores and a scalloped skirt."""
    segments = 32
    vertices, faces, palette = [], [], []
    # Reefed deployment grows in radius while its package first stretches vertically.
    width = radius * (.045 + .955 * opening)
    height = radius * (.35 + .4 * opening)
    for j, angle in enumerate(np.linspace(.12, np.pi / 2, 7)):
        for k in range(segments):
            a = 2 * np.pi * k / segments
            scallop = 1 - (.045 * (1 + np.cos(a * 16))) * j / 6
            flutter = 1 + .012 * np.sin(phase * 4 + a * 6) * opening
            vertices.append((width * np.sin(angle) * np.cos(a) * flutter,
                             width * np.sin(angle) * np.sin(a) * flutter,
                             height * np.cos(angle) * scallop))
    base = np.asarray(colors.to_rgb(color))
    for j in range(6):
        for k in range(segments):
            faces.append([j * segments + k, (j + 1) * segments + k,
                          (j + 1) * segments + (k + 1) % segments,
                          j * segments + (k + 1) % segments])
            palette.append(base if k // 2 % 2 else np.array([.94, .93, .84]))
    return Mesh(np.asarray(vertices), faces, np.asarray(palette))


def cloud_sprite(seed):
    """Procedural soft cloud texture, generated in memory without asset files."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:64j, -1:1:128j]
    density = np.zeros_like(x)
    for _ in range(13):
        cx, cy = rng.uniform(-.6, .6), rng.uniform(-.13, .25)
        rx, ry = rng.uniform(.10, .25), rng.uniform(.12, .37)
        density += np.exp(-((x - cx) / rx) ** 2 - ((y - cy) / ry) ** 2)
    noise = .04 * (np.sin(x * 54 + y * 13) + np.sin(x * 27 - y * 47))
    alpha = smoothstep((density + noise - .06) / .85) * .64
    light = .80 + .15 * smoothstep((y + .2) * 2)
    return np.dstack((light * .98, light, light, alpha))


class FramePresenter:
    """Cache static UI pixels and draw moving artists in their actual layer order."""

    def __init__(self, figure, artists):
        self.fig, self.canvas, self.artists = figure, figure.canvas, artists
        self.enabled = bool(self.canvas.supports_blit and "macosx" not in matplotlib.get_backend().lower())
        self.background = None
        for artist in artists:
            artist.set_animated(self.enabled)
        self.connection = self.canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, _event=None):
        if not self.enabled:
            return
        self.background = self.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_artists()

    def _draw_artists(self):
        for artist in self.artists:
            if artist.get_visible():
                self.fig.draw_artist(artist)

    def present(self):
        if not self.enabled:
            self.canvas.draw_idle()
        elif self.background is None:
            self.canvas.draw()
        else:
            self.canvas.restore_region(self.background)
            self._draw_artists()
            self.canvas.blit(self.fig.bbox)

    def suspend(self):
        self.background = None
        for artist in self.artists:
            artist.set_animated(False)
        self.enabled = False


class FlightReplay:
    """Perspective chase camera with recorded telemetry and continuous-time playback."""

    def __init__(self, data, source_name="Simulation", speed=1, fps=60,
                 drogue_time=None, main_time=None, main_altitude=1000,
                 rocket_length=ROCKET_LENGTH_M, drogue_cutaway=False):
        self.flight = FlightData(data, drogue_time, main_time, main_altitude)
        self.source_name = source_name
        self.software = False
        self.length = rocket_length
        self.drogue_cutaway = drogue_cutaway
        self.t = self.flight.start
        self.speed, self.fps = speed, fps
        self.paused = self.finished = self.closed = False
        self.camera_mode = 0
        self.hud_visible = True
        self.animation = None
        self._timer = None
        self._presenter = None
        self._scrubbing = False
        self._pending_seek = None
        self._resize_pending = False
        self._frame_costs = deque(maxlen=120)
        self._exporting = False
        self._last_clock = None
        self._setting_slider = False
        self.fig = plt.figure(figsize=(13.6, 7.65), dpi=90, facecolor=BACKGROUND)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Loxodon-1 | Cinematic Flight")
        self.ax = self.fig.add_axes((0, .12, 1, .80), facecolor="#6699b2")
        self.ax.set(xlim=(-1, 1), ylim=(-1, 1))
        self.ax.axis("off")
        self.scene = PerspectiveScene(self.ax)
        self._cloud_sprites = [Image.fromarray((cloud_sprite(i + 3)[::-1] * 255).astype(np.uint8))
                               for i in range(18)]
        self._cloud_sizes = {}
        self.cloud_layer = self.ax.imshow(np.zeros((288, 512, 4), dtype=np.uint8),
                                          extent=(-1, 1, -1, 1), origin="upper",
                                          aspect="auto", zorder=-2, interpolation="bilinear")
        self.landscape = build_landscape()
        self.rocket = rocket_mesh(self.length, self.flight.diameter / 2)
        # A photographic-style sky gradient lives behind the world geometry.
        sky = np.linspace(0, 1, 320)[:, None, None]
        low, high = np.array([.69, .79, .80]), np.array([.12, .34, .52])
        gradient = np.repeat(low[None, None, :] * (1 - sky) + high[None, None, :] * sky, 2, axis=1)
        self.ax.imshow((gradient * 255).astype(np.uint8), extent=(-1, 1, -1, 1),
                       origin="lower", aspect="auto", zorder=-5)
        self._build_hud(source_name)
        self._build_controls()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        self.fig.canvas.mpl_connect("button_press_event", self._on_pointer_down)
        self.fig.canvas.mpl_connect("button_release_event", self._on_pointer_up)
        self._prepare_smoke()
        self.draw_time(self.t)

    def _build_hud(self, source):
        self.fig.text(.035, .955, "L O X O D O N", color=WHITE, fontsize=19, weight="bold")
        self.fig.text(.22, .955, "FLIGHT CINEMA  /  01", color=CYAN, fontsize=10)
        self.fig.text(.965, .955, source[:60], ha="right", color=MUTED, fontsize=9)
        self.hud = []

        def label(x, y, text="", **kwargs):
            artist = self.ax.text(x, y, text, transform=self.ax.transAxes,
                                  color=WHITE, family="monospace", zorder=20, **kwargs)
            self.hud.append(artist)
            return artist

        self.stage_text = label(.035, .92, fontsize=13, weight="bold")
        self.clock_text = label(.035, .875, fontsize=10)
        self.event_text = label(.035, .82, fontsize=10)
        self.event_text.set_color(ORANGE)
        self.camera_text = label(.965, .92, ha="right", fontsize=9)
        self.telemetry = {}
        for label_name, y in (("ALTITUDE AGL", .81), ("SPEED", .68), ("VERTICAL", .55), ("WIND FROM", .42)):
            title = label(.965, y, label_name, ha="right", fontsize=8)
            title.set_color("#d1e4e8")
            self.telemetry[label_name] = label(.965, y - .048, ha="right", fontsize=18, weight="bold")
        self.provenance = label(.035, .04, fontsize=7)
        self.provenance.set_color("#d5e4e6")
        # Small altitude strip gives the movie a flight context without splitting the scene.
        self.graph_ax = self.fig.add_axes((.035, .18, .22, .10), facecolor="none")
        self.graph_ax.plot(self.flight.times, self.flight.heights, color=WHITE, alpha=.22, lw=1)
        self.graph_line, = self.graph_ax.plot([], [], color=CYAN, lw=1.4)
        self.graph_point, = self.graph_ax.plot([], [], "o", color=ORANGE, ms=3)
        self.graph_ax.set_xlim(self.flight.start, self.flight.end)
        self.graph_ax.set_ylim(min(self.flight.heights), max(self.flight.heights) + 1)
        self.graph_ax.axis("off")
        self.nav_ax = self.fig.add_axes((.855, .18, .105, .185), facecolor="none")
        self.nav_ax.set(xlim=(-1.16, 1.16), ylim=(-1.18, 1.18), aspect="equal")
        self.nav_ax.axis("off")
        size = 96
        u, v = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        self.nav_mask = u * u + v * v <= 1
        self.nav_normals = np.stack((u, v, np.sqrt(np.maximum(0, 1 - u * u - v * v))), axis=-1)
        self.nav_image = self.nav_ax.imshow(np.zeros((size, size, 4)), extent=(-1, 1, -1, 1), origin="lower")
        self.nav_ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor="#bacbd0", lw=1.2))
        self.nav_ax.plot([-.35, -.12, 0, .12, .35], [0, 0, -.09, 0, 0], color=ORANGE, lw=2)
        self.prograde, = self.nav_ax.plot([], [], "o", markerfacecolor="none", markeredgecolor=CYAN, ms=8)
        self.nav_label = self.nav_ax.text(0, -1.16, "ATTITUDE", color=WHITE, ha="center", fontsize=7)
        self.nav_angles = label(.91, .035, ha="center", fontsize=7)

    def _build_controls(self):
        self.slider_ax = self.fig.add_axes((.035, .092, .93, .014), facecolor="#283d48")
        self.slider = Slider(self.slider_ax, "", self.flight.start, self.flight.end,
                             valinit=self.flight.start, color=CYAN)
        self.slider.drawon = False
        self.slider.valtext.set_visible(False)
        self.slider.on_changed(self.seek)
        self.buttons = []
        specs = (("Pause", self.toggle_pause, .035, .075),
                 ("Replay", self.repeat, .12, .075),
                 ("Camera", self.cycle_camera, .205, .085),
                 ("Speed +", self.faster, .30, .085),
                 ("Speed -", self.slower, .395, .085))
        for text, callback, x, width in specs:
            axis = self.fig.add_axes((x, .025, width, .039))
            button = Button(axis, text, color="#18313e", hovercolor="#2b4b59", useblit=False)
            button.label.set_color(WHITE)
            button.label.set_fontsize(9)
            button.on_clicked(callback)
            for spine in axis.spines.values():
                spine.set_visible(False)
            self.buttons.append(button)
        self.status_text = self.fig.text(.505, .04, "", color=CYAN, fontsize=9)
        self.fig.text(.965, .04, "SPACE pause    C camera    H HUD    ← → seek",
                      color=MUTED, ha="right", fontsize=8)

    def _rig(self, t):
        f = self.flight
        t = min(t, f.ground_time)
        pos = f.position(t)
        pos[2] += self.length * .535
        axis, roll = f.attitude(t)
        event = f.drogue_time if f.drogue_time is not None else f.main_time
        if event is not None and t >= event:
            elapsed = t - event
            blend = smoothstep(elapsed / 2.5)
            relative_air = f.wind(t) - f.vector(f.velocities, t)
            # Tether tension suspends the airframe nose-up. Oscillations decay
            # after opening; they are illustrative, never written into CSV data.
            settling = smoothstep((f.ground_time - t) / 5)
            amplitude = (.20 * np.exp(-elapsed / 15) + .035) * settling
            tilt = np.array([amplitude * np.sin(1.9 * elapsed),
                             .7 * amplitude * np.cos(1.3 * elapsed), 1.])
            tilt[:2] += relative_air[:2] * .004
            axis = unit_vector(axis * (1 - blend) + unit_vector(tilt) * blend)
            roll *= 1 - blend
        return pos, rotation_from_axis(axis, roll)

    def _draw_recovery(self, t, pos, rotation):
        f = self.flight
        t = min(t, f.ground_time)
        attachment = pos + rotation @ [0, 0, self.length * .32]
        relative_air = f.wind(t) - f.vector(f.velocities, t)
        first_event = f.drogue_time if f.drogue_time is not None else f.main_time
        elapsed = max(0, t - (first_event if first_event is not None else t))
        main_open = f.opening(t, f.main_time)
        for main in (False, True):
            event = f.main_time if main else f.drogue_time
            if event is None or t <= event:
                continue
            visibility = float(smoothstep((t - event) / .35))
            if not main and self.drogue_cutaway:
                visibility *= 1 - main_open
            if visibility < .001:
                continue
            opening = f.opening(t, event)
            radius = self.length * (1.02 if main else .26)
            tether = self.length * (2.0 if main else 1.25)
            sway = np.array([np.sin(elapsed * (1.1 if main else 1.8)),
                             np.cos(elapsed * .9), 0]) * (.14 + .08 * (1 - main_open))
            lateral = relative_air * .055 + sway
            if not main:
                # Keep the smaller drogue outboard of the main's suspension lines.
                lateral += np.array([.65 * main_open, .36 * main_open, 0])
            direction = unit_vector(np.array([lateral[0], lateral[1], 1]))
            extension = smoothstep((t - event) / .8)
            center = attachment + direction * tether * extension
            basis = rotation_from_axis(direction)
            color = "#277fc4" if main else "#e97029"
            package = float(smoothstep((t - event) / .45))
            mesh = canopy_mesh(radius * package, opening, color, t * .45)
            self.scene.mesh(mesh, center, basis, alpha=visibility)
            width = radius * package * (.045 + .955 * opening)
            angles = np.arange(16) * 2 * np.pi / 16
            rims = np.column_stack((width * np.cos(angles), width * np.sin(angles), np.zeros(16)))
            rims = rims @ basis.T + center
            risers = np.stack((np.broadcast_to(attachment, rims.shape), rims), axis=1)
            self.scene.line_segments(risers, "#e9e6d4", .65, .86 * visibility)
            # The alternating gores define fabric panels. Coplanar decorative
            # seam lines caused depth flicker as the cloth moved, so omit them.
            self.scene.line([pos, attachment], WHITE, 1)

    def _prepare_smoke(self):
        f = self.flight
        emitted = np.arange(f.start, f.end, .125)
        powered = np.interp(emitted, f.times, f.column("thrust")) > max(1, f.maximum_thrust * .015)
        self._smoke_times = emitted[powered]
        positions = []
        for stamp in self._smoke_times:
            point, basis = self._rig(float(stamp))
            positions.append(point + basis @ [0, 0, -self.length * .62])
        self._smoke_origins = np.asarray(positions).reshape(-1, 3)
        self._smoke_wind = np.array([f.wind_displacement(t) for t in self._smoke_times]).reshape(-1, 3)

    def _draw_exhaust(self, t, pos, rotation):
        f = self.flight
        thrust = f.value("thrust", t)
        ratio = thrust / max(1, f.maximum_thrust)
        if ratio > .015:
            nozzle = pos + rotation @ [0, 0, -self.length * .53]
            flicker = 1 + .035 * np.sin(t * 9) + .018 * np.sin(t * 17)
            fade = float(smoothstep(ratio / .08))
            flame_length = self.length * (.35 + .9 * np.sqrt(ratio)) * flicker * fade
            for scale, radius, color, alpha in ((1, .17, "#ff681d", .7),
                                                (.8, .105, "#ffbd55", .9),
                                                (.5, .06, "#fff3bc", 1)):
                flame = surface_mesh([(0, radius * .7), (-flame_length * scale * .2, radius),
                                      (-flame_length * scale, .001)], 12, color=color)
                self.scene.mesh(flame, nozzle, rotation, alpha=alpha * fade, emissive=True)
        # Persistent particles: fixed births, continuous aging and wind advection.
        ages = t - self._smoke_times
        live = np.flatnonzero((ages > 0) & (ages < 9))
        displacement = f.wind_displacement(t)
        for i in live:
            age, emitted = ages[i], self._smoke_times[i]
            point = self._smoke_origins[i] + displacement - self._smoke_wind[i]
            point += np.array([np.sin(emitted * 9), np.cos(emitted * 13), .5]) * age * .17
            fade = float(smoothstep(age / .3) * (1 - smoothstep((age - 6) / 3)))
            self.scene.particle(point, .13 + age * .22, "#c5d0cc", .18 * fade)

    def _draw_wind(self, t, pos, distance):
        wind = self.flight.wind(t)
        speed = np.linalg.norm(wind)
        if speed < .05:
            return
        direction = wind / speed
        # A local field of short tracers communicates air motion with no fake
        # change to the exported trajectory. Density is intentionally restrained.
        for i in range(32):
            seed = np.array([np.sin(i * 93.2), np.cos(i * 71.7), np.sin(i * 37.9)])
            travelled = np.linalg.norm(self.flight.wind_displacement(t))
            phase = (travelled * .10 + i * .618) % 2 - 1
            point = pos + seed * distance * .65 + direction * phase * distance
            fade = float(smoothstep((1 - abs(phase)) / .3))
            self.scene.line([point, point + direction * (.25 + speed * .10)], CYAN, .65, .20 * fade)
        # Animated windsock beside the launch pad, anchored in world space.
        pole = np.array([8., 4., 6.])
        self.scene.line([[8, 4, 0], pole], "#c8d0c9", 1)
        end = pole + direction * 2 + [0, 0, -.35]
        sleeve = surface_mesh([(0, .33), (1.8, .14)], 12,
                              stripe=lambda k, j: "#f58d44" if k % 4 < 2 else "#efe4cb")
        self.scene.mesh(sleeve, pole, rotation_from_axis(end - pole))

    def _camera(self, t, pos, rotation):
        f = self.flight
        main = 0.0 if f.main_time is None else float(smoothstep((t - f.main_time + 1.2) / 3.8))
        drogue = 0.0 if f.drogue_time is None else float(smoothstep((t - f.drogue_time + .6) / 3))
        recovery = max(main, drogue * .55)
        target = pos + [0, 0, self.length * recovery * 1.05]
        # Framing expands continuously to contain airframe, risers and canopies.
        distance = self.length * (2.3 + recovery * 2.9)
        if self.camera_mode == 1:
            azimuth = -.85 + (t - f.start) * .055
        elif self.camera_mode == 2:
            distance *= 3.8
            azimuth = -1.0
        else:
            azimuth = -1.08 + .19 * np.sin(t * .055)
        elevation = .13 + .1 * smoothstep(pos[2] / 400)
        eye = target + distance * np.array([np.cos(azimuth) * np.cos(elevation),
                                            np.sin(azimuth) * np.cos(elevation),
                                            np.sin(elevation)])
        eye[2] = max(1, eye[2])
        return eye, target, distance

    def _draw_navball(self, rotation, velocity):
        world_up_in_body = rotation.T @ [0, 0, 1]
        heights = self.nav_normals @ world_up_in_body
        image = np.empty((*heights.shape, 4))
        image[:, :, :3] = np.where((heights > 0)[:, :, None],
                                   np.array([.14, .39, .55]), np.array([.46, .30, .18]))
        horizon = abs(heights) < .025
        image[horizon, :3] = [.88, .83, .65]
        for band in (-.5, .5):
            image[abs(heights - band) < .012, :3] *= 1.4
        image[:, :, 3] = self.nav_mask.astype(float)
        self.nav_image.set_data(np.clip(image, 0, 1))
        relative = rotation.T @ unit_vector(velocity)
        self.prograde.set_data([relative[0] * .85], [relative[1] * .85])
        self.prograde.set_marker("x" if relative[2] < 0 else "o")
        self.prograde.set_visible(bool(np.linalg.norm(velocity) > .3))

    def _draw_clouds(self, t):
        # Composite cloud sprites once into a small texture. Eighteen separate
        # resampled image artists were more expensive than the entire airframe.
        sheet = Image.new("RGBA", (512, 288))
        for i, sprite in enumerate(self._cloud_sprites):
            center = np.array([np.sin(i * 17.3) * 6500,
                               np.cos(i * 11.7) * 6500, 950 + (i % 3) * 250.])
            center += self.flight.wind_displacement(t)
            screen, depth = self.scene.project(center[None, :])
            visible = bool(depth[0] > 150 and np.all(abs(screen[0]) < 1.5))
            if visible:
                width = (400 + i % 4 * 80) * 1.75 / depth[0] / self.scene.aspect
                height = width * self.scene.aspect * .34
                x, y = screen[0]
                w, h = max(1, round(width * 512)), max(1, round(height * 288))
                if w > 2048 or h > 1152:
                    continue
                key = (i, w, h)
                if key not in self._cloud_sizes:
                    if len(self._cloud_sizes) > 160:
                        self._cloud_sizes.clear()
                    self._cloud_sizes[key] = sprite.resize((w, h), Image.Resampling.BILINEAR)
                left = round((x - width + 1) * 256)
                top = round((1 - y - height) * 144)
                sheet.alpha_composite(self._cloud_sizes[key], dest=(left, top))
        self.cloud_layer.set_data(np.asarray(sheet))

    def draw_time(self, t):
        self.t = float(np.clip(t, self.flight.start, self.flight.end))
        t, f = self.t, self.flight
        pos, rotation = self._rig(t)
        eye, target, distance = self._camera(t, pos, rotation)
        aspect = max(.5, self.ax.bbox.width / max(1, self.ax.bbox.height))
        self.scene.clear()
        self.scene.camera(eye, target, aspect)
        self._draw_clouds(t)
        self.scene.mesh(self.landscape)
        # Launch rail and gantry supply a stable scale reference during liftoff.
        self.scene.line([[.5, .5, 0], [.5, .5, 5]], "#c3ccc5", 1.2)
        for height in np.arange(0, 5, .65):
            self.scene.line([[.5, .5, height], [.8, .5, height + .6], [.5, .5, height + .6]],
                            "#748a8d", .6)
        # Wide tracking exposes the full recorded trajectory for context.
        if self.camera_mode == 2:
            path_times = np.linspace(f.start, t, 170)
            path = np.array([f.position(sample) for sample in path_times])
            self.scene.line(path, CYAN, .7, .3)
        self._draw_wind(t, pos, distance)
        self._draw_exhaust(t, pos, rotation)
        self.scene.mesh(self.rocket, pos, rotation)
        self._draw_recovery(t, pos, rotation)
        self.scene.finish()
        self.stage_text.set_text(f.stage(t))
        self.clock_text.set_text(f"MISSION TIME   T+{t - f.start:07.2f} s")
        recent = [(name, when) for name, when, _ in f.events
                  if name not in ("SIM_WARN", "SIMULATION_END") and 0 <= t - when < 2.8]
        self.event_text.set_text(recent[-1][0].replace("_", " ") if recent else "")
        mode = ("CINEMATIC CHASE", "ORBIT CAMERA", "WIDE TRACKING")[self.camera_mode]
        self.camera_text.set_text(f"{mode}   /   {self.speed:g}×")
        height = f.value("altitude", t) - f.heights[0]
        velocity = f.vector(f.velocities, t)
        self.telemetry["ALTITUDE AGL"].set_text(f"{height:,.0f} ft")
        speed = f.value("velocity", t, float(np.linalg.norm(velocity)))
        self.telemetry["SPEED"].set_text(f"{speed:,.1f} m/s")
        vz = f.value("vertical_velocity", t, velocity[2])
        self.telemetry["VERTICAL"].set_text(f"{vz:+.1f} m/s")
        self.telemetry["WIND FROM"].set_text(
            f"{f.value('wind_speed', t):.1f} m/s / {np.degrees(f.circular('wind_direction', t)) % 360:03.0f}°"
            if f.has_wind else "NO WIND DATA")
        recovery = ((f.drogue_time is not None and t >= f.drogue_time)
                    or (f.main_time is not None and t >= f.main_time))
        source = "RECORDED ATTITUDE" if f.has_attitude else "ATTITUDE FROM TRAJECTORY"
        if recovery:
            source = "ILLUSTRATIVE RECOVERY SWAY"
        path_source = "RECORDED 3D PATH" if f.has_position else "VERTICAL PATH / LATERAL DATA MISSING"
        self.provenance.set_text(f"{path_source}  ·  {source}  ·  {f.recovery_source}\n"
                                 "Illustrative airframe, terrain and recovery rig; geometry is not certified.")
        axis = rotation[:, 2]
        elevation = np.degrees(np.arcsin(np.clip(axis[2], -1, 1)))
        heading = np.degrees(np.arctan2(axis[0], axis[1])) % 360
        self.nav_angles.set_text(f"EL {elevation:+.0f}° / HDG {heading:03.0f}°")
        self._draw_navball(rotation, velocity)
        before = f.times <= t
        self.graph_line.set_data(np.append(f.times[before], t),
                                  np.append(f.heights[before], f.value("altitude", t)))
        self.graph_point.set_data([t], [f.value("altitude", t)])
        state = "FINISHED" if self.finished else "PAUSED" if self.paused else "PLAYING"
        self.status_text.set_text(f"{state}  /  {self.speed:g}×")
        if hasattr(self, "slider"):
            self._setting_slider = True
            # Do not schedule an extra figure redraw during each movie frame.
            old_drawon = self.slider.drawon
            self.slider.drawon = False
            self.slider.set_val(t)
            self.slider.drawon = old_drawon
            self._setting_slider = False
        return ()

    def draw_sample(self, index):
        """Compatibility helper for inspecting an exact CSV sample."""
        return self.draw_time(self.flight.times[index])

    def _tick(self, _frame=None):
        started = time.perf_counter()
        now = time.monotonic()
        elapsed = 0.0 if self._last_clock is None else max(0.0, now - self._last_clock)
        self._last_clock = now
        if self.closed or self._exporting:
            return False
        changed = self._resize_pending
        self._resize_pending = False
        if self._pending_seek is not None:
            self.t = self._pending_seek
            self._pending_seek = None
            self.finished = False
            changed = True
        elif not (self.paused or self._scrubbing):
            # Slow frames may slow playback slightly; never teleport after an
            # OS stall, window resize, or a long button/slider interaction.
            elapsed = min(elapsed, .05) if elapsed < .25 else 0.0
            self.t = min(self.flight.end, self.t + elapsed * self.speed)
            changed = True
            if self.t >= self.flight.end:
                self.finished = self.paused = True
                self.buttons[0].label.set_text("Play")
                if self._presenter is not None:
                    self._presenter.background = None
        if changed:
            self.draw_time(self.t)
            if self._presenter is not None:
                self._presenter.present()
            else:
                self.fig.canvas.draw_idle()
        cost = time.perf_counter() - started
        if changed:
            self._frame_costs.append(cost)
        if self._timer is not None:
            # The old timer waited a full frame interval AFTER a slow redraw.
            self._timer.interval = max(1, round((1 / self.fps - cost) * 1000)) if changed else 25
        return True

    def _on_resize(self, _event):
        self._resize_pending = True
        self._last_clock = None
        if self._presenter is not None:
            self._presenter.background = None

    def _on_pointer_down(self, event):
        if event.inaxes is self.slider_ax and event.button == 1:
            self._scrubbing = True
            self._last_clock = None

    def _on_pointer_up(self, _event):
        if self._scrubbing:
            self._scrubbing = False
            self._last_clock = None

    def _refresh(self):
        self._last_clock = None
        self.draw_time(self.t)
        self.fig.canvas.draw_idle()

    def seek(self, value):
        if self._setting_slider:
            return
        self.finished = False
        value = float(np.clip(value, self.flight.start, self.flight.end))
        self._last_clock = None
        if self._timer is not None:
            self._pending_seek = value
        else:
            self.draw_time(value)
            self.fig.canvas.draw_idle()

    def toggle_pause(self, _event=None):
        if self.finished:
            self.repeat()
            return
        self.paused = not self.paused
        self.buttons[0].label.set_text("Play" if self.paused else "Pause")
        self._refresh()

    def repeat(self, _event=None):
        self.paused = self.finished = False
        self._pending_seek = None
        self._scrubbing = False
        self.buttons[0].label.set_text("Pause")
        self.t = self.flight.start
        self._refresh()

    def cycle_camera(self, _event=None):
        self.camera_mode = (self.camera_mode + 1) % 3
        self._refresh()

    def faster(self, _event=None):
        self.speed = min(64, self.speed * 2)
        self._refresh()

    def slower(self, _event=None):
        self.speed = max(.25, self.speed / 2)
        self._refresh()

    def _on_key(self, event):
        if event.key in (" ", "space"):
            self.toggle_pause()
        elif event.key in ("r", "R"):
            self.repeat()
        elif event.key in ("c", "C"):
            self.cycle_camera()
        elif event.key in ("+", "="):
            self.faster()
        elif event.key == "-":
            self.slower()
        elif event.key in ("left", "right"):
            self.seek(self.t + (5 if event.key == "right" else -5))
        elif event.key in ("h", "H"):
            self.hud_visible = not self.hud_visible
            for artist in self.hud:
                artist.set_visible(self.hud_visible)
            self.graph_ax.set_visible(self.hud_visible)
            self.nav_ax.set_visible(self.hud_visible)
            self.fig.canvas.draw_idle()

    def _on_close(self, _event):
        self.closed = True
        if self._timer is not None:
            self._timer.stop()

    def _moving_artists(self):
        return [self.cloud_layer, self.scene.polygons, self.scene.lines, self.scene.particles,
                *self.hud, self.graph_ax, self.nav_ax, self.slider_ax, self.status_text]

    def show(self):
        if not self.software and importlib.util.find_spec("pyglet"):
            player = GPUFlightWindow(self)
            plt.close(self.fig)
            self.closed = False
            player.run()
            return
        self._presenter = FramePresenter(self.fig, self._moving_artists())
        self.fig.canvas.draw()
        self._last_clock = None
        self._timer = self.fig.canvas.new_timer(interval=1)
        self._timer.add_callback(self._tick)
        self._timer.start()
        plt.show()

    def export(self, path, start=None, end=None, dpi=110):
        start = self.flight.start if start is None else max(start, self.flight.start)
        end = self.flight.end if end is None else min(end, self.flight.end)
        if end <= start:
            raise ValueError("Export end must be later than start, within the flight.")
        suffix = path.suffix.lower()
        if suffix == ".mp4":
            if not writers.is_available("ffmpeg"):
                raise ValueError("MP4 export needs ffmpeg on PATH. Use --export flight.gif instead.")
            writer = FFMpegWriter(fps=self.fps, codec="libx264",
                                  extra_args=["-pix_fmt", "yuv420p", "-crf", "19"],
                                  metadata={"title": "Loxodon-1 cinematic flight replay"})
        elif suffix == ".gif":
            writer = PillowWriter(fps=self.fps)
        else:
            raise ValueError("Movie export must end in .mp4 or .gif.")
        frame_count = max(2, math.ceil((end - start) / self.speed * self.fps) + 1)
        if suffix == ".gif" and frame_count > 900:
            raise ValueError("Use MP4 for a long movie, or choose a shorter GIF with --start and --end.")
        old_state = self.t, self.paused, self.finished
        self._exporting = True
        if self._timer is not None:
            self._timer.stop()
        if self._presenter is not None:
            self._presenter.suspend()
        # Movie frames follow simulation time at a constant speed, independent
        # of how long rendering takes. Interactive controls are hidden in exports.
        for button in self.buttons:
            button.ax.set_visible(False)
        self.slider_ax.set_visible(False)
        self.status_text.set_visible(False)
        self.fig.texts[-1].set_visible(False)
        try:
            with writer.saving(self.fig, str(path), dpi):
                for frame, t in enumerate(np.linspace(start, end, frame_count)):
                    self.draw_time(t)
                    writer.grab_frame()
                    if frame % max(1, self.fps * 2) == 0:
                        print(f"Rendering {100 * frame / (frame_count - 1):.0f}%", flush=True)
        finally:
            for button in self.buttons:
                button.ax.set_visible(True)
            self.slider_ax.set_visible(True)
            self.status_text.set_visible(True)
            self.fig.texts[-1].set_visible(True)
            self.t, self.paused, self.finished = old_state
            self._exporting = False
            self._last_clock = None
            self.draw_time(self.t)
            if self._presenter is not None:
                self.fig.canvas.mpl_disconnect(self._presenter.connection)
                self._presenter = FramePresenter(self.fig, self._moving_artists())
                self.fig.canvas.draw_idle()
            if self._timer is not None and not self.closed:
                self._timer.start()
        print(f"Saved movie: {path}", flush=True)



class GLFlightRenderer:
    """Persistent GPU buffers, real depth testing and multisampled perspective."""

    VERTEX = """#version 330 core
    in vec3 position; in vec3 normal; in vec4 color;
    uniform mat4 vp; uniform mat4 model;
    out vec3 world_position; out vec3 world_normal; out vec4 tint;
    void main() {
        vec4 world = model * vec4(position, 1.0);
        world_position = world.xyz; world_normal = mat3(model) * normal;
        tint = color; gl_Position = vp * world;
    }"""
    FRAGMENT = """#version 330 core
    in vec3 world_position; in vec3 world_normal; in vec4 tint;
    uniform vec3 eye; uniform float opacity; uniform float emissive;
    out vec4 final_color;
    void main() {
        vec3 rgb = tint.rgb;
        if (emissive < 0.5) {
            vec3 n = normalize(world_normal);
            vec3 light = normalize(vec3(0.35, -0.6, 0.8));
            float diffuse = 0.50 + 0.50 * max(dot(n, light), 0.0);
            vec3 half_vector = normalize(light + normalize(eye - world_position));
            float shine = 0.10 * pow(max(dot(n, half_vector), 0.0), 48.0);
            rgb = rgb * diffuse + shine;
        }
        float fog = clamp((length(world_position - eye) - 100.0) / 6500.0, 0.0, 0.85);
        rgb = mix(rgb, vec3(0.46, 0.65, 0.72), fog);
        final_color = vec4(rgb, tint.a * opacity);
    }"""
    SCREEN_VERTEX = """#version 330 core
    in vec2 position; out vec2 uv;
    void main() { uv = position; gl_Position = vec4(position, 0.0, 1.0); }
    """
    SKY_FRAGMENT = """#version 330 core
    in vec2 uv; out vec4 final_color;
    void main() {
        vec3 low = vec3(0.69,0.79,0.80); vec3 high = vec3(0.12,0.34,0.52);
        final_color = vec4(mix(low, high, clamp(uv.y * 0.5 + 0.5,0.0,1.0)),1.0);
    }"""
    NAV_FRAGMENT = """#version 330 core
    in vec2 uv; uniform vec3 body_up; uniform vec3 velocity; out vec4 final_color;
    void main() {
        float radius = length(uv); if (radius > 1.0) discard;
        vec3 sphere = vec3(uv, sqrt(max(0.0,1.0-dot(uv,uv))));
        float height = dot(sphere, body_up);
        vec3 rgb = height > 0.0 ? vec3(0.14,0.39,0.55) : vec3(0.46,0.30,0.18);
        if(abs(height)<0.018) rgb = vec3(0.88,0.83,0.65);
        if(abs(abs(height)-0.5)<0.009) rgb *= 1.35;
        if(radius>0.97) rgb=vec3(0.72,0.80,0.82);
        if(abs(uv.y)<0.018 && abs(uv.x)<0.35) rgb=vec3(1.0,0.7,0.35);
        float marker=length(uv-velocity.xy*0.85);
        if(abs(marker-0.055)<0.012) rgb=velocity.z>0.0 ? vec3(0.4,0.9,0.95) : vec3(1.0,0.7,0.35);
        final_color=vec4(rgb,1.0);
    }"""
    CLOUD_VERTEX = """#version 330 core
    in vec3 position; in vec2 corner; in vec2 texcoord; in vec2 size;
    uniform mat4 vp; uniform vec3 right; uniform vec3 up; uniform vec3 drift;
    out vec2 uv;
    void main() {
        vec3 world=position+drift+right*corner.x*size.x+up*corner.y*size.y;
        gl_Position=vp*vec4(world,1.0); uv=texcoord;
    }"""
    CLOUD_FRAGMENT = """#version 330 core
    in vec2 uv; uniform sampler2D cloud; out vec4 final_color;
    void main() { final_color=texture(cloud,uv); if(final_color.a<0.004) discard; }
    """
    POINT_VERTEX = """#version 330 core
    in vec3 position; in vec4 color; in float radius;
    uniform mat4 vp; uniform float viewport_height; out vec4 tint;
    void main() {
        gl_Position=vp*vec4(position,1.0); tint=color;
        gl_PointSize=clamp(radius*vp[1][1]*viewport_height/gl_Position.w,1.0,128.0);
    }"""
    POINT_FRAGMENT = """#version 330 core
    in vec4 tint; out vec4 final_color;
    void main() {
        float d=length(gl_PointCoord*2.0-1.0); if(d>1.0) discard;
        final_color=vec4(tint.rgb,tint.a*exp(-d*d*3.5)*(1.0-smoothstep(0.6,1.0,d)));
    }"""

    def __init__(self, window, replay):
        import pyglet
        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram
        self.pyglet, self.gl, self.window, self.replay = pyglet, gl, window, replay
        self.program = ShaderProgram(Shader(self.VERTEX, "vertex"), Shader(self.FRAGMENT, "fragment"))
        self.sky = ShaderProgram(Shader(self.SCREEN_VERTEX, "vertex"), Shader(self.SKY_FRAGMENT, "fragment"))
        self.nav = ShaderProgram(Shader(self.SCREEN_VERTEX, "vertex"), Shader(self.NAV_FRAGMENT, "fragment"))
        self.cloud = ShaderProgram(Shader(self.CLOUD_VERTEX, "vertex"), Shader(self.CLOUD_FRAGMENT, "fragment"))
        self.smoke = ShaderProgram(Shader(self.POINT_VERTEX, "vertex"), Shader(self.POINT_FRAGMENT, "fragment"))
        quad = [-1.,-1.,1.,-1.,1.,1.,-1.,-1.,1.,1.,-1.,1.]
        self.sky_quad = self.sky.vertex_list(6, gl.GL_TRIANGLES, position=("f", quad))
        self.nav_quad = self.nav.vertex_list(6, gl.GL_TRIANGLES, position=("f", quad))
        self._mesh_buffers = []
        self._line_buffer = self._point_buffer = None
        self._original_scene = replay.scene
        self._angles, self._corners, self._uvs, self._sizes = [], [], [], []
        corners = np.asarray(quad).reshape(6,2)
        for i in range(18):
            center = [np.sin(i*17.3)*6500, np.cos(i*11.7)*6500, 950+(i%3)*250]
            self._angles.extend(center * 6)
            self._corners.extend(corners.ravel())
            self._uvs.extend(((corners+1)*.5).ravel())
            self._sizes.extend([400+i%4*80, (400+i%4*80)*.34]*6)
        self.cloud_buffer = self.cloud.vertex_list(108, gl.GL_TRIANGLES,
            position=("f", self._angles), corner=("f", self._corners),
            texcoord=("f", self._uvs), size=("f", self._sizes))
        pixels = (cloud_sprite(3)*255).astype(np.uint8)
        self.texture = pyglet.image.ImageData(128,64,"RGBA",pixels.tobytes(),pitch=128*4).get_texture()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glClearColor(.035,.07,.1,1)
        self._closed = False

    @staticmethod
    def _matrix(value):
        return tuple(np.asarray(value,dtype=np.float32).T.ravel())

    def mesh(self, mesh, position=None, rotation=None, alpha: float = 1.0, emissive=False):
        gl = self.gl
        slot = self._mesh_slot
        self._mesh_slot += 1
        if slot >= len(self._mesh_buffers):
            self._mesh_buffers.append({"source": None,"buffer": None})
        cached = self._mesh_buffers[slot]
        if cached["source"] is not mesh:
            indices, owners = [], []
            for owner, face in enumerate(mesh.faces):
                for j in range(1,len(face)-1):
                    indices.extend((face[0],face[j],face[j+1]));owners.extend((owner,owner,owner))
            count=len(indices)
            buffer=cached["buffer"]
            if buffer is None:
                buffer=self.program.vertex_list(count,gl.GL_TRIANGLES,position="f",normal="f",color="f")
                cached["buffer"]=buffer
            elif buffer.count != count:
                buffer.resize(count)
            getattr(buffer, "position")[:] = mesh.vertices[indices].astype(np.float32).ravel()
            getattr(buffer, "normal")[:] = mesh.normals[owners].astype(np.float32).ravel()
            rgba=np.column_stack((mesh.palette[owners],np.ones(count)))
            getattr(buffer, "color")[:] = rgba.astype(np.float32).ravel()
            cached["source"]=mesh
        model=np.eye(4)
        if rotation is not None:model[:3,:3]=rotation
        if position is not None:model[:3,3]=position
        self.program.use()
        self.program["model"]=self._matrix(model)
        self.program["opacity"]=float(alpha)
        self.program["emissive"]=1.0 if emissive else 0.0
        gl.glDepthMask(gl.GL_FALSE if alpha<.999 else gl.GL_TRUE)
        cached["buffer"].draw(gl.GL_TRIANGLES)
        gl.glDepthMask(gl.GL_TRUE)

    def line(self, points, color=WHITE, width=.65, alpha: float = 1.0):
        points=np.asarray(points).reshape(-1,3)
        if len(points)>1:
            self.line_segments(np.stack((points[:-1],points[1:]),axis=1),color,width,alpha)

    def line_segments(self, segments, color=WHITE, width=.65, alpha: float = 1.0):
        points=np.asarray(segments).reshape(-1,3)
        self._lines.extend(points)
        self._line_colors.extend([colors.to_rgba(color,alpha)]*len(points))

    def particle(self, position, radius, color, alpha):
        self._points.append(position)
        self._point_radii.append(radius)
        self._point_colors.append(colors.to_rgba(color,alpha))

    def _draw_lines_and_smoke(self):
        gl=self.gl
        if self._lines:
            count=len(self._lines)
            if self._line_buffer is None:
                self._line_buffer=self.program.vertex_list(count,gl.GL_LINES,position="f",normal="f",color="f")
            elif self._line_buffer.count!=count:self._line_buffer.resize(count)
            getattr(self._line_buffer, "position")[:]=np.asarray(self._lines,dtype=np.float32).ravel()
            getattr(self._line_buffer, "normal")[:]=np.zeros(count*3,dtype=np.float32)
            getattr(self._line_buffer, "color")[:]=np.asarray(self._line_colors,dtype=np.float32).ravel()
            self.program.use();self.program["model"]=self._matrix(np.eye(4))
            self.program["emissive"]=1.0;self.program["opacity"]=1.0
            gl.glDepthMask(gl.GL_FALSE);self._line_buffer.draw(gl.GL_LINES)
        if self._points:
            count=len(self._points)
            if self._point_buffer is None:
                self._point_buffer=self.smoke.vertex_list(count,gl.GL_POINTS,position="f",color="f",radius="f")
            elif self._point_buffer.count!=count:self._point_buffer.resize(count)
            points=np.asarray(self._points)
            order=np.argsort(-np.linalg.norm(points-self.eye,axis=1),kind="stable")
            getattr(self._point_buffer, "position")[:]=points[order].astype(np.float32).ravel()
            getattr(self._point_buffer, "color")[:]=np.asarray(self._point_colors,dtype=np.float32)[order].ravel()
            getattr(self._point_buffer, "radius")[:]=np.asarray(self._point_radii,dtype=np.float32)[order]
            self.smoke.use();self.smoke["vp"]=self._matrix(self.vp)
            self.smoke["viewport_height"]=float(self._viewport_height)
            gl.glDepthMask(gl.GL_FALSE);self._point_buffer.draw(gl.GL_POINTS)
        gl.glDepthMask(gl.GL_TRUE)

    def draw(self,t,show_nav=True):
        gl,r=self.gl,self.replay
        # UI batches restore their own blend state. Re-establish the world
        # state every frame so transparent clouds/wind never turn opaque.
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_CULL_FACE)
        framebuffer_w,framebuffer_h=self.window.get_framebuffer_size()
        scale=framebuffer_w/max(1,self.window.width)
        width,height=framebuffer_w,max(1,framebuffer_h-round(125*scale))
        self._viewport_height=height
        gl.glViewport(0,round(70*scale),width,height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.sky.use();self.sky_quad.draw(gl.GL_TRIANGLES)
        gl.glEnable(gl.GL_DEPTH_TEST)
        pos,rotation=r._rig(t)
        eye,target,distance=r._camera(t,pos,rotation)
        self.eye=eye
        forward=unit_vector(target-eye)
        right=unit_vector(np.cross(forward,[0,0,1]),[1,0,0]);up=np.cross(right,forward)
        view=np.eye(4);view[:3,:3]=np.array([right,up,-forward]);view[:3,3]=-view[:3,:3]@eye
        near,far=.08,60000.
        projection=np.zeros((4,4));projection[0,0]=1.75/(width/height);projection[1,1]=1.75
        projection[2,2]=(far+near)/(near-far);projection[2,3]=2*far*near/(near-far);projection[3,2]=-1
        self.vp=projection@view
        self.program.use();self.program["vp"]=self._matrix(self.vp);self.program["eye"]=tuple(eye)
        self._mesh_slot=0
        self._lines,self._line_colors,self._points,self._point_colors,self._point_radii=[],[],[],[],[]
        self.mesh(r.landscape)
        self.cloud.use();self.cloud["vp"]=self._matrix(self.vp)
        self.cloud["right"]=tuple(right);self.cloud["up"]=tuple(up)
        self.cloud["drift"]=tuple(r.flight.wind_displacement(t));self.cloud["cloud"]=0
        gl.glActiveTexture(gl.GL_TEXTURE0);gl.glBindTexture(gl.GL_TEXTURE_2D,self.texture.id)
        gl.glDepthMask(gl.GL_FALSE);self.cloud_buffer.draw(gl.GL_TRIANGLES);gl.glDepthMask(gl.GL_TRUE)
        self.mesh(r.rocket,pos,rotation)
        r.scene=self
        try:
            self.line([[.5,.5,0],[.5,.5,5]],"#c3ccc5",1.2)
            r._draw_wind(t,pos,distance)
            r._draw_exhaust(t,pos,rotation)
            r._draw_recovery(t,pos,rotation)
            if r.camera_mode==2:
                self.line([r.flight.position(sample) for sample in np.linspace(r.flight.start,t,160)],CYAN,.7,.3)
        finally:r.scene=self._original_scene
        self._draw_lines_and_smoke()
        gl.glDisable(gl.GL_DEPTH_TEST)
        if show_nav:
            gl.glViewport(round((self.window.width-190)*scale),round(120*scale),round(140*scale),round(140*scale))
            self.nav.use();self.nav["body_up"]=tuple(rotation.T@np.array([0.,0.,1.]))
            velocity=r.flight.vector(r.flight.velocities,t)
            self.nav["velocity"]=tuple(rotation.T@unit_vector(velocity))
            self.nav_quad.draw(gl.GL_TRIANGLES)
        gl.glViewport(0,0,framebuffer_w,framebuffer_h)

    def close(self):
        if self._closed:return
        self._closed=True
        for item in self._mesh_buffers:
            if item["buffer"] is not None:item["buffer"].delete()
        for item in (self._line_buffer,self._point_buffer,self.sky_quad,self.nav_quad,self.cloud_buffer):
            if item is not None:item.delete()
        for program in (self.program,self.sky,self.nav,self.cloud,self.smoke):program.delete()
        self.texture.delete()


class GPUFlightWindow:
    """V-synchronised native player; all controls and text stay on the GPU."""

    def __init__(self, replay):
        import pyglet

        # This must precede importing pyglet.gl/window: GL error checks add a
        # driver round trip after every call and are meant for development.
        pyglet.options["debug_gl"] = False
        pyglet.options["dpi_scaling"] = "stretch"
        from pyglet import gl, shapes
        from pyglet.window import key, mouse

        self.pyglet, self.gl, self.key, self.mouse = pyglet, gl, key, mouse
        self.replay = replay
        self._closing = False
        self._scheduled = False
        self._reset_clock = True
        self._scrubbing = False
        self._last_text = 0.0
        self._hovered = None
        options = dict(major_version=3, minor_version=3, forward_compatible=True,
                       depth_size=24, double_buffer=True)
        try:
            config = gl.Config(**options, sample_buffers=1, samples=4)
            self.window = pyglet.window.Window(width=1280, height=720,
                                               caption="Loxodon-1 | Flight Cinema",
                                               config=config, resizable=True,
                                               vsync=True)
        except pyglet.window.NoSuchConfigException:
            config = gl.Config(**options)
            self.window = pyglet.window.Window(width=1280, height=720,
                                               caption="Loxodon-1 | Flight Cinema",
                                               config=config, resizable=True,
                                               vsync=True)
        self.window.set_minimum_size(960, 600)
        try:
            renderer_type = globals()["GLFlightRenderer"]
            self.painter = renderer_type(self.window, replay)
        except Exception:
            self.window.close()
            raise

        self._chrome = pyglet.graphics.Batch()
        self._hud = pyglet.graphics.Batch()
        self._shapes = []
        self._labels = []
        self._button_items = []
        self._hud_labels = {}
        self._white = (235, 243, 246, 255)
        self._muted = (151, 177, 188, 255)
        self._cyan = (112, 220, 224, 255)
        self._orange = (255, 179, 90, 255)

        def rect(x, y, width, height, color, batch=None):
            shape = shapes.Rectangle(x, y, width, height, color=color,
                                     batch=self._chrome if batch is None else batch)
            self._shapes.append(shape)
            return shape

        def label(text, x: float = 0.0, y: float = 0.0, size=11, color=None, anchor="left",
                  bold=False, batch=None):
            artist = pyglet.text.Label(text, font_name="Helvetica Neue",
                                       font_size=size, weight="bold" if bold else "normal",
                                       color=self._white if color is None else color,
                                       x=x, y=y, anchor_x=("center" if anchor == "center" else "right" if anchor == "right" else "left"), anchor_y="center",
                                       batch=self._chrome if batch is None else batch)
            self._labels.append(artist)
            return artist

        self._header = rect(0, 0, 1280, 55, (7, 18, 29, 255))
        self._footer = rect(0, 0, 1280, 76, (7, 18, 29, 255))
        self._header_rule = rect(0, 0, 1280, 1, (39, 71, 84, 255))
        self._footer_rule = rect(0, 75, 1280, 1, (39, 71, 84, 255))
        self._brand = label("L O X O D O N", 30, 0, 19, bold=True)
        self._subtitle = label("FLIGHT CINEMA", 236, 0, 10, self._cyan)
        source = str(getattr(replay, "source_name", "LIVE FLIGHT REPLAY"))[:64]
        self._source = label(source, 0, 0, 9, self._muted, anchor="right")
        self._track = rect(30, 58, 1220, 3, (38, 62, 75, 255))
        self._track_fill = rect(30, 58, 0, 3, self._cyan)
        self._knob = shapes.Circle(30, 59.5, 5, color=self._white, batch=self._chrome)
        self._shapes.append(self._knob)
        specs = (("Pause", self.toggle_pause, 30, 76),
                 ("Replay", self.repeat, 114, 76),
                 ("Camera", self.cycle_camera, 198, 88),
                 ("Speed +", self.faster, 294, 80),
                 ("Speed −", self.slower, 382, 80))
        for text, callback, x, width in specs:
            background = rect(x, 11, width, 30, (24, 49, 62, 255))
            caption = label(text, x + width / 2, 26, 10, anchor="center")
            self._button_items.append((background, caption, callback))
        self._state = label("", 480, 26, 10, self._cyan)
        self._help = label("SPACE pause    C camera    H HUD    ← → seek", 0, 26,
                           9, self._muted, anchor="right")

        for name, size, color, anchor in (
            ("stage", 15, self._white, "left"),
            ("clock", 11, self._white, "left"),
            ("event", 10, self._orange, "left"),
            ("camera", 10, self._white, "right"),
            ("alt_title", 9, self._muted, "right"),
            ("alt", 22, self._white, "right"),
            ("speed_title", 9, self._muted, "right"),
            ("speed", 22, self._white, "right"),
            ("vz_title", 9, self._muted, "right"),
            ("vz", 19, self._white, "right"),
            ("wind_title", 9, self._muted, "right"),
            ("wind", 12, self._white, "right"),
            ("nav", 9, self._white, "center"),
            ("angles", 9, self._muted, "center"),
            ("provenance", 8, self._muted, "left"),
        ):
            self._hud_labels[name] = label("", size=size, color=color,
                                           anchor=anchor, batch=self._hud)
        for name, text in (("alt_title", "ALTITUDE AGL"),
                           ("speed_title", "SPEED"),
                           ("vz_title", "VERTICAL"),
                           ("wind_title", "WIND FROM"),
                           ("nav", "ATTITUDE")):
            self._hud_labels[name].text = text

        self._layout(self.window.width, self.window.height)
        self._refresh_text(force=True)
        self.window.push_handlers(self)

    def _layout(self, width, height):
        self._header.position = (0, height - 55)
        self._header.width = width
        self._header_rule.position = (0, height - 55)
        self._header_rule.width = width
        self._footer.width = self._footer_rule.width = width
        self._brand.y = self._subtitle.y = self._source.y = height - 28
        self._source.x = width - 30
        self._source.text = (str(getattr(self.replay, "source_name", "LIVE FLIGHT REPLAY"))[:64]
                             if width >= 1100 else "")
        self._help.x = width - 30
        self._help.text = ("SPACE pause    C camera    H HUD    ← → seek"
                           if width >= 1200 else "SPACE pause  ·  C camera")
        self._timeline_start = 30.0
        self._timeline_end = float(width - 30)
        self._track.width = self._timeline_end - self._timeline_start
        places = {
            "stage": (30, height - 86), "clock": (30, height - 111),
            "event": (30, height - 142), "camera": (width - 30, height - 83),
            "alt_title": (width - 30, height - 114),
            "alt": (width - 30, height - 143),
            "speed_title": (width - 30, height - 184),
            "speed": (width - 30, height - 213),
            "vz_title": (width - 30, height - 254),
            "vz": (width - 30, height - 281),
            "wind_title": (width - 30, height - 321),
            "wind": (width - 30, height - 346),
            "nav": (width - 120, 106), "angles": (width - 120, 90),
            "provenance": (30, 95),
        }
        for name, (x, y) in places.items():
            self._hud_labels[name].x = x
            self._hud_labels[name].y = y
        self._update_timeline()

    def _update_timeline(self):
        f = self.replay.flight
        duration = max(1e-9, f.end - f.start)
        amount = max(0.0, min(1.0, (self.replay.t - f.start) / duration))
        filled = amount * (self._timeline_end - self._timeline_start)
        self._track_fill.width = filled
        self._knob.x = self._timeline_start + filled

    def _refresh_text(self, force=False):
        import math
        import time

        now = time.monotonic()
        if not force and now - self._last_text < .1:
            return
        self._last_text = now
        replay, f = self.replay, self.replay.flight
        t = replay.t
        labels = self._hud_labels
        labels["stage"].text = f.stage(t)
        labels["clock"].text = f"MISSION TIME   T+{t - f.start:07.2f} s"
        recent = [(name, when) for name, when, _ in f.events
                  if name not in ("SIM_WARN", "SIMULATION_END") and 0 <= t - when < 2.8]
        labels["event"].text = recent[-1][0].replace("_", " ") if recent else ""
        labels["camera"].text = ("CINEMATIC CHASE", "ORBIT CAMERA", "WIDE TRACKING")[replay.camera_mode]
        velocity = f.vector(f.velocities, t)
        total_speed = math.sqrt(sum(float(v) ** 2 for v in velocity))
        labels["alt"].text = f"{f.value('altitude', t) - f.heights[0]:,.0f} ft"
        labels["speed"].text = f"{f.value('velocity', t, total_speed):,.1f} m/s"
        labels["vz"].text = f"{f.value('vertical_velocity', t, float(velocity[2])):+.1f} m/s"
        labels["wind"].text = (
            f"{f.value('wind_speed', t):.1f} m/s / {math.degrees(f.circular('wind_direction', t)) % 360:03.0f}°"
            if f.has_wind else "NO WIND DATA")
        _, rotation = replay._rig(t)
        axis = rotation[:, 2]
        elevation = math.degrees(math.asin(max(-1, min(1, float(axis[2])))))
        heading = math.degrees(math.atan2(float(axis[0]), float(axis[1]))) % 360
        labels["angles"].text = f"EL {elevation:+.0f}° / HDG {heading:03.0f}°"
        recovery = ((f.drogue_time is not None and t >= f.drogue_time)
                    or (f.main_time is not None and t >= f.main_time))
        attitude = ("ILLUSTRATIVE RECOVERY SWAY" if recovery else
                    "RECORDED ATTITUDE" if f.has_attitude else "ATTITUDE FROM TRAJECTORY")
        labels["provenance"].text = f"{attitude}  ·  {f.recovery_source}"
        state = "FINISHED" if replay.finished else "PAUSED" if replay.paused else "PLAYING"
        self._state.text = f"{state}  /  {replay.speed:g}×"
        self._button_items[0][1].text = "Play" if replay.paused or replay.finished else "Pause"

    def update(self, dt):
        import math

        replay, f = self.replay, self.replay.flight
        if self._closing or replay.closed:
            return
        # A resize, debugger stop or menu interaction should hold the shot,
        # rather than skip seconds of flight when the event loop resumes.
        if self._reset_clock or not math.isfinite(dt) or dt > .25:
            dt = 0.0
            self._reset_clock = False
        dt = max(0.0, min(.05, dt))
        pending = getattr(replay, "_pending_seek", None)
        if pending is not None:
            replay._pending_seek = None
            replay.t = max(f.start, min(f.end, float(pending)))
            replay.finished = False
        elif not replay.paused and not replay.finished and not self._scrubbing:
            replay.t = min(f.end, replay.t + dt * replay.speed)
            if replay.t >= f.end:
                replay.finished = replay.paused = True
        self._update_timeline()
        self._refresh_text()

    def on_draw(self):
        import time

        if self._closing:
            return
        began = time.perf_counter()
        self.window.clear()
        self.painter.draw(self.replay.t, show_nav=bool(self.replay.hud_visible))
        # The world uses a depth buffer. Flat UI is composited afterwards,
        # with the logical window projection managed by pyglet's WindowBlock.
        gl = self.gl
        gl.glViewport(0, 0, *self.window.get_framebuffer_size())
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self._chrome.draw()
        if self.replay.hud_visible:
            self._hud.draw()
        self.replay._frame_costs.append(time.perf_counter() - began)

    def on_resize(self, width, height):
        self._layout(width, height)
        self._reset_clock = True
        # Return None so Window's default handler updates its orthographic
        # projection, including Retina backing-scale changes.

    def _changed(self):
        self._reset_clock = True
        self._refresh_text(force=True)
        self._update_timeline()

    def toggle_pause(self):
        if self.replay.finished:
            self.repeat()
            return
        self.replay.paused = not self.replay.paused
        self._changed()

    def repeat(self):
        self.replay.t = self.replay.flight.start
        self.replay.paused = self.replay.finished = False
        self.replay._pending_seek = None
        self._changed()

    def cycle_camera(self):
        self.replay.camera_mode = (self.replay.camera_mode + 1) % 3
        self._changed()

    def faster(self):
        self.replay.speed = min(64.0, self.replay.speed * 2)
        self._changed()

    def slower(self):
        self.replay.speed = max(.25, self.replay.speed / 2)
        self._changed()

    def _seek(self, value):
        f = self.replay.flight
        self.replay.t = max(f.start, min(f.end, float(value)))
        self.replay._pending_seek = None
        self.replay.finished = False
        self._changed()

    def on_key_press(self, symbol, modifiers):
        key = self.key
        if symbol == key.SPACE:
            self.toggle_pause()
        elif symbol == key.R:
            self.repeat()
        elif symbol == key.C:
            self.cycle_camera()
        elif symbol == key.H:
            self.replay.hud_visible = not self.replay.hud_visible
            self._changed()
        elif symbol in (key.PLUS, key.EQUAL, key.NUM_ADD):
            self.faster()
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            self.slower()
        elif symbol in (key.LEFT, key.RIGHT):
            self._seek(self.replay.t + (5 if symbol == key.RIGHT else -5))

    def _scrub_to(self, x):
        fraction = max(0.0, min(1.0, (x - self._timeline_start)
                                      / max(1.0, self._timeline_end - self._timeline_start)))
        f = self.replay.flight
        self._seek(f.start + fraction * (f.end - f.start))

    def on_mouse_press(self, x, y, button, modifiers):
        if button != self.mouse.LEFT:
            return
        if self._timeline_start - 8 <= x <= self._timeline_end + 8 and abs(y - 59.5) <= 13:
            self._scrubbing = self.replay._scrubbing = True
            self._scrub_to(x)
            return self.pyglet.event.EVENT_HANDLED
        for rectangle, _, callback in self._button_items:
            if rectangle.x <= x <= rectangle.x + rectangle.width and rectangle.y <= y <= rectangle.y + rectangle.height:
                callback()
                return self.pyglet.event.EVENT_HANDLED

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._scrubbing:
            self._scrub_to(x)
            return self.pyglet.event.EVENT_HANDLED

    def on_mouse_release(self, x, y, button, modifiers):
        if button == self.mouse.LEFT and self._scrubbing:
            self._scrub_to(x)
            self._scrubbing = self.replay._scrubbing = False
            self._reset_clock = True
            return self.pyglet.event.EVENT_HANDLED

    def on_mouse_motion(self, x, y, dx, dy):
        for rectangle, _, _ in self._button_items:
            inside = rectangle.x <= x <= rectangle.x + rectangle.width and rectangle.y <= y <= rectangle.y + rectangle.height
            rectangle.color = (43, 75, 89, 255) if inside else (24, 49, 62, 255)

    def _shutdown(self):
        if self._closing:
            return
        self._closing = True
        self.replay.closed = True
        self.replay._scrubbing = False
        self.pyglet.clock.unschedule(self.update)
        self.window.remove_handlers(self)
        self.painter.close()

    def on_close(self):
        self._shutdown()
        # The default Window.on_close handler releases the native window.

    def close(self):
        self._shutdown()
        self.window.close()

    def run(self):
        interval = 1 / max(1, self.replay.fps)
        if not self._scheduled:
            self.pyglet.clock.schedule_interval(self.update, interval)
            self._scheduled = True
        try:
            self.pyglet.app.run(interval=interval)
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path)
    parser.add_argument("--config", type=Path, help="Tracker-compatible column/unit JSON")
    parser.add_argument("--speed", type=float, default=1, help="Simulation seconds per screen second (default 1, real time)")
    parser.add_argument("--fps", type=int, default=60, help="Target screen/movie frame rate (default 60)")
    parser.add_argument("--camera", choices=("cinematic", "orbit", "wide"), default="cinematic")
    parser.add_argument("--software", action="store_true", help="Use the software viewer instead of GPU rendering")
    parser.add_argument("--rocket-length", type=float, default=ROCKET_LENGTH_M, help="Illustrative airframe length in m")
    parser.add_argument("--drogue-time", type=float, help="Override drogue event time in CSV seconds")
    parser.add_argument("--main-time", type=float, help="Override main event time in CSV seconds")
    parser.add_argument("--main-altitude", type=float, default=MAIN_DEPLOY_ALTITUDE_FT,
                        help="Fallback main deployment height AGL in ft when no events exist")
    parser.add_argument("--drogue-cutaway", action="store_true", help="Hide drogue after main deployment")
    parser.add_argument("--export", type=Path, help="Render MP4 (ffmpeg) or GIF (Pillow) instead of opening a window")
    parser.add_argument("--start", type=float, help="Movie start time in CSV seconds")
    parser.add_argument("--end", type=float, help="Movie end time in CSV seconds")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0 or not 1 <= args.fps <= 120:
        parser.error("Use a positive finite speed and fps between 1 and 120.")
    if not math.isfinite(args.rocket_length) or args.rocket_length <= 0:
        parser.error("Rocket length must be positive and finite.")
    for key in ("start", "end", "drogue_time", "main_time", "main_altitude"):
        value = getattr(args, key)
        if value is not None and not math.isfinite(value):
            parser.error(f"{key.replace('_', '-')} must be finite.")
    if args.csv is None:
        folder = Path(__file__).resolve().parent
        candidates = list((folder / "simulations").glob("*.csv")) + list(folder.glob("*.csv"))
        if not candidates:
            parser.error("No CSV found. Supply a path or put a CSV in simulations/.")
        args.csv = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    try:
        config = json.loads(args.config.expanduser().read_text()) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object.")
        data = load_flight(args.csv.expanduser().resolve(), config)
        dashboard = FlightReplay(data, args.csv.name, speed=args.speed, fps=args.fps,
                                 drogue_time=args.drogue_time, main_time=args.main_time,
                                 main_altitude=args.main_altitude, rocket_length=args.rocket_length,
                                 drogue_cutaway=args.drogue_cutaway)
        dashboard.camera_mode = ("cinematic", "orbit", "wide").index(args.camera)
        dashboard.software = args.software
        print(f"Loxodon-1 replay: {args.csv.name}", flush=True)
        print(f"Recovery: {dashboard.flight.recovery_source}; airframe and recovery rig are illustrative.")
        if args.export:
            dashboard.export(args.export.expanduser().resolve(), args.start, args.end)
            plt.close(dashboard.fig)
        else:
            dashboard.show()
        return 0
    except (ValueError, KeyError, OSError, ImportError, IndexError) as exc:
        print(f"Animation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
