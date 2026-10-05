"""Telestration: the lines, arrows and shapes an analyst draws on the match.

Every drawing is plain data (a dict), saved in a JSON file, so any app can read it:

    {"kind": "pass",     "players": [17, 14],        "first": 200, "last": 260, "color": "yellow"}
    {"kind": "triangle", "players": [4, 8, 17],      "first": 1,   "last": 445, "color": "cyan"}
    {"kind": "zone",     "players": [2, 6, 7, 10],   ...}     any number of players, a filled shape
    {"kind": "arrow",    "points": [[x, y], [x, y]], ...}     points in PITCH coordinates (M4)
    {"kind": "line",     "points": [[x, y], [x, y]], ...}
    {"kind": "arrow",    "player": 17, "offset": [dx, dy], ...}   an arrow that runs with player 17

Two families:
  - player-linked (pass, triangle, zone): drawn between players' feet, so they move
    with the players frame by frame, like Metrica's team-shape lines.
  - pitch-fixed (line, arrow): stored in pitch coordinates, so they stay on the same
    spot of grass while the camera pans.
  - an arrow that starts on a player is tied to him: it starts at his feet in every
    frame and points the same way and distance on the pitch (his run, his pass option).
Everything is painted only on grass pixels, so players stay in front of the drawings.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import pandas as pd

from match_video_tracker.camera import from_pitch, to_pitch

COLORS = {"yellow": (0, 230, 255), "white": (255, 255, 255), "cyan": (255, 230, 0),
          "magenta": (255, 0, 255), "red": (40, 40, 255), "green": (60, 220, 60)}   # BGR
PLAYER_KINDS = {"pass": 2, "triangle": 3, "zone": 3}    # how many players each needs (zone: at least)
PITCH_KINDS = {"line": 2, "arrow": 2}                   # how many clicked points each needs
FILL_ALPHA = 0.25       # how see-through a filled triangle or zone is
LINE_ALPHA = 0.9


def make(kind, first, last, color="yellow", players=None, points=None, player=None, offset=None):
    """A new drawing, checked: the right number of players or points for its kind."""
    if kind == "arrow" and player is not None:
        dx, dy = (float(v) for v in offset)
        return {"kind": kind, "player": int(player), "offset": [dx, dy], "first": int(first),
                "last": int(last), "color": color}
    if kind in PLAYER_KINDS:
        players = [int(p) for p in players or []]
        if len(players) < PLAYER_KINDS[kind] or (kind != "zone" and len(players) != PLAYER_KINDS[kind]):
            raise ValueError(f"A {kind} needs {PLAYER_KINDS[kind]} players.")
        if len(set(players)) != len(players):
            raise ValueError("Pick different players.")
        return {"kind": kind, "players": players, "first": int(first), "last": int(last), "color": color}
    if kind in PITCH_KINDS:
        points = [[float(x), float(y)] for x, y in points or []]
        if len(points) != PITCH_KINDS[kind]:
            raise ValueError(f"A {kind} needs {PITCH_KINDS[kind]} points.")
        return {"kind": kind, "points": points, "first": int(first), "last": int(last), "color": color}
    raise ValueError(f"Unknown drawing: {kind}")


def save(drawings, path):
    Path(path).write_text(json.dumps(drawings, indent=1))


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else []


def describe(d):
    """A short label for the list of drawings, e.g. 'pass 17 → 14, frames 200-260'."""
    if d["kind"] in PLAYER_KINDS:
        who = (" → " if d["kind"] == "pass" else ", ").join(str(p) for p in d["players"])
    elif d.get("player") is not None:
        who = f"with player {d['player']}"
    else:
        who = "on the pitch"
    return f"{d['kind']} {who}, frames {d['first']}-{d['last']}"


def feet_at(rows, players):
    """The feet (x, y) of these players in one frame's rows, in the same order, or None
    if one of them is not in the frame."""
    if rows.empty:
        return None
    by_id = rows.set_index("id")
    if not all(p in by_id.index for p in players):
        return None
    x, y = ("foot_x_smooth", "foot_y_smooth") if "foot_x_smooth" in by_id else ("foot_x", "foot_y")
    return [(float(by_id.loc[p, x]), float(by_id.loc[p, y])) for p in players]


def arrow_head(layer, a, b, value, width):
    """Two short strokes at end b, so a line becomes an arrow."""
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    length = np.hypot(*d)
    if length < 1:
        return
    d /= length
    size = max(12.0, 4.0 * width)
    for turn in (0.5, -0.5):                              # +- about 30 degrees
        c, s = np.cos(np.pi - turn), np.sin(np.pi - turn)
        tip = b + size * np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        cv2.line(layer, tuple(int(v) for v in b), tuple(int(v) for v in tip), value, width, cv2.LINE_AA)


def shape_points(d, n, rows, camera):
    """Where drawing d is in frame n, in pixels: a list of points, or None if hidden."""
    if not d["first"] <= n <= d["last"]:
        return None
    if d["kind"] in PLAYER_KINDS:
        return feet_at(rows, d["players"])
    if d.get("player") is not None:                       # an arrow that runs with a player
        feet = feet_at(rows, [d["player"]])
        if feet is None:
            return None
        (x, y), = feet
        start = to_pitch(pd.DataFrame({"frame": [n], "x": [x], "y": [y]}), camera)
        end = (start["pitch_x"].iat[0] + d["offset"][0], start["pitch_y"].iat[0] + d["offset"][1])
        return [(x, y), tuple(from_pitch([end], camera, n)[0])]
    return [tuple(p) for p in from_pitch(d["points"], camera, n)]


def draw_all(frame, n, drawings, rows, camera, grass):
    """Paint every drawing that is on in frame n. rows = that frame's tracks; grass =
    the frame's grass mask (True = grass). Returns a new frame."""
    out = frame.astype(np.float32)
    scale = max(0.5, frame.shape[1] / 1920)
    width = max(2, int(3 * scale))
    grass = grass.astype(np.float32)
    for d in drawings:
        pts = shape_points(d, n, rows, camera)
        if pts is None:
            continue
        alpha = np.zeros(frame.shape[:2], np.float32)
        ipts = np.array([[int(x), int(y)] for x, y in pts], np.int32)
        if d["kind"] in ("triangle", "zone"):
            if d["kind"] == "zone":                      # a zone goes around its players
                ipts = cv2.convexHull(ipts).reshape(-1, 2)
            cv2.fillPoly(alpha, [ipts], FILL_ALPHA, cv2.LINE_AA)
            cv2.polylines(alpha, [ipts], True, LINE_ALPHA, width, cv2.LINE_AA)
        else:
            cv2.line(alpha, tuple(ipts[0]), tuple(ipts[1]), LINE_ALPHA, width, cv2.LINE_AA)
            if d["kind"] in ("pass", "arrow"):
                arrow_head(alpha, ipts[0], ipts[1], LINE_ALPHA, width)
        a = (alpha * grass)[:, :, None]                   # under the players: grass only
        out = out * (1 - a) + np.array(COLORS.get(d["color"], COLORS["yellow"]), np.float32) * a
    return out.astype(np.uint8)
