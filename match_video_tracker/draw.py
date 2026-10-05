"""Draw the results on a frame (Skill 3, step 4)."""
from collections import Counter

import cv2
import numpy as np

GREEN = (0, 200, 0)         # BGR: OpenCV's order is blue, green, red
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def to_point(x, y):
    """OpenCV wants whole-number pixel positions."""
    return int(x), int(y)


def draw_people(frame, players, crowd, show_crowd=True):
    """Green box = player. Thin red box = removed as crowd, so you can check the filter."""
    out = frame.copy()                      # draw on a copy and keep the original clean

    if show_crowd:
        for x1, y1, x2, y2, _ in crowd:
            cv2.rectangle(out, to_point(x1, y1), to_point(x2, y2), RED, 1)
    for x1, y1, x2, y2, _ in players:
        cv2.rectangle(out, to_point(x1, y1), to_point(x2, y2), GREEN, 2)

    put_label(out, f"players: {len(players)}   crowd removed: {len(crowd)}")
    return out


def put_label(out, text):
    """White text on a black box in the bottom-left corner, so it never hides the scoreboard."""
    scale = max(0.6, out.shape[1] / 1600)   # bigger text on bigger frames
    thick = max(2, round(2 * scale))
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    h = out.shape[0]
    cv2.rectangle(out, (10, h - th - 30), (30 + tw, h - 10), BLACK, -1)
    cv2.putText(out, text, (20, h - 20), FONT, scale, WHITE, thick, cv2.LINE_AA)


def put_legend(out, items):
    """Like put_label, but with a coloured square before each text: [colour, text] pairs."""
    scale = max(0.6, out.shape[1] / 1600)
    thick = max(2, round(2 * scale))
    sizes = [cv2.getTextSize(text, FONT, scale, thick)[0] for _, text in items]
    th = max(s[1] for s in sizes)           # text height = size of the coloured squares
    gap = int(25 * scale)                   # space between two items
    width = sum(th + 8 + tw for tw, _ in sizes) + gap * (len(items) - 1)
    h = out.shape[0]
    cv2.rectangle(out, (10, h - th - 30), (30 + width, h - 10), BLACK, -1)
    x = 20
    for (color, text), (tw, _) in zip(items, sizes):
        cv2.rectangle(out, (x, h - 20 - th), (x + th, h - 20), color, -1)
        x += th + 8
        cv2.putText(out, text, (x, h - 20), FONT, scale, WHITE, thick, cv2.LINE_AA)
        x += tw + gap


def id_color(track_id):
    """The same ID always gets the same colour, so an ID switch is easy to spot."""
    hue = (track_id * 37) % 180                 # jump around the colour wheel (Skill 3: HSV)
    hsv = np.uint8([[[hue, 220, 255]]])         # one pixel: (hue, saturation, value)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def text_color(color):
    """Black text on a light background, white text on a dark one."""
    b, g, r = color
    brightness = 0.114 * b + 0.587 * g + 0.299 * r  # how bright the colour looks to us
    return BLACK if brightness > 140 else WHITE


def marker_size(out):
    """Bigger markers on bigger frames."""
    scale = max(0.5, out.shape[1] / 1920)
    thick = max(2, round(2 * scale))
    return scale, thick


def draw_marker(out, box, label, color, scale, thick):
    """An ellipse under the player's feet with his ID below it, like TV graphics."""
    x1, y1, x2, y2 = box
    cx, feet = int((x1 + x2) / 2), int(y2)
    half_w = max(int((x2 - x1) / 2), 8)
    # An open ellipse under the feet: from -45 to 235 degrees leaves a gap at the top.
    cv2.ellipse(out, (cx, feet), (half_w, max(half_w // 3, 3)), 0, -45, 235, color, thick, cv2.LINE_AA)

    # The ID in a small coloured box just below the ellipse.
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.6 * scale, thick)
    top = feet + half_w // 3 + 4
    cv2.rectangle(out, (cx - tw // 2 - 4, top), (cx + tw // 2 + 4, top + th + 8), color, -1)
    cv2.putText(out, label, (cx - tw // 2, top + th + 4), FONT, 0.6 * scale,
                text_color(color), thick, cv2.LINE_AA)


def draw_tracks(frame, tracks):
    """M2: one colour per ID. tracks = (id, x1, y1, x2, y2, conf) per player."""
    out = frame.copy()
    scale, thick = marker_size(out)
    for track_id, x1, y1, x2, y2, _ in tracks:
        draw_marker(out, (x1, y1, x2, y2), str(track_id), id_color(track_id), scale, thick)
    put_label(out, f"players tracked: {len(tracks)}")
    return out


def draw_teams(frame, players, colors, names):
    """M3: one colour per team. players = (id, x1, y1, x2, y2, team) per player.

    colors and names map each team number to its colour and its name in the legend.
    """
    out = frame.copy()
    scale, thick = marker_size(out)
    for track_id, x1, y1, x2, y2, team in players:
        draw_marker(out, (x1, y1, x2, y2), str(track_id), colors[team], scale, thick)
    on_screen = Counter(team for *_, team in players)      # how many of each team in this frame
    put_legend(out, [(colors[t], f"{name}: {on_screen[t]}") for t, name in names.items()])
    return out
