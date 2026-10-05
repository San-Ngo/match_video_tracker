"""Shirt colour, measured in the Lab colour space (Skill 6, step 1).

Lab splits a colour into L (lightness), a* (green to red) and b* (blue to yellow).
Ignoring L means shadows and sunshine change the colour less, and distances in
a*/b* match how different two shirts look to your eyes: red is far from blue.
"""
import math

import cv2


def to_lab(frame):
    """Convert a BGR frame to Lab once, so every player in it can reuse it."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)


def shirt_colour(lab, grass, box):
    """Average (a*, b*) of a player's shirt, or (nan, nan) if we can't see it.

    The shirt is the upper middle of the box: 15-50 % down, the middle half across.
    Grass pixels inside that patch are ignored (grass = True/False mask from grass.py).
    """
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    ys, ye = int(y1 + 0.15 * h), int(y1 + 0.5 * h)
    xs, xe = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    ys, xs = max(ys, 0), max(xs, 0)
    patch = lab[ys:ye, xs:xe]
    not_grass = ~grass[ys:ye, xs:xe]
    if not_grass.sum() < 5:                 # too small or all grass: we can't tell
        return math.nan, math.nan
    a, b = patch[not_grass][:, 1:].mean(axis=0)     # columns 1 and 2 = a* and b*
    return float(a), float(b)
