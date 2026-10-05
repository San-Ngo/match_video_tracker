"""Shirt colour, measured in the Lab colour space (Skill 6, step 1).

Lab splits a colour into L (lightness), a* (green to red) and b* (blue to yellow).
Ignoring L means shadows and sunshine change the colour less, and distances in
a*/b* match how different two shirts look to your eyes: red is far from blue.
"""
import math

import cv2
import numpy as np

# Where the shirt is inside a player's box, as fractions of the box.
SHIRT_TOP, SHIRT_BOTTOM = 0.15, 0.5     # from 15 % to 50 % of the way down
SHIRT_SIDE = 0.25                       # skip 25 % on the left and right: keep the middle half


def to_lab(frame):
    """Convert a BGR frame to Lab once, so every player in it can reuse it."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)


def shirt_colour(lab, grass, box):
    """Median (a*, b*) of a player's shirt, or (nan, nan) if the box is too small.

    Grass pixels inside the shirt patch are ignored. But if almost the whole patch
    looks like grass, the shirt itself is green or yellow (a keeper, a referee), so
    we keep every pixel instead of throwing the shirt away.
    The median ignores the odd pixel (a shirt number, an arm) better than the mean.
    """
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    ys, ye = int(y1 + SHIRT_TOP * h), int(y1 + SHIRT_BOTTOM * h)
    xs, xe = int(x1 + SHIRT_SIDE * w), int(x2 - SHIRT_SIDE * w)
    ys, xs = max(ys, 0), max(xs, 0)
    patch = lab[ys:ye, xs:xe].reshape(-1, 3)        # one row per pixel: L, a*, b*
    if len(patch) < 5:                              # box too small or off the screen
        return math.nan, math.nan
    not_grass = ~grass[ys:ye, xs:xe].reshape(-1)
    if not_grass.sum() >= max(5, 0.3 * len(patch)):   # a normal shirt: drop the grass
        patch = patch[not_grass]
    a, b = np.median(patch[:, 1:], axis=0)          # columns 1 and 2 = a* and b*
    return float(a), float(b)


def shirt_to_bgr(a, b, lightness=140, boost=2.0):
    """Turn a shirt colour (a*, b*) back into a bright BGR colour we can draw with.

    We pick the lightness ourselves and push the colour further away from grey
    (boost), so even a dark navy shirt gets a marker you can see on the grass.
    """
    a = 128 + boost * (a - 128)                     # 128 = grey in OpenCV's Lab
    b = 128 + boost * (b - 128)
    pixel = np.array([[[lightness, a, b]]]).round().clip(0, 255).astype(np.uint8)
    blue, green, red = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return int(blue), int(green), int(red)
