"""The broadcast-style spotlight: a glowing ring under one player and a trail of where
he ran, both drawn UNDER the players (Skill 10: alpha blending and masks).

1. Paint the ring and the trail on an empty "alpha" layer: 0 = nothing, 1 = full colour.
   The trail fades: old points are faint, new points are strong.
2. Multiply that layer by the grass mask, so it is 0 wherever there is no grass:
   on a boot, a leg, the ball. That is the chroma key trick TV graphics use, and it
   puts the drawing under the players.
3. Blend: picture = frame x (1 - alpha) + colour x alpha.
"""
import cv2
import numpy as np

RING_ALPHA = 0.85       # how strong the ring is (1 = solid colour)
TRAIL_ALPHA = 0.7       # how strong the newest part of the trail is
TRAIL_SECONDS = 3       # how much of his run the trail shows


def ring_layer(alpha, box, scale):
    """Add a glowing ring under the feet of `box` to the alpha layer."""
    x1, y1, x2, y2 = box
    cx, feet = int((x1 + x2) / 2), int(y2)
    rx = max(int(0.75 * (x2 - x1)), 10)               # a bit wider than the player
    ry = max(int(0.3 * rx), 4)                        # flat: it lies on the pitch
    # Work in a small box around the feet only: blurring the whole frame is slow.
    pad = rx + 20
    top, left = max(feet - pad, 0), max(cx - pad, 0)
    bottom, right = min(feet + pad, alpha.shape[0]), min(cx + pad, alpha.shape[1])
    if bottom <= top or right <= left:
        return
    glow = np.zeros((bottom - top, right - left), np.float32)
    centre = (cx - left, feet - top)
    cv2.ellipse(glow, centre, (rx, ry), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), max(2, 0.25 * ry))               # soft edge
    cv2.ellipse(glow, centre, (rx, ry), 0, 0, 360, 1.0, max(2, int(3 * scale)), cv2.LINE_AA)
    region = alpha[top:bottom, left:right]
    np.maximum(region, glow * RING_ALPHA, out=region)


def trail_layer(alpha, points, scale):
    """Add a fading line through `points` (oldest first) to the alpha layer.

    Oldest first, so each newer, stronger piece is painted over the older ones.
    """
    n = len(points)
    for i in range(1, n):
        strength = TRAIL_ALPHA * i / n                # older = fainter
        width = max(2, int((2 + 6 * i / n) * scale))  # older = thinner
        a, b = tuple(int(v) for v in points[i - 1]), tuple(int(v) for v in points[i])
        cv2.line(alpha, a, b, strength, width, cv2.LINE_AA)


def blend(frame, alpha, grass, color):
    """Paint `color` onto the frame with strength `alpha`, but only on grass pixels."""
    a = (alpha * grass)[:, :, None]                   # 0 on players, boots and ball
    out = frame.astype(np.float32) * (1 - a) + np.array(color, np.float32) * a
    return out.astype(np.uint8)


def draw_spotlight(frame, grass, box, trail, color):
    """The whole spotlight on one frame. box = the player's box or None if he is not
    in this frame; trail = his recent foot positions in this frame's pixels."""
    scale = max(0.5, frame.shape[1] / 1920)
    alpha = np.zeros(frame.shape[:2], np.float32)
    if len(trail) > 1:
        trail_layer(alpha, trail, scale)
    if box is not None:
        ring_layer(alpha, box, scale)
    return blend(frame, alpha, grass.astype(np.float32), color)
