"""Find the grass, so we can tell players (on the pitch) from the crowd (in the stands).

Skill 3, step 5: HSV splits a colour into hue (which colour), saturation (how strong)
and value (how bright). Grass is one range of hue, so it is easy to find in HSV.
"""
import cv2

# OpenCV's hue runs from 0 to 179, so green is roughly 30 to 90.
# These are the numbers to tune if your pitch looks different.
GRASS_LOW = (30, 40, 40)        # (hue, saturation, value)
GRASS_HIGH = (90, 255, 255)


def grass_mask(frame, low=GRASS_LOW, high=GRASS_HIGH):
    """Return a True/False image: True where the pixel looks like grass."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)      # 255 = grass, 0 = not grass
    return mask > 0                         # turn 0/255 into False/True


def foot_grass_fraction(mask, box):
    """Share of grass (0 to 1) in a thin strip around a person's feet.

    box = (x1, y1, x2, y2): the top-left and bottom-right corners, in pixels.
    A player stands on grass, so this is high. A fan in the stands gets about 0.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    # The strip: a bit wider than the person, from just above the feet to just below.
    sx1, sx2 = x1 - 0.3 * w, x2 + 0.3 * w
    sy1, sy2 = y2 - 0.1 * h, y2 + 0.1 * h

    # Keep the strip inside the image, and turn the floats into whole pixel numbers.
    img_h, img_w = mask.shape
    sx1, sx2 = int(max(0, sx1)), int(min(img_w, sx2))
    sy1, sy2 = int(max(0, sy1)), int(min(img_h, sy2))
    if sx2 <= sx1 or sy2 <= sy1:            # the strip is outside the image
        return 0.0

    strip = mask[sy1:sy2, sx1:sx2]          # rows (y) first, then columns (x): Skill 2, step 3
    return float(strip.mean())              # the mean of True/False = the share of True
