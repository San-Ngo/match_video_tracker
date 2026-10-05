import math

import numpy as np

from match_video_tracker.colour import shirt_colour, to_lab
from match_video_tracker.grass import grass_mask


def player_image(shirt_bgr):
    """A 100 x 60 green picture with a 'player' (a coloured rectangle) in the middle."""
    img = np.zeros((100, 60, 3), dtype=np.uint8)
    img[:] = (40, 160, 40)                  # grass
    img[10:90, 15:45] = shirt_bgr           # the player
    return img


def test_red_and_blue_shirts_are_far_apart():
    box = (15, 10, 45, 90)
    red = player_image((40, 40, 220))       # BGR
    blue = player_image((200, 60, 30))
    ra, rb = shirt_colour(to_lab(red), grass_mask(red), box)
    ba, bb = shirt_colour(to_lab(blue), grass_mask(blue), box)
    assert ra > ba                          # a* is higher for red
    assert math.hypot(ra - ba, rb - bb) > 30


def test_all_grass_gives_no_colour():
    img = player_image((40, 160, 40))       # the "player" is grass-coloured
    a, b = shirt_colour(to_lab(img), grass_mask(img), (15, 10, 45, 90))
    assert math.isnan(a) and math.isnan(b)
