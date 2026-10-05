import math

import numpy as np

from match_video_tracker.colour import shirt_colour, shirt_to_bgr, to_lab
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


def test_a_green_shirt_still_gets_a_colour():
    img = player_image((60, 220, 120))      # a keeper in lime green: the grass mask says "grass"
    a, b = shirt_colour(to_lab(img), grass_mask(img), (15, 10, 45, 90))
    assert not math.isnan(a)
    assert a < 128                          # a* below 128 = green


def test_a_tiny_box_gives_no_colour():
    img = player_image((40, 40, 220))
    a, b = shirt_colour(to_lab(img), grass_mask(img), (20, 20, 22, 22))     # 2 x 2 pixels
    assert math.isnan(a) and math.isnan(b)


def test_shirt_to_bgr_turns_a_red_shirt_into_a_red_marker():
    blue, green, red = shirt_to_bgr(155, 140)   # (a*, b*) of a red shirt
    assert red > green and red > blue
