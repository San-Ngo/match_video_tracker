import numpy as np

from match_video_tracker.grass import foot_grass_fraction, grass_mask

GREEN = (40, 160, 40)       # BGR: a grass-like green
GREY = (120, 120, 120)      # the stands


def make_pitch_image():
    """A fake 200 x 300 frame: grey stands on top, green grass from row 100 down."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:100] = GREY
    img[100:] = GREEN
    return img


def test_grass_mask_finds_only_the_green_part():
    mask = grass_mask(make_pitch_image())
    assert mask.shape == (200, 300)
    assert not mask[:100].any()             # no grass in the stands
    assert mask[100:].all()                 # all grass on the pitch


def test_player_on_the_pitch_has_grass_at_the_feet():
    mask = grass_mask(make_pitch_image())
    player = (140, 120, 160, 170)           # x1, y1, x2, y2: feet at y = 170, on the grass
    assert foot_grass_fraction(mask, player) > 0.9


def test_fan_in_the_stands_has_no_grass_at_the_feet():
    mask = grass_mask(make_pitch_image())
    fan = (140, 20, 160, 70)                # feet at y = 70, in the grey stands
    assert foot_grass_fraction(mask, fan) == 0.0


def test_box_outside_the_image_gives_zero():
    mask = grass_mask(make_pitch_image())
    assert foot_grass_fraction(mask, (400, 300, 420, 350)) == 0.0
