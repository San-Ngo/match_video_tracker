import numpy as np

from match_video_tracker.detect import split_players_and_crowd
from match_video_tracker.draw import draw_people


def test_split_keeps_players_on_grass_and_drops_the_crowd():
    mask = np.zeros((200, 300), dtype=bool)
    mask[100:] = True                       # grass from row 100 down
    player = (140, 120, 160, 170, 0.9)      # x1, y1, x2, y2, confidence
    fan = (140, 20, 160, 70, 0.8)
    players, crowd = split_players_and_crowd([player, fan], mask)
    assert players == [player]
    assert crowd == [fan]


def test_draw_keeps_the_original_frame_clean():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_people(frame, players=[(10, 10, 50, 90, 0.9)], crowd=[])
    assert out.shape == frame.shape
    assert frame.sum() == 0                 # the original is still all black
    assert out.sum() > 0                    # the copy has the drawing
