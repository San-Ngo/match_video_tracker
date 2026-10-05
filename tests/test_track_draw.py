import numpy as np

from match_video_tracker.draw import draw_tracks, id_color


def test_same_id_always_gets_the_same_colour():
    assert id_color(7) == id_color(7)
    assert id_color(7) != id_color(8)


def test_draw_tracks_keeps_the_original_frame_clean():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    out = draw_tracks(frame, [(7, 100, 50, 130, 150, 0.9)])
    assert frame.sum() == 0
    assert out.sum() > 0
