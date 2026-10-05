import numpy as np
import pandas as pd

from match_video_tracker.camera import from_pitch, to_pitch
from match_video_tracker.spotlight import draw_spotlight


def test_spotlight_paints_only_on_grass():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    grass = np.ones((200, 300), dtype=bool)
    grass[:, 150:] = False                          # right half: a "player" standing on the ring
    out = draw_spotlight(frame, grass, (100, 50, 200, 150), [], (0, 0, 255))
    assert out[150, 120, 2] > 100                   # ring on the grass: red
    assert out[150, 180].sum() == 0                 # on the player: untouched


def test_from_pitch_undoes_to_pitch():
    camera = pd.DataFrame({"to_first_00": [1.0], "to_first_01": [0.0], "to_first_02": [-40.0],
                           "to_first_10": [0.0], "to_first_11": [1.0], "to_first_12": [5.0]},
                          index=pd.Index([7], name="frame"))
    pts = pd.DataFrame({"frame": [7], "x": [300.0], "y": [200.0]})
    pitch = to_pitch(pts, camera)[["pitch_x", "pitch_y"]].to_numpy()
    back = from_pitch(pitch, camera, 7)
    assert np.allclose(back, [[300.0, 200.0]])


def test_trail_shows_only_the_last_seconds():
    from match_video_tracker.spotlight import box_at, follow, trail_at
    frames = list(range(1, 11))
    camera = pd.DataFrame({"to_first_00": 1.0, "to_first_01": 0.0, "to_first_02": 0.0,
                           "to_first_10": 0.0, "to_first_11": 1.0, "to_first_12": 0.0, "cut": False},
                          index=pd.Index(frames, name="frame"))
    tracks = pd.DataFrame({"frame": frames, "id": 8, "x1": 0.0, "y1": 0.0, "x2": 20.0, "y2": 50.0,
                           "foot_x_smooth": [10.0 * f for f in frames], "foot_y_smooth": 50.0})
    me = follow(tracks, camera, 8)
    trail = trail_at(me, camera, 10, keep_frames=3)
    assert [x for x, _ in trail] == [70.0, 80.0, 90.0, 100.0]     # frames 7-10 only
    assert box_at(me, 10) == (0.0, 0.0, 20.0, 50.0)
    assert box_at(me, 11) is None
