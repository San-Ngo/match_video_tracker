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
