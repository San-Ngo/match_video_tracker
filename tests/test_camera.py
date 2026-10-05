import numpy as np
import pandas as pd

from match_video_tracker.camera import to_pitch, track_camera


def textured_frame(shift_x=0):
    """A random 'stadium' picture; shift_x moves everything right, like a camera panning left."""
    rng = np.random.default_rng(0)
    big = (rng.random((400, 1100)) * 255).astype(np.uint8)
    big = np.kron(big, np.ones((3, 3), dtype=np.uint8))[:600, :2000]      # blocks = corners to follow
    img = big[:, 100 - shift_x:100 - shift_x + 1600]
    return np.dstack([img, img, img])


def test_track_camera_measures_a_pan():
    frames = [(1, textured_frame(0), []), (2, textured_frame(12), [])]
    camera = track_camera(iter(frames))
    assert abs(camera.loc[2, "dx"] - 12) < 1         # the picture slid 12 px to the right
    assert not camera["cut"].any()


def test_to_pitch_takes_the_pan_out():
    frames = [(1, textured_frame(0), []), (2, textured_frame(12), [])]
    camera = track_camera(iter(frames))
    still = pd.DataFrame({"frame": [1, 2], "x": [500.0, 512.0], "y": [300.0, 300.0]})   # lying still
    out = to_pitch(still, camera)
    assert abs(out["pitch_x"].iloc[1] - out["pitch_x"].iloc[0]) < 1
