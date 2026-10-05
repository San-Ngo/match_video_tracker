import numpy as np
import pandas as pd

from match_video_tracker.ball import ball_path, standing_still


def test_standing_still_finds_a_spare_ball_but_not_the_match_ball():
    frames = np.arange(1, 21)
    spare = pd.DataFrame({"frame": frames, "tracklet": 0, "pitch_x": 100.0, "pitch_y": 50.0})
    moving = pd.DataFrame({"frame": frames, "tracklet": 1, "pitch_x": 300.0 + 10 * frames, "pitch_y": 80.0})
    balls = pd.concat([spare, moving], ignore_index=True)
    balls["scale"] = 100.0                          # players are 100 px tall
    still = standing_still(balls)
    assert still[balls["tracklet"] == 0].all()
    assert not still[balls["tracklet"] == 1].any()


def test_ball_path_fills_short_holes_and_drops_a_jump():
    rows = [(1, 100.0), (2, 102.0), (3, 900.0), (4, 106.0), (8, 114.0)]    # frame 3 jumps away
    balls = pd.DataFrame({"frame": [f for f, _ in rows], "x": [x for _, x in rows], "y": 50.0,
                          "scale": 100.0, "conf": 0.5, "keep": True})
    path = ball_path(balls, n_frames=10, fps=60)
    assert path.loc[3, "source"] == "filled"        # the jump was dropped, then filled in
    assert abs(path.loc[3, "x"] - 104) < 1e-6
    assert path.loc[6, "source"] == "filled"        # the hole 5-7 is short: filled
    assert path.loc[10, "source"] == "lost"         # after the last sighting: unknown


def test_tiles_cover_the_whole_frame_and_overlap():
    from match_video_tracker.ball import tiles
    parts = tiles(2304, 1172)
    assert len(parts) == 4
    assert min(x1 for x1, _, _, _ in parts) == 0 and max(x2 for _, _, x2, _ in parts) == 2304
    assert min(y1 for _, y1, _, _ in parts) == 0 and max(y2 for _, _, _, y2 in parts) == 1172
    left, right = parts[0], parts[1]
    assert left[2] - right[0] == 200              # 100 px overlap on each side of the middle


def test_ball_class_works_for_both_kinds_of_model():
    from match_video_tracker.ball import ball_class, is_football_model

    class Coco:
        names = {0: "person", 32: "sports ball"}

    class Football:
        names = {0: "ball"}

    assert ball_class(Coco()) == 32 and not is_football_model(Coco())
    assert ball_class(Football()) == 0 and is_football_model(Football())
