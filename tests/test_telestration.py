import numpy as np
import pandas as pd
import pytest

from match_video_tracker.telestration import describe, draw_all, load, make, save


def camera_still(frames):
    return pd.DataFrame({"to_first_00": 1.0, "to_first_01": 0.0, "to_first_02": 0.0,
                         "to_first_10": 0.0, "to_first_11": 1.0, "to_first_12": 0.0, "cut": False},
                        index=pd.Index(frames, name="frame"))


def rows(feet):
    """One frame's tracks: feet = {player id: (x, y)}."""
    return pd.DataFrame([{"id": p, "foot_x": x, "foot_y": y} for p, (x, y) in feet.items()])


def test_make_checks_the_number_of_players():
    assert make("pass", 1, 10, players=[17, 14])["players"] == [17, 14]
    with pytest.raises(ValueError):
        make("triangle", 1, 10, players=[1, 2])
    with pytest.raises(ValueError):
        make("pass", 1, 10, players=[5, 5])


def test_save_and_load_give_back_the_same_drawings(tmp_path):
    drawings = [make("triangle", 1, 10, players=[1, 2, 3]), make("arrow", 5, 8, points=[[0, 0], [9, 9]])]
    save(drawings, tmp_path / "d.json")
    assert load(tmp_path / "d.json") == drawings
    assert describe(drawings[0]) == "triangle 1, 2, 3, frames 1-10"


def test_a_pass_line_follows_the_players():
    frame = np.zeros((100, 200, 3), np.uint8)
    grass = np.ones((100, 200), bool)
    d = [make("pass", 1, 2, color="white", players=[1, 2])]
    a = draw_all(frame, 1, d, rows({1: (20, 50), 2: (100, 50)}), camera_still([1, 2]), grass)
    b = draw_all(frame, 2, d, rows({1: (20, 80), 2: (100, 80)}), camera_still([1, 2]), grass)
    assert a[50, 60].sum() > 0 and a[80, 60].sum() == 0      # frame 1: line at y=50
    assert b[80, 60].sum() > 0 and b[50, 60].sum() == 0      # frame 2: the players moved to y=80


def test_drawings_stay_under_the_players_and_only_in_their_frames():
    frame = np.zeros((100, 200, 3), np.uint8)
    grass = np.ones((100, 200), bool)
    grass[:, 55:65] = False                                  # a player's leg across the line
    d = [make("line", 1, 1, points=[[20, 50], [100, 50]])]
    out = draw_all(frame, 1, d, rows({}), camera_still([1, 2]), grass)
    assert out[50, 30].sum() > 0                             # on the grass: drawn
    assert out[50, 60].sum() == 0                            # on the leg: not drawn
    later = draw_all(frame, 2, d, rows({}), camera_still([1, 2]), grass)
    assert later.sum() == 0                                  # frame 2: the line is over


def test_an_arrow_tied_to_a_player_runs_with_him():
    frame = np.zeros((100, 200, 3), np.uint8)
    grass = np.ones((100, 200), bool)
    d = [make("arrow", 1, 2, color="white", player=7, offset=[60, 0])]
    assert describe(d[0]) == "arrow with player 7, frames 1-2"
    a = draw_all(frame, 1, d, rows({7: (20, 30)}), camera_still([1, 2]), grass)
    b = draw_all(frame, 2, d, rows({7: (40, 70)}), camera_still([1, 2]), grass)
    assert a[30, 50].sum() > 0 and a[70, 70].sum() == 0      # frame 1: from (20, 30) to (80, 30)
    assert b[70, 70].sum() > 0 and b[30, 50].sum() == 0      # frame 2: he ran, the arrow came along
    gone = draw_all(frame, 1, d, rows({}), camera_still([1, 2]), grass)
    assert gone.sum() == 0                                   # he is not on screen: no arrow
