import math

from match_video_tracker.tracks import (drop_short, fill_gaps, id_summary, link_broken_tracks,
                                        renumber, smooth, to_table)

RED = (155.0, 140.0)        # shirt colour as (a*, b*) in Lab
BLUE = (135.0, 105.0)


def make_rows():
    """Player 7 is missing in frame 3 (YOLO missed him). Player 9 is seen once."""
    return [
        (1, 7, 90, 100, 110, 200, 0.9, *RED),
        (2, 7, 94, 101, 114, 201, 0.9, *RED),
        (4, 7, 102, 103, 122, 203, 0.8, *RED),
        (1, 9, 300, 100, 320, 220, 0.7, *BLUE),
    ]


def test_to_table_puts_the_feet_at_the_bottom_middle():
    df = to_table(make_rows())
    first = df.iloc[0]
    assert first["foot_x"] == 100           # (90 + 110) / 2
    assert first["foot_y"] == 200           # y2
    assert first["h"] == 100                # 200 - 100


def test_id_summary_counts_frames_and_holes():
    s = id_summary(to_table(make_rows()))
    assert s.loc[7, "seen"] == 3
    assert s.loc[7, "span"] == 4            # frames 1 to 4
    assert s.loc[7, "missing"] == 1         # frame 3
    assert s.loc[9, "seen"] == 1


def test_fill_gaps_draws_a_straight_line_through_the_hole():
    clean = fill_gaps(to_table(make_rows()))
    p7 = clean[clean["id"] == 7].set_index("frame")
    assert list(p7.index) == [1, 2, 3, 4]
    assert p7.loc[3, "foot_x"] == 108       # halfway between 104 (frame 2) and 112 (frame 4)
    assert bool(p7.loc[3, "filled"]) is True
    assert bool(p7.loc[2, "filled"]) is False


def test_smooth_removes_zigzag_jitter():
    rows = [(f, 1, x, 0, x, 100, 0.9, *RED) for f, x in enumerate([0, 10, 0, 10, 0], start=1)]
    out = smooth(to_table(rows), window=3)
    middle = out["foot_x_smooth"].iloc[1:4]
    assert middle.max() - middle.min() < 10  # the 0-10-0 zigzag is flattened


def test_drop_short_removes_ids_seen_only_briefly():
    df = drop_short(to_table(make_rows()), min_frames=2)
    assert set(df["id"]) == {7}             # player 9 was seen in only 1 frame


def test_link_joins_a_player_who_comes_back_with_a_new_id():
    rows = [
        (1, 8, 100, 100, 120, 200, 0.9, *RED),      # player 8 runs behind a defender...
        (2, 8, 104, 100, 124, 200, 0.9, *RED),
        (32, 29, 160, 100, 180, 200, 0.9, *RED),    # ...and comes back 30 frames later as "29"
        (33, 29, 164, 100, 184, 200, 0.9, *RED),
        (32, 30, 170, 100, 190, 200, 0.9, *BLUE),   # a blue player starts nearby: must NOT join
    ]
    linked, joins = link_broken_tracks(to_table(rows), fps=60)
    assert joins == 1
    assert set(linked[linked["frame"] >= 32]["id"]) == {8, 30}   # 29 became 8, 30 stayed 30


def test_link_ignores_a_player_who_is_too_far_away():
    rows = [
        (1, 8, 100, 100, 120, 200, 0.9, *RED),
        (5, 29, 900, 100, 920, 200, 0.9, *RED),     # 8 body heights away after 4 frames: impossible
    ]
    _, joins = link_broken_tracks(to_table(rows), fps=60)
    assert joins == 0


def test_renumber_gives_short_ids_in_order_of_appearance():
    rows = [(1, 57, 0, 0, 10, 10, 0.9, *RED), (2, 3, 0, 0, 10, 10, 0.9, *RED),
            (3, 57, 0, 0, 10, 10, 0.9, *RED)]
    out = renumber(to_table(rows))
    assert list(out["id"]) == [1, 2, 1]


def test_missing_colour_never_joins():
    rows = [(1, 8, 100, 100, 120, 200, 0.9, math.nan, math.nan),
            (3, 29, 102, 100, 122, 200, 0.9, math.nan, math.nan)]
    _, joins = link_broken_tracks(to_table(rows), fps=60)
    assert joins == 0
