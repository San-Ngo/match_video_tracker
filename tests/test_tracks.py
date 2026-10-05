from match_video_tracker.tracks import drop_short, fill_gaps, id_summary, smooth, to_table


def make_rows():
    """Player 7 is missing in frame 3 (YOLO missed him). Player 9 is seen once."""
    return [
        (1, 7, 90, 100, 110, 200, 0.9),
        (2, 7, 94, 101, 114, 201, 0.9),
        (4, 7, 102, 103, 122, 203, 0.8),
        (1, 9, 300, 100, 320, 220, 0.7),
    ]


def test_to_table_puts_the_feet_at_the_bottom_middle():
    df = to_table(make_rows())
    first = df.iloc[0]
    assert first["foot_x"] == 100           # (90 + 110) / 2
    assert first["foot_y"] == 200           # y2


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
    rows = [(f, 1, x, 0, x, 100, 0.9) for f, x in enumerate([0, 10, 0, 10, 0], start=1)]
    out = smooth(to_table(rows), window=3)
    middle = out["foot_x_smooth"].iloc[1:4]
    assert middle.max() - middle.min() < 10  # the 0-10-0 zigzag is flattened


def test_drop_short_removes_ids_seen_only_briefly():
    df = drop_short(to_table(make_rows()), min_frames=2)
    assert set(df["id"]) == {7}             # player 9 was seen in only 1 frame
