"""Clean up the tracker's output with pandas (Skill 7).

The table has one row per player per frame: frame, id, x1, y1, x2, y2, conf.
"""
import pandas as pd

COLUMNS = ["frame", "id", "x1", "y1", "x2", "y2", "conf"]
POSITION = ["x1", "y1", "x2", "y2", "foot_x", "foot_y"]


def to_table(rows):
    """Turn a list of (frame, id, x1, y1, x2, y2, conf) into a DataFrame."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["foot_x"] = (df["x1"] + df["x2"]) / 2    # the middle of the box...
    df["foot_y"] = df["y2"]                     # ...at the bottom: where the feet are
    return df


def id_summary(df):
    """One row per ID: when it was first and last seen, and how many frames it is missing."""
    s = df.groupby("id")["frame"].agg(first="min", last="max", seen="count")
    s["span"] = s["last"] - s["first"] + 1      # frames from the first to the last sighting
    s["missing"] = s["span"] - s["seen"]        # holes inside that span
    return s.sort_values("seen", ascending=False)


def drop_short(df, min_frames):
    """Remove IDs seen in fewer than min_frames frames: usually flickers, not real players."""
    seen = df.groupby("id")["frame"].transform("count")     # for every row: how often its ID was seen
    return df[seen >= min_frames]


def fill_gaps(df, max_gap=60):
    """Fill the holes in each player's track with straight lines (interpolate).

    A hole is a frame where the tracker kept the ID but YOLO missed the player.
    Only holes of up to max_gap frames are filled. Filled rows get filled=True.
    """
    pieces = []
    for pid, track in df.groupby("id"):
        track = track.set_index("frame").sort_index()
        every_frame = range(track.index.min(), track.index.max() + 1)
        track = track.reindex(every_frame)      # the missing frames appear as empty (NaN) rows
        track.index.name = "frame"
        track["filled"] = track["conf"].isna()  # remember which rows we are about to invent
        track[POSITION] = track[POSITION].interpolate(limit=max_gap, limit_area="inside")
        track["id"] = pid
        pieces.append(track.dropna(subset=["foot_x"]).reset_index())
    return pd.concat(pieces, ignore_index=True)


def smooth(df, window=5):
    """Average each foot position with its neighbours to remove jitter (rolling mean)."""
    out = df.sort_values(["id", "frame"]).reset_index(drop=True)
    for col in ["foot_x", "foot_y"]:
        out[col + "_smooth"] = out.groupby("id")[col].transform(
            lambda s: s.rolling(window, center=True, min_periods=1).mean())
    return out
