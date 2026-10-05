"""Clean up the tracker's output with pandas (Skill 7).

The table has one row per player per frame:
frame, id, x1, y1, x2, y2, conf, shirt_a, shirt_b
"""
import math

import pandas as pd

COLUMNS = ["frame", "id", "x1", "y1", "x2", "y2", "conf", "shirt_a", "shirt_b"]
POSITION = ["x1", "y1", "x2", "y2", "foot_x", "foot_y"]


def to_table(rows):
    """Turn a list of (frame, id, x1, y1, x2, y2, conf, shirt_a, shirt_b) into a DataFrame."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["foot_x"] = (df["x1"] + df["x2"]) / 2    # the middle of the box...
    df["foot_y"] = df["y2"]                     # ...at the bottom: where the feet are
    df["h"] = df["y2"] - df["y1"]               # box height = the player's size on screen
    return df


def id_summary(df):
    """One row per ID: when it was first and last seen, and how many frames it is missing."""
    s = df.groupby("id")["frame"].agg(first="min", last="max", seen="count")
    s["span"] = s["last"] - s["first"] + 1      # frames from the first to the last sighting
    s["missing"] = s["span"] - s["seen"]        # holes inside that span
    return s.sort_values("seen", ascending=False)


def link_broken_tracks(df, fps, max_gap_s=1.5, max_colour_dist=15):
    """Join tracks that are really one player, then return (new table, number of joins).

    When a player is hidden behind someone for a moment, the tracker often gives
    him a new ID when he comes back. We join track A to track B when:
      - B starts at most max_gap_s seconds after A ends,
      - B starts where A could have run to in that time (about 4 body heights a second),
      - and both have the same shirt colour (so a red player never joins a blue one).
    """
    t = (df.sort_values("frame").groupby("id")
         .agg(start=("frame", "min"), end=("frame", "max"),
              x_start=("foot_x", "first"), y_start=("foot_y", "first"),
              x_end=("foot_x", "last"), y_end=("foot_y", "last"),
              h=("h", "median"), a=("shirt_a", "median"), b=("shirt_b", "median")))

    pairs = []                                  # (cost, A, B) for every possible join
    for a_id, A in t.iterrows():
        for b_id, B in t.iterrows():
            gap = B["start"] - A["end"]         # frames between A's end and B's start
            if a_id == b_id or gap <= 0 or gap > max_gap_s * fps:
                continue
            moved = math.hypot(B["x_start"] - A["x_end"], B["y_start"] - A["y_end"]) / A["h"]
            allowed = 0.5 + 4 * gap / fps       # body heights he could have run
            colour = math.hypot(B["a"] - A["a"], B["b"] - A["b"])
            if moved <= allowed and colour <= max_colour_dist:     # NaN colour fails here
                pairs.append((moved / allowed + gap / fps, a_id, b_id))

    # Best joins first. Each track can have one track before it and one after it.
    nxt, has_prev = {}, set()
    for _, a_id, b_id in sorted(pairs):
        if a_id not in nxt and b_id not in has_prev:
            nxt[a_id] = b_id
            has_prev.add(b_id)

    # Follow each chain A -> B -> C ... and give all of it the first track's ID.
    new_id = {}
    for first in t.index:
        if first in has_prev:                   # not the start of a chain
            continue
        cur = first
        while True:
            new_id[cur] = first
            if cur not in nxt:
                break
            cur = nxt[cur]

    out = df.copy()
    out["id"] = out["id"].map(new_id)
    return out, len(nxt)


def renumber(df):
    """Give the players short IDs 1, 2, 3 ... in the order they first appear."""
    first_seen = df.groupby("id")["frame"].min().sort_values(kind="stable")
    new_id = {old: i for i, old in enumerate(first_seen.index, start=1)}
    out = df.copy()
    out["id"] = out["id"].map(new_id)
    return out


def drop_short(df, min_frames):
    """Remove IDs seen in fewer than min_frames frames: usually flickers, not real players."""
    seen = df.groupby("id")["frame"].transform("count")     # for every row: how often its ID was seen
    return df[seen >= min_frames]


def short_holes(missing, max_len):
    """True for the missing rows that belong to a hole of at most max_len rows in a row."""
    hole_id = (~missing).cumsum()                       # rows of the same hole share a number
    hole_len = missing.groupby(hole_id).transform("sum")    # how long each row's hole is
    return missing & (hole_len <= max_len)


def fill_gaps(df, max_gap=60):
    """Fill the holes in each player's track with straight lines (interpolate).

    A hole is a frame where we kept the ID but YOLO missed the player.
    Only holes of up to max_gap frames are filled; longer holes stay empty, because a
    straight line over a long time is a guess, not the player. Filled rows get filled=True.
    """
    pieces = []
    for pid, track in df.groupby("id"):
        track = track.set_index("frame").sort_index()
        every_frame = range(track.index.min(), track.index.max() + 1)
        track = track.reindex(every_frame)      # the missing frames appear as empty (NaN) rows
        track.index.name = "frame"
        track["filled"] = track["conf"].isna()  # remember which rows we are about to invent
        fill = short_holes(track["foot_x"].isna(), max_gap)
        guess = track[POSITION].interpolate(limit_area="inside")
        track.loc[fill, POSITION] = guess.loc[fill]
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
