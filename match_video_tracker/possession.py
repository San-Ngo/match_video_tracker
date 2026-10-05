"""Who has the ball, and possession per team (Skill 7, steps 4-6).

1. In each frame, the team player closest to the ball has it, if the ball is within
   POSSESSION_DIST body heights of his feet. Otherwise nobody has it (a pass in the air).
   The ball must also be low and slow: a shot flying past a defender is nobody's.
2. Flickers are ignored: a player must keep the ball for MIN_TOUCH seconds to count,
   and in a duel the player who had the ball keeps it unless the other is clearly closer.
3. A team keeps possession for up to CARRY seconds after its last touch, so a pass in
   the air still counts for the team that made it.
4. Passes and turnovers come from the order of the players who had the ball.
"""
import numpy as np
import pandas as pd

from match_video_tracker.teams import OTHER

POSSESSION_DIST = 0.5   # body heights between the ball and a player's feet
MIN_TOUCH = 0.1         # seconds: a shorter "possession" is just noise in a duel
MAX_CONTROL = 6         # body heights per second (about 11 m/s): faster = a pass or shot in flight
FEET = 0.25             # the ball must be at most this far above his feet (body heights)
STICKY = 0.3            # body heights: how much closer a challenger must be to take the ball
CARRY = 1.5             # seconds a team keeps possession without a touch (a pass in the air)


def ball_to_players(ball, tracks):
    """For every frame with a ball: every team player's distance to it (body heights) and
    how high above his feet it is. Referees and keepers (team OTHER) are left out: we
    can't tell which team a keeper is on.
    """
    players = tracks[tracks["team"] != OTHER][["frame", "id", "team", "foot_x", "foot_y", "y1", "y2"]]
    b = ball.dropna(subset=["x"]).reset_index()            # frame, x, y, source
    pairs = b.merge(players, on="frame")                   # every ball x every player in that frame
    h = pairs["y2"] - pairs["y1"]
    pairs["dist"] = np.hypot(pairs["foot_x"] - pairs["x"], pairs["foot_y"] - pairs["y"]) / h
    pairs["above"] = (pairs["foot_y"] - pairs["y"]) / h   # how high above his feet the ball is
    return pairs[["frame", "id", "team", "dist", "above"]]


def holders(pairs, speed, fps, max_dist=POSSESSION_DIST, min_touch=MIN_TOUCH):
    """The ID of the player on the ball in each frame (NaN = nobody), without flickers.

    A player can have the ball when it is close to his feet, low (not flying past at
    knee height) and slow enough to control (speed = ball_speed, per frame). If two
    players are close, the one who already had it keeps it unless the other one is
    clearly closer (by STICKY body heights): that's how a duel looks to a fan.
    """
    ok = (pairs["dist"] <= max_dist) & (pairs["above"] <= FEET)
    ok &= pairs["frame"].map(speed) <= MAX_CONTROL
    close = {f: g.set_index("id")["dist"] for f, g in pairs[ok].groupby("frame")}
    holder, current = [], None
    for f in speed.index:
        d = close.get(f)
        if d is None:
            holder.append(np.nan)                          # nobody has it (lost, in flight)
            continue
        best = d.idxmin()
        if current in d.index and d[current] <= d[best] + STICKY:
            best = current                                 # he keeps it in a close duel
        holder.append(best)
        current = best
    holder = pd.Series(holder, index=speed.index, dtype=float)
    run = (holder != holder.shift()).cumsum()              # number each run of the same holder
    run_len = holder.groupby(run).transform("size")
    return holder.where(run_len >= max(1, round(min_touch * fps)))


def team_in_possession(holder, team_of, fps, carry_s=CARRY):
    """The team in possession in each frame: the team of the last player who had the ball,
    for up to carry_s seconds after his last touch (a pass in the air). After that
    nobody (the ball is dead, out, or in the net).
    """
    return holder.map(team_of).ffill(limit=int(carry_s * fps))


def spells(holder):
    """Each time a player had the ball: his ID and the first and last frame, in order."""
    h = holder.dropna().astype(int)
    if h.empty:
        return pd.DataFrame(columns=["id", "first", "last"])
    run = (h != h.shift()).cumsum()                        # the same player again = same spell
    frames = h.index.to_series()
    return pd.DataFrame({"id": h.groupby(run).first(),
                         "first": frames.groupby(run).min(),
                         "last": frames.groupby(run).max()}).reset_index(drop=True)


def events(spell_table, team_of):
    """Passes (to a teammate) and turnovers (to an opponent), from one spell to the next."""
    rows = []
    for before, after in zip(spell_table.itertuples(), spell_table.iloc[1:].itertuples()):
        same_team = team_of[before.id] == team_of[after.id]
        rows.append({"frame": after.first, "from_id": before.id, "to_id": after.id,
                     "event": "pass" if same_team else "turnover",
                     "team": team_of[before.id]})
    return pd.DataFrame(rows, columns=["frame", "from_id", "to_id", "event", "team"])


def possession_share(team):
    """Share of the frames each team had the ball: {0: 0.4, 1: 0.6}. NaN frames don't count."""
    counts = team.dropna().astype(int).value_counts()
    return {t: counts.get(t, 0) / max(counts.sum(), 1) for t in (0, 1)}
