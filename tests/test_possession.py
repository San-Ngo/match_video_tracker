import numpy as np
import pandas as pd

from match_video_tracker.possession import events, holders, spells, team_in_possession


def pairs_for(frames, dists):
    """Ball-to-player distances: dists = {player id: distance} the same in every frame."""
    rows = [{"frame": f, "id": pid, "team": 0, "dist": d, "above": 0.0}
            for f in frames for pid, d in dists.items()]
    return pd.DataFrame(rows)


def test_holder_keeps_the_ball_in_a_close_duel():
    frames = range(1, 11)
    first = pairs_for(range(1, 6), {7: 0.2})                       # 7 alone on the ball
    duel = pairs_for(range(6, 11), {7: 0.3, 9: 0.2})               # 9 a bit closer, not clearly
    speed = pd.Series(2.0, index=frames)
    h = holders(pd.concat([first, duel]), speed, fps=10)
    assert set(h.dropna()) == {7}


def test_a_fast_ball_belongs_to_nobody():
    speed = pd.Series(30.0, index=range(1, 6))                     # a shot flying past
    h = holders(pairs_for(range(1, 6), {7: 0.1}), speed, fps=10)
    assert h.isna().all()


def test_team_keeps_possession_only_for_a_moment_after_the_last_touch():
    holder = pd.Series([7, 7, np.nan, np.nan, np.nan, np.nan], index=range(1, 7), dtype=float)
    team = team_in_possession(holder, {7: 1}, fps=2, carry_s=1.5)  # 1.5 s = 3 frames
    assert list(team.iloc[:5]) == [1, 1, 1, 1, 1]
    assert np.isnan(team.iloc[5])


def test_events_tell_passes_from_turnovers():
    holder = pd.Series([7, 7, 8, 8, 20, 20], index=range(1, 7), dtype=float)
    ev = events(spells(holder), {7: 0, 8: 0, 20: 1})
    assert list(ev["event"]) == ["pass", "turnover"]


def test_a_spell_on_guesses_alone_is_not_possession():
    frames = range(1, 11)
    speed = pd.Series(2.0, index=frames)
    seen = pd.Series([True] + [False] * 9, index=frames)          # seen once, then only filled in
    h = holders(pairs_for(frames, {7: 0.2}), speed, fps=10, seen=seen)
    assert h.isna().all()
