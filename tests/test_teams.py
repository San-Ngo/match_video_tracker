import numpy as np
import pandas as pd

from match_video_tracker.teams import (OTHER, find_switch, fit_teams, label_players, player_colours,
                                       split_team_switches)

RED = (155.0, 140.0)        # shirt colour as (a*, b*) in Lab
BLUE = (135.0, 105.0)
YELLOW = (112.0, 177.0)     # the referee


def shirts(players, seed=0):
    """A small tracks table. players = [(colour, frames), ...]; player 1 is the first one."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid, (colour, frames) in enumerate(players, start=1):
        for f in range(1, frames + 1):
            a, b = np.array(colour) + rng.normal(0, 3, size=2)     # real colours wobble a bit
            rows.append({"frame": f, "id": pid, "shirt_a": a, "shirt_b": b})
    return pd.DataFrame(rows)


def near(centre, colour):
    return np.linalg.norm(np.array(centre) - np.array(colour)) < 5


def test_fit_teams_finds_both_kits_and_the_referee_is_other():
    # 5 blue, 4 red, and a referee who is on screen 3 times longer than anybody
    df = shirts([(BLUE, 100)] * 5 + [(RED, 100)] * 4 + [(YELLOW, 300)])
    centres = fit_teams(player_colours(df))
    assert near(centres[0], BLUE)           # the team with the most players comes first
    assert near(centres[1], RED)
    teams = label_players(df, centres)["team"]
    assert list(teams) == [0] * 5 + [1] * 4 + [OTHER]


def test_fit_teams_works_when_one_team_is_small():
    df = shirts([(BLUE, 445)] * 9 + [(RED, 100)] * 3 + [(YELLOW, 445)])
    centres = fit_teams(player_colours(df))
    assert near(centres[0], BLUE) and near(centres[1], RED)


def test_find_switch_finds_where_a_track_changes_team():
    team = np.array([0.0] * 40 + [1.0] * 40)
    assert find_switch(team, min_frames=30) == 40


def test_a_player_who_passes_an_opponent_is_not_a_switch():
    team = np.array([0.0] * 50 + [1.0] * 10 + [0.0] * 50)   # 10 frames of red: someone ran past
    assert find_switch(team, min_frames=30) is None


def test_split_team_switches_gives_the_second_player_a_new_id():
    blue = shirts([(BLUE, 40)])
    red = shirts([(RED, 40)], seed=1)
    red["frame"] += 40                      # the same ID 1 jumps to a red player at frame 41
    df = pd.concat([blue, red], ignore_index=True)
    centres = np.array([BLUE, RED])
    out, cuts = split_team_switches(df, centres, min_frames=30)
    assert cuts == 1
    assert set(out[out["frame"] <= 40]["id"]) == {1}
    assert set(out[out["frame"] > 40]["id"]) == {2}
