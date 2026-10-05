"""Split the players into two teams by shirt colour (Skill 6, steps 2-4).

1. Every player gets one colour: the median of his shirt colours over all frames.
2. K-Means finds the two team colours (see fit_teams for how referees are kept out).
3. Anyone far from both team colours is OTHER: a referee, a keeper, a sub in a bib.
4. If an ID jumps from a blue player to a red one (two players collided and the
   tracker swapped them), the colour change gives it away: we cut the track there.
"""
import numpy as np
from sklearn.cluster import KMeans

OTHER = -1          # the "team" of referees, keepers and anyone else
OTHER_DIST = 20     # further than this from both team colours (in a*/b* units) = OTHER
SPARE = 2           # extra K-Means groups that catch the referees and keepers


def player_colours(df):
    """One row per ID: his median shirt colour (a*, b*) and the frames we measured it in."""
    seen = df.dropna(subset=["shirt_a", "shirt_b"])
    out = seen.groupby("id")[["shirt_a", "shirt_b"]].median()
    out["frames"] = seen.groupby("id").size()
    return out


def distances(colours, centres):
    """Distance from every colour (one per row) to every team colour (one per column)."""
    X = np.asarray(colours, dtype=float)
    return np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)


def fit_teams(colours):
    """Find the 2 team colours with K-Means, without letting referees or keepers pull them.

    K-Means puts everybody in a group. With k=2 a referee joins one of the teams and
    drags its colour towards his shirt. So we do it in 3 steps:
      1. K-Means with k=4: 2 groups for the teams, plus spare groups for the others.
      2. The group with the most players is a team. The next biggest group with a
         clearly different colour is the other team (a big team is sometimes split
         into 2 groups). A referee is one person, so his group stays small.
      3. K-Means again with k=2, only on the players close to those 2 colours.
    Every player (ID) counts once, however long he is on screen.
    Returns the 2 team colours as a (2, 2) array, the team with the most players first.
    """
    X = colours[["shirt_a", "shirt_b"]].to_numpy()
    if len(X) < 2:
        raise ValueError("Need at least 2 players to find 2 teams.")

    # 1. K-Means with spare groups
    k = min(2 + SPARE, len(X))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    size = np.bincount(km.labels_, minlength=k)                # players in each group
    groups = km.cluster_centers_[np.argsort(size, kind="stable")[::-1]]   # biggest group first

    # 2. the biggest group, and the biggest group that looks different from it
    first = groups[0]
    second = next((g for g in groups[1:] if np.linalg.norm(g - first) > OTHER_DIST), groups[1])
    centres = np.array([first, second])

    # 3. K-Means again, starting from those 2 colours, without the far-away people
    near = distances(X, centres).min(axis=1) <= OTHER_DIST
    if near.sum() >= 2:
        centres = KMeans(n_clusters=2, init=centres, n_init=1).fit(X[near]).cluster_centers_
    team = distances(X[near], centres).argmin(axis=1)
    size = np.bincount(team, minlength=2)
    return centres[np.argsort(size, kind="stable")[::-1]]      # biggest team first


def team_of(colours, centres):
    """Team (0 or 1) of each colour, OTHER if it is far from both, NaN if we have no colour."""
    d = distances(colours, centres)
    team = d.argmin(axis=1).astype(float)
    team[d.min(axis=1) > OTHER_DIST] = OTHER
    team[np.isnan(d).any(axis=1)] = np.nan
    return team


def find_switch(team, min_frames, purity=0.8):
    """The row where one track changes team for good, or None.

    team has 0, 1 or NaN (unknown) for each frame of ONE track. We look for the row
    with at least `purity` (80 %) of one team before it and of the other team after
    it, with at least min_frames known frames on each side, so a player who just
    runs past an opponent for a moment is not a switch.
    """
    t = np.asarray(team, dtype=float)
    known = ~np.isnan(t)
    ones = np.cumsum(np.where(known, t, 0))     # frames of team 1 up to each row
    seen = np.cumsum(known)                     # known frames up to each row
    best, best_jump = None, 0.0
    for i in range(1, len(t)):
        before, after = seen[i - 1], seen[-1] - seen[i - 1]
        if before < min_frames or after < min_frames:
            continue
        p_before = ones[i - 1] / before                     # share of team 1 before row i
        p_after = (ones[-1] - ones[i - 1]) / after          # share of team 1 from row i on
        changed = ((p_before <= 1 - purity and p_after >= purity) or
                   (p_before >= purity and p_after <= 1 - purity))
        if changed and abs(p_after - p_before) > best_jump:
            best, best_jump = i, abs(p_after - p_before)
    return best


def split_points(team, min_frames):
    """Every row where a track changes team (it can happen more than once), in order."""
    i = find_switch(team, min_frames)
    if i is None:
        return []
    before = split_points(team[:i], min_frames)
    after = [i + j for j in split_points(team[i:], min_frames)]
    return before + [i] + after


def split_team_switches(df, centres, min_frames):
    """Cut each track where it jumps from a player of one team to a player of the other.

    The part after a cut gets a new ID. Returns (new table, number of cuts).
    """
    df = df.sort_values(["id", "frame"]).reset_index(drop=True)
    team = team_of(df[["shirt_a", "shirt_b"]], centres)
    team[team == OTHER] = np.nan                # only team-to-team changes count
    new_id = df["id"].to_numpy().copy()
    next_id = int(new_id.max()) + 1
    cuts = 0
    for rows in df.groupby("id").indices.values():     # the row numbers of one ID
        for cut in split_points(team[rows], min_frames):
            new_id[rows[cut:]] = next_id                # from the cut on: a new player
            next_id += 1
            cuts += 1
    df["id"] = new_id
    return df, cuts


def label_players(df, centres):
    """One row per ID: his shirt colour, the frames we measured it in, and his team."""
    players = player_colours(df)
    players["team"] = team_of(players[["shirt_a", "shirt_b"]], centres).astype(int)
    return players
