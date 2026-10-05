"""Milestone 3: split the players into two teams by shirt colour.

Run it AFTER m2_track.py, on the same video, with your .venv active:
    python m3_teams.py data/clip.mov

It starts from M2's outputs/tracks_clean.csv, so there is no YOLO here and it is fast:
    1. measure: read the video and measure every player's shirt colour in every frame
    2. teams:   K-Means finds the 2 team colours; referees and keepers become "other"
    3. repair:  cut IDs that jumped to a player of the other team, then join the pieces again
    4. draw:    read the video again and draw every marker in its team's colour

It saves in outputs/:
    m3_teams.mp4      the video with a team-coloured marker under every player
    m3_check.jpg      3 frames of that video, to check the teams by eye
    tracks_teams.csv  the tracks plus a team column: 0 = Team 1, 1 = Team 2, -1 = other
    teams.csv         one row per player: his shirt colour, frames seen and team
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from match_video_tracker.colour import shirt_colour, shirt_to_bgr, to_lab
from match_video_tracker.draw import WHITE, draw_teams
from match_video_tracker.grass import grass_mask
from match_video_tracker.teams import (OTHER, fit_teams, label_players, player_colours,
                                       split_team_switches)
from match_video_tracker.tracks import fill_gaps, link_broken_tracks, smooth
from match_video_tracker.video import open_writer, video_info

OUTPUTS = Path("outputs")
CHECK_AT = (0.25, 0.5, 0.75)        # m3_check.jpg shows the frames at 25 %, 50 % and 75 % of the clip


def measure(df, source):
    """Pass 1: measure every player's shirt colour in every frame. We already know the boxes."""
    real = df[~df["filled"].astype(bool)]           # filled rows were invented by M2: nothing to measure
    by_frame = {f: g for f, g in real.groupby("frame")}
    cap = cv2.VideoCapture(source)
    colours = {}                                    # row number in df -> (a*, b*)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        rows = by_frame.get(n)
        if rows is None:
            continue
        lab, grass = to_lab(frame), grass_mask(frame)
        for row, x1, y1, x2, y2 in rows[["x1", "y1", "x2", "y2"]].itertuples():
            colours[row] = shirt_colour(lab, grass, (x1, y1, x2, y2))
    cap.release()
    shirts = pd.DataFrame.from_dict(colours, orient="index", columns=["shirt_a", "shirt_b"])
    return df.drop(columns=["shirt_a", "shirt_b"]).join(shirts)


def draw(df, source, colors, names):
    """Pass 2: read the video again and draw every player in his team's colour."""
    by_frame = {f: g for f, g in df.groupby("frame")}
    cap = cv2.VideoCapture(source)
    fps, w, h, total = video_info(cap)
    writer = open_writer(OUTPUTS / "m3_teams.mp4", fps, (w, h))
    check_frames = {int(total * share) for share in CHECK_AT}
    checks = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        rows = by_frame.get(n)
        players = [] if rows is None else list(rows[["id", "x1", "y1", "x2", "y2", "team"]]
                                               .itertuples(index=False, name=None))
        out = draw_teams(frame, players, colors, names)
        writer.write(out)
        if n in check_frames:
            checks.append(cv2.resize(out, (w // 2, h // 2)))    # half size is enough to check
    cap.release()
    writer.release()
    if checks:
        cv2.imwrite(str(OUTPUTS / "m3_check.jpg"), np.vstack(checks))  # the 3 frames on top of each other


def main():
    parser = argparse.ArgumentParser(description="Milestone 3: split the players into two teams")
    parser.add_argument("source", help="the same video you gave to m2_track.py")
    parser.add_argument("--tracks", default=str(OUTPUTS / "tracks_clean.csv"), help="M2's clean tracks")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}. Check the file name and folder.")
    fps, w, h, total = video_info(cap)
    cap.release()
    df = pd.read_csv(args.tracks)
    if total and df["frame"].max() > total:
        raise SystemExit(f"{args.tracks} does not belong to {args.source}. Run m2_track.py on it first.")

    # 1. measure
    print("measuring shirt colours ...")
    df = measure(df, args.source)

    # 2. teams (Skill 6: K-Means)
    centres = fit_teams(player_colours(df))

    # 3. repair: cut IDs that jumped to an opponent, then join the pieces with M2's link step
    df, cuts = split_team_switches(df, centres, min_frames=int(fps / 2))
    df, joins = link_broken_tracks(df, fps)
    df = smooth(fill_gaps(df, max_gap=int(fps)), window=5)     # fill the holes the joins left

    players = label_players(df, centres)
    df["team"] = df["id"].map(players["team"]).fillna(OTHER).astype(int)
    df.to_csv(OUTPUTS / "tracks_teams.csv", index=False)
    players.to_csv(OUTPUTS / "teams.csv")

    # 4. draw: each team in (a brighter version of) its own shirt colour, other people in white
    colors = {0: shirt_to_bgr(*centres[0]), 1: shirt_to_bgr(*centres[1]), OTHER: WHITE}
    names = {0: "Team 1", 1: "Team 2", OTHER: "Other"}
    print("drawing the video ...")
    draw(df, args.source, colors, names)

    count = players["team"].value_counts()
    (a1, b1), (a2, b2) = centres
    print("\nDone -> outputs/m3_teams.mp4 and outputs/m3_check.jpg")
    print(f"Team colours (a*, b*): Team 1 = ({a1:.0f}, {b1:.0f})   Team 2 = ({a2:.0f}, {b2:.0f})")
    print(f"Players: Team 1 = {count.get(0, 0)}   Team 2 = {count.get(1, 0)}   "
          f"Other = {count.get(OTHER, 0)} (referees, keepers, subs in bibs)")
    print(f"Cut {cuts} IDs that jumped to a player of the other team, then joined {joins} broken tracks.")
    print("\nEvery player (outputs/teams.csv):")
    print(players.round(1).to_string())


if __name__ == "__main__":
    main()
