"""Milestone 5: find the ball and work out who has it.

Run it AFTER m3_teams.py and m4_camera.py, on the same video, with your .venv active:
    python m5_possession.py data/clip.mov

YOLO only looks for the ball here; the players come from M3, the camera from M4.
    1. detect:     every ball-like box in every frame (the slow part, about a minute)
    2. ball path:  keep the boxes that can be the match ball (white, round, near a
                   player, and MOVING on the pitch: spare balls by the boards lie still),
                   drop jumps, fill holes
    3. possession: the ball is a player's when it is at his feet, low and slow
    4. draw:       a triangle over the ball, one over the player on the ball, possession bar

If ball.pt is in the project folder (a YOLO model fine-tuned on football, see
notebooks/train_ball.ipynb), it is used instead of yolo26m.pt and finds the ball more often.

To try other settings without running YOLO again, add --reuse:
    python m5_possession.py data/clip.mov --reuse --dist 0.75

It saves in outputs/:
    m5_possession.mp4    the video with the ball and the possession bar
    ball_candidates.csv  every ball-like box YOLO found (what --reuse reads)
    ball.csv             one ball position per frame: seen, filled or lost
    possession.csv       who has the ball in each frame and which team is in possession
    events.csv           passes and turnovers, in order
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

from match_video_tracker.ball import (add_nearest_player, ball_path, ball_speed, detect_balls,
                                      is_football_model, keep_match_ball, link_tracklets, to_table,
                                      whiteness)
from match_video_tracker.camera import to_pitch
from match_video_tracker.colour import shirt_to_bgr
from match_video_tracker.detect import load_model, pick_device
from match_video_tracker.draw import WHITE, draw_possession
from match_video_tracker.possession import (POSSESSION_DIST, ball_to_players, events, holders,
                                            possession_share, spells, team_in_possession)
from match_video_tracker.teams import OTHER
from match_video_tracker.video import open_writer, video_info

OUTPUTS = Path("outputs")


def detect(model, args):
    """Pass 1: every ball-like box in every frame, with how white it is."""
    cap = cv2.VideoCapture(args.source)
    fps, w, h, total = video_info(cap)
    rows = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for x1, y1, x2, y2, conf in detect_balls(model, frame, conf=0.05, imgsz=args.imgsz,
                                                 device=args.device, tiled=args.tiled):
            rows.append((n, x1, y1, x2, y2, conf, whiteness(hsv, (x1, y1, x2, y2))))
        if n % 25 == 0:
            print(f"looking for the ball: frame {n}/{total}")
    cap.release()
    return to_table(rows)


def draw(args, tracks, ball, possession, colors, names):
    """Pass 2: read the video again and draw everything."""
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    cap = cv2.VideoCapture(args.source)
    fps, w, h, total = video_info(cap)
    writer = open_writer(OUTPUTS / "m5_possession.mp4", fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        rows = by_frame.get(n)
        players = [] if rows is None else list(rows[["id", "x1", "y1", "x2", "y2", "team"]]
                                               .itertuples(index=False, name=None))
        x, y, source = ball.loc[n, ["x", "y", "source"]] if n in ball.index else (float("nan"),) * 3
        p = possession.loc[n] if n in possession.index else None
        holder = None if p is None or pd.isna(p["holder"]) else int(p["holder"])
        shares = None if p is None or pd.isna(p["share_1"]) else {0: p["share_1"], 1: p["share_2"]}
        writer.write(draw_possession(frame, players, colors, names,
                                     None if pd.isna(x) else (x, y, source == "seen"), holder, shares))
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Milestone 5: find the ball and who has it")
    parser.add_argument("source", help="the same video you gave to m2, m3 and m4")
    parser.add_argument("--reuse", action="store_true",
                        help="skip YOLO and reuse outputs/ball_candidates.csv from the last run")
    parser.add_argument("--dist", type=float, default=POSSESSION_DIST,
                        help="possession distance in body heights (default 0.5)")
    parser.add_argument("--model", default=None,
                        help="ball model (default: ball.pt if you trained one, else yolo26m.pt)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="default: 640 per tile for ball.pt, 1280 for the whole frame with yolo26m.pt")
    parser.add_argument("--device", default=None, help="mps or cpu (default: mps when available)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}. Check the file name and folder.")
    fps, w, h, total = video_info(cap)
    cap.release()
    tracks = pd.read_csv(OUTPUTS / "tracks_teams.csv")
    if total and tracks["frame"].max() > total:
        raise SystemExit("outputs/tracks_teams.csv does not belong to this video. Run m2 and m3 on it first.")
    if not (OUTPUTS / "camera.csv").exists():
        raise SystemExit("outputs/camera.csv is missing. Run m4_camera.py on this video first.")
    camera = pd.read_csv(OUTPUTS / "camera.csv", index_col="frame")

    # 1. detect (or reuse the last detection)
    if args.reuse:
        candidates = pd.read_csv(OUTPUTS / "ball_candidates.csv")
    else:
        args.device = args.device or pick_device()
        args.model = args.model or ("ball.pt" if Path("ball.pt").exists() else "yolo26m.pt")
        model = load_model(args.model)
        args.tiled = is_football_model(model)               # football models look at 4 tiles
        args.imgsz = args.imgsz or (640 if args.tiled else 1280)
        kind = "football model, 4 tiles" if args.tiled else "standard COCO model, whole frame"
        print(f"Loaded {args.model} on {args.device} ({kind})")
        candidates = detect(model, args)
        candidates.to_csv(OUTPUTS / "ball_candidates.csv", index=False)

    # 2. the ball's path (pitch coordinates from M4 tell a moving ball from one lying still)
    balls = add_nearest_player(to_pitch(candidates, camera), tracks)
    balls = keep_match_ball(link_tracklets(balls, fps), tracks)
    ball = ball_path(balls, total, fps)
    seen = to_pitch(ball.dropna(subset=["x"]).reset_index(), camera).set_index("frame")
    body = (tracks["y2"] - tracks["y1"]).groupby(tracks["frame"]).median()     # pixels per body height
    ball["speed"] = ball_speed(seen, body, fps)            # body heights per second
    ball.to_csv(OUTPUTS / "ball.csv")

    # 3. possession
    team_of = tracks.groupby("id")["team"].first()
    holder = holders(ball_to_players(ball, tracks), ball["speed"], fps, max_dist=args.dist,
                     seen=ball["source"] == "seen")
    team = team_in_possession(holder, team_of, fps)        # kept for a moment after the last touch
    running = pd.DataFrame({0: (team == 0).cumsum(), 1: (team == 1).cumsum()})
    total_frames = (running[0] + running[1]).where(lambda s: s > 0)
    possession = pd.DataFrame({"holder": holder, "team": team,
                               "share_1": running[0] / total_frames,
                               "share_2": running[1] / total_frames})
    possession.to_csv(OUTPUTS / "possession.csv")
    event_table = events(spells(holder), team_of)
    event_table.to_csv(OUTPUTS / "events.csv", index=False)

    # 4. draw, in the team colours from M3
    players = pd.read_csv(OUTPUTS / "teams.csv")
    kit = players[players["team"] != OTHER].groupby("team")[["shirt_a", "shirt_b"]].median()
    colors = {0: shirt_to_bgr(*kit.loc[0]), 1: shirt_to_bgr(*kit.loc[1]), OTHER: WHITE}
    names = {0: "Team 1", 1: "Team 2", OTHER: "Other"}
    print("drawing the video ...")
    draw(args, tracks, ball, possession, colors, names)

    seen = (ball["source"] == "seen").sum()
    filled = (ball["source"] == "filled").sum()
    share = possession_share(team)
    print("\nDone -> outputs/m5_possession.mp4")
    print(f"Ball: seen in {seen} of {total} frames, filled in {filled} more, lost in {total - seen - filled}.")
    print(f"Possession (ball within {args.dist} body heights): Team 1 {share[0]:.0%}   Team 2 {share[1]:.0%}")
    passes = (event_table["event"] == "pass").sum()
    print(f"Passes: {passes}   Turnovers: {len(event_table) - passes}")
    if len(event_table):
        print("\nEvents (outputs/events.csv):")
        print(event_table.to_string(index=False))


if __name__ == "__main__":
    main()
