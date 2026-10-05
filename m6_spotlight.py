"""Milestone 6: a broadcast-style spotlight on one player.

Run it AFTER m4_camera.py (and m5_possession.py, to pick the player automatically):
    python m6_spotlight.py data/clip.mov                 the player who had the ball the most
    python m6_spotlight.py data/clip.mov --player 8      any player ID you see in the M3 video

What you get, on every frame:
    - a glowing ring under his feet, in his team's colour
    - a trail of where he ran in the last few seconds. The trail is kept in pitch
      coordinates (M4), so it stays on the grass where he really ran while the camera pans
    - both drawn only on grass pixels, so his boots and legs stay in front of them

It saves in outputs/:
    m6_spotlight.mp4     the clip with the spotlight
    m6_spotlight.jpg     one frame before (left) and after (right)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from match_video_tracker.camera import from_pitch, to_pitch
from match_video_tracker.colour import shirt_to_bgr
from match_video_tracker.draw import BLACK, FONT, text_color
from match_video_tracker.grass import grass_mask
from match_video_tracker.spotlight import TRAIL_SECONDS, draw_spotlight
from match_video_tracker.teams import OTHER
from match_video_tracker.video import open_writer, video_info

OUTPUTS = Path("outputs")


def pick_player(tracks):
    """The player who had the ball the most (M5), or else the one seen the longest."""
    possession = OUTPUTS / "possession.csv"
    if possession.exists():
        holder = pd.read_csv(possession)["holder"].dropna()
        if len(holder):
            return int(holder.mode().iat[0])
    return int(tracks.groupby("id")["frame"].count().idxmax())


def name_tag(out, box, label, color):
    """A small tag with his ID above his head."""
    scale = max(0.5, out.shape[1] / 1920)
    thick = max(2, round(2 * scale))
    x1, y1, x2, _ = box
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.7 * scale, thick)
    cx, top = int((x1 + x2) / 2), int(y1) - int(12 * scale)
    cv2.rectangle(out, (cx - tw // 2 - 6, top - th - 10), (cx + tw // 2 + 6, top), color, -1)
    cv2.rectangle(out, (cx - tw // 2 - 6, top - th - 10), (cx + tw // 2 + 6, top), BLACK, 1)
    cv2.putText(out, label, (cx - tw // 2, top - 5), FONT, 0.7 * scale, text_color(color), thick, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Milestone 6: spotlight one player")
    parser.add_argument("source", help="the same video you gave to m2 - m5")
    parser.add_argument("--player", type=int, default=None, help="the player's ID (default: most time on the ball)")
    parser.add_argument("--seconds", type=float, default=TRAIL_SECONDS, help="length of the trail in seconds")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}. Check the file name and folder.")
    fps, w, h, total = video_info(cap)
    for needed in ("tracks_teams.csv", "camera.csv", "teams.csv"):
        if not (OUTPUTS / needed).exists():
            raise SystemExit(f"outputs/{needed} is missing. Run m3_teams.py and m4_camera.py first.")
    tracks = pd.read_csv(OUTPUTS / "tracks_teams.csv")
    camera = pd.read_csv(OUTPUTS / "camera.csv", index_col="frame")

    pid = args.player if args.player is not None else pick_player(tracks)
    me = tracks[tracks["id"] == pid].sort_values("frame")
    if me.empty:
        raise SystemExit(f"There is no player {pid}. Open outputs/m3_teams.mp4 to see the IDs.")
    me = to_pitch(me, camera, x="foot_x_smooth", y="foot_y_smooth").set_index("frame")

    players = pd.read_csv(OUTPUTS / "teams.csv", index_col="id")
    team = players.loc[pid, "team"] if pid in players.index else OTHER
    kit = players[players["team"] == team][["shirt_a", "shirt_b"]].median()
    color = shirt_to_bgr(*kit, lightness=170) if team != OTHER else (255, 255, 255)
    cuts = camera.index[camera["cut"]].tolist()             # a trail never crosses a scene cut
    keep = int(args.seconds * fps)

    writer = open_writer(OUTPUTS / "m6_spotlight.mp4", fps, (w, h))
    still_at = int(me.index[len(me) // 2])                  # the before/after picture
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        start = max([n - keep] + [c for c in cuts if c <= n])
        recent = me.loc[(me.index >= start) & (me.index <= n), ["pitch_x", "pitch_y"]]
        trail = from_pitch(recent.to_numpy(), camera, n) if len(recent) > 1 else []
        box = tuple(me.loc[n, ["x1", "y1", "x2", "y2"]]) if n in me.index else None
        out = draw_spotlight(frame, grass_mask(frame), box, trail, color)
        if box is not None:
            name_tag(out, box, str(pid), color)
        writer.write(out)
        if n == still_at:
            cv2.imwrite(str(OUTPUTS / "m6_spotlight.jpg"), np.hstack([frame, out])[:, ::2][::2])
    cap.release()
    writer.release()

    print(f"\nDone -> outputs/m6_spotlight.mp4 and outputs/m6_spotlight.jpg")
    print(f"Spotlight on player {pid} (team {'other' if team == OTHER else team + 1}), "
          f"seen in {len(me)} of {total} frames, trail of {args.seconds:g} s.")
    print("Try another player: python m6_spotlight.py " + args.source + " --player <ID>")


if __name__ == "__main__":
    main()
