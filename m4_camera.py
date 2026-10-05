"""Milestone 4: measure how the camera moves, and each player's real movement.

Run it AFTER m3_teams.py, on the same video, with your .venv active:
    python m4_camera.py data/clip.mov

No YOLO here, so it is fast. For every frame it:
    1. follows ~500 points on the pitch, ads and stands with optical flow (players masked out)
    2. lets RANSAC pick the one camera move most points agree on (a pan, a zoom)
    3. adds the moves up, to turn every pixel into "pitch coordinates" (where it would be
       in the first frame): the camera's pan is taken out, the player's own run stays

It saves in outputs/:
    camera.csv         one row per frame: the camera move, and the matrix to pitch coordinates
    tracks_pitch.csv   tracks_teams.csv plus pitch_x / pitch_y for every player's feet
    m4_camera.png      the camera pan over the clip, and one player's path with and without it
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                   # draw to a file, no window
import matplotlib.pyplot as plt
import pandas as pd

from match_video_tracker.camera import to_pitch, track_camera
from match_video_tracker.video import video_info

OUTPUTS = Path("outputs")


def frames_with_boxes(source, tracks):
    """Yield (frame number, frame, player boxes) for every frame of the video."""
    boxes = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in tracks.groupby("frame")}
    cap = cv2.VideoCapture(source)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % 50 == 0:
            print(f"following the camera: frame {n}")
        yield n, frame, boxes.get(n, [])
    cap.release()


def plot(camera, tracks, fps, path):
    """Left: how far the camera has panned. Right: the longest-tracked player's path
    on the screen (with the pan in it) and on the pitch (pan taken out)."""
    seconds = camera.index / fps
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.plot(seconds, -camera["to_first_02"], label="sideways")
    left.plot(seconds, -camera["to_first_12"], label="up / down")
    for t in seconds[camera["cut"]]:
        left.axvline(t, color="grey", linestyle=":")
    left.set(title="How far the picture has slid since the first frame", xlabel="seconds", ylabel="pixels")
    left.legend()

    pid = tracks.groupby("id")["frame"].count().idxmax()
    p = tracks[tracks["id"] == pid].sort_values("frame")
    right.plot(p["foot_x"], p["foot_y"], label="on the screen (camera pan included)")
    right.plot(p["pitch_x"], p["pitch_y"], label="on the pitch (pan taken out)")
    right.invert_yaxis()                # image y goes down
    right.set(title=f"Player {pid}'s path", xlabel="x (pixels)", ylabel="y (pixels)")
    right.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser(description="Milestone 4: measure the camera's movement")
    parser.add_argument("source", help="the same video you gave to m2 and m3")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}. Check the file name and folder.")
    fps, w, h, total = video_info(cap)
    cap.release()
    tracks = pd.read_csv(OUTPUTS / "tracks_teams.csv")
    if total and tracks["frame"].max() > total:
        raise SystemExit("outputs/tracks_teams.csv does not belong to this video. Run m2 and m3 on it first.")

    camera = track_camera(frames_with_boxes(args.source, tracks))
    camera.to_csv(OUTPUTS / "camera.csv")
    tracks = to_pitch(tracks, camera, x="foot_x", y="foot_y")
    tracks.to_csv(OUTPUTS / "tracks_pitch.csv", index=False)
    plot(camera, tracks, fps, OUTPUTS / "m4_camera.png")

    pan_x, pan_y = -camera["to_first_02"].iloc[-1], -camera["to_first_12"].iloc[-1]
    print("\nDone -> outputs/camera.csv, outputs/tracks_pitch.csv, outputs/m4_camera.png")
    print(f"Camera moved {pan_x:+.0f} px left/right and {pan_y:+.0f} px up/down over the clip.")
    print(f"Points that agreed with the camera move: {camera['agree'].iloc[1:].median():.0f} per frame (median).")
    print(f"Scene cuts: {int(camera['cut'].sum())}")


if __name__ == "__main__":
    main()
