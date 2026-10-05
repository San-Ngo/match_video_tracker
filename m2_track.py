"""Milestone 2: give every player an ID that stays with him for the whole clip.

Run it from the project folder, with your .venv active:
    python m2_track.py data/clip.mov

It works in two passes over the video:
    1. analyse: YOLO + tracker on every frame, plus each player's shirt colour
    2. clean up with pandas: join broken tracks, drop flickers, fill holes, smooth
    3. draw:    read the video again and draw the CLEAN tracks

It saves three files in outputs/:
    m2_track.mp4      the video with an ID under every player
    tracks.csv        one row per player per frame, straight from the tracker
    tracks_clean.csv  after joining broken tracks, filling holes and smoothing
"""
import argparse
from pathlib import Path

import cv2

from match_video_tracker.colour import shirt_colour, to_lab
from match_video_tracker.detect import load_model, pick_device
from match_video_tracker.draw import draw_tracks
from match_video_tracker.grass import foot_grass_fraction, grass_mask
from match_video_tracker.track import track_people
from match_video_tracker.tracks import (drop_short, fill_gaps, id_summary, link_broken_tracks,
                                        renumber, smooth, to_table)
from match_video_tracker.video import open_writer, video_info

OUTPUTS = Path("outputs")


def analyse(model, args):
    """Pass 1: track every player and measure his shirt colour. Returns rows and fps."""
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}. Check the file name and folder.")
    fps, w, h, total = video_info(cap)

    rows = []           # one (frame, id, x1, y1, x2, y2, conf, shirt_a, shirt_b) per player per frame
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tracks = track_people(model, frame, tracker=args.tracker, conf=args.conf,
                              imgsz=args.imgsz, device=args.device)
        mask = grass_mask(frame)
        lab = to_lab(frame)
        for t in tracks:
            box = t[1:5]
            if foot_grass_fraction(mask, box) >= args.min_grass:   # same crowd filter as M1
                rows.append((n, *t, *shirt_colour(lab, mask, box)))
        if n % 25 == 0:
            print(f"analysing frame {n}/{total}")
    cap.release()
    return rows, fps


def draw(clean, args):
    """Pass 3: read the video again and draw the clean tracks on every frame."""
    by_frame = {f: g for f, g in clean.groupby("frame")}      # frame number -> its rows
    cap = cv2.VideoCapture(args.source)
    fps, w, h, total = video_info(cap)
    writer = open_writer(OUTPUTS / "m2_track.mp4", fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        rows = by_frame.get(n)
        tracks = [] if rows is None else list(rows[["id", "x1", "y1", "x2", "y2", "conf"]]
                                              .itertuples(index=False, name=None))
        writer.write(draw_tracks(frame, tracks))
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Milestone 2: track the players")
    parser.add_argument("source", help="a video (.mp4, .mov)")
    parser.add_argument("--model", default="yolo26m.pt", help="YOLO weights")
    parser.add_argument("--tracker", default="trackers/tracktrack.yaml",
                        help="tracker settings: trackers/tracktrack.yaml or trackers/botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.1, help="low on purpose: the tracker uses weak boxes")
    parser.add_argument("--imgsz", type=int, default=1280, help="bigger finds far-away players but is slower")
    parser.add_argument("--min-grass", type=float, default=0.15, help="share of grass needed at the feet")
    parser.add_argument("--device", default=None, help="mps or cpu (default: mps when available)")
    parser.add_argument("--out", default="outputs", help="folder for the results (analyse.py uses outputs/<clip name>)")
    args = parser.parse_args()
    global OUTPUTS
    OUTPUTS = Path(args.out)                                # every file of this run goes here
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    args.device = args.device or pick_device()
    print(f"Loading {args.model} on {args.device} ...")
    model = load_model(args.model)

    # 1. analyse
    rows, fps = analyse(model, args)
    raw = to_table(rows)
    raw.to_csv(OUTPUTS / "tracks.csv", index=False)

    # 2. clean up with pandas (Skill 7)
    linked, joins = link_broken_tracks(raw, fps)                # one ID per player, even after he was hidden
    clean = renumber(drop_short(linked, min_frames=int(fps / 2)))   # drop flickers, IDs 1, 2, 3 ...
    clean = smooth(fill_gaps(clean, max_gap=int(fps)), window=5)    # fill holes up to 1 s, smooth
    clean.to_csv(OUTPUTS / "tracks_clean.csv", index=False)

    # 3. draw
    print("drawing the video ...")
    draw(clean, args)

    summary = id_summary(clean)
    print(f"\nDone -> outputs/m2_track.mp4")
    print(f"Tracker IDs: {raw['id'].nunique()}   joined broken tracks: {joins}   "
          f"players after cleaning: {len(summary)}")
    print(f"Filled {int(clean['filled'].sum())} missing boxes -> outputs/tracks_clean.csv")
    print("\nThe 10 longest tracks:")
    print(summary.head(10).to_string())


if __name__ == "__main__":
    main()
