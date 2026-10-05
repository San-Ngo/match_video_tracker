"""Milestone 2: give every player an ID that stays with him for the whole clip.

Run it from the project folder, with your .venv active:
    python m2_track.py data/clip.mov
It saves three files in outputs/:
    m2_track.mp4      the video with an ID under every player
    tracks.csv        one row per player per frame, straight from the tracker
    tracks_clean.csv  the same, with short holes filled and the paths smoothed
"""
import argparse
from pathlib import Path

import cv2

from match_video_tracker.detect import load_model, pick_device
from match_video_tracker.draw import draw_tracks
from match_video_tracker.grass import foot_grass_fraction, grass_mask
from match_video_tracker.track import track_people
from match_video_tracker.tracks import drop_short, fill_gaps, id_summary, smooth, to_table
from match_video_tracker.video import open_writer, video_info

OUTPUTS = Path("outputs")


def main():
    parser = argparse.ArgumentParser(description="Milestone 2: track the players")
    parser.add_argument("source", help="a video (.mp4, .mov)")
    parser.add_argument("--model", default="yolo26m.pt", help="YOLO weights")
    parser.add_argument("--tracker", default="trackers/botsort.yaml", help="tracker settings file")
    parser.add_argument("--conf", type=float, default=0.1, help="low on purpose: the tracker uses weak boxes")
    parser.add_argument("--imgsz", type=int, default=1280, help="bigger finds far-away players but is slower")
    parser.add_argument("--min-grass", type=float, default=0.15, help="share of grass needed at the feet")
    parser.add_argument("--device", default=None, help="mps or cpu (default: mps when available)")
    args = parser.parse_args()

    OUTPUTS.mkdir(exist_ok=True)
    args.device = args.device or pick_device()
    print(f"Loading {args.model} on {args.device} ...")
    model = load_model(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}. Check the file name and folder.")
    fps, w, h, total = video_info(cap)
    writer = open_writer(OUTPUTS / "m2_track.mp4", fps, (w, h))

    rows = []           # one (frame, id, x1, y1, x2, y2, conf) per player per frame
    short_id = {}       # the tracker's ID -> 1, 2, 3 ... in the order players first appear
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tracks = track_people(model, frame, tracker=args.tracker, conf=args.conf,
                              imgsz=args.imgsz, device=args.device)

        # Same crowd filter as Milestone 1: keep only people standing on grass.
        mask = grass_mask(frame)
        players = [t for t in tracks if foot_grass_fraction(mask, t[1:5]) >= args.min_grass]

        # Fans get tracker IDs too, so the tracker's numbers grow fast (437, 507 ...).
        # Give each player a short number instead, in the order they first appear.
        for t in players:
            if t[0] not in short_id:
                short_id[t[0]] = len(short_id) + 1
        players = [(short_id[t[0]], *t[1:]) for t in players]

        rows.extend((n, *t) for t in players)
        writer.write(draw_tracks(frame, players))
        if n % 25 == 0:
            print(f"frame {n}/{total}: {len(players)} players tracked")
    cap.release()
    writer.release()

    # --- pandas part (Skill 7) ---
    df = to_table(rows)
    df.to_csv(OUTPUTS / "tracks.csv", index=False)
    clean = drop_short(df, min_frames=int(fps / 2))              # drop IDs seen for less than 0.5 s
    clean = smooth(fill_gaps(clean, max_gap=int(fps)), window=5)  # fill holes up to 1 s, then smooth
    clean.to_csv(OUTPUTS / "tracks_clean.csv", index=False)

    summary = id_summary(df)
    long_ids = summary[summary["seen"] >= fps]          # IDs that lasted at least 1 second
    print(f"\nDone: {n} frames -> outputs/m2_track.mp4")
    print(f"Unique player IDs: {len(summary)}  (lasted 1 s or more: {len(long_ids)})")
    print(f"Filled {int(clean['filled'].sum())} missing boxes -> outputs/tracks_clean.csv")
    print("\nThe 10 longest tracks:")
    print(summary.head(10).to_string())


if __name__ == "__main__":
    main()
