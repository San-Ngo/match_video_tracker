"""Milestone 1: find the players in an image or a video, and remove the crowd.

Run it from the project folder, with your .venv active:
    python m1_detect.py data/frame.png
    python m1_detect.py data/clip.mov
The results are saved in the outputs/ folder.
"""
import argparse
from pathlib import Path

import cv2

from match_video_tracker.detect import detect_people, load_model, pick_device, split_players_and_crowd
from match_video_tracker.draw import draw_people
from match_video_tracker.grass import grass_mask
from match_video_tracker.video import open_writer, video_info

VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
OUTPUTS = Path("outputs")


def process_frame(model, frame, args):
    """One frame: detect people -> split players from crowd -> draw."""
    people = detect_people(model, frame, conf=args.conf, imgsz=args.imgsz, device=args.device)
    mask = grass_mask(frame)
    players, crowd = split_players_and_crowd(people, mask, min_grass=args.min_grass)
    return draw_people(frame, players, crowd), players, crowd


def run_on_image(model, path, args):
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read {path}. Check the file name and folder.")

    out, players, crowd = process_frame(model, frame, args)
    cv2.imwrite(str(OUTPUTS / "m1_detect.jpg"), out)
    # Also save the grass mask (white = grass), so you can see what the filter sees.
    cv2.imwrite(str(OUTPUTS / "m1_grass_mask.png"), grass_mask(frame).astype("uint8") * 255)
    print(f"{len(players)} players, {len(crowd)} crowd removed -> outputs/m1_detect.jpg")


def run_on_video(model, path, args):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}. Check the file name and folder.")

    fps, w, h, total = video_info(cap)
    writer = open_writer(OUTPUTS / "m1_detect.mp4", fps, (w, h))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:                          # no more frames: the video is over
            break
        out, players, crowd = process_frame(model, frame, args)
        writer.write(out)
        n += 1
        if n % 25 == 0:                     # a progress line every 25 frames
            print(f"frame {n}/{total}: {len(players)} players, {len(crowd)} crowd removed")

    cap.release()
    writer.release()
    print(f"Done: {n} frames -> outputs/m1_detect.mp4")


def main():
    parser = argparse.ArgumentParser(description="Milestone 1: detect players and remove the crowd")
    parser.add_argument("source", help="an image (.jpg, .png) or a video (.mp4, .mov)")
    parser.add_argument("--model", default="yolo26m.pt", help="YOLO weights, downloaded the first time")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum YOLO confidence, 0 to 1")
    parser.add_argument("--imgsz", type=int, default=1280, help="bigger finds far-away players but is slower")
    parser.add_argument("--min-grass", type=float, default=0.15, help="share of grass needed at the feet")
    parser.add_argument("--device", default=None, help="mps or cpu (default: mps when available)")
    args = parser.parse_args()

    OUTPUTS.mkdir(exist_ok=True)
    args.device = args.device or pick_device()
    print(f"Loading {args.model} on {args.device} ...")
    model = load_model(args.model)

    path = Path(args.source)
    if path.suffix.lower() in VIDEO_TYPES:
        run_on_video(model, path, args)
    else:
        run_on_image(model, path, args)


if __name__ == "__main__":      # True only when you run this file directly
    main()
