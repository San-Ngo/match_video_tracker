"""Analyse one clip for the Studio app: runs M2, M3, M4 and M5 in a row.

    python analyse.py data/clip2.mov

Every clip gets its own folder, so clips never mix: data/clip2.mov -> outputs/clip2/
(tracks, teams, camera, ball, possession, and your drawings for that clip).
At the end it makes the light browser copy of the clip that the app plays.
On a laptop this takes a few minutes per 10 seconds of video (YOLO runs in M2 and M5).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from match_video_tracker.web import clip_folder, prepare

STEPS = ["m2_track.py", "m3_teams.py", "m4_camera.py", "m5_possession.py"]


def main():
    parser = argparse.ArgumentParser(description="Analyse a clip for the Studio app (M2 - M5)")
    parser.add_argument("source", help="a video in data/, e.g. data/clip2.mov")
    parser.add_argument("--from-step", type=int, default=2, choices=[2, 3, 4, 5],
                        help="start at this milestone (to redo only the later steps)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"There is no {source}. Put the clip in data/ first.")
    folder = clip_folder(source)
    print(f"Results go to {folder}/")
    for step in STEPS[args.from_step - 2:]:
        print(f"\n=== {step} ===", flush=True)
        start = time.time()
        # sys.executable = the Python of your .venv, so the same packages are used
        if subprocess.run([sys.executable, step, str(source), "--out", str(folder)]).returncode != 0:
            raise SystemExit(f"{step} stopped with an error (see above).")
        print(f"({time.time() - start:.0f} s)", flush=True)

    print("\n=== browser copy for the Studio app ===", flush=True)
    video, data = prepare(source, folder)
    print(f"\nDone. Open the app with: streamlit run app.py   ({video}, {data})")


if __name__ == "__main__":
    main()
