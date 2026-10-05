# Match Video Tracker

Football match analysis from broadcast video: detect every player, track them across frames, split them into teams, follow the ball and draw broadcast-style player spotlights.

> 🚧 Work in progress, built one milestone a week while learning computer vision.

## Roadmap

| Milestone | What it does | Status |
| --- | --- | --- |
| M1 Detect | Find the players in a frame and remove the crowd | ✅ Done |
| M2 Track | Keep one ID per player across frames | 🔨 In progress |
| M3 Teams | Split the players into two teams by shirt colour | ⏳ |
| M4 Ball and possession | Find the ball and compute possession | ⏳ |
| M5 Camera motion | Separate camera movement from player movement | ⏳ |
| M6 Spotlight | Ring and trace under a player, drawn under the players | ⏳ |
| M7 Studio app | Click a player in a Streamlit app and export an mp4 | ⏳ |

Releases: v1 = M1–M3, v2 = M4–M6, v3 = M7.

## Tech stack

Python, NumPy, OpenCV, Ultralytics YOLO, TrackTrack and BoT-SORT tracking, pandas and pytest so far. scikit-learn and Streamlit join as the milestones need them.

## Setup (macOS)

```bash
git clone https://github.com/San-Ngo/match_video_tracker.git
cd match_video_tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Put your own match clips in `data/`. Videos and model weights stay on your computer and are not stored in the repo.

## Run it

Activate the environment first (`source .venv/bin/activate`), then:

```bash
python m1_detect.py data/frame.png
python m1_detect.py data/clip.mov
python m2_track.py data/clip.mov
```

- **M1** saves `outputs/m1_detect.jpg` (or `.mp4`). Green boxes are players; thin red boxes are people removed as crowd.
- **M2** saves `outputs/m2_track.mp4` with an ID under every player (TrackTrack by default; add `--tracker trackers/botsort.yaml` to compare), plus `outputs/tracks.csv` (one row per player per frame) and `outputs/tracks_clean.csv`. The clean table joins broken tracks: when a player is hidden behind someone and comes back with a new ID, he gets his old ID back if the timing, position and shirt colour all match.

## Project structure

```
match_video_tracker/   the code, one module per job
tests/                 pytest tests
data/                  your clips (not on GitHub)
outputs/               results (not on GitHub)
practice/              small learning scripts
trackers/              tracker settings (TrackTrack, BoT-SORT)
m1_detect.py           Milestone 1: detect players, remove the crowd
m2_track.py            Milestone 2: track players with IDs
notes.md               what I learned each week
```

## License

AGPL-3.0, the same license as Ultralytics YOLO, which this project builds on.
