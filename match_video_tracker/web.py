"""Get a clip ready for the browser player in the Studio app.

The browser cannot run YOLO, so Python sends it everything it needs, once:
  - a light H.264 copy of the clip (every browser plays H.264; iPhone .mov files are
    often HEVC, which Chrome may not play). It is built from the same OpenCV frames the
    analysis used, at a constant frame rate, so frame n is always at time (n - 1) / fps.
  - one JSON file: every player's box and team per frame, the team colours, and the
    camera matrix per frame (pixels -> pitch, M4), so drawings can stay on the grass.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from match_video_tracker.colour import shirt_to_bgr
from match_video_tracker.teams import OTHER

OUTPUTS = Path("outputs")
WEB_WIDTH = 1280                  # the browser copy is this wide: light, and sharp enough
NEEDED = ("tracks_teams.csv", "camera.csv", "teams.csv")     # made by m2, m3 and m4


def clip_folder(source):
    """Each clip keeps its own results: data/clip2.mov -> outputs/clip2/."""
    return OUTPUTS / Path(source).stem


def is_analysed(folder):
    return all((Path(folder) / f).exists() for f in NEEDED)


def ffmpeg_exe():
    """ffmpeg from the imageio-ffmpeg package (pip installs it), or the system one."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def open_h264(path, fps, size):
    """An ffmpeg process that turns raw BGR frames (written to its stdin) into an H.264
    mp4 that plays in every browser and on phones. size = (width, height), both even."""
    w, h = size
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def even_size(w, h, width):
    """(width, height) scaled to `width`, both even (H.264 needs even sizes)."""
    width = min(width, w) // 2 * 2
    return width, int(round(h * width / w / 2)) * 2


def write_web_video(source, path, width=WEB_WIDTH):
    """Copy the clip to an H.264 mp4, `width` pixels wide, frame by frame."""
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = None
    proc = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if proc is None:
            size = even_size(frame.shape[1], frame.shape[0], width)
            proc = open_h264(path, fps, size)
        proc.stdin.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA).tobytes())
        n += 1
    cap.release()
    if proc is None:
        raise SystemExit(f"Could not read {source}.")
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg could not write the browser video. Try: pip install imageio-ffmpeg")
    return n


def hex_colour(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def team_colours(players):
    """Team colour per team, as '#rrggbb' for the browser (brightened shirt colours)."""
    kit = players[players["team"] != OTHER].groupby("team")[["shirt_a", "shirt_b"]].median()
    return {str(t): hex_colour(shirt_to_bgr(*kit.loc[t])) for t in kit.index} | {str(OTHER): "#ffffff"}


def player_frames(tracks, total):
    """For every frame 1..total, a list of [id, team, x1, y1, x2, y2, foot_x, foot_y]
    (rounded to 0.1 pixel to keep the file small)."""
    fx, fy = ("foot_x_smooth", "foot_y_smooth") if "foot_x_smooth" in tracks else ("foot_x", "foot_y")
    cols = ["id", "team", "x1", "y1", "x2", "y2", fx, fy]
    frames = [[] for _ in range(total)]
    for row in tracks[["frame"] + cols].itertuples(index=False, name=None):
        n = int(row[0])
        if 1 <= n <= total:
            frames[n - 1].append([int(row[1]), int(row[2])] + [round(float(v), 1) for v in row[3:]])
    return frames


def camera_matrices(camera, total):
    """For every frame, [a, b, c, d, e, f]: pitch_x = a*x + b*y + c, pitch_y = d*x + e*y + f
    (the to_first matrix of M4). Frames without a camera row get the one before."""
    cols = [f"to_first_{i}{j}" for i in range(2) for j in range(3)]
    cam = camera.reindex(range(1, total + 1))[cols].ffill().fillna({c: v for c, v in zip(cols, [1, 0, 0, 0, 1, 0])})
    return np.round(cam.to_numpy(float), 6).tolist()


def clip_data(tracks, camera, players, fps, width, height, total, possession=None):
    """Everything the browser player needs, as one dict (saved as JSON)."""
    data = {"fps": fps, "total": total, "width": width, "height": height,
            "colours": team_colours(players),
            "frames": player_frames(tracks, total),
            "camera": camera_matrices(camera, total)}
    if possession is not None:
        holder = possession.set_index("frame")["holder"].reindex(range(1, total + 1))
        data["holder"] = [None if pd.isna(h) else int(h) for h in holder]
    return data


def prepare(source, folder, web_width=WEB_WIDTH):
    """Write folder/web.mp4 and folder/web.json for this clip (skipped if they are newer
    than the analysis). Returns the two paths."""
    folder = Path(folder)
    video, data = folder / "web.mp4", folder / "web.json"
    analysis = max((folder / f).stat().st_mtime for f in NEEDED)
    possession = folder / "possession.csv"
    if possession.exists():
        analysis = max(analysis, possession.stat().st_mtime)
    if not video.exists() or video.stat().st_size == 0:
        write_web_video(source, video, web_width)
    if not data.exists() or data.stat().st_mtime < analysis:
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        cap.release()
        h, w = frame.shape[:2]
        tracks = pd.read_csv(folder / "tracks_teams.csv")
        total = max(total, int(tracks["frame"].max()))
        d = clip_data(tracks, pd.read_csv(folder / "camera.csv", index_col="frame"),
                      pd.read_csv(folder / "teams.csv"), fps, w, h, total,
                      pd.read_csv(possession) if possession.exists() else None)
        data.write_text(json.dumps(d, separators=(",", ":")))
    return video, data
