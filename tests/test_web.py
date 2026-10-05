import json
import subprocess

import cv2
import numpy as np
import pandas as pd

from match_video_tracker.web import (camera_matrices, clip_data, clip_folder, even_size, ffmpeg_exe,
                                     player_frames, write_web_video)


def test_each_clip_has_its_own_folder():
    assert str(clip_folder("data/clip2.mov")) == "outputs/clip2"
    assert clip_folder("data/clip.mov") != clip_folder("data/clip1.mov")


def test_even_size_for_h264():
    assert even_size(2778, 1376, 1280) == (1280, 634)
    assert even_size(1001, 501, 1280) == (1000, 500)          # never bigger than the clip


def tracks():
    return pd.DataFrame({"frame": [1, 1, 3], "id": [4, 7, 4], "team": [0, 1, 0],
                         "x1": [10, 50, 12], "y1": [20, 20, 20], "x2": [20, 60, 22], "y2": [60, 60, 60],
                         "foot_x": [15, 55, 17], "foot_y": [60, 60, 60]})


def test_player_frames_keeps_every_frame():
    frames = player_frames(tracks(), total=3)
    assert len(frames) == 3
    assert [r[0] for r in frames[0]] == [4, 7]
    assert frames[1] == []                                     # nobody in frame 2
    assert frames[2][0] == [4, 0, 12.0, 20.0, 22.0, 60.0, 17.0, 60.0]


def test_camera_matrices_fill_missing_frames():
    cam = pd.DataFrame({"frame": [1, 2], "to_first_00": [1, 1], "to_first_01": [0, 0], "to_first_02": [0, 5],
                        "to_first_10": [0, 0], "to_first_11": [1, 1], "to_first_12": [0, 0]}).set_index("frame")
    m = camera_matrices(cam, total=3)
    assert m[0] == [1, 0, 0, 0, 1, 0]
    assert m[2] == m[1] == [1, 0, 5, 0, 1, 0]                  # frame 3 keeps frame 2's camera


def test_clip_data_is_json():
    players = pd.DataFrame({"id": [4, 7], "team": [0, 1], "shirt_a": [120, 170], "shirt_b": [100, 150]})
    cam = pd.DataFrame({"frame": [1], **{f"to_first_{i}{j}": [float(i == j)] for i in range(2) for j in range(3)}})
    poss = pd.DataFrame({"frame": [1, 2, 3], "holder": [4, None, 7]})
    d = json.loads(json.dumps(clip_data(tracks(), cam.set_index("frame"), players, 60, 100, 80, 3, poss)))
    assert d["total"] == 3 and len(d["frames"]) == 3 and len(d["camera"]) == 3
    assert set(d["colours"]) == {"0", "1", "-1"} and d["colours"]["-1"] == "#ffffff"
    assert d["holder"] == [4, None, 7]


def test_web_video_has_every_frame(tmp_path):
    src = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    n = write_web_video(src, tmp_path / "web.mp4", width=32)
    assert n == 10
    probe = subprocess.run([ffmpeg_exe(), "-i", str(tmp_path / "web.mp4")], capture_output=True, text=True).stderr
    assert "h264" in probe and "32x24" in probe
