"""How the camera moves, so we can tell camera movement from player movement (Skill 8, steps 3-6).

1. In each frame, find corners on the pitch, ads and stands (players are masked out,
   because they move by themselves).
2. Follow those corners into the next frame with optical flow (Lucas-Kanade).
3. RANSAC finds the one camera move (shift, zoom, small turn) that most corners agree on,
   and ignores the corners that disagree (a scoreboard, a moving fan).
4. Chain the moves: every pixel of every frame can then be turned into "pitch
   coordinates": where it would be in the FIRST frame. Something lying still on the
   pitch, like a spare ball, keeps the same pitch coordinates while the camera pans.
"""
import cv2
import numpy as np
import pandas as pd

WORK_WIDTH = 960        # optical flow on a smaller copy of the frame: much faster, still exact enough
MIN_INLIERS = 30        # fewer corners agreeing than this = a scene cut (or a frame we can't trust)


def small_gray(frame):
    """A grey, smaller copy of the frame and the factor we shrank it by."""
    f = WORK_WIDTH / frame.shape[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), f


def background_mask(shape, boxes, f):
    """255 where we may look for corners, 0 on the players (their boxes, a bit bigger)."""
    mask = np.full(shape, 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        pad = 0.2 * (y2 - y1)
        cv2.rectangle(mask, (int((x1 - pad) * f), int((y1 - pad) * f)),
                      (int((x2 + pad) * f), int((y2 + pad) * f)), 0, -1)
    return mask


def camera_move(prev_gray, gray, mask, f):
    """The camera move from prev to this frame as a 3x3 matrix in full-size pixels, and how
    many corners agreed. Returns (None, 0) when it can't be measured."""
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01,
                                      minDistance=8, mask=mask)
    if corners is None or len(corners) < MIN_INLIERS:
        return None, 0
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None)
    ok = status.ravel() == 1
    if ok.sum() < MIN_INLIERS:
        return None, 0
    # Shift + zoom + small turn (no stretching), fitted with RANSAC.
    move, inliers = cv2.estimateAffinePartial2D(corners[ok] / f, moved[ok] / f,
                                                method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if move is None:
        return None, 0
    return np.vstack([move, [0, 0, 1]]), int(inliers.sum())


def track_camera(frames_and_boxes):
    """Measure the camera in every frame. frames_and_boxes yields (frame number, frame, boxes).

    Returns one row per frame: the move since the previous frame (dx, dy in pixels,
    zoom), how many corners agreed, whether it is a scene cut, and the 3x3 matrix
    (to_first_*) that turns this frame's pixels into pitch coordinates.
    """
    rows = []
    to_first = np.eye(3)
    prev = None
    for n, frame, boxes in frames_and_boxes:
        gray, f = small_gray(frame)
        move, agree, cut = np.eye(3), 0, False
        if prev is not None:
            prev_gray, prev_mask = prev
            measured, agree = camera_move(prev_gray, gray, prev_mask, f)
            if measured is None:
                cut = True                      # new shot: start the pitch coordinates again
                to_first = np.eye(3)
            else:
                move = measured
                to_first = to_first @ np.linalg.inv(move)      # this frame -> previous -> ... -> first
        prev = (gray, background_mask(gray.shape, boxes, f))
        rows.append({"frame": n, "dx": move[0, 2], "dy": move[1, 2],
                     "zoom": float(np.hypot(move[0, 0], move[1, 0])),
                     "agree": agree, "cut": cut,
                     **{f"to_first_{i}{j}": to_first[i, j] for i in range(2) for j in range(3)}})
    return pd.DataFrame(rows).set_index("frame")


def to_pitch(df, camera, x="x", y="y"):
    """Add pitch_x and pitch_y: where each (x, y) would be in the clip's first frame."""
    def get(name):                              # this column of the camera table, for every row of df
        return camera.loc[df["frame"], name].to_numpy()

    a, b, c = get("to_first_00"), get("to_first_01"), get("to_first_02")
    d, e, g = get("to_first_10"), get("to_first_11"), get("to_first_12")
    out = df.copy()
    out["pitch_x"] = a * df[x].to_numpy() + b * df[y].to_numpy() + c
    out["pitch_y"] = d * df[x].to_numpy() + e * df[y].to_numpy() + g
    return out
