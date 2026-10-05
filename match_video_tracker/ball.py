"""Find the match ball (Skill 8, steps 1-2).

YOLO's "sports ball" class finds the ball, but also spare balls lying by the
advertising boards, bright boots and other white blobs. So we:
  1. keep every ball-like box, even weak ones (the real ball is small and often blurred),
  2. drop boxes that are not white and round (yellow boots, shirts),
  3. connect the same object across frames into a short path (a "tracklet"),
  4. drop paths that don't move on the pitch (M5's camera motion takes the pan out):
     spare balls lying by the boards, the boots of a player standing still,
     and drop boxes far from every player,
  5. in each frame, keep the most confident box that is left,
  6. drop impossible jumps and fill the frames where the ball is hidden.
All distances are in body heights (the height of a nearby player's box), so the
same settings work for close-ups and wide shots.
"""
import cv2
import numpy as np
import pandas as pd

from match_video_tracker.tracks import short_holes

SPORTS_BALL = 32        # the COCO class number of a ball
NEAR = 1.0              # a ball within 1 body height of a player's feet is "near" him
MAX_SPEED = 25          # body heights per second, about 45 m/s: faster than any shot
STILL = 0.2             # a tracklet that moves less than this on the pitch is lying still


def ball_class(model):
    """The class number of the ball in this model: "sports ball" (32) in the standard COCO
    model, "ball" in a model fine-tuned on football (often 0)."""
    for number, name in model.names.items():
        if name in ("ball", "football", "sports ball"):
            return number
    return SPORTS_BALL


def is_football_model(model):
    """True for a model trained on football footage: it has a "ball" class and no "sports ball"."""
    return "sports ball" not in model.names.values()


def tiles(w, h, overlap=100):
    """4 overlapping tiles (x1, y1, x2, y2) that cover a w x h frame: top-left, top-right,
    bottom-left, bottom-right. A ball cut by a tile edge is whole in the next tile."""
    tw, th = min(w, w // 2 + overlap), min(h, h // 2 + overlap)
    return [(x, y, x + tw, y + th) for y in (0, h - th) for x in (0, w - tw)]


def detect_balls(model, frame, conf=0.05, imgsz=1280, device="cpu", tiled=False):
    """Every ball-like box in one frame: [(x1, y1, x2, y2, conf), ...].

    tiled=True looks at 4 overlapping tiles instead of the whole frame, each tile at
    imgsz, so a 12-pixel ball is twice as big for YOLO. A model fine-tuned on Roboflow's
    football ball set was trained on tiles like these, so use it with tiled=True.
    """
    cls = ball_class(model)
    if not tiled:
        result = model.predict(frame, classes=[cls], conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        return [(x1, y1, x2, y2, c) for (x1, y1, x2, y2), c
                in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]

    h, w = frame.shape[:2]
    boxes, scores = [], []
    parts = tiles(w, h)
    crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in parts]
    results = model.predict(crops, classes=[cls], conf=conf, imgsz=imgsz, device=device, verbose=False)
    for (tx, ty, _, _), result in zip(parts, results):
        for (x1, y1, x2, y2), c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            boxes.append([x1 + tx, y1 + ty, x2 - x1, y2 - y1])     # back to frame pixels, (x, y, w, h)
            scores.append(c)
    # The same ball seen in two overlapping tiles: keep the more confident box.
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.3) if boxes else []
    return [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3], scores[i])
            for i in np.array(keep).flatten()]


def whiteness(hsv, box):
    """Share of the box's pixels that look white or light grey (a ball, also in shadow)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    patch = hsv[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1].reshape(-1, 3)
    if len(patch) == 0:
        return 0.0
    light_grey = (patch[:, 1] < 70) & (patch[:, 2] > 110)    # little colour, not dark
    return float(light_grey.mean())


def to_table(rows):
    """Turn (frame, x1, y1, x2, y2, conf, white) rows into a table with the ball's centre."""
    df = pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "white"])
    df["x"] = (df["x1"] + df["x2"]) / 2
    df["y"] = (df["y1"] + df["y2"]) / 2
    df["aspect"] = (df["x2"] - df["x1"]) / (df["y2"] - df["y1"])    # 1.0 = as wide as tall
    return df


def add_nearest_player(balls, tracks):
    """For every ball box: the nearest person (any team), and his distance in body heights.

    Adds the columns near_id, near_team, dist (body heights) and scale (that frame's
    typical body height in pixels, used to turn pixels into body heights).
    """
    people = tracks[["frame", "id", "team", "foot_x", "foot_y", "y1", "y2"]].copy()
    people["h"] = people["y2"] - people["y1"]
    scale = people.groupby("frame")["h"].median().rename("scale")
    pairs = balls.reset_index().merge(people, on="frame", how="left")    # every ball x every person
    pairs["dist"] = np.hypot(pairs["foot_x"] - pairs["x"], pairs["foot_y"] - pairs["y"]) / pairs["h"]
    nearest = pairs.loc[pairs.groupby("index")["dist"].idxmin().dropna()].set_index("index")
    out = balls.copy()
    out["near_id"] = nearest["id"]
    out["near_team"] = nearest["team"]
    out["dist"] = nearest["dist"]
    out["scale"] = out["frame"].map(scale)
    return out


def link_tracklets(balls, fps, max_gap=5):
    """Give boxes that are the same object in nearby frames the same tracklet number.

    A box joins the tracklet whose last box is closest, if that box is at most
    max_gap frames old and the ball could have moved that far in the time.
    """
    balls = balls.sort_values("frame").copy()
    tracklet = pd.Series(-1, index=balls.index)
    last = {}                                       # tracklet -> (frame, x, y) of its last box
    next_id = 0
    for frame, rows in balls.groupby("frame"):
        free = dict(last)
        for i, r in rows.sort_values("conf", ascending=False).iterrows():
            scale = r["scale"] if r["scale"] == r["scale"] else 100      # NaN -> 100 px
            best, best_d = None, None
            for t, (f, x, y) in free.items():
                gap = frame - f
                reach = (MAX_SPEED * gap / fps + 0.3) * scale            # pixels it could move
                d = np.hypot(r["x"] - x, r["y"] - y)
                if gap <= max_gap and d <= reach and (best_d is None or d < best_d):
                    best, best_d = t, d
            if best is None:                        # nothing close: a new tracklet starts
                best, next_id = next_id, next_id + 1
            else:
                del free[best]                      # one box per tracklet per frame
            tracklet[i] = best
            last[best] = (frame, r["x"], r["y"])
    balls["tracklet"] = tracklet
    return balls


def standing_still(balls, still=STILL, min_boxes=8):
    """True for boxes whose tracklet never moves on the pitch: spare balls, boots of a
    player who stands still. Needs pitch_x / pitch_y from camera.to_pitch (M5).

    In pitch coordinates the camera's pan is taken out, so a ball lying by the boards
    keeps the same position while the camera moves. The match ball in play keeps moving.
    """
    middle = balls.groupby("tracklet")[["pitch_x", "pitch_y"]].transform("median")
    off = np.hypot(balls["pitch_x"] - middle["pitch_x"], balls["pitch_y"] - middle["pitch_y"])
    off = off / balls["scale"]                                       # in body heights
    moves = off.groupby(balls["tracklet"]).transform(lambda s: s.quantile(0.9))
    boxes = balls.groupby("tracklet")["frame"].transform("count")
    return (moves < still) & (boxes >= min_boxes)


def on_a_body(balls, tracks, knee=0.75):
    """True for boxes inside a player's box and above his knees: white shorts, socks, a head.

    The ball at a player's feet sits in the bottom quarter of his box (below knee=75 %).
    """
    people = tracks[["frame", "x1", "y1", "x2", "y2"]]
    pairs = balls[["frame", "x", "y"]].reset_index().merge(people, on="frame")
    knee_y = pairs["y1"] + knee * (pairs["y2"] - pairs["y1"])
    inside = pairs["x"].between(pairs["x1"], pairs["x2"]) & pairs["y"].between(pairs["y1"], knee_y)
    hit = inside.groupby(pairs["index"]).any()
    return hit.reindex(balls.index, fill_value=False)


def keep_match_ball(balls, tracks, min_white=0.25):
    """Mark the boxes that can be the match ball: keep = True.

    A box is kept if it is white and round, near a player but not on his body above
    the knees, and its tracklet is not standing still on the pitch (a spare ball by
    the boards, the boots of a player standing still).
    """
    looks_right = (balls["white"] >= min_white) & balls["aspect"].between(0.6, 1.7)
    near = balls["dist"] <= NEAR
    out = balls.copy()
    out["keep"] = looks_right & near & ~on_a_body(balls, tracks) & ~standing_still(balls)
    return out


def ball_path(balls, n_frames, fps, max_gap_s=1.0):
    """One ball position per frame: the best kept box, without jumps, holes filled.

    Returns a table indexed by frame (1 .. n_frames) with x, y and how the position
    was found: "seen" (a box) or "filled" (a straight line between two sightings).
    """
    kept = balls[balls["keep"]].sort_values("conf", ascending=False)
    path = kept.groupby("frame").head(1).set_index("frame")[["x", "y", "scale"]].sort_index()

    # Drop impossible jumps: a sighting that is too far from both of its neighbours.
    for _ in range(3):
        if len(path) < 3:
            break
        gap_prev = path.index.to_series().diff()
        gap_next = -path.index.to_series().diff(-1)
        step_prev = np.hypot(path["x"].diff(), path["y"].diff()) / path["scale"]
        step_next = np.hypot(path["x"].diff(-1), path["y"].diff(-1)) / path["scale"]
        too_fast_prev = step_prev > MAX_SPEED * gap_prev / fps + 0.5
        too_fast_next = step_next > MAX_SPEED * gap_next / fps + 0.5
        jump = too_fast_prev & too_fast_next                 # both neighbours disagree
        if not jump.any():
            break
        path = path[~jump]

    full = path.reindex(range(1, n_frames + 1))
    full.index.name = "frame"
    full["source"] = np.where(full["x"].notna(), "seen", "lost")
    fill = short_holes(full["x"].isna(), int(max_gap_s * fps))     # holes of up to 1 s
    guess = full[["x", "y"]].interpolate(limit_area="inside")      # straight lines
    full.loc[fill, ["x", "y"]] = guess.loc[fill]
    full.loc[fill & full["x"].notna(), "source"] = "filled"
    return full[["x", "y", "source"]]


def ball_speed(path, scale, fps, window=5):
    """The ball's speed in body heights per second, from its pitch coordinates.

    path needs pitch_x / pitch_y (camera.to_pitch), so a camera pan is not counted as
    speed. scale = each frame's typical body height in pixels. A rolling mean over
    `window` frames smooths out the wobble of the detections.
    """
    step = np.hypot(path["pitch_x"].diff(), path["pitch_y"].diff()) / scale.reindex(path.index)
    return (step * fps).rolling(window, center=True, min_periods=1).mean()
