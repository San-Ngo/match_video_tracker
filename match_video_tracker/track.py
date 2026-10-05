"""Tracking: give every person an ID that stays the same from frame to frame.

Skill 5: tracking by detection. YOLO finds the people in each frame, then BoT-SORT
matches the new boxes to the players it already knows. A match keeps the old ID;
a box with no match gets a new ID.
"""
from .detect import PERSON


def track_people(model, frame, tracker="trackers/botsort.yaml", conf=0.1, imgsz=1280, device="cpu"):
    """Return a list of (id, x1, y1, x2, y2, confidence) for one frame.

    persist=True makes Ultralytics keep the tracker's memory between calls,
    which is what lets a player keep his ID. conf is low (0.1) on purpose:
    the tracker uses weak boxes to hold on to half-hidden players.
    """
    result = model.track(frame, persist=True, tracker=tracker, classes=[PERSON],
                         conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    if result.boxes.id is None:             # nobody is being tracked in this frame
        return []
    ids = result.boxes.id.int().tolist()    # [3, 7, 12, ...]
    boxes = result.boxes.xyxy.tolist()      # [[x1, y1, x2, y2], ...]
    confs = result.boxes.conf.tolist()
    return [(i, x1, y1, x2, y2, c) for i, (x1, y1, x2, y2), c in zip(ids, boxes, confs)]
