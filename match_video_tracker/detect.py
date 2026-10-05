"""Find people with YOLO, then keep the players and drop the crowd.

Skill 4: YOLO returns a box, a class and a confidence for every object it finds.
The model was trained on the COCO dataset, where class 0 is "person". COCO has
no "player" class, so we use the grass under each person's feet instead.
"""
from .grass import foot_grass_fraction

PERSON = 0      # the COCO class number for "person"


def pick_device():
    """Use the Mac's GPU (called MPS) when it exists, otherwise the CPU."""
    import torch    # imported here, so the tests can run without loading torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name="yolo26m.pt"):
    """Load a YOLO model. The first time, Ultralytics downloads the weights file."""
    from ultralytics import YOLO    # imported here for the same reason as torch

    return YOLO(name)


def detect_people(model, frame, conf=0.25, imgsz=1280, device="cpu"):
    """Return a list of (x1, y1, x2, y2, confidence), one for every person YOLO finds."""
    result = model.predict(frame, classes=[PERSON], conf=conf, imgsz=imgsz,
                           device=device, verbose=False)[0]
    boxes = result.boxes.xyxy.tolist()      # [[x1, y1, x2, y2], ...]
    confs = result.boxes.conf.tolist()      # [0.91, 0.87, ...]
    return [(x1, y1, x2, y2, c) for (x1, y1, x2, y2), c in zip(boxes, confs)]


def split_players_and_crowd(people, mask, min_grass=0.15):
    """Players stand on grass. Everyone else (fans, people on the track) is crowd."""
    players, crowd = [], []
    for person in people:
        box = person[:4]                    # (x1, y1, x2, y2) without the confidence
        if foot_grass_fraction(mask, box) >= min_grass:
            players.append(person)
        else:
            crowd.append(person)
    return players, crowd
