"""Small video helpers shared by every milestone script (Skill 3, steps 2-3)."""
import cv2


def video_info(cap):
    """Frames per second, width, height and number of frames of an open video."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, w, h, total


def open_writer(path, fps, size):
    """Try H.264 first (plays in QuickTime), then MPEG-4 as a backup.

    size = (width, height): the opposite order of img.shape.
    """
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise SystemExit("Could not create the output video.")
