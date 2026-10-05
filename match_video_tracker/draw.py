"""Draw the results on a frame (Skill 3, step 4)."""
import cv2
import numpy as np

GREEN = (0, 200, 0)         # BGR: OpenCV's order is blue, green, red
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def to_point(x, y):
    """OpenCV wants whole-number pixel positions."""
    return int(x), int(y)


def draw_people(frame, players, crowd, show_crowd=True):
    """Green box = player. Thin red box = removed as crowd, so you can check the filter."""
    out = frame.copy()                      # draw on a copy and keep the original clean

    if show_crowd:
        for x1, y1, x2, y2, _ in crowd:
            cv2.rectangle(out, to_point(x1, y1), to_point(x2, y2), RED, 1)
    for x1, y1, x2, y2, _ in players:
        cv2.rectangle(out, to_point(x1, y1), to_point(x2, y2), GREEN, 2)

    put_label(out, f"players: {len(players)}   crowd removed: {len(crowd)}")
    return out


def put_label(out, text):
    """White text on a black box in the bottom-left corner, so it never hides the scoreboard."""
    scale = max(0.6, out.shape[1] / 1600)   # bigger text on bigger frames
    thick = max(2, round(2 * scale))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    h = out.shape[0]
    cv2.rectangle(out, (10, h - th - 30), (30 + tw, h - 10), BLACK, -1)
    cv2.putText(out, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)


def id_color(track_id):
    """The same ID always gets the same colour, so an ID switch is easy to spot."""
    hue = (track_id * 37) % 180                 # jump around the colour wheel (Skill 3: HSV)
    hsv = np.uint8([[[hue, 220, 255]]])         # one pixel: (hue, saturation, value)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_tracks(frame, tracks):
    """An ellipse under each player's feet with his ID below it, like TV graphics."""
    out = frame.copy()
    scale = max(0.5, out.shape[1] / 1920)       # bigger labels on bigger frames
    thick = max(2, round(2 * scale))
    for track_id, x1, y1, x2, y2, _ in tracks:
        color = id_color(track_id)
        cx, feet = int((x1 + x2) / 2), int(y2)
        half_w = max(int((x2 - x1) / 2), 8)
        # An open ellipse under the feet: from -45 to 235 degrees leaves a gap at the top.
        cv2.ellipse(out, (cx, feet), (half_w, max(half_w // 3, 3)), 0, -45, 235, color, thick, cv2.LINE_AA)

        # The ID in a small coloured box just below the ellipse.
        label = str(track_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, thick)
        top = feet + half_w // 3 + 4
        cv2.rectangle(out, (cx - tw // 2 - 4, top), (cx + tw // 2 + 4, top + th + 8), color, -1)
        cv2.putText(out, label, (cx - tw // 2, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, BLACK, thick, cv2.LINE_AA)

    put_label(out, f"players tracked: {len(tracks)}")
    return out
