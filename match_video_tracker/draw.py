"""Draw the results on a frame (Skill 3, step 4)."""
import cv2

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

    # The count in the top-left corner: black outline first, white text on top.
    label = f"players: {len(players)}   crowd removed: {len(crowd)}"
    scale = max(0.6, out.shape[1] / 1600)   # bigger text on bigger frames
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 5, cv2.LINE_AA)
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 2, cv2.LINE_AA)
    return out
