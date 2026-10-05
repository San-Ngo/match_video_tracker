import numpy as np

from match_video_tracker.draw import BLACK, WHITE, draw_teams, draw_tracks, id_color, text_color


def test_same_id_always_gets_the_same_colour():
    assert id_color(7) == id_color(7)
    assert id_color(7) != id_color(8)


def test_draw_tracks_keeps_the_original_frame_clean():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    out = draw_tracks(frame, [(7, 100, 50, 130, 150, 0.9)])
    assert frame.sum() == 0
    assert out.sum() > 0


def test_text_is_black_on_light_colours_and_white_on_dark_ones():
    assert text_color(WHITE) == BLACK
    assert text_color((120, 0, 0)) == WHITE        # dark blue (BGR)


def test_draw_teams_paints_the_marker_in_the_team_colour():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    colors = {0: (255, 0, 0), 1: (0, 0, 255), -1: WHITE}     # team 1 blue, team 2 red, other white
    names = {0: "Team 1", 1: "Team 2", -1: "Other"}
    out = draw_teams(frame, [(7, 100, 20, 130, 120, 1)], colors, names)
    marker = out[110:130, 95:135]                  # the ellipse under the feet (y2 = 120)
    reds = (marker[:, :, 2] > 200) & (marker[:, :, 0] < 50)
    assert reds.any()                              # player 7 is in team 2: red marker
    assert frame.sum() == 0
