"""Milestone 7: the Studio app. Click any player and get his spotlight clip.

Run it from the project folder, with your .venv active:
    streamlit run app.py
It opens in your browser (stop it with Ctrl + C in the Terminal).

It uses what M2-M5 saved in outputs/, so run those on a clip first. M6 is not needed:
the app draws the spotlight itself, for whichever player you click.
"""
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from match_video_tracker.colour import shirt_to_bgr
from match_video_tracker.draw import WHITE, draw_teams, name_tag
from match_video_tracker.grass import grass_mask
from match_video_tracker.possession import possession_share, spells
from match_video_tracker.spotlight import box_at, draw_spotlight, follow, spotlight_colour, trail_at
from match_video_tracker.teams import OTHER
from match_video_tracker.video import open_writer, video_info

OUTPUTS, DATA = Path("outputs"), Path("data")
VIEW_WIDTH = 1100                       # how wide the frame is shown in the browser
NAMES = {0: "Team 1", 1: "Team 2", OTHER: "Other"}

st.set_page_config(page_title="Match Video Tracker Studio", layout="wide")


# ---------- loading (cached: Streamlit re-runs this file on every click) ----------

@st.cache_data
def load_tables():
    """Everything M2-M5 saved. Possession and events are optional (M5)."""
    tables = {"tracks": pd.read_csv(OUTPUTS / "tracks_teams.csv"),
              "camera": pd.read_csv(OUTPUTS / "camera.csv", index_col="frame"),
              "players": pd.read_csv(OUTPUTS / "teams.csv")}
    for name in ("possession", "events"):
        path = OUTPUTS / f"{name}.csv"
        tables[name] = pd.read_csv(path) if path.exists() else None
    return tables


@st.cache_data
def clip_info(source):
    """fps and number of frames of a video."""
    cap = cv2.VideoCapture(source)
    fps, w, h, total = video_info(cap)
    cap.release()
    return fps, total


@st.cache_data(max_entries=50)
def read_frame(source, n):
    """Frame n (counting from 1, like the CSV files) of the video."""
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def team_colours(players):
    kit = players[players["team"] != OTHER].groupby("team")[["shirt_a", "shirt_b"]].median()
    return {0: shirt_to_bgr(*kit.loc[0]), 1: shirt_to_bgr(*kit.loc[1]), OTHER: WHITE}


# ---------- drawing ----------

def picture(frame, n, tracks, camera, players, pid, trail_s, fps):
    """Frame n with the spotlight under player pid and every player's team marker."""
    rows = tracks[tracks["frame"] == n]
    out = frame
    me = None
    if pid is not None:
        me = follow(tracks, camera, pid)
        team = players.set_index("id")["team"].get(pid, OTHER)
        colour = spotlight_colour(players, team)
        out = draw_spotlight(frame, grass_mask(frame), box_at(me, n),
                             trail_at(me, camera, n, int(trail_s * fps)), colour)
    out = draw_teams(out, list(rows[["id", "x1", "y1", "x2", "y2", "team"]]
                               .itertuples(index=False, name=None)), team_colours(players), NAMES)
    if me is not None and box_at(me, n) is not None:
        name_tag(out, box_at(me, n), str(pid), colour)
    return out


def player_at(rows, x, y):
    """The player whose box (plus a bit under the feet) contains (x, y), or None."""
    h = rows["y2"] - rows["y1"]
    hit = rows[(rows["x1"] <= x) & (x <= rows["x2"]) & (rows["y1"] <= y) & (y <= rows["y2"] + 0.3 * h)]
    if hit.empty:
        return None
    feet = ((hit["x1"] + hit["x2"]) / 2 - x) ** 2 + (hit["y2"] - y) ** 2
    return int(hit.loc[feet.idxmin(), "id"])                # two boxes overlap: the closer feet


def export(source, tracks, camera, players, pid, first, last, trail_s, fps, size):
    """Write frames first..last with the spotlight to outputs/studio_<id>_<first>-<last>.mp4."""
    path = OUTPUTS / f"studio_player{pid}_{first}-{last}.mp4"
    writer = open_writer(path, fps, size)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
    bar = st.progress(0.0, text="Exporting ...")
    for n in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(picture(frame, n, tracks, camera, players, pid, trail_s, fps))
        bar.progress((n - first + 1) / (last - first + 1), text=f"Exporting frame {n} of {last}")
    cap.release()
    writer.release()
    bar.empty()
    return path


# ---------- the page ----------

st.title("Match Video Tracker Studio")

missing = [f for f in ("tracks_teams.csv", "camera.csv", "teams.csv") if not (OUTPUTS / f).exists()]
if missing:
    st.error(f"Missing in outputs/: {', '.join(missing)}. Run m2_track.py, m3_teams.py and "
             "m4_camera.py on your clip first.")
    st.stop()

tables = load_tables()
tracks, camera, players = tables["tracks"], tables["camera"], tables["players"]

with st.sidebar:
    st.header("Clip")
    clips = sorted(str(p) for p in DATA.glob("*") if p.suffix.lower() in (".mp4", ".mov", ".m4v"))
    if not clips:
        st.error("Put a clip in the data/ folder first.")
        st.stop()
    analysed = int(tracks["frame"].max())                   # the clip outputs/ was made from has this length
    best = next((i for i, c in enumerate(clips) if clip_info(c)[1] == analysed), 0)
    source = st.selectbox("Video", clips, index=best)
    fps, total = clip_info(source)
    if tracks["frame"].max() > total:
        st.error("outputs/ was made from a different clip. Run m2 - m5 on this one, then reload.")
        st.stop()
    trail_s = st.slider("Trail length (seconds)", 0.0, 6.0, 3.0, 0.5)
    st.caption("The analysis comes from outputs/, made by the last m2 - m5 run.")

if "player" not in st.session_state:                       # start on the player with the most ball
    holder = tables["possession"]["holder"].dropna() if tables["possession"] is not None else pd.Series()
    st.session_state.player = int(holder.mode().iat[0]) if len(holder) else None
    st.session_state.last_click = None

video_col, stats_col = st.columns([3, 1], gap="large")

with video_col:
    n = st.slider("Frame", 1, total, min(total, 200), help="Drag to move through the clip")
    st.caption(f"{n / fps:.2f} s of {total / fps:.1f} s · click a player to put the spotlight on him")
    frame = read_frame(source, n)
    if frame is None:
        st.error(f"Could not read frame {n}.")
        st.stop()
    shown = picture(frame, n, tracks, camera, players, st.session_state.player, trail_s, fps)
    small = cv2.resize(shown, (VIEW_WIDTH, int(shown.shape[0] * VIEW_WIDTH / shown.shape[1])))
    click = streamlit_image_coordinates(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), key="frame")
    if click and click != st.session_state.last_click:     # a new click (old ones stay in `click`)
        st.session_state.last_click = click
        scale = shown.shape[1] / click["width"]             # the page may show the picture smaller
        pid = player_at(tracks[tracks["frame"] == n], click["x"] * scale, click["y"] * scale)
        if pid is not None and pid != st.session_state.player:
            st.session_state.player = pid
            st.rerun()

with stats_col:
    pid = st.session_state.player
    st.subheader("Player" if pid is None else f"Player {pid}")
    if pid is not None:
        team = players.set_index("id")["team"].get(pid, OTHER)
        seen = (tracks["id"] == pid).sum()
        st.write(f"**{NAMES[team]}** · on screen for {seen / fps:.1f} s")
        if tables["possession"] is not None:
            holder = tables["possession"].set_index("frame")["holder"]
            mine = spells(holder)
            mine = mine[mine["id"] == pid]
            on_ball = (mine["last"] - mine["first"] + 1).sum() / fps
            st.write(f"On the ball: {on_ball:.1f} s in {len(mine)} spell(s)")
        if st.button("Clear spotlight"):
            st.session_state.player = None
            st.rerun()

    if tables["possession"] is not None:
        st.subheader("Possession")
        share = possession_share(tables["possession"]["team"])
        for t in (0, 1):
            st.write(f"{NAMES[t]}: **{share[t]:.0%}**")
            st.progress(float(share[t]))
    if tables["events"] is not None and len(tables["events"]):
        st.subheader("Passes and turnovers")
        ev = tables["events"].assign(second=lambda d: (d["frame"] / fps).round(1))
        st.dataframe(ev[["second", "from_id", "to_id", "event"]], hide_index=True)

    st.subheader("Players")
    counts = players["team"].value_counts()
    st.write(" · ".join(f"{NAMES[t]}: {counts.get(t, 0)}" for t in (0, 1, OTHER)))

st.divider()
st.subheader("Export a spotlight clip")
if st.session_state.player is None:
    st.info("Click a player first.")
else:
    first_s, last_s = st.slider("Part of the clip (seconds)", 0.0, total / fps,
                                (0.0, total / fps), 0.1)
    if st.button(f"Export player {st.session_state.player}"):
        first, last = max(1, int(first_s * fps) + 1), min(total, int(last_s * fps))
        h, w = frame.shape[:2]
        path = export(source, tracks, camera, players, st.session_state.player,
                      first, last, trail_s, fps, (w, h))
        st.success(f"Saved {path}")
        st.video(str(path))
        st.download_button("Download the mp4", path.read_bytes(), file_name=path.name, mime="video/mp4")
