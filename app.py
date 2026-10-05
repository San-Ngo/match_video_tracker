"""The Studio app: play a clip, pause it, and draw the tactics on it.

Run it from the project folder, with your .venv active:
    streamlit run app.py
It opens in your browser (stop it with Ctrl + C in the Terminal).

Each clip in data/ is analysed once (python analyse.py data/clip.mov, or the button in
the app), into its own folder outputs/<clip name>/. The player itself is a small web page
(studio/player.js) that plays the video and draws on top of it; this file feeds it the
analysis, saves the drawings, shows the stats and exports the video.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from match_video_tracker import telestration as tele
from match_video_tracker.draw import WHITE, draw_teams, name_tag
from match_video_tracker.colour import shirt_to_bgr
from match_video_tracker.grass import grass_mask
from match_video_tracker.possession import possession_share, spells
from match_video_tracker.spotlight import box_at, draw_spotlight, follow, spotlight_colour, trail_at
from match_video_tracker.teams import OTHER
from match_video_tracker.video import video_info
from match_video_tracker.web import clip_folder, even_size, is_analysed, open_h264, prepare

DATA, STATIC, STUDIO = Path("data"), Path("static"), Path("studio")
NAMES = {0: "Team 1", 1: "Team 2", OTHER: "Other"}
VIDEO_TYPES = (".mp4", ".mov", ".m4v")

st.set_page_config(page_title="Match Video Tracker Studio", layout="wide")

# The player: HTML/CSS/JS in studio/, mounted with Streamlit's custom components (v2).
player = st.components.v2.component("mvt_player", html="<span></span>",
                                    css=(STUDIO / "player.css").read_text(),
                                    js=(STUDIO / "player.js").read_text())


# ---------- loading (cached: Streamlit re-runs this file on every click) ----------

@st.cache_data
def load_tables(folder, stamp):
    """What analyse.py saved for one clip. stamp = when it was saved (a new run reloads)."""
    folder = Path(folder)
    tables = {"tracks": pd.read_csv(folder / "tracks_teams.csv"),
              "camera": pd.read_csv(folder / "camera.csv", index_col="frame"),
              "players": pd.read_csv(folder / "teams.csv")}
    for name in ("possession", "events"):
        path = folder / f"{name}.csv"
        tables[name] = pd.read_csv(path) if path.exists() else None
    return tables


def team_colours(players):
    kit = players[players["team"] != OTHER].groupby("team")[["shirt_a", "shirt_b"]].median()
    return {0: shirt_to_bgr(*kit.loc[0]), 1: shirt_to_bgr(*kit.loc[1]), OTHER: WHITE}


def publish(path, name):
    """Copy a file to static/, which Streamlit serves to the browser at app/static/...
    Returns its web address (with the file time, so a new version is not cached)."""
    target = STATIC / name
    if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return f"./app/static/{name}?v={int(path.stat().st_mtime)}"


def run_analysis(source):
    """Run analyse.py on the clip and show its last lines while it works."""
    box = st.empty()
    lines = []
    proc = subprocess.Popen([sys.executable, "-u", "analyse.py", str(source)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-15:]))
    return proc.wait() == 0


# ---------- export ----------

def picture(frame, n, tracks, camera, players, pid, trail_s, fps, drawings):
    """Frame n with the drawings and spotlight under the players, team markers on top
    (the same picture as the browser player, made with OpenCV for the mp4)."""
    rows = tracks[tracks["frame"] == n]
    grass = grass_mask(frame)
    out = tele.draw_all(frame, n, drawings, rows, camera, grass) if drawings else frame
    colour = None
    if pid is not None:
        me = follow(tracks, camera, pid)
        team = players.set_index("id")["team"].get(pid, OTHER)
        colour = spotlight_colour(players, team)
        out = draw_spotlight(out, grass, box_at(me, n), trail_at(me, camera, n, int(trail_s * fps)), colour)
    out = draw_teams(out, list(rows[["id", "x1", "y1", "x2", "y2", "team"]]
                               .itertuples(index=False, name=None)), team_colours(players), NAMES)
    if colour is not None and box_at(follow(tracks, camera, pid), n) is not None:
        name_tag(out, box_at(follow(tracks, camera, pid), n), str(pid), colour)
    return out


def export(source, folder, tables, pid, first, last, trail_s, drawings, pauses):
    """Write frames first..last to folder/studio_<first>-<last>.mp4 (H.264, plays everywhere).
    pauses: hold the picture on each drawing's first frame for its "hold" seconds."""
    cap = cv2.VideoCapture(str(source))
    fps, w, h, total = video_info(cap)
    size = even_size(w, h, w)
    path = Path(folder) / f"studio_{first}-{last}.mp4"
    proc = open_h264(path, fps, size)
    hold = {d["first"]: d.get("hold", 2) for d in drawings} if pauses else {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
    bar = st.progress(0.0, text="Exporting ...")
    for n in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        out = picture(frame, n, tables["tracks"], tables["camera"], tables["players"], pid, trail_s, fps, drawings)
        out = cv2.resize(out, size) if out.shape[1::-1] != size else out
        for _ in range(1 + int(hold.get(n, 0) * fps)):       # a pause = the same picture again
            proc.stdin.write(out.tobytes())
        bar.progress((n - first + 1) / (last - first + 1), text=f"Exporting frame {n} of {last}")
    cap.release()
    proc.stdin.close()
    proc.wait()
    bar.empty()
    return path


# ---------- the page ----------

st.title("Match Video Tracker Studio")

with st.sidebar:
    st.header("Clip")
    clips = sorted(p for p in DATA.glob("*") if p.suffix.lower() in VIDEO_TYPES)
    if not clips:
        st.error("Put a clip (.mp4 or .mov) in the data/ folder first.")
        st.stop()
    source = st.selectbox("Video", clips, format_func=lambda p: p.name)
    folder = clip_folder(source)
    trail_s = st.slider("Spotlight trail (seconds)", 0.0, 6.0, 3.0, 0.5)
    st.caption(f"Analysis and drawings of this clip: `{folder}/`")

if not is_analysed(folder):
    st.info(f"**{source.name}** is not analysed yet. The app needs the players, teams and camera "
            "of this clip (M2 - M5). It takes a few minutes, once per clip.")
    st.code(f"python analyse.py {source}", language="bash")
    if st.button("Analyse this clip now", type="primary"):
        with st.status(f"Analysing {source.name} ...", expanded=True) as status:
            ok = run_analysis(source)
            status.update(label="Done" if ok else "Stopped with an error", state="complete" if ok else "error")
        if ok:
            st.rerun()
    st.stop()

with st.spinner("Getting the clip ready for the browser (only the first time) ..."):
    web_video, web_data = prepare(source, folder)
stamp = max((folder / f).stat().st_mtime for f in ("tracks_teams.csv", "camera.csv", "teams.csv"))
tables = load_tables(str(folder), stamp)
info = json.loads(web_data.read_text())
fps, total = info["fps"], info["total"]
drawings_file = folder / "drawings.json"
key = f"player_{folder.name}"                      # one player per clip: a new clip starts fresh

holder = tables["possession"]["holder"].dropna() if tables["possession"] is not None else pd.Series()
first_pid = int(holder.mode().iat[0]) if len(holder) else None     # start on the player with the most ball

video_col, stats_col = st.columns([4, 1], gap="medium")
with video_col:
    result = player(key=key, data={"clip": folder.name,
                                   "video_url": publish(web_video, f"{folder.name}/web.mp4"),
                                   "data_url": publish(web_data, f"{folder.name}/web.json"),
                                   "drawings": tele.load(drawings_file),
                                   "player": first_pid, "trail_s": trail_s},
                    default={"player": first_pid, "frame": 1},
                    on_drawings_change=lambda: None, on_player_change=lambda: None,
                    on_frame_change=lambda: None)
    drawings = tele.load(drawings_file)
    if result.get("drawings") is not None and result["drawings"] != drawings:
        drawings = result["drawings"]
        tele.save(drawings, drawings_file)                     # saved for next time
    st.caption("Space = play / pause · ← → = one frame · Esc = cancel a drawing · "
               "drawings are saved in " + f"`{drawings_file}`")

tracks, players = tables["tracks"], tables["players"]
pid = result.get("player")
with stats_col:
    st.subheader("Spotlight" if pid is None else f"Player {pid}")
    if pid is None:
        st.caption("Pick 👆 Spotlight and click a player.")
    else:
        team = players.set_index("id")["team"].get(pid, OTHER)
        st.write(f"**{NAMES[team]}** · on screen for {(tracks['id'] == pid).sum() / fps:.1f} s")
        if tables["possession"] is not None:
            mine = spells(tables["possession"].set_index("frame")["holder"])
            mine = mine[mine["id"] == pid]
            st.write(f"On the ball: {(mine['last'] - mine['first'] + 1).sum() / fps:.1f} s "
                     f"in {len(mine)} spell(s)")
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

st.divider()
st.subheader("Export a video")
st.caption("With your drawings, the team markers, and the spotlight if a player is picked.")
first_s, last_s = st.slider("Part of the clip (seconds)", 0.0, total / fps, (0.0, total / fps), 0.1)
pauses = st.checkbox("Pause on each drawing (like the player)", value=True)
if st.button("Export the video"):
    first, last = max(1, int(first_s * fps) + 1), min(total, int(last_s * fps))
    path = export(source, folder, tables, pid, first, last, trail_s, drawings, pauses)
    st.success(f"Saved {path}")
    st.video(str(path))
    st.download_button("Download the mp4", path.read_bytes(), file_name=f"{folder.name}_{path.name}",
                       mime="video/mp4")
