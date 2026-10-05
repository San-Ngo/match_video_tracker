// The Studio's video player: play, pause, and draw tactics on the paused picture.
//
// Python (app.py) sends: the browser copy of the clip (web.mp4), the analysis as JSON
// (web.json: every player's box and team per frame, and the camera per frame), and the
// saved drawings. This file draws everything on a <canvas> on top of the <video>:
//   - the drawings and the spotlight are painted on grass pixels only, so the players
//     stay in front of them (the same chroma key as the Python version),
//   - the team markers and IDs are painted on top, and are what you click.
// Drawings are plain data, the same as match_video_tracker/telestration.py:
//   {kind: "pass" | "triangle" | "zone", players: [ids], first, last, color, hold}
//   {kind: "arrow" | "line", points: [[pitch x, pitch y], ...], first, last, color, hold}
//   {kind: "arrow", player: id, offset: [pitch dx, pitch dy], ...}   an arrow that runs with a player
// "hold" = seconds the video pauses on the drawing in presentation mode (and in the export).

const COLORS = { yellow: "#ffe600", white: "#ffffff", cyan: "#00e6ff", magenta: "#ff00ff",
                 red: "#ff2828", green: "#3cdc3c" };
const NEED = { pass: 2, triangle: 3, zone: 3, arrow: 2, line: 2 };      // zone: at least 3
const PLAYER_KINDS = ["pass", "triangle", "zone"];
const TOOLS = [
  ["select", "👆 Spotlight", "Click a player to put the spotlight on him (click him again to remove it)."],
  ["pass", "➡ Pass", "Click the passer, then the receiver."],
  ["triangle", "△ Triangle", "Click 3 players. The triangle moves with them."],
  ["zone", "⬠ Zone", "Click 3 or more players, then Finish (or press Enter)."],
  ["arrow", "↗ Arrow", "Click a player (the arrow runs with him) or a spot on the grass, then where it points."],
  ["line", "／ Line", "Click 2 spots on the grass (an offside line, a gap)."],
];
const SHOW = [[2, "2 s"], [4, "4 s"], [6, "6 s"], [10, "10 s"], [0, "to the end"]];
const GRASS = { hLow: 30, hHigh: 90, sLow: 40, vLow: 40 };       // OpenCV HSV, as in grass.py
const MASK_W = 480;                                               // grass mask resolution

const HTML = `
<div class="mvt">
  <div class="stage">
    <video playsinline muted preload="auto"></video>
    <canvas></canvas>
    <button class="big" title="Play (space)">▶</button>
    <div class="badge hidden"></div>
  </div>
  <div class="bar">
    <button class="play" title="Play / pause (space)">▶</button>
    <button class="back" title="One frame back (←)">⏮</button>
    <button class="fwd" title="One frame forward (→)">⏭</button>
    <input class="seek" type="range" min="1" value="1" step="1">
    <span class="time"></span>
    <select class="speed" title="Speed">
      <option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option>
    </select>
  </div>
  <div class="tools"></div>
  <div class="hint"></div>
  <div class="list"></div>
</div>`;

export default function (component) {
  const { parentElement, data } = component;
  let p = parentElement.__mvt;
  if (!p) {                                       // first time: build the player once
    p = parentElement.__mvt = new Player(parentElement);
  }
  p.api = component;                              // the newest setStateValue etc.
  p.update(data);
  return () => {};
}

class Player {
  constructor(root) {
    const host = document.createElement("div");      // root may be a shadow root (no insertAdjacentHTML)
    host.innerHTML = HTML;
    root.appendChild(host);
    const $ = (s) => host.querySelector(s);
    this.el = { video: $("video"), canvas: $("canvas"), big: $(".big"), badge: $(".badge"), play: $(".play"),
                back: $(".back"), fwd: $(".fwd"), seek: $(".seek"), time: $(".time"), speed: $(".speed"),
                tools: $(".tools"), hint: $(".hint"), list: $(".list") };
    this.clip = null; this.info = null;
    this.drawings = []; this.pending = []; this.mouse = null;
    this.tool = "select"; this.color = "yellow"; this.show = 4; this.hold = 2; this.present = true;
    this.player = null; this.trailS = 3;
    this.frame = 1; this.lastFrame = 1; this.holding = null;
    this.layer = document.createElement("canvas");
    this.mask = document.createElement("canvas");
    this.buildTools();
    this.wire();
  }

  // ---------- data from Python ----------
  update(d) {
    if (!d) return;
    this.trailS = d.trail_s ?? 3;
    if (d.clip !== this.clip) {                   // a new clip: start again from frame 1
      this.clip = d.clip;
      this.info = null;
      this.drawings = (d.drawings || []).map((x) => ({ hold: 2, ...x }));
      this.player = d.player ?? null;
      this.pending = [];
      this.el.video.src = d.video_url;
      fetch(d.data_url).then((r) => r.json()).then((info) => {
        this.info = info;
        this.el.seek.max = info.total;
        this.render();
      });
      this.renderList();
    }
  }

  send(name, value) {                             // to Python (app.py reruns with it)
    if (this.api && this.api.setStateValue) this.api.setStateValue(name, value);
  }

  // ---------- controls ----------
  buildTools() {
    const t = this.el.tools;
    t.innerHTML = "";
    for (const [id, label, hint] of TOOLS) {
      const b = document.createElement("button");
      b.textContent = label; b.dataset.tool = id; b.title = hint;
      b.onclick = () => this.setTool(id);
      t.appendChild(b);
    }
    this.finishBtn = this.button(t, "✓ Finish zone", () => this.finishZone());
    this.cancelBtn = this.button(t, "✕ Cancel", () => { this.pending = []; this.refresh(); });
    t.appendChild(Object.assign(document.createElement("span"), { className: "sep" }));
    for (const name of Object.keys(COLORS)) {
      const b = document.createElement("button");
      b.className = "swatch"; b.dataset.color = name; b.title = name; b.style.background = COLORS[name];
      b.onclick = () => { this.color = name; this.refresh(); };
      t.appendChild(b);
    }
    t.appendChild(Object.assign(document.createElement("span"), { className: "sep" }));
    const show = document.createElement("label");
    show.innerHTML = "Show for <select class='show'></select>";
    t.appendChild(show);
    const sel = show.querySelector("select");
    for (const [v, txt] of SHOW) sel.add(new Option(txt, v, false, v === this.show));
    sel.onchange = () => { this.show = +sel.value; };
    const pres = document.createElement("label");
    pres.innerHTML = "<input type='checkbox' checked> Pause at each drawing for <select class='hold'></select>";
    t.appendChild(pres);
    pres.querySelector("input").onchange = (e) => { this.present = e.target.checked; };
    const hold = pres.querySelector("select");
    for (const v of [1, 2, 3, 5]) hold.add(new Option(`${v} s`, v, false, v === this.hold));
    hold.onchange = () => { this.hold = +hold.value; };
    t.appendChild(Object.assign(document.createElement("span"), { className: "sep" }));
    this.undoBtn = this.button(t, "↶ Undo", () => this.remove(this.drawings.length - 1));
    this.refresh();
  }

  button(parent, text, fn) {
    const b = document.createElement("button");
    b.textContent = text; b.onclick = fn;
    parent.appendChild(b);
    return b;
  }

  setTool(id) {
    this.tool = id; this.pending = []; this.anchor = null;
    this.played = true; this.updateButtons();    // the big ▶ would cover the players
    if (id !== "select") this.pause();             // you draw on a paused picture
    this.refresh();
  }

  refresh() {                                     // buttons, hint and picture after any change
    for (const b of this.el.tools.querySelectorAll("[data-tool]")) b.classList.toggle("on", b.dataset.tool === this.tool);
    for (const b of this.el.tools.querySelectorAll("[data-color]")) b.classList.toggle("on", b.dataset.color === this.color);
    this.finishBtn.style.display = this.tool === "zone" ? "" : "none";
    this.finishBtn.disabled = this.pending.length < 3;
    this.cancelBtn.disabled = !this.pending.length;
    this.undoBtn.disabled = !this.drawings.length;
    const hint = TOOLS.find((x) => x[0] === this.tool)[2];
    const picked = this.pending.length
      ? (PLAYER_KINDS.includes(this.tool) ? ` Picked: <b>${this.pending.join(", ")}</b>` : ` <b>${this.pending.length}</b> spot picked.`)
      : "";
    this.el.hint.innerHTML = hint + picked;
    this.render();
  }

  wire() {
    const v = this.el.video;
    this.el.play.onclick = this.el.big.onclick = () => this.toggle();
    this.el.back.onclick = () => this.step(-1);
    this.el.fwd.onclick = () => this.step(1);
    this.el.speed.onchange = () => { v.playbackRate = +this.el.speed.value; };
    this.el.seek.oninput = () => this.seek(+this.el.seek.value);
    v.addEventListener("loadeddata", () => { this.sizeCanvas(); this.render(); });
    v.addEventListener("seeked", () => this.render());
    v.addEventListener("pause", () => { this.updateButtons(); this.render(); this.send("frame", this.frame); });
    v.addEventListener("play", () => { this.updateButtons(); this.loop(); });
    v.addEventListener("ended", () => this.updateButtons());
    new ResizeObserver(() => { this.sizeCanvas(); this.render(); }).observe(this.el.canvas);
    const c = this.el.canvas;
    c.addEventListener("click", (e) => this.click(e));
    c.addEventListener("dblclick", () => { if (this.tool === "zone") this.finishZone(); });
    c.addEventListener("mousemove", (e) => { this.mouse = this.toVideo(e); if (v.paused) this.render(); });
    c.addEventListener("mouseleave", () => { this.mouse = null; if (v.paused) this.render(); });
    this.keys = (e) => {
      const tag = (e.composedPath()[0].tagName || "").toLowerCase();
      if (["input", "textarea", "select"].includes(tag) || !this.el.canvas.isConnected) return;
      if (e.code === "Space") { e.preventDefault(); this.toggle(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); this.step(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); this.step(1); }
      else if (e.key === "Escape") { this.pending = []; this.refresh(); }
      else if (e.key === "Enter" && this.tool === "zone") this.finishZone();
    };
    window.addEventListener("keydown", this.keys);
  }

  updateButtons() {
    const playing = !this.el.video.paused;
    this.el.play.textContent = playing ? "⏸" : "▶";
    if (playing) this.played = true;
    this.el.big.classList.toggle("hidden", playing || this.played);   // only before the first play
  }

  toggle() {
    if (this.el.video.paused) this.play(); else this.pause();
  }

  play() {
    this.cancelHold();
    this.pending = [];
    if (this.frame >= (this.info ? this.info.total : 1)) this.seek(1);
    this.lastFrame = this.frame;
    this.el.video.play();
  }

  pause() {
    this.cancelHold();
    if (!this.el.video.paused) this.el.video.pause();
  }

  seek(n) {
    if (!this.info) return;
    if (!this.played) { this.played = true; this.updateButtons(); }
    n = Math.max(1, Math.min(this.info.total, n));
    this.frame = this.lastFrame = n;
    this.el.video.currentTime = (n - 1 + 0.5) / this.info.fps;      // the middle of frame n
    this.render();
  }

  step(d) {
    this.pause();
    this.seek(this.frame + d);
  }

  // ---------- presentation: pause on each drawing ----------
  loop() {
    const v = this.el.video;
    const tick = (now, meta) => {
      if (v.paused || !this.info) return;
      this.frame = Math.min(this.info.total, Math.round(meta.mediaTime * this.info.fps) + 1);
      const stop = this.present && this.drawings.find((d) => d.first > this.lastFrame && d.first <= this.frame);
      this.lastFrame = this.frame;
      if (stop) this.holdOn(stop);
      this.render();
      if (!v.paused) v.requestVideoFrameCallback(tick);
    };
    if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(tick);
    else {                                        // very old browsers
      const raf = () => { if (!v.paused) { tick(0, { mediaTime: v.currentTime }); requestAnimationFrame(raf); } };
      requestAnimationFrame(raf);
    }
  }

  holdOn(d) {                                     // freeze on a drawing, then play on
    this.el.video.pause();
    this.seek(d.first);
    this.el.badge.textContent = `⏸ ${d.hold ?? this.hold} s`;
    this.el.badge.classList.remove("hidden");
    this.holding = setTimeout(() => { this.holding = null; this.el.badge.classList.add("hidden"); this.el.video.play(); },
                              1000 * (d.hold ?? this.hold));
  }

  cancelHold() {
    if (this.holding) clearTimeout(this.holding);
    this.holding = null;
    this.el.badge.classList.add("hidden");
  }

  // ---------- clicks ----------
  toVideo(e) {                                    // mouse -> pixels of the original video
    const r = this.el.canvas.getBoundingClientRect();
    if (!this.info) return null;
    return [(e.clientX - r.left) * this.info.width / r.width, (e.clientY - r.top) * this.info.height / r.height];
  }

  rows(n) { return (this.info && this.info.frames[n - 1]) || []; }

  playerAt(x, y, n = this.frame) {
    // The player whose box (a bit wider, and down to his marker) holds the click; if two
    // do, the one with the closer feet. Else the closest feet within half a body.
    let best = null, bestD = Infinity;
    for (const [id, team, x1, y1, x2, y2, fx, fy] of this.rows(n)) {
      const w = x2 - x1, h = y2 - y1;
      const inside = x >= x1 - 0.25 * w && x <= x2 + 0.25 * w && y >= y1 && y <= y2 + 0.45 * h;
      const d = Math.hypot(x - fx, y - fy) - (inside ? 1e6 : 0);
      if ((inside || Math.hypot(x - fx, y - fy) < 0.6 * h) && d < bestD) { best = id; bestD = d; }
    }
    return best;
  }

  click(e) {
    if (!this.info) return;
    const v = this.el.video;
    if (!v.paused) {                              // a click while playing pauses first
      this.pause();
      if (this.tool !== "select") return;
    }
    const [x, y] = this.toVideo(e);
    const n = this.frame;
    if (this.tool === "select") {
      const id = this.playerAt(x, y);
      if (id === null) { this.toggle(); return; }  // a click on the grass = play / pause
      this.player = id === this.player ? null : id;
      this.send("player", this.player);
      this.render();
      return;
    }
    if (PLAYER_KINDS.includes(this.tool)) {
      const id = this.playerAt(x, y);
      if (id === null) { this.flash("No player there: click on a player (his marker or his body)."); return; }
      if (this.tool === "zone" && this.pending.length >= 3 && id === this.pending[0]) { this.finishZone(); return; }
      if (this.pending.includes(id)) { this.flash(`Player ${id} is already picked.`); return; }
      this.pending.push(id);
      if (this.tool !== "zone" && this.pending.length === NEED[this.tool]) this.add({ players: this.pending });
    } else {                                      // arrow, line: spots on the pitch
      if (!this.pending.length) {
        // an arrow that starts on a player is tied to him: it moves with him
        this.anchor = this.tool === "arrow" ? this.playerAt(x, y) : null;
        const start = this.anchor !== null ? this.feet(this.anchor, n) : [x, y];
        this.pending.push(this.toPitch(start[0], start[1], n));
      } else {
        this.pending.push(this.toPitch(x, y, n));
      }
      if (this.pending.length === NEED[this.tool]) {
        const [[ax, ay], [bx, by]] = this.pending;
        if (this.anchor !== null) this.add({ player: this.anchor, offset: [bx - ax, by - ay] });
        else this.add({ points: this.pending });
        this.anchor = null;
      }
    }
    this.refresh();
  }

  flash(text) { this.el.hint.innerHTML = `<b>${text}</b>`; }

  finishZone() {
    if (this.tool === "zone" && this.pending.length >= 3) { this.add({ players: this.pending }); this.refresh(); }
  }

  add(what) {
    const fps = this.info.fps, total = this.info.total;
    const d = { kind: this.tool, ...what, first: this.frame,
                last: this.show ? Math.min(total, this.frame + Math.round(this.show * fps)) : total,
                color: this.color, hold: this.hold };
    this.drawings = [...this.drawings, d];
    this.pending = [];
    this.lastFrame = this.frame;                  // do not pause on it again right away
    this.renderList();
    this.send("drawings", this.drawings);
  }

  remove(i) {
    if (i < 0) return;
    this.drawings = this.drawings.filter((_, j) => j !== i);
    this.renderList(); this.refresh();
    this.send("drawings", this.drawings);
  }

  renderList() {
    const l = this.el.list, fps = this.info ? this.info.fps : 60;
    l.innerHTML = this.drawings.length ? "" : "<i>No drawings yet. Pause, pick a tool and click on the picture.</i>";
    this.drawings.forEach((d, i) => {
      const row = document.createElement("div");
      row.className = "row";
      const who = PLAYER_KINDS.includes(d.kind) ? d.players.join(d.kind === "pass" ? " → " : ", ")
        : d.player != null ? `with player ${d.player}` : "on the pitch";
      row.innerHTML = `<span class="dot" style="background:${COLORS[d.color] || "#fff"}"></span>
        <span class="what" title="Go to this moment">${d.kind} ${who} · ${((d.first - 1) / fps).toFixed(1)} s</span>`;
      row.querySelector(".what").onclick = () => { this.pause(); this.seek(d.first); };
      this.button(row, "✕", () => this.remove(i)).title = "Delete this drawing";
      l.appendChild(row);
    });
    if (this.undoBtn) this.undoBtn.disabled = !this.drawings.length;
  }

  // ---------- camera (M4): pixels <-> pitch ----------
  toPitch(x, y, n) {
    const [a, b, c, d, e, f] = this.info.camera[n - 1];
    return [a * x + b * y + c, d * x + e * y + f];
  }

  fromPitch(px, py, n) {
    const [a, b, c, d, e, f] = this.info.camera[n - 1];
    const det = a * e - b * d;
    return [(e * (px - c) - b * (py - f)) / det, (-d * (px - c) + a * (py - f)) / det];
  }

  feet(id, n) {
    for (const r of this.rows(n)) if (r[0] === id) return [r[6], r[7]];
    return null;
  }

  row(id, n) {
    for (const r of this.rows(n)) if (r[0] === id) return r;
    return null;
  }

  // ---------- drawing ----------
  sizeCanvas() {
    const c = this.el.canvas, r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!r.width) return;
    c.width = this.layer.width = Math.round(r.width * dpr);
    c.height = this.layer.height = Math.round(r.height * dpr);
  }

  render() {
    if (!this.info) return;
    const v = this.el.video, c = this.el.canvas;
    if (!c.width) this.sizeCanvas();
    if (v.paused) this.frame = Math.min(this.info.total, Math.floor(v.currentTime * this.info.fps + 1e-3) + 1);
    const n = this.frame, s = c.width / this.info.width;
    this.el.seek.value = n;
    const fps = this.info.fps;
    this.el.time.textContent = `${((n - 1) / fps).toFixed(2)} s · frame ${n}/${this.info.total}`;

    // 1. drawings + spotlight on a layer, then keep only its grass pixels
    const L = this.layer.getContext("2d");
    L.setTransform(1, 0, 0, 1, 0, 0);
    L.globalCompositeOperation = "source-over";
    L.clearRect(0, 0, this.layer.width, this.layer.height);
    L.setTransform(s, 0, 0, s, 0, 0);             // draw in video pixels
    const unit = this.info.width / 1920;          // lines as thick as on a 1920 px video
    for (const d of this.drawings) if (d.first <= n && n <= d.last) this.paint(L, d, n, unit);
    this.paintPending(L, n, unit);
    if (this.player !== null) this.paintSpotlight(L, n, unit);
    if (v.readyState >= 2) this.keepGrass(L);

    const C = c.getContext("2d");
    C.setTransform(1, 0, 0, 1, 0, 0);
    C.clearRect(0, 0, c.width, c.height);
    C.drawImage(this.layer, 0, 0);

    // 2. team markers and IDs on top (these are what you click)
    C.setTransform(s, 0, 0, s, 0, 0);
    const picked = PLAYER_KINDS.includes(this.tool) ? this.pending : [];
    const hover = this.mouse && v.paused && this.tool !== "line" && !(this.tool === "arrow" && this.pending.length)
      ? this.playerAt(this.mouse[0], this.mouse[1]) : null;
    for (const r of this.rows(n)) this.marker(C, r, unit, picked.includes(r[0]), r[0] === hover);
  }

  keepGrass(L) {
    // The grass mask (as grass.py): small copy of the frame -> HSV test -> alpha.
    const m = this.mask, W = MASK_W, H = Math.round(MASK_W * this.info.height / this.info.width);
    if (m.width !== W) { m.width = W; m.height = H; }
    const M = m.getContext("2d", { willReadFrequently: true });
    M.drawImage(this.el.video, 0, 0, W, H);
    const img = M.getImageData(0, 0, W, H), p = img.data;
    for (let i = 0; i < p.length; i += 4) {
      const r = p[i], g = p[i + 1], b = p[i + 2];
      const max = Math.max(r, g, b), min = Math.min(r, g, b), delta = max - min;
      let grass = false;
      if (max >= GRASS.vLow && delta > 0 && (255 * delta) / max >= GRASS.sLow) {
        let h;                                    // hue in degrees
        if (max === r) h = 60 * (((g - b) / delta) % 6);
        else if (max === g) h = 60 * ((b - r) / delta + 2);
        else h = 60 * ((r - g) / delta + 4);
        if (h < 0) h += 360;
        h /= 2;                                   // OpenCV hue: 0 - 179
        grass = h >= GRASS.hLow && h <= GRASS.hHigh;
      }
      p[i + 3] = grass ? 255 : 0;
    }
    M.putImageData(img, 0, 0);
    L.setTransform(1, 0, 0, 1, 0, 0);
    L.globalCompositeOperation = "destination-in";      // keep the layer only where there is grass
    L.drawImage(m, 0, 0, this.layer.width, this.layer.height);
    L.globalCompositeOperation = "source-over";
  }

  points(d, n) {                                  // where drawing d is in frame n (video pixels)
    if (PLAYER_KINDS.includes(d.kind)) {
      const pts = d.players.map((id) => this.feet(id, n));
      return pts.some((q) => q === null) ? null : pts;
    }
    if (d.player !== undefined && d.player !== null) {      // an arrow tied to a player
      const f = this.feet(d.player, n);
      if (!f) return null;
      const [px, py] = this.toPitch(f[0], f[1], n);
      return [f, this.fromPitch(px + d.offset[0], py + d.offset[1], n)];
    }
    return d.points.map(([px, py]) => this.fromPitch(px, py, n));
  }

  paint(L, d, n, unit, colour) {
    const pts = this.points(d, n);
    if (!pts) return;
    const col = colour || COLORS[d.color] || COLORS.yellow;
    L.strokeStyle = L.fillStyle = col;
    L.lineWidth = 4 * unit; L.lineCap = L.lineJoin = "round";
    if (d.kind === "triangle" || d.kind === "zone") {
      const shape = d.kind === "zone" ? hull(pts) : pts;
      L.beginPath(); shape.forEach(([x, y], i) => (i ? L.lineTo(x, y) : L.moveTo(x, y))); L.closePath();
      L.globalAlpha = 0.25; L.fill();
      L.globalAlpha = 0.95; L.stroke();
      L.globalAlpha = 1;
      for (const [x, y] of shape) { L.beginPath(); L.arc(x, y, 6 * unit, 0, 7); L.fill(); }
      return;
    }
    const [[x1, y1], [x2, y2]] = pts;
    L.globalAlpha = 0.95;
    if (d.kind === "pass") L.setLineDash([18 * unit, 12 * unit]);      // a pass: dashed, like TV
    L.beginPath(); L.moveTo(x1, y1); L.lineTo(x2, y2); L.stroke();
    L.setLineDash([]);
    if (d.kind === "pass" || d.kind === "arrow") arrowHead(L, x1, y1, x2, y2, 22 * unit);
    L.globalAlpha = 1;
  }

  paintPending(L, n, unit) {                      // the drawing you are making, to the mouse
    if (!this.pending.length || this.tool === "select") return;
    const pts = PLAYER_KINDS.includes(this.tool)
      ? this.pending.map((id) => this.feet(id, n)).filter(Boolean)
      : this.pending.map(([px, py]) => this.fromPitch(px, py, n));
    if (this.anchor != null && this.feet(this.anchor, n)) pts[0] = this.feet(this.anchor, n);
    if (this.mouse) {
      const id = PLAYER_KINDS.includes(this.tool) ? this.playerAt(this.mouse[0], this.mouse[1]) : null;
      pts.push(id !== null && this.feet(id, n) ? this.feet(id, n) : this.mouse);
    }
    if (pts.length < 2) return;
    L.globalAlpha = 0.7;
    L.strokeStyle = COLORS[this.color]; L.lineWidth = 3 * unit; L.setLineDash([10 * unit, 8 * unit]);
    L.beginPath(); pts.forEach(([x, y], i) => (i ? L.lineTo(x, y) : L.moveTo(x, y)));
    if (this.tool !== "pass" && this.tool !== "arrow" && this.tool !== "line" && pts.length > 2) L.closePath();
    L.stroke(); L.setLineDash([]); L.globalAlpha = 1;
  }

  paintSpotlight(L, n, unit) {
    // A glowing ring under the player and his trail of the last few seconds. The trail is
    // kept in pitch coordinates, so it stays where he really ran while the camera pans.
    const me = this.row(this.player, n);
    if (!me) return;
    const col = this.info.colours[String(me[1])] || "#ffffff";
    const keep = Math.round(this.trailS * this.info.fps), trail = [];
    for (let k = Math.max(1, n - keep); k <= n; k++) {
      const f = this.feet(this.player, k);
      if (f) { const [px, py] = this.toPitch(f[0], f[1], k); trail.push(this.fromPitch(px, py, n)); }
    }
    L.strokeStyle = col; L.lineCap = L.lineJoin = "round";
    for (let i = 1; i < trail.length; i++) {      // fades out towards the past
      L.globalAlpha = 0.15 + 0.75 * (i / trail.length);
      L.lineWidth = (2 + 5 * (i / trail.length)) * unit;
      L.beginPath(); L.moveTo(...trail[i - 1]); L.lineTo(...trail[i]); L.stroke();
    }
    const [, , x1, y1, x2, y2, fx, fy] = me, w = Math.max(x2 - x1, 0.45 * (y2 - y1));
    L.globalAlpha = 0.35; L.fillStyle = col;
    L.beginPath(); L.ellipse(fx, fy, 0.95 * w, 0.32 * w, 0, 0, 7); L.fill();
    L.globalAlpha = 1; L.lineWidth = 5 * unit;
    L.beginPath(); L.ellipse(fx, fy, 0.8 * w, 0.27 * w, 0, 0, 7); L.stroke();
  }

  marker(C, r, unit, picked, hover) {
    const [id, team, x1, y1, x2, y2, fx, fy] = r;
    const col = this.info.colours[String(team)] || "#ffffff";
    const w = Math.max(10 * unit, 0.5 * (x2 - x1));
    C.strokeStyle = col; C.lineWidth = (picked || hover ? 5 : 3) * unit;
    C.beginPath(); C.ellipse(fx, fy, w, 0.35 * w, 0, -0.25 * Math.PI, 1.25 * Math.PI); C.stroke();
    if (picked || hover) {
      C.strokeStyle = picked ? COLORS[this.color] : "#ffffff"; C.lineWidth = 2.5 * unit;
      C.strokeRect(x1 - 4 * unit, y1 - 4 * unit, x2 - x1 + 8 * unit, y2 - y1 + 8 * unit);
    }
    const size = 22 * unit, text = String(id);
    C.font = `bold ${size}px sans-serif`;
    const tw = C.measureText(text).width + 8 * unit, ty = fy + 0.35 * w + 4 * unit;
    C.fillStyle = col; C.fillRect(fx - tw / 2, ty, tw, size + 4 * unit);
    C.fillStyle = textColour(col); C.textAlign = "center"; C.textBaseline = "top";
    C.fillText(text, fx, ty + 3 * unit);
  }
}

function arrowHead(L, x1, y1, x2, y2, size) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  L.setLineDash([]);
  L.beginPath();
  L.moveTo(x2, y2); L.lineTo(x2 - size * Math.cos(a - 0.45), y2 - size * Math.sin(a - 0.45));
  L.moveTo(x2, y2); L.lineTo(x2 - size * Math.cos(a + 0.45), y2 - size * Math.sin(a + 0.45));
  L.stroke();
}

function hull(pts) {                              // convex hull (Andrew's monotone chain)
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

function textColour(hex) {                        // black text on light colours, else white
  const v = parseInt(hex.slice(1), 16), r = v >> 16, g = (v >> 8) & 255, b = v & 255;
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#000000" : "#ffffff";
}
