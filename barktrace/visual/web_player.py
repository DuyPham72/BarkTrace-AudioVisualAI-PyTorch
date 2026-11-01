"""Browser player: the ORIGINAL video with the run's boxes drawn live from metadata.json (metadata.py),
so the app can show results without rendering a new video (render.py stays available as an export).
Looks like render.py: dark box + dog_id normally; thick, blinking bright box + "| BARKING" while that
dog is the predicted barker. Uses requestVideoFrameCallback for frame-exact boxes where the browser has it.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

_JS = r"""
function setup() {
  const raw = props.value;
  if (!raw) { element.innerHTML = ""; return; }
  const d = typeof raw === "string" ? JSON.parse(raw) : raw;
  const m = d.meta, fps = m.video.fps, st = m.style;
  element.innerHTML = '<div style="position:relative;display:inline-block;max-width:100%">' +
    '<video controls playsinline preload="auto" style="display:block;max-width:100%;max-height:70vh"></video>' +
    '<canvas style="position:absolute;left:0;top:0;pointer-events:none"></canvas></div>' +
    '<div style="font-size:12px;opacity:.7;margin-top:4px"></div>';
  const video = element.querySelector("video"), canvas = element.querySelector("canvas"), ctx = canvas.getContext("2d");
  video.addEventListener("error", () => {  // e.g. a codec this browser cannot decode (HEVC, or H.264 in plain Chromium)
    const msg = element.querySelector("div > div:last-child") || element;
    msg.innerHTML = '<b>This browser cannot play this video file.</b> Choose "Rendered video file" to get it rendered, ' +
      'or open the app in Chrome, Edge, Safari or Firefox. The boxes are still in metadata.json.';
    msg.style.color = "#c00"; msg.style.opacity = "1";
  });
  video.src = d.video_url;
  // metadata schema 2: per dog, segments of keyframes; between two keys the box is their linear
  // interpolation, rounded half up (= metadata.box_at); drawn in increasing track_id order
  const segs = [];
  for (const [dog, list] of Object.entries(m.boxes)) for (const sg of list) segs.push([sg.track_id, dog, sg.keys]);
  segs.sort((a, b) => a[0] - b[0]);
  function boxesAt(f) {
    const out = [];
    for (const [tid, dog, k] of segs) {
      if (f < k[0][0] || f > k[k.length - 1][0]) continue;
      let lo = 0, hi = k.length - 1;  // last key with frame <= f
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (k[mid][0] <= f) lo = mid; else hi = mid - 1; }
      const a = k[lo], b = k[Math.min(lo + 1, k.length - 1)];
      const w = b[0] === a[0] ? 0 : (f - a[0]) / (b[0] - a[0]);
      out.push([dog, ...[1, 2, 3, 4].map(i => Math.floor(a[i] + w * (b[i] - a[i]) + 0.5))]);
    }
    return out;
  }
  const color = {}; for (const g of m.dogs) color[g.dog_id] = g.color_rgb;
  const rgb = (c, k) => `rgb(${Math.round(c[0] * k)},${Math.round(c[1] * k)},${Math.round(c[2] * k)})`;
  const barking = (dog, t) => m.barks.find(b => b.barker_dog_id === dog && b.start <= t && t <= b.end);
  const VERB = {bark: 'BARKING', howl: 'HOWLING', whine: 'WHINING', growl: 'GROWLING', grunt: 'GRUNTING', moan: 'MOANING', pant: 'PANTING'};
  function draw(frame) {
    const w = video.clientWidth, h = video.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const sx = w / m.video.width, sy = h / m.video.height, t = frame / fps;
    ctx.clearRect(0, 0, w, h);
    for (const [dog, x1, y1, x2, y2] of boxesAt(frame)) {
      const c = color[dog] || [255, 255, 255], bark = barking(dog, t);
      const on = bark && Math.floor(frame / st.flash_interval_frames) % 2 === 0;
      const col = rgb(c, on ? 1 : st.dark_scale);
      ctx.lineWidth = (bark ? st.barking_thickness : st.normal_thickness) * Math.max(sx, 0.5);
      ctx.strokeStyle = col; ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      const label = bark ? `${dog} | ${VERB[bark.type] || 'BARKING'}` : dog;
      ctx.font = "bold 13px sans-serif";
      const tw = ctx.measureText(label).width, top = Math.max(0, y1 * sy - 22);
      ctx.fillStyle = col; ctx.fillRect(x1 * sx, top, tw + 12, 22);
      const sum = c.reduce((a, v) => a + v * (on ? 1 : st.dark_scale), 0);
      ctx.fillStyle = sum < 400 ? "#fff" : "#000"; ctx.fillText(label, x1 * sx + 6, top + 16);
    }
  }
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    const cb = (now, md) => { draw(Math.round(md.mediaTime * fps)); video.requestVideoFrameCallback(cb); };
    video.requestVideoFrameCallback(cb);
  } else {
    const loop = () => { draw(Math.round(video.currentTime * fps)); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
  video.addEventListener("seeked", () => draw(Math.round(video.currentTime * fps)));
  video.addEventListener("loadeddata", () => draw(Math.round(video.currentTime * fps)));
  new ResizeObserver(() => draw(Math.round(video.currentTime * fps))).observe(video);
}
watch("value", setup);
setup();
"""


def player_component(label: str = "Result player", visible: bool = True) -> gr.HTML:
    return gr.HTML(value="", html_template="<div></div>", js_on_load=_JS, label=label, visible=visible)


def player_value(video_path: str | Path, metadata_json: str | Path) -> str:
    """Value for player_component(): the original video (served by Gradio from its upload cache) + metadata."""
    return json.dumps({"video_url": f"/gradio_api/file={Path(video_path).resolve()}",
                       "meta": json.loads(Path(metadata_json).read_text())})
