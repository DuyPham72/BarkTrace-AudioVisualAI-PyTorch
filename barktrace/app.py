#!/usr/bin/env python3
"""BarkTrace web app: persistent dog_id tracking (visual branch) + automatic dog sound detection (audio branch: noise-robust
vocalization BEATs x FlexSED, audio/vocalization_detector.py "dogvoc"), run at the same time on one GPU, then the
dog that made each sound is picked (Synchformer audio-visual sync + motion). See barktrace/pipeline.py and README.md.

Run:  python barktrace/app.py   (prints its URL: the first free port from 7860 up;
      tunnel it over SSH if remote: ssh -L <port>:127.0.0.1:<port> <this-host>)
"""
from __future__ import annotations

import html
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("DOG_PIPELINE_KEEP_MODELS", "1")  # keep the visual models loaded between runs (barktrace/pipeline.py)

import gradio as gr
import pandas as pd

from pipeline import DEFAULT_MODE, RUNS_DIR, PipelineError, run_pipeline, warm_models

sys.path.insert(0, str(Path(__file__).resolve().parent / "visual"))
from web_player import player_component, player_value  # noqa: E402
from video_renderer import load_metadata, render_metadata  # noqa: E402

# The app keeps nothing: each run (and each recreated video) goes to APP_TMP and is deleted as soon as the page
# has shown it -- Gradio has copied what it serves, and metadata.json is downloaded or viewed from the page.
# KEEP_HOURS only sweeps folders left by a failed or interrupted run. Command-line runs (barktrace/run_video.py,
# barktrace/run_batch.py) still write to runs/.
APP_TMP = RUNS_DIR / "app_tmp"
RECREATED_DIR = APP_TMP / "recreated"
KEEP_HOURS = 1


def clean_tmp(all_: bool = False):
    """Delete app run folders (and recreated videos) older than KEEP_HOURS, or all of them."""
    for p in list(APP_TMP.glob("*")) + list(RECREATED_DIR.glob("*")):
        if p != RECREATED_DIR and (all_ or time.time() - p.stat().st_mtime > KEEP_HOURS * 3600):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


def delete_now(path):
    """Delete one run folder or recreated video once the page has it."""
    if path and Path(path).exists():
        shutil.rmtree(path, ignore_errors=True) if Path(path).is_dir() else Path(path).unlink(missing_ok=True)


OUTPUT_CHOICES = [("Player: boxes drawn over the original video (fast)", "player"),
                  ("Rendered video file (slower)", "render")]

VERB = {"bark": "bark", "howl": "howl", "whine": "whine", "growl": "growl", "grunt": "grunt", "moan": "moan"}


# ---------- result views (all from metadata.json, so a saved result looks the same as a fresh one) ----------

def _cards(items: list[tuple[str, str]]) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="num">{html.escape(str(v))}</div><div class="lbl">{html.escape(k)}</div></div>'
        for k, v in items) + "</div>"


def _summary_cards(m: dict, seconds: float | None = None) -> str:
    named = sum(1 for b in m["barks"] if b["barker_dog_id"])
    items = [("dogs", len(m["dogs"])), ("dog sounds", len(m["barks"])), ("given to a dog", named),
             ("outside / unsure", len(m["barks"]) - named), ("video length", f"{m['video']['duration_seconds']:.0f} s")]
    if seconds is not None:
        items.append(("run time", f"{seconds:.0f} s"))
    return _cards(items)


def _sounds_df(m: dict, pred: pd.DataFrame | None = None) -> pd.DataFrame:
    why = {}
    if pred is not None and len(pred) and "reason" in pred:
        why = dict(zip(pred["event_id"], pred["reason"].fillna("")))
    rows = [{"#": i + 1, "start (s)": round(b["start"], 2), "end (s)": round(b["end"], 2),
             "sound": VERB.get(b.get("type"), b.get("type") or "-"),
             "made by": b["barker_dog_id"] or "outside / unsure", "why": why.get(b["event_id"], b["status"])}
            for i, b in enumerate(m["barks"])]
    return pd.DataFrame(rows, columns=["#", "start (s)", "end (s)", "sound", "made by", "why"])


def _metadata_view(m: dict) -> dict:
    """metadata.json for the on-page viewer: everything but the per-frame boxes, which are summarized
    (they make the file large; the download has them all)."""
    v = {k: val for k, val in m.items() if k != "boxes"}
    v["boxes"] = {dog: f"{len(segs)} segment(s), {sum(len(s['keys']) for s in segs)} keyframes (in the download)"
                  for dog, segs in m["boxes"].items()}
    return v


def _download_copy(meta_path: str, video_file: str) -> str:
    """metadata.json under a name that says which video it belongs to (the browser saves it under this name)."""
    out = Path(meta_path).with_name(f"{Path(video_file).stem}_metadata.json")
    shutil.copy2(meta_path, out)
    return str(out)


def _error(exc: Exception) -> str:
    if isinstance(exc, PipelineError):
        return f'<div class="err"><b>Error:</b> {html.escape(str(exc))}</div>'
    return (f'<div class="err"><b>Unexpected error:</b> {html.escape(str(exc))}'
            f'<pre>{html.escape(traceback.format_exc()[-3000:])}</pre></div>')


# ---------- handlers ----------

def on_run(video_path, timestamps_text, output, progress=gr.Progress()):
    hide = gr.update(value=None, visible=False)
    clean_tmp()
    try:
        r = run_pipeline(video_path, timestamps_text, DEFAULT_MODE, progress_cb=lambda f, d: progress(f, desc=d),
                         output=output, runs_dir=APP_TMP)
    except Exception as exc:  # noqa: BLE001 - shown to the user
        return (hide, gr.update(value="", visible=False), _error(exc), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), hide, gr.update(value=None), "", None, gr.update(visible=False))
    m = json.loads(Path(r["metadata"]).read_text())
    det, bs = r["detection"], r["branch_seconds"]
    source = (f"sounds detected automatically (threshold {det['threshold']})" if det else "typed sound intervals")
    note = (f'<p class="muted">{source} · GPU {r["gpu"]} · branches {"in parallel" if r["parallel"] else "one after the other"}: '
            f'visual {bs.get("visual", 0):.0f} s, audio {bs.get("audio", 0):.0f} s</p>')
    player = player_value(r["video"], r["metadata"]) if output == "player" else ""
    # only the chosen output is shown (no empty video box under the player, or the other way round)
    return (gr.update(value=r["annotated_video"], visible=output == "render"),
            gr.update(value=player, visible=output != "render"),
            _summary_cards(m, r["total_seconds"]) + note,
            _sounds_df(m, r["predictions"]), r["events"].round(3), r["profiles"],
            gr.update(value=_download_copy(r["metadata"], r["video"]), visible=True),
            gr.update(value=_metadata_view(m)), r["log"], str(r["run_dir"]), gr.update(visible=True))


def on_open_saved(video_file, meta_file):
    blank = ("", "", pd.DataFrame(), gr.update(value=None), gr.update())
    if not video_file or not meta_file:
        return ("", '<div class="err">Load both the original clip and its metadata.json.</div>', *blank[2:])
    try:
        m, video = load_metadata(meta_file, video_file)
    except Exception as exc:  # noqa: BLE001 - shown to the user
        return ("", _error(exc), *blank[2:])
    return (player_value(video, meta_file), _summary_cards(m), _sounds_df(m),
            gr.update(value=_metadata_view(m)), gr.update(value=None, visible=False))  # it is already downloaded


def on_recreate_saved(video_file, meta_file, progress=gr.Progress()):
    if not video_file or not meta_file:
        return gr.update(value=None, visible=False), '<div class="err">Load both the original clip and its metadata.json.</div>', None
    try:
        m, video = load_metadata(meta_file, video_file)
        progress(0.1, desc="Recreating the annotated video from metadata.json")
        RECREATED_DIR.mkdir(parents=True, exist_ok=True)
        out = RECREATED_DIR / f"{Path(m['video']['file']).stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        render_metadata(meta_file, out, video)
    except Exception as exc:  # noqa: BLE001
        return gr.update(value=None, visible=False), _error(exc), None
    return gr.update(value=str(out), visible=True), _summary_cards(m), str(out)


# ---------- page ----------

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#hero {padding: 18px 22px; border-radius: 14px; margin-bottom: 6px;
       background: linear-gradient(120deg, var(--primary-600), var(--primary-400)); color: white;}
#hero h1 {margin: 0; font-size: 1.6rem; color: white;}
#hero p {margin: 4px 0 0; opacity: .92; color: white;}
.cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: 10px; margin: 4px 0 8px;}
.card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 12px;
       background: var(--background-fill-secondary);}
.card .num {font-size: 1.45rem; font-weight: 700; color: var(--primary-600);}
.card .lbl {font-size: .8rem; opacity: .75;}
.muted {opacity: .7; font-size: .85rem; margin: 2px 0;}
.err {border: 1px solid #e5484d; background: rgba(229, 72, 77, .08); border-radius: 10px; padding: 10px 12px;}
.err pre {white-space: pre-wrap; font-size: .75rem;}
#run-btn {font-size: 1.05rem;}
"""

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

HOW = ("**How it works.** Dogs are tracked and given persistent ids (`dog_1`, `dog_2`, …); dog sounds are detected "
       "from the audio; each sound is given to the visible dog whose movement is in sync with it, or to "
       "*outside / unsure* when no visible dog clearly made it. While a dog is making a sound its box blinks.")

with gr.Blocks(title="BarkTrace") as demo:
    gr.HTML('<div id="hero"><h1>BarkTrace</h1>'
            '<p>Track every dog, detect every dog sound, and find which dog made it.</p></div>')

    with gr.Tab("Analyze a video"):
        run_dir_state = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                video_in = gr.Video(label="Video", sources=["upload"])
                output_in = gr.Radio(OUTPUT_CHOICES, value="player", label="Show the result as")
                with gr.Accordion("Advanced: type the sound intervals yourself", open=False):
                    ts_in = gr.Textbox(label="One 'start,end' in seconds per line (replaces automatic detection)",
                                       lines=4, placeholder="12.0,13.5\n40.2,41.0")
                run_btn = gr.Button("Analyze", variant="primary", size="lg", elem_id="run-btn")
                gr.Markdown(HOW)
            with gr.Column(scale=6):
                player_out = player_component("Result")
                video_out = gr.Video(label="Annotated video", visible=False)
                status_out = gr.HTML('<p class="muted">Upload a video and press <b>Analyze</b>. '
                                     'The result appears here.</p>')
        with gr.Group(visible=False) as results:
            with gr.Tabs():
                with gr.Tab("Who made each sound"):
                    sounds_out = gr.Dataframe(wrap=True, interactive=False)
                with gr.Tab("Details"):
                    gr.Markdown("**Detected sounds** (audio branch)")
                    events_out = gr.Dataframe(wrap=True, interactive=False)
                    gr.Markdown("**Tracks and their persistent dog id**")
                    prof_out = gr.Dataframe(wrap=True, interactive=False)
                with gr.Tab("Log"):
                    log_out = gr.Textbox(show_label=False, lines=16, max_lines=40)

    with gr.Tab("Open a saved result"):
        gr.Markdown("Load the **original clip** and its **metadata.json** to see the result again without running "
                    "the pipeline. The clip must be the one the metadata was made from (same size, frame count and "
                    "frame rate).")
        recreated_state = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                saved_video_in = gr.File(label="Original clip", file_types=["video"])
                saved_meta_in = gr.File(label="metadata.json", file_types=[".json"])
                with gr.Row():
                    open_btn = gr.Button("Open", variant="primary")
                    recreate_btn = gr.Button("Recreate annotated video")
            with gr.Column(scale=6):
                saved_player_out = player_component("Saved result")
                saved_video_out = gr.Video(label="Recreated annotated video", visible=False)
                saved_status_out = gr.HTML()
        gr.Markdown("**Who made each sound**")
        saved_sounds_out = gr.Dataframe(wrap=True, interactive=False)

    with gr.Tab("metadata.json"):
        gr.Markdown("The full result of the last analyzed or opened video: video info, dogs, sounds and every box. "
                    "Nothing is kept on the server, so **download it to keep the result**; the *Open a saved result* "
                    "tab shows it again and can recreate the annotated video.")
        meta_dl = gr.DownloadButton("Download metadata.json", visible=False, variant="primary")
        meta_view = gr.JSON(label="Contents (per-frame boxes summarized; the download has them all)", open=True)

    run_btn.click(on_run, inputs=[video_in, ts_in, output_in],
                  outputs=[video_out, player_out, status_out, sounds_out, events_out, prof_out,
                           meta_dl, meta_view, log_out, run_dir_state, results],
                  show_progress_on=player_out  # one progress bar, not one per output
                  ).then(delete_now, inputs=run_dir_state)  # the page has everything: keep nothing
    open_btn.click(on_open_saved, inputs=[saved_video_in, saved_meta_in],
                   outputs=[saved_player_out, saved_status_out, saved_sounds_out, meta_view, meta_dl],
                   show_progress_on=saved_player_out)
    recreate_btn.click(on_recreate_saved, inputs=[saved_video_in, saved_meta_in],
                       outputs=[saved_video_out, saved_status_out, recreated_state],
                       show_progress_on=saved_status_out
                       ).then(delete_now, inputs=recreated_state)


def _warm():
    try:
        t = warm_models()
        print("visual models loaded on GPU " + ", ".join(f"{g} ({s:.0f} s)" for g, s in t.items())
              + "; each run uses the least busy of them (models move to RAM after 60 idle s; workers exit after 12 idle h)",
              flush=True)
    except Exception as exc:  # noqa: BLE001 - the first run then loads them instead
        print(f"model warm-up skipped: {exc}", flush=True)


if __name__ == "__main__":
    from pipeline import STARTUP_GLOBS, prefetch
    clean_tmp(all_=True)
    threading.Thread(target=prefetch, args=(STARTUP_GLOBS,), daemon=True).start()  # warm the page cache
    threading.Thread(target=_warm, daemon=True).start()
    # port: first free from 7860 up (GRADIO_SERVER_PORT to force one)
    demo.queue().launch(server_name="127.0.0.1", share=False, theme=THEME, css=CSS)
