"""Merged pipeline: the updated visual pipeline (persistent dog_id) + the updated audio pipeline's
fine-tuned bark detector, with both branches running AT THE SAME TIME on one GPU.

  visual branch  Stage 1a  visual/dog_tracker_lvmt.py: LVMT video instance segmentation, every frame (the only
                           mode since 2026-10-18; the former SAM 3 mode is no longer offered)
                 Stage 2   visual/identity_merger.py: precision identity merger
                 Stage 2b  visual/real_dog_check.py: SigLIP2 real-dog check
  audio branch   Stage 0   audio/vocalization_detector.py --method dogvoc: every dog vocalization (bark, howl, whine, growl,
                           grunt, moan; not panting) = vocalization BEATs trained with background noise x FlexSED
                           dog prompts, threshold 0.80 (precise); the type of each sound (sound_models/sound_types.py)
  then           Stage 1b  visual/barker_attribution.py: which dog made each sound (FINAL v3, dog fine-tuned Synchformer,
                           2026-10-26): the only dog on screen if its score >= 0.4, among several the best
                           Synchformer + motion score if >= 0.9 (>= 0.5 next to a bark given to that dog), otherwise
                           "outside" (runs on the visual branch's GPU);
                           detected sounds are first cut into single barks, one decision per bark (typed ones stay whole)
                 Render    visual/video_renderer.py: persistent dog_id boxes; the barking dog flashes

Typed bark intervals override Stage 0.
GPU (plan_gpus): a GPU is usable when its usage (the higher of utilization % and memory use %, not counting
this app's own idle workers) is below BUSY_PCT. Two usable GPUs with room: visual branch on one, audio
branch on the other, at the same time. One usable GPU: everything on it, at the same time only if both
branches fit in its memory, otherwise one after the other. None: the least busy GPU, one after the other.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VISUAL, AUDIO = HERE / "visual", HERE / "audio"
RUNS_DIR = HERE / "runs"
PY = sys.executable
CACHE = Path(os.environ.get("BARKTRACE_MODELS", REPO / "models"))  # every model weight (README.md, Models)
os.environ["BARKTRACE_MODELS"] = str(CACHE)  # the stage scripts read it
# every tool cache inside the project; inherited by the stage subprocesses
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch_cache"))
os.environ.setdefault("HF_HOME", str(CACHE / "hf_cache"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "mpl_config"))
os.environ.setdefault("TMPDIR", str(REPO / ".tmp"))
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

LVMT_TRACKER = VISUAL / "dog_tracker_lvmt.py"
PRECISION_MERGER = VISUAL / "identity_merger.py"
VERIFY_PROFILES = VISUAL / "real_dog_check.py"
ASSIGN_BARKER = VISUAL / "barker_attribution.py"
DETECT_BARKS = AUDIO / "vocalization_detector.py"
BARK_METHOD = "dogvoc"

_SAM3_MERGE = ["--gap", "1000000", "--big-gap", "-1", "--partner-events", "false", "--size-jump", "-1",
               "--split-threshold", "0", "--multidog-split-threshold", "0.7", "--jump-cut", "0.7", "--dedup-iou", "0.5", "--panel-sim", "0.92",
               "--min-cl-decide", "3", "--part-iomin", "0.85", "--cl-rule", "count", "--same-id-iomin", "0.3",
               "--dup-track-sim", "0.8", "--lookalike-quantile", "0.75", "--parallel-seeds", "true",
               "--complete-min-sim", "0.3", "--min-profile-dets", "20", "--cl-min-dets", "10", "--complete-shared-ok", "true"]
# The only visual mode (2026-10-18): LVMT tracks (visual/dog_tracker_lvmt.py) + the merger and real-dog check. 21 tracking-GT
# videos: HOTA 52.6 vs 51.6 for the former SAM 3 mode, IDF1 56.8 vs 55.5, identity purity 97.6 vs 97.8,
# ID switches 38 vs 56, missed boxes 3331 vs 3545, false boxes 2313 vs 2196
MODES = {"lvmt": {"label": "LVMT", "tracker": LVMT_TRACKER, "track": [], "merge": _SAM3_MERGE,
                  "track_label": "Stage 1a (LVMT detection + tracking)"}}
DEFAULT_MODE = "lvmt"

# Peak GPU memory on this server (README.md; 723 clip, Fast): visual branch 5.0 GB, audio branch 16.1 GB,
# both at the same time 20.2 GB. If a branch still fails in parallel (e.g. out of memory on a long
# video), it is retried alone after the other branch.
NEED_ONE_MB = 17000
NEED_BOTH_MB = 21000

# weights the later stages load (SigLIP2, DINOv3): read into the OS page cache while LVMT runs
PREFETCH_GLOBS = [str(CACHE / "hf_cache/hub/models--google--siglip2-so400m-patch14-384/snapshots/*/model.safetensors"),
                  str(CACHE / "hf_cache/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/*/model.safetensors")]
# every weight a run loads (LVMT, Synchformer, BEATs, FlexSED/CLAP, sound-type models), read at app start so the first
# run doesn't wait on the disk
STARTUP_GLOBS = PREFETCH_GLOBS + [
    str(CACHE / "lvmt/ovis_dinov2_online_55.6.pth"), str(CACHE / "synchformer/24-01-04T16-39-21/*"),
    str(CACHE / "synchformer/dog_ft_20261026/*.pt"),
    str(CACHE / "lab_beats_ft/final_vocal_noise.pt"), str(CACHE / "lab_beats_ft/final_dog.pt"), str(CACHE / "FlexSED/ckpts/flexsed_as.pt"),
    str(CACHE / "hf_cache/hub/models--laion--clap-htsat-unfused/snapshots/*/*"),
    str(CACHE / "hf_cache/hub/models--AndreasXi--FineLAP/snapshots/*/*.safetensors"), str(CACHE / "PretrainedSED/resources/ATST-F_strong_1.pt"), str(CACHE / "dog_sound_types/classifier.joblib"),
    str(Path(os.environ.get("BARKTRACE_BEATS_SED", REPO / "third_party" / "beats_sed")) / "**/BEATS_iter3_plus_AS2M.pt")]

sys.path.insert(0, str(VISUAL))
from video_renderer import render  # noqa: E402
from metadata_writer import write_metadata  # noqa: E402
from praat_export import convert as convert_to_praat  # noqa: E402

_WORKER_STAGES = {str(PRECISION_MERGER): "precision_merger", str(VERIFY_PROFILES): "verify_profiles"}

for _p in (LVMT_TRACKER, PRECISION_MERGER, VERIFY_PROFILES, DETECT_BARKS, ASSIGN_BARKER):
    if not _p.exists():
        raise FileNotFoundError(f"Expected file missing: {_p}")


class PipelineError(RuntimeError):
    pass


# Keep the visual models loaded between runs (barktrace/app.py turns this on): SAM 3 tracking, the merger, the real-dog
# check run inside one long-lived visual/model_worker.py process (same scripts, same arguments, same results).
KEEP_MODELS = os.environ.get("DOG_PIPELINE_KEEP_MODELS", "0") == "1"
WORKER_IDLE_MINUTES = 720  # a worker exits after 12 h without a job (idle models already move to RAM after 60 s)
WARM_MIN_FREE_MB = 11000   # app start: warm a worker on every GPU with this much free memory
STAGE_WORKER = VISUAL / "model_worker.py"
# With the worker (measured 2026-10-12): audio branch peak 16.1 GB; worker 7.3 GB while running, 2.5 GB while
# idle (models offloaded to RAM); worker + audio on one GPU peaked at 23.4 GB. The audio branch runs next to
# the worker on the SAME GPU only if the free memory covers it + what the worker still adds + a margin;
# otherwise on ANOTHER GPU with >= NEED_ONE_MB free, still in parallel; otherwise after the visual branch.
AUDIO_PEAK_MB, WORKER_RUN_MB, WORKER_IDLE_MB, MARGIN_MB = 16100, 7300, 2500, 1000


def same_gpu_parallel_ok(free_mb: float, worker_alive: bool) -> bool:
    return free_mb >= AUDIO_PEAK_MB + MARGIN_MB + WORKER_RUN_MB - (WORKER_IDLE_MB if worker_alive else 0)


class StageWorker:
    """The app's handle on one visual/model_worker.py, pinned to one GPU; restarted if it idled out or died.
    There is one per GPU (WORKERS), so every run can use the least busy GPU with its models loaded."""

    def __init__(self, gpu: str):
        self.proc, self.conn, self.gpu = None, None, gpu
        self.lock = threading.Lock()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.conn is not None

    def pids(self) -> set:
        """The worker and its child processes (the merger's helper pool)."""
        if not self.alive():
            return set()
        out = subprocess.run(["pgrep", "-P", str(self.proc.pid)], capture_output=True, text=True).stdout.split()
        return {self.proc.pid, *map(int, out)}

    def ensure(self) -> str:
        with self.lock:
            if self.alive():
                return self.gpu
            import secrets
            from multiprocessing.connection import Client
            self.stop()
            gpu = self.gpu
            sock = REPO / ".tmp" / f"merged_worker_{os.getpid()}_gpu{gpu}.sock"
            sock.parent.mkdir(exist_ok=True); sock.unlink(missing_ok=True)
            key = secrets.token_hex(16)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "HF_HUB_OFFLINE": "1"}
            logf = open(REPO / ".tmp" / f"merged_worker_{os.getpid()}_gpu{gpu}.log", "a")
            self.proc = subprocess.Popen([PY, str(STAGE_WORKER), str(sock), key, str(WORKER_IDLE_MINUTES)],
                                         cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
            for _ in range(1200):
                if self.proc.poll() is not None:
                    raise PipelineError(f"stage worker failed to start (see {logf.name})")
                try:
                    self.conn = Client(str(sock), family="AF_UNIX", authkey=bytes.fromhex(key))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    time.sleep(0.1)
            else:
                self.stop()
                raise PipelineError("stage worker did not come up")
            return gpu

    def call(self, job: dict) -> dict:
        with self.lock:
            self.conn.send(job)
            return self.conn.recv()

    def stop(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc, self.conn = None, None


WORKERS: dict[str, StageWorker] = {}  # GPU index -> its model-keeping worker
WORKER: StageWorker | None = None    # the visual worker of the current run


def worker_for(gpu: str) -> StageWorker:
    if gpu not in WORKERS:
        WORKERS[gpu] = StageWorker(gpu)
    return WORKERS[gpu]


def _own_gpu_mb() -> dict[str, float]:
    """GPU memory held by this app's own workers (and their helpers), per GPU index."""
    pids = set().union(*(w.pids() for w in WORKERS.values())) if WORKERS else set()
    if not pids:
        return {}
    q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout  # noqa: E731
    idx = dict(l.split(", ")[::-1] for l in q("--query-gpu=index,uuid").strip().splitlines())
    own: dict[str, float] = {}
    for line in q("--query-compute-apps=pid,gpu_uuid,used_memory").strip().splitlines():
        pid, uuid, mb = (x.strip() for x in line.split(","))
        if int(pid) in pids:
            own[idx.get(uuid, "?")] = own.get(idx.get(uuid, "?"), 0) + float(mb)
    return own


def warm_models() -> dict[str, float]:
    """App start: a worker on every GPU with room, with the visual models loaded. -> {gpu: seconds}"""
    own = _own_gpu_mb()
    gpus = [g for g, _, f, _ in _gpu_stats(1) if f + own.get(g, 0) >= WARM_MIN_FREE_MB]
    if os.environ.get("DOG_PIPELINE_GPU"):
        gpus = [os.environ["DOG_PIPELINE_GPU"]]
    times = {}
    for g in gpus:
        w = worker_for(g); w.ensure()
        res = w.call({"script": "warm"})
        if res["rc"] != 0:
            raise PipelineError(f"model warm-up on GPU {g} failed:\n" + res["stderr"][-2000:])
        times[g] = res["seconds"]
    return times


def parse_timestamps(text: str) -> pd.DataFrame:
    """'start,end' lines (seconds) -> timestamps.csv rows; empty text -> empty frame (automatic detection)."""
    rows = []
    for i, raw in enumerate((text or "").strip().splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise PipelineError(f"Line {i + 1} isn't 'start,end' (two numbers): {raw!r}")
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise PipelineError(f"Line {i + 1}: not numeric: {raw!r}") from exc
        if end <= start:
            raise PipelineError(f"Line {i + 1}: end must be after start: {raw!r}")
        rows.append({"event_id": f"event_{len(rows):03d}", "start_time": start, "end_time": end})
    return pd.DataFrame(rows, columns=["event_id", "start_time", "end_time"])


def _run(cmd: list, log: list, label: str, env: dict, show_stdout: bool = True) -> None:
    t0 = time.time()
    stage = _WORKER_STAGES.get(str(cmd[1])) if KEEP_MODELS and len(cmd) > 1 else None
    if stage:
        job = {"script": stage, "argv": [str(c) for c in cmd[2:]], "cwd": str(REPO)}
        for attempt in (1, 2):  # the worker may have idled out just now: restart it once
            try:
                WORKER.ensure()
                r = WORKER.call(job)
                break
            except (EOFError, OSError, BrokenPipeError):
                WORKER.stop()
                if attempt == 2:
                    raise
        res = subprocess.CompletedProcess(cmd, r["rc"], r["stdout"], r["stderr"])
    else:
        res = subprocess.run([str(c) for c in cmd], cwd=REPO, capture_output=True, text=True, env=env)
    log.append(f"$ {' '.join(str(c) for c in cmd)}")
    if res.stdout and show_stdout:
        log.append(res.stdout[-3000:])
    if res.returncode != 0:
        log.append(f"--- {label} FAILED (exit {res.returncode}) ---\n{res.stderr[-3000:]}")
        raise PipelineError(f"{label} failed (exit {res.returncode}) - see log")
    log.append(f"--- {label} OK ({time.time() - t0:.0f} s) ---")


def _gpu_stats(samples: int = 5) -> list[tuple[str, float, float, float]]:
    """[(index, mean utilization % over ~2 s, min free MB, total MB)]; other users' load changes from run to run."""
    acc: dict[str, list] = {}
    for i in range(samples):
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        for line in out.strip().splitlines():
            idx, util, free, total = (x.strip() for x in line.split(","))
            acc.setdefault(idx, []).append((float(util), float(free), float(total)))
        if i + 1 < samples:
            time.sleep(0.5)
    return [(k, sum(v[0] for v in vs) / len(vs), min(v[1] for v in vs), vs[0][2]) for k, vs in acc.items()]


BUSY_PCT = 70  # a GPU at or above this usage (utilization % or memory use %) is left alone when another is usable


def plan_gpus(stats: list[tuple[str, float, float, float]], own: dict[str, float] | None = None,
              worker: dict[str, bool] | None = None) -> tuple[str, str | None, bool]:
    """-> (visual GPU, audio GPU or None = the same GPU, run both branches at the same time?).
    stats from _gpu_stats; own: MB this app's idle workers hold per GPU (counted as free for the visual branch,
    which runs inside that worker); worker: whether a worker is alive per GPU (with keep-models)."""
    own, worker = own or {}, worker or {}
    def usage(s):
        g, util, free, total = s
        return max(util, 100 * (total - free - own.get(g, 0)) / total)
    def fits_visual(s):  # the worker's models while running, or a cold visual branch
        return s[2] + own.get(s[0], 0) >= (WORKER_RUN_MB if worker.get(s[0]) else NEED_ONE_MB - AUDIO_PEAK_MB + WORKER_RUN_MB) + MARGIN_MB
    def fits_both(s):
        return same_gpu_parallel_ok(s[2], worker.get(s[0], False)) if worker else s[2] >= NEED_BOTH_MB
    usable = sorted([s for s in stats if usage(s) < BUSY_PCT], key=lambda s: (usage(s), -s[2]))
    if len(usable) >= 2:  # audio (the big branch) on the usable GPU with most free memory, visual on another
        for a in sorted(usable, key=lambda s: -s[2]):
            v = [s for s in usable if s[0] != a[0] and fits_visual(s)]
            if a[2] >= AUDIO_PEAK_MB + MARGIN_MB and v:
                return v[0][0], a[0], True
    if usable:  # one GPU does everything; in parallel only if both fit
        g = usable[0]
        return g[0], None, fits_both(g)
    g = min(stats, key=lambda s: (usage(s), -s[2]))  # everything busy: the least busy, one after the other
    return g[0], None, False


def prefetch(globs=PREFETCH_GLOBS) -> None:
    """Read files into the OS page cache (errors ignored)."""
    for g in globs:
        for f in glob.glob(g, recursive=True):
            if not os.path.isfile(f):
                continue
            try:
                with open(f, "rb") as fh:
                    while fh.read(64 << 20):
                        pass
            except OSError:
                pass


def render_run(video_path: str, run_dir: Path) -> str:
    """The annotated video of a finished run (the app's "Export annotated video"; output="render")."""
    run_dir = Path(run_dir)
    shown = run_dir / "stage1" / "predictions_shown.csv"
    out = run_dir / "annotated.mp4"
    render(video_path=Path(video_path), tracks_csv=run_dir / "stage2" / "tracks_pieces.csv",
           predictions_csv=shown if shown.exists() else run_dir / "stage1" / "predictions.csv",
           track_to_profile_csv=run_dir / "stage2" / "track_to_all_dog_profile.csv", output_path=out)
    if (run_dir / "stage_log.txt").exists():
        append_stage_log(run_dir, f"annotated video rendered: {out.name}")
    return str(out)


def append_stage_log(run_dir: Path, *lines: str) -> None:
    """Add lines to a run's stage_log.txt (how the run went; the results are in metadata.json)."""
    with open(Path(run_dir) / "stage_log.txt", "a") as f:
        f.write("\n".join(lines) + "\n")


def run_pipeline(video_path: str, timestamps_text: str = "", mode: str = DEFAULT_MODE, parallel: bool | None = None,
                 progress_cb=None, output: str = "player", runs_dir: Path | None = None) -> dict:
    """parallel: None = decide from free GPU memory; True / False forces it.
    runs_dir: parent of the run folder (default RUNS_DIR; the app uses a temporary one).
    output: "player" (metadata.json for the browser player; no video file) | "render" (also annotated.mp4).
    Results: <run_dir>/metadata.json (+ Praat .TextGrid/.Table); how the run went: <run_dir>/stage_log.txt
    (also written when a stage fails)."""
    if not video_path:
        raise PipelineError("Upload a video first.")
    if mode not in MODES:
        raise PipelineError(f"Unknown mode {mode!r}; choose one of {sorted(MODES)}")
    video_path = str(Path(video_path).expanduser().resolve())
    t_start = time.time()
    log: list[str] = []
    run_dir = (runs_dir or RUNS_DIR) / time.strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True, exist_ok=True)
    log += [f"run {run_dir.name}  started {time.strftime('%Y-%m-%d %H:%M:%S')}", f"video: {video_path}",
            f"tracking mode: {mode}; models kept loaded: {KEEP_MODELS}; output: {output}"]
    try:
        return _run_pipeline(video_path, timestamps_text, mode, parallel, progress_cb, output, log, run_dir, t_start)
    except Exception as exc:
        log.append(f"FAILED: {exc}")
        raise
    finally:
        (run_dir / "stage_log.txt").write_text("\n".join(log) + "\n")


def _run_pipeline(video_path, timestamps_text, mode, parallel, progress_cb, output, log, run_dir, t_start) -> dict:
    progress = progress_cb or (lambda f, d: None)
    stage0, stage1, stage2 = run_dir / "stage0", run_dir / "stage1", run_dir / "stage2"
    for d in (stage0, stage1):
        d.mkdir(parents=True, exist_ok=True)
    timestamps_csv = stage0 / "timestamps.csv"

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    audio_gpu = None
    if os.environ.get("DOG_PIPELINE_GPU"):
        gpu, auto_parallel = os.environ["DOG_PIPELINE_GPU"], True
    elif KEEP_MODELS:  # every run: the plan, then that GPU's worker (own idle workers count as free)
        global WORKER
        own = _own_gpu_mb()
        gpu, audio_gpu, auto_parallel = plan_gpus(_gpu_stats(), own, {g: w.alive() for g, w in WORKERS.items()})
        WORKER = worker_for(gpu)
    else:
        gpu, audio_gpu, auto_parallel = plan_gpus(_gpu_stats())
    parallel = auto_parallel if parallel is None else parallel
    env["CUDA_VISIBLE_DEVICES"] = gpu
    audio_env = {**env, "CUDA_VISIBLE_DEVICES": audio_gpu or gpu}
    log.append(f"GPU {gpu}" + (f" (audio branch on GPU {audio_gpu})" if audio_gpu else "")
               + f"; visual and audio branches {'in parallel' if parallel else 'one after the other'}; mode {mode}")

    manual = parse_timestamps(timestamps_text)
    cfg = MODES[mode]
    visual = [([PY, cfg["tracker"], "--video", video_path, "--output-dir", stage1, "--device", "cuda:0",
                *cfg["track"]], cfg["track_label"]),
              ([PY, PRECISION_MERGER, "--video", video_path, "--tracks", stage1 / "tracks.csv", "--output-dir", stage2,
                "--device", "cuda:0", *cfg["merge"]], "Stage 2 (precision identity merger)"),
              ([PY, VERIFY_PROFILES, "--video", video_path, "--stage2-dir", stage2, "--device", "cuda:0"],
               "Stage 2b (real-dog verification)")]
    audio = []
    if len(manual):
        manual.to_csv(timestamps_csv, index=False)
        log.append(f"Using {len(manual)} typed bark interval(s); automatic detection skipped")
    else:
        audio = [([PY, DETECT_BARKS, "--video", video_path, "--output-dir", stage0, "--method", BARK_METHOD],
                  "Stage 0 (dog sound detection: every dog vocalization: noise-robust vocalization BEATs x FlexSED, threshold 0.80)")]

    threading.Thread(target=prefetch, daemon=True).start()
    branch_logs, branch_secs, errors = {"visual": [], "audio": []}, {}, {}

    def branch(name, steps):
        t0 = time.time()
        try:
            for cmd, label in steps:
                _run(cmd, branch_logs[name], label, audio_env if name == "audio" else env)
        except PipelineError as e:
            errors[name] = e
        branch_secs[name] = time.time() - t0

    progress(0.05, "Visual (SAM 3 tracking + identity) and audio (bark detection) branches"
             + (" in parallel" if parallel else ", one after the other"))
    if parallel:
        threads = [threading.Thread(target=branch, args=(n, s)) for n, s in (("visual", visual), ("audio", audio))]
        [t.start() for t in threads]; [t.join() for t in threads]
        for name, steps in (("visual", visual), ("audio", audio)):
            if name in errors:  # e.g. out of GPU memory next to the other branch: once more, alone
                branch_logs[name].append(f"{name} branch failed in parallel ({errors.pop(name)}); retrying it alone")
                progress(0.5, f"Retrying the {name} branch alone")
                branch(name, steps)
    else:
        branch("visual", visual)
        if "visual" not in errors:
            progress(0.4, "Audio branch (bark detection)")
            branch("audio", audio)
    log += branch_logs["visual"] + branch_logs["audio"]
    log.append(f"Branch time: visual {branch_secs.get('visual', 0):.0f} s, audio {branch_secs.get('audio', 0):.0f} s, "
               f"both done after {time.time() - t_start:.0f} s")
    if errors:
        raise next(iter(errors.values()))

    tracks_csv = stage2 / "tracks_pieces.csv"
    if not tracks_csv.exists() or pd.read_csv(tracks_csv).empty:
        raise PipelineError("No dogs were found in this video.")
    events = pd.read_csv(timestamps_csv)
    predictions_csv = stage1 / "predictions.csv"
    progress(0.7, f"Stage 1b which dog made each sound ({len(events)} sound(s); Synchformer + motion)")
    _run([PY, ASSIGN_BARKER, "--video", video_path, "--events", timestamps_csv, "--stage2-dir", stage2,
          "--output", predictions_csv, "--device", "cuda:0", *([] if len(manual) else ["--split"])],  # typed intervals stay whole
         log, "Stage 1b (which dog made each sound)", env)
    pred = pd.read_csv(predictions_csv)
    shown_csv = predictions_csv
    meta = write_metadata(video_path, tracks_csv, shown_csv, stage2 / "track_to_all_dog_profile.csv",
                          run_dir / "metadata.json",
                          barks_source="typed" if len(manual) else f"{BARK_METHOD} (every dog vocalization: noise-robust vocalization BEATs x FlexSED, threshold 0.80)",
                          sound_types_csv=None if len(manual) else stage0 / "sound_types.csv")
    praat = convert_to_praat(meta)  # <video>.TextGrid + <video>.Table next to metadata.json
    log.append(f"metadata.json written ({meta.stat().st_size / 1e3:.0f} KB); Praat: " + ", ".join(p.name for p in praat))
    out = None
    if output == "render":
        progress(0.9, "Rendering")
        t0 = time.time()
        out = render_run(video_path, run_dir)
        log.append(f"--- Render OK ({time.time() - t0:.0f} s) ---")
    log.append(f"Total {time.time() - t_start:.0f} s")
    progress(1.0, "Done")
    det = stage0 / "bark_detection.json"
    profiles = stage2 / "track_to_all_dog_profile.csv"
    return {"run_dir": run_dir, "annotated_video": out, "metadata": str(meta),
            "video": video_path, "events": events,
            "detection": json.loads(det.read_text()) if det.exists() else None,
            "predictions": pred,
            "profiles": pd.read_csv(profiles) if profiles.exists() else pd.DataFrame(),
            "parallel": parallel, "gpu": gpu if not audio_gpu else f"{gpu} (audio on {audio_gpu})", "branch_seconds": branch_secs, "total_seconds": time.time() - t_start,
            "log": "\n".join(log)}


if __name__ == "__main__":  # self-check of the GPU plan (24 GB cards)
    T = 24564
    idle = [("0", 0, 24000, T), ("1", 0, 24000, T)]
    assert plan_gpus(idle) == ("0", "1", True) or plan_gpus(idle) == ("1", "0", True)    # both free: one branch each
    assert plan_gpus([("0", 0, 23000, T), ("1", 80, 20000, T)]) == ("0", None, True)    # GPU 1 busy (util 80%): all on 0, fits both
    assert plan_gpus([("0", 0, 18000, T), ("1", 10, 6000, T)]) == ("0", None, False)    # GPU 1 memory 76% used: all on 0, one after the other
    assert plan_gpus([("0", 0, 23000, T), ("1", 0, 15000, T)]) == ("1", "0", True)      # both usable: audio where it fits (0)
    assert plan_gpus([("0", 0, 16000, T), ("1", 0, 15000, T)]) == ("0", None, False)    # usable, but audio fits on neither beside visual
    assert plan_gpus([("0", 90, 3000, T), ("1", 95, 9000, T)]) == ("0", None, False)    # all busy: least busy, one after the other
    # own idle worker (2.5 GB) on GPU 1 does not make it busy; with live workers the same-GPU rule applies
    assert plan_gpus([("0", 0, 24000, T), ("1", 0, 21500, T)], {"1": 2500}, {"0": True, "1": True})[2]
    assert plan_gpus([("0", 85, 2000, T), ("1", 0, 21500, T)], {"1": 2500}, {"1": True}) == ("1", None, False)  # 21.5 GB free < 21.9 needed
    print("GPU plan OK; now:", plan_gpus(_gpu_stats()))
