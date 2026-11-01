#!/usr/bin/env python3
"""Run the pipeline on every video under a folder; each result goes to <output>/<case>/<name>/ with the
video (with --render: the annotated video instead) and its metadata.json. Finished videos are skipped, so a
stopped batch can be restarted. Each run folder under runs/ is deleted once its results are copied.

    DOG_PIPELINE_GPU=0 python barktrace/run_batch.py --input clips/ --output results/ \
        [--shard 0/2] [--render]
"""
from __future__ import annotations

import argparse
import shutil
import traceback
from pathlib import Path

import pipeline

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th video starting at i")
ap.add_argument("--render", action="store_true", help="store the annotated video (boxes drawn) instead of the original")
a = ap.parse_args()
i, n = map(int, a.shard.split("/"))
src, dst = Path(a.input).resolve(), Path(a.output).resolve()
pipeline.RUNS_DIR = pipeline.RUNS_DIR / f"batch_{src.name}_{i}"  # own run folders per shard (names are timestamps)

videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"))[i::n]
for k, v in enumerate(videos, 1):
    out = dst / v.parent.relative_to(src) / v.stem
    if (out / (f"{v.stem}_annotated.mp4" if a.render else "metadata.json")).exists():
        continue
    print(f"[{k}/{len(videos)}] {v.relative_to(src)}", flush=True)
    try:
        r = pipeline.run_pipeline(str(v), output="render" if a.render else "player")
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(traceback.format_exc())
        print(f"  FAILED (see {out / 'error.txt'})", flush=True)
        continue
    out.mkdir(parents=True, exist_ok=True)
    if a.render:
        shutil.copy2(r["annotated_video"], out / f"{v.stem}_annotated.mp4")
        (out / v.name).unlink(missing_ok=True)
    else:
        shutil.copy2(v, out / v.name)
    shutil.copy2(r["metadata"], out / "metadata.json")
    shutil.rmtree(r["run_dir"])
    (out / "error.txt").unlink(missing_ok=True)
    print(f"  {len(r['events'])} sounds, {r['total_seconds']:.0f} s -> {out}", flush=True)
