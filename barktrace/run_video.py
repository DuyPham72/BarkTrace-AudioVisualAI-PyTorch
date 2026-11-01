#!/usr/bin/env python3
"""Run the merged pipeline on one video from the command line.

    python run_video.py --video clip.mp4 [--mode lvmt]
        [--timestamps "12.0,13.5\\n40.2,41.0"] [--sequential]
"""
from __future__ import annotations

import argparse

from pipeline import DEFAULT_MODE, MODES, run_pipeline

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--video", required=True)
ap.add_argument("--mode", default=DEFAULT_MODE, choices=list(MODES))
ap.add_argument("--timestamps", default="", help="typed bark intervals ('start,end' per line); default: automatic detection")
ap.add_argument("--sequential", action="store_true", help="run the visual and audio branches one after the other")
a = ap.parse_args()
r = run_pipeline(a.video, a.timestamps.replace("\\n", "\n"), a.mode, parallel=False if a.sequential else None,
                 progress_cb=lambda f, d: print(f"[{f:.2f}] {d}", flush=True))
print(r["log"])
print(f"\n{len(r['events'])} bark events, {r['profiles']['persistent_profile_id'].nunique() if len(r['profiles']) else 0} dogs; "
      f"total {r['total_seconds']:.0f} s -> {r['annotated_video']}")
