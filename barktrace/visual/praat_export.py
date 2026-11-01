#!/usr/bin/env python3
"""metadata.json -> Praat files: a TextGrid (open in Praat next to the audio) and a Praat Table (the
tab-separated "tmin tier text tmax" format of Praat's "Down to Table", as in the Dog play pant labels:
CRLF line ends, 6 decimals).

Tiers:
  barks          interval tier: each bark, labelled with the chosen barker dog_ID
                 ("no dog visible" when no dog was on screen, "uncertain" for a hidden near-tie pick)
  dog_1, ...     interval tier per dog: when that dog is on screen (label = its dog_ID)
  dog_1 box, ... point tier per dog: its box keyframes (metadata schema 2), label "x1 y1 x2 y2" in video
                 pixels; between two points the box moves linearly
Usage: praat_export.py <metadata.json> [--out <prefix>]   -> <prefix>.TextGrid and <prefix>.Table
       (default prefix: next to the metadata, named after the video)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _q(text: str) -> str:
    return '"' + str(text).replace('"', '""') + '"'


def tiers_from_metadata(m: dict) -> tuple[float, list[dict]]:
    fps, dur = m["video"]["fps"], m["video"]["duration_seconds"]
    t = lambda f: min(dur, f / fps)  # noqa: E731  frame -> seconds (a frame lasts until the next one)
    barks = []
    for b in sorted(m["barks"], key=lambda b: b["start"]):
        label = (b["barker_dog_id"] if b.get("barker_dog_id") else
                 "uncertain" if b.get("status") == "uncertain" else "no dog visible")
        if barks and (b["start"], min(b["end"], dur)) == barks[-1][3:]:  # a second barker of the same sound: "dog_1 + dog_2"
            barks[-1] = (*barks[-1][:2], f"{barks[-1][2]} + {label}", *barks[-1][3:]); continue
        start = max(b["start"], barks[-1][1]) if barks else b["start"]  # Praat intervals may not overlap
        if min(b["end"], dur) > start:
            barks.append((start, min(b["end"], dur), label, b["start"], min(b["end"], dur)))
    tiers = [{"name": "barks", "class": "IntervalTier", "items": [x[:3] for x in barks]}]
    for d in m["dogs"]:
        dog, segs = d["dog_id"], m["boxes"].get(d["dog_id"], [])
        spans = sorted((t(s["keys"][0][0]), t(s["keys"][-1][0] + 1)) for s in segs)
        merged = []
        for a, b in spans:  # union of the dog's segments (two tracks of one dog can overlap)
            if merged and a <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        tiers.append({"name": dog, "class": "IntervalTier", "items": [(a, b, dog) for a, b in merged]})
        points: dict[float, list[str]] = {}
        for s in segs:
            for f, x1, y1, x2, y2 in s["keys"]:
                points.setdefault(round(t(f), 6), []).append(f"{x1} {y1} {x2} {y2}")
        tiers.append({"name": f"{dog} box", "class": "TextTier",
                      "items": [(tm, " | ".join(v)) for tm, v in sorted(points.items())]})
    return dur, tiers


def write_textgrid(path: Path, dur: float, tiers: list[dict]) -> None:
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {dur}",
           "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for n, tier in enumerate(tiers, start=1):
        out += [f"    item [{n}]:", f'        class = "{tier["class"]}"', f"        name = {_q(tier['name'])}",
                "        xmin = 0", f"        xmax = {dur}"]
        if tier["class"] == "IntervalTier":
            filled, last = [], 0.0  # an interval tier covers 0..xmax: empty intervals fill the gaps
            for a, b, label in tier["items"]:
                if a > last:
                    filled.append((last, a, ""))
                filled.append((a, b, label)); last = b
            if last < dur:
                filled.append((last, dur, ""))
            out.append(f"        intervals: size = {len(filled)}")
            for i, (a, b, label) in enumerate(filled, start=1):
                out += [f"        intervals [{i}]:", f"            xmin = {a}", f"            xmax = {b}",
                        f"            text = {_q(label)}"]
        else:
            out.append(f"        points: size = {len(tier['items'])}")
            for i, (tm, label) in enumerate(tier["items"], start=1):
                out += [f"        points [{i}]:", f"            number = {tm}", f"            mark = {_q(label)}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_table(path: Path, tiers: list[dict]) -> None:
    """Praat's TextGrid "Down to Table" (no line numbers, 6 decimals, tier names, no empty intervals):
    rows sorted by the printed start time, then end time; tab-separated, CRLF line ends."""
    rows = []
    for k, tier in enumerate(tiers):
        for item in tier["items"]:
            a, b, label = item if tier["class"] == "IntervalTier" else (item[0], item[0], item[1])
            if label:
                rows.append((a, b, k, f"{a:.6f}\t{tier['name']}\t{label}\t{b:.6f}"))
    rows.sort(key=lambda r: (round(r[0], 6), round(r[1], 6), r[2]))
    path.write_bytes(("\r\n".join(["tmin\ttier\ttext\ttmax"] + [r[3] for r in rows]) + "\r\n").encode("utf-8"))


def convert(metadata_json: Path, prefix: Path | None = None) -> tuple[Path, Path]:
    m = json.loads(Path(metadata_json).read_text())
    if m.get("schema_version", 1) < 2:
        raise ValueError(f"{metadata_json}: metadata schema 2 needed (re-run the pipeline)")
    prefix = prefix or Path(metadata_json).with_name(Path(m["video"]["file"]).stem)
    dur, tiers = tiers_from_metadata(m)
    tg, tb = Path(f"{prefix}.TextGrid"), Path(f"{prefix}.Table")
    write_textgrid(tg, dur, tiers); write_table(tb, tiers)
    return tg, tb


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output prefix (without extension)")
    a = ap.parse_args()
    for f in convert(a.metadata, a.out):
        print(f)
