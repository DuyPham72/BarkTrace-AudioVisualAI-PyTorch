#!/usr/bin/env python3
"""Long-lived stage worker: runs the pipeline's GPU stages inside ONE process that keeps their models
loaded between runs (SigLIP2, DINOv3), instead of a fresh Python
process per stage that re-imports PyTorch and reloads every model.

Each job calls a stage script's own main() with the same command-line arguments and working directory
the subprocess would get, so the results are the same. Started by barktrace/pipeline.py (KEEP_MODELS); pinned to
the GPU it was started on; exits after IDLE_MINUTES without a job (frees the GPU on this shared server)
or when the app goes away. Protocol: multiprocessing.connection on a local socket with a random key;
job = {"script": name, "argv": [...], "cwd": str} -> {"rc", "stdout", "stderr", "seconds"}.

Usage (barktrace/pipeline.py does this): model_worker.py <socket path> <authkey hex> <idle minutes>
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULES: dict = {}
# When idle for OFFLOAD_AFTER_S, move every loaded model to RAM and free the GPU (other apps/users share it);
# the next job moves them back first (a few seconds; the same weights, so the same results).
OFFLOAD_AFTER_S = float(os.environ.get("DOG_WORKER_OFFLOAD_AFTER_S", "60"))
_offloaded = False


def _loaded_modules():
    """Every torch module the stage caches hold (models are created lazily, so look them up each time)."""
    import torch.nn as nn
    mods = []

    def add(obj):
        if isinstance(obj, nn.Module):
            mods.append(obj)
        elif obj is not None and hasattr(obj, "__dict__"):
            mods.extend(v for v in vars(obj).values() if isinstance(v, nn.Module))
            for v in vars(obj).values():  # one level deeper (e.g. the extractor's DINOv3 backend)
                if not isinstance(v, nn.Module) and hasattr(v, "__dict__") and type(v).__module__ != "builtins":
                    mods.extend(w for w in vars(v).values() if isinstance(w, nn.Module))
    if "verify_profiles" in MODULES:
        for v in MODULES["verify_profiles"]._VERIFIERS.values():
            add(v)
    if "precision_merger" in MODULES:
        for ex in MODULES["precision_merger"]._EXTRACTORS.values():
            add(ex)
    return list({id(m): m for m in mods}.values())


def _move_models(device: str) -> None:
    import torch
    for m in _loaded_modules():
        m.to(device)
    torch.cuda.empty_cache()


def _load(name: str):
    """Stage modules, imported once."""
    if name not in MODULES:
        if name == "precision_merger":
            import identity_merger as m
        elif name == "verify_profiles":
            import real_dog_check as m
        else:
            raise ValueError(f"unknown stage {name}")
        MODULES[name] = m
    return MODULES[name]


def run_job(job: dict) -> dict:
    m = _load(job["script"])
    out, err, rc, t0, cwd = io.StringIO(), io.StringIO(), 0, time.time(), os.getcwd()
    argv = sys.argv
    try:
        os.chdir(job["cwd"]); sys.argv = [job["script"] + ".py", *job["argv"]]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                m.main()
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception:  # noqa: BLE001 - reported back like a failed subprocess
        rc = 1; err.write(traceback.format_exc())
    finally:
        sys.argv = argv; os.chdir(cwd)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return {"rc": rc, "stdout": out.getvalue(), "stderr": err.getvalue(), "seconds": time.time() - t0}


def warm() -> None:
    """Load every model now, so the first run is fast too."""
    import torch
    torch.zeros(1).cuda()
    # SAM 3 and the YOLO confirmer are no longer warmed: LVMT is the only tracker (it runs as its own process)
    _load("verify_profiles").get_verifier("cuda:0")
    _load("precision_merger").get_extractor("cuda:0")


def _shutdown() -> None:
    """Stop the merger's helper processes (its parallel-seed pool): they would outlive this worker as
    orphans, each holding ~0.6 GB of GPU memory."""
    pm = MODULES.get("precision_merger")
    pool = getattr(pm, "_POOL", None) if pm is not None else None
    if pool is not None:
        for proc in list(getattr(pool, "_processes", {}).values()):
            proc.terminate()
        pool.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    import signal
    from multiprocessing.connection import Listener
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # stopped by the app: still clean up (finally below)
    address, authkey, idle = sys.argv[1], bytes.fromhex(sys.argv[2]), float(sys.argv[3]) * 60
    with Listener(address, family="AF_UNIX", authkey=authkey) as listener:
        conn = listener.accept()
        try:
            global _offloaded
            waited = 0.0
            while True:
                if not conn.poll(min(OFFLOAD_AFTER_S, idle - waited) if not _offloaded else idle - waited):
                    waited += OFFLOAD_AFTER_S if not _offloaded else idle
                    if waited >= idle:
                        break
                    if not _offloaded:  # idle: give the GPU back
                        _move_models("cpu"); _offloaded = True
                    continue
                waited = 0.0
                job = conn.recv()
                if _offloaded:
                    _move_models("cuda:0"); _offloaded = False
                if job.get("script") == "warm":
                    t0 = time.time()
                    try:
                        warm(); conn.send({"rc": 0, "stdout": "", "stderr": "", "seconds": time.time() - t0})
                    except Exception:  # noqa: BLE001
                        conn.send({"rc": 1, "stdout": "", "stderr": traceback.format_exc(), "seconds": time.time() - t0})
                    continue
                conn.send(run_job(job))
        except (EOFError, OSError):  # the app went away
            pass
        finally:
            _shutdown()


if __name__ == "__main__":
    main()
