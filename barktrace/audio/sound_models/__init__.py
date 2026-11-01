# Hugging Face models (FineLAP, CLAP) come from the project cache, never ~/.cache, even when a bark module is
# imported outside barktrace/pipeline.py (which sets the same values)
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(os.environ.get("BARKTRACE_MODELS", Path(__file__).resolve().parents[3] / "models")) / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
