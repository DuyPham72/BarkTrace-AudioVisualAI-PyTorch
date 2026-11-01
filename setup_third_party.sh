#!/usr/bin/env bash
# Third-party code at the exact commits this project used (all unmodified clones).
#   third_party/LVMT, third_party/detectron2   dog tracking (Detectron2 is used as pure Python: no build step)
#   $BARKTRACE_MODELS/FlexSED, /PretrainedSED  sound models (their checkpoints go inside them, README.md "Models")
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
MODELS="${BARKTRACE_MODELS:-$REPO/models}"
clone() {  # url commit dir
  if [ -d "$3/.git" ] || [ -L "$3" ]; then echo "exists: $3"; return; fi
  git clone --quiet "$1" "$3" && git -C "$3" checkout --quiet "$2" && echo "cloned $1 @ $2 -> $3"
}
mkdir -p "$REPO/third_party" "$MODELS"
clone https://github.com/tue-mps/lvmt a0fa70c "$REPO/third_party/LVMT"
clone https://github.com/facebookresearch/detectron2 1e3e13b "$REPO/third_party/detectron2"
clone https://github.com/JHU-LCAP/FlexSED 89aac33 "$MODELS/FlexSED"
clone https://github.com/fschmid56/PretrainedSED 1aa47e4 "$MODELS/PretrainedSED"
