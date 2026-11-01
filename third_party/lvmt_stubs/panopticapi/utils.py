"""Import stub for panopticapi (only rgb2id is used by LVMT, for panoptic training data)."""
import numpy as np


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


class IdGenerator:  # used only by LVMT's panoptic (VPS) evaluator, never at inference
    def __init__(self, categories):
        raise RuntimeError("panopticapi stub: panoptic evaluation is not available")
