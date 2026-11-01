"""Import stub for wandb (LVMT's training script logs to it; logging is off for our fine-tuning): every call is a no-op."""


def _noop(*args, **kwargs):
    return None


def __getattr__(name):
    return _noop
