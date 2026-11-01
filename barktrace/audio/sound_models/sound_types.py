"""Type of each detected dog vocalization: bark, growl, grunt, howl, moan, pant, whine.

Per clip, two frame-level models: PretrainedSED ATST-F (AudioSet-Strong) frame embeddings + class probabilities, and
FineLAP dense frame embeddings + prompt scores. Per sound: the mean and max of each over the sound plus log duration ->
logistic regression (models/dog_sound_types/classifier.joblib), trained on
  - AudioSet-Strong dog events from internet videos (data/audioset_strong_animals, human-labelled times): bark (Bark,
    Yip, Bow-wow), whine (Whimper (dog)), howl, growl, pant (Pant (dog));
  - the play sessions (expert labels; all types, the only grunt and moan examples), also mixed into AudioSet
    backgrounds without dogs so that recording conditions do not decide the type;
  - Barkopedia barks.
Replaced on 2026-10-29 the classifier on class scores only (play sessions + Barkopedia barks, howl and whine rules on
top; archived): every internet-video sound in its training set was a bark, so it called internet-video whines bark
(chihuahua vids/15: 46 of 50 whines called bark).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SED_CLASSES = ["Bark", "Yip", "Bow-wow", "Dog", "Howl", "Growling", "Whimper (dog)", "Whimper", "Pant (dog)", "Pant",
               "Grunt", "Groan", "Wail, moan", "Squeal", "Snort", "Sigh", "Canidae, wild dogs, wolves", "Animal",
               "Domestic animals, pets", "Roar", "Purr", "Crying, sobbing", "Speech", "Music"]
PROMPTS = ["Bark", "Howl", "Whimper", "dog whining", "Growling", "dog growling", "Grunt", "Moan", "Pant", "dog panting"]
CLASSIFIER = Path(os.environ.get("BARKTRACE_MODELS", Path(__file__).resolve().parents[3] / "models")) / "dog_sound_types" / "classifier.joblib"


class TypeTracks:
    """frame features of one clip: {fps_s, emb_s[T, 768], sed[T, 24], fps_f, emb_f[T, 1024], fl[T, 10]}"""

    def __init__(self, device: str = "cuda:0"):
        from .finelap import FineLAPSED
        from .pretrained_sed import IDX, StrongSED
        self.sed = StrongSED("ATST-F", device)
        self.finelap = FineLAPSED(device)
        self.idx = [IDX[c] for c in SED_CLASSES]

    def __call__(self, audio16k: np.ndarray) -> dict:
        fps_s, p, emb_s = self.sed.frame_probs(audio16k, emb=True)
        emb_f = self.finelap.frame_embeds(audio16k)
        return {"fps_s": fps_s, "emb_s": emb_s.astype(np.float32), "sed": p[:, self.idx].astype(np.float32),
                "fps_f": self.finelap.fps, "emb_f": emb_f.astype(np.float32),
                "fl": self.finelap.prompt_probs(emb_f, PROMPTS).astype(np.float32)}


def features(tr: dict, start: float, end: float) -> np.ndarray:
    """mean and max over the sound of every frame feature, plus log duration"""
    def seg(x, fps):
        a = int(start * fps); b = max(a + 1, int(np.ceil(end * fps)))
        y = x[a:b] if a < len(x) else x[-1:]
        return np.r_[y.mean(0), y.max(0)]
    return np.r_[seg(tr["emb_s"], tr["fps_s"]), seg(tr["sed"], tr["fps_s"]), seg(tr["emb_f"], tr["fps_f"]),
                 seg(tr["fl"], tr["fps_f"]), np.log(max(end - start, 0.05))].astype(np.float32)


def classify_tracks(model, tr: dict, events) -> list[tuple[str, float]]:
    """(type, confidence) per (start, end) from a clip's frame features: one sound per interval, as in training (a
    detection that holds several sounds is typed per sound: the pipeline types the barker step's single sounds)"""
    if not len(events):
        return []
    P = model.predict_proba(np.stack([features(tr, s, e) for s, e in events]))
    return [(str(model.classes_[i]), float(p[i])) for p, i in zip(P, P.argmax(1))]


def save_tracks(tr: dict, path: Path) -> None:
    np.savez_compressed(path, **{k: np.asarray(v, np.float16 if np.ndim(v) else np.float64) for k, v in tr.items()})


def load_tracks(path: Path) -> dict:
    z = np.load(path)
    return {k: float(z[k]) if z[k].ndim == 0 else z[k].astype(np.float32) for k in z.files}


def load_model(path: Path = CLASSIFIER):
    import joblib
    return joblib.load(path)


class DogTypeClassifier:
    def __init__(self, device: str = "cuda:0", path: Path = CLASSIFIER):
        self.model = load_model(path)
        self.tracks = TypeTracks(device)

    def classify(self, audio16k: np.ndarray, events, tracks: dict | None = None) -> list[tuple[str, float]]:
        if not events:
            return []
        return classify_tracks(self.model, tracks if tracks is not None else self.tracks(audio16k), events)


if __name__ == "__main__":  # self-check: the saved frame features give the same types as the live ones
    import tempfile
    rng = np.random.default_rng(0)
    tr = {"fps_s": 25.0, "emb_s": rng.normal(size=(250, 768)).astype(np.float32), "sed": rng.random((250, 24), np.float32),
          "fps_f": 6.25, "emb_f": rng.normal(size=(63, 1024)).astype(np.float32), "fl": rng.random((63, 10), np.float32)}
    with tempfile.TemporaryDirectory() as d:
        save_tracks(tr, Path(d) / "t.npz"); t2 = load_tracks(Path(d) / "t.npz")
    m = load_model(); ev = [(0.5, 1.2), (3.0, 3.4), (9.5, 10.0)]
    a, b = classify_tracks(m, tr, ev), classify_tracks(m, t2, ev)
    assert [x[0] for x in a] == [x[0] for x in b] and all(abs(x[1] - y[1]) < 0.02 for x, y in zip(a, b)), (a, b)
    assert features(tr, 9.9, 10.5).shape == features(tr, 0, 1).shape  # past the end: last frame
    print("dog_types self-check ok", a)
