"""Study One entry point and shared MobileNetV2 training code for all three studies."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
from http.client import HTTPException
import json
from math import ceil
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import time
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import torch


IMAGE_SIZE = 224
FEATURE_SIZE = 1280
TOP_K = 4
TOP_K_WEIGHT = 0.25
SPLIT_SEED = 42
SEEDS = (42, 43, 44)
LEARNING_RATE = 0.001
ARMS = ("reference", "global_average", "global_topk")
OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "sgd": keras.optimizers.SGD,
    "rmsprop": keras.optimizers.RMSprop,
}
SCORES = ("accuracy", "precision", "recall", "f1", "mcc", "kappa", "auc", "ap",
          "defect_recall")
REVISION = "52fe75dec70c1e8046493b4e3256f8f752a4d39d"
AM_URL = f"https://codeload.github.com/mahsan2/AMdataset/zip/{REVISION}"
WEIGHTS_SHA256 = "f8aff69536bd77a692c594f559c798c19bf7f3f36668fc9fa00b21c6aab4797c"
EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class Study:
    name: str
    classes: tuple[str, str]  # paper ordering: defect=0, normal=1
    counts: tuple[int, int]
    split_sizes: tuple[int, int, int]
    member_prefix: str
    inventory_sha256: str
    url: str
    archive_name: str
    epochs: int
    batch_size: int
    paper: dict[str, tuple[float, ...]]


STUDY = Study(
    name="one",
    classes=("badcylinder", "goodcylinder"),
    counts=(136, 135),
    split_sizes=(189, 27, 55),
    member_prefix=f"AMdataset-{REVISION}/dataset/",
    inventory_sha256="80d2453793591e2077100429d38cf1a2ddd618f2035b19129d3fb13864bf2c8d",
    url=AM_URL,
    archive_name="amdataset.zip",
    epochs=30,
    batch_size=8,
    # Test rows of the author's Studyone/{Adam,Sgd,rmsprop}/mobilenetmodel_results.xlsx.
    # The final field is the author's specificity, i.e. recall for the defect class.
    paper={
        "adam": (1.0,) * len(SCORES),
        "sgd": (54 / 55, 28 / 29, 1.0, 56 / 57, 0.9642392563832106,
                0.9636002647253474, 1.0, 1.0, 26 / 27),
        "rmsprop": (1.0,) * len(SCORES),
    },
)


def download_archive(study, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(f".{os.getpid()}.part")
    print(f"Downloading {study.url} to {path}", flush=True)
    request = Request(study.url, headers={"User-Agent": "AM3d-MobileNetV2-reproduction"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=180) as source, pending.open("wb") as target:
                shutil.copyfileobj(source, target)
            with ZipFile(pending):
                pass
            pending.replace(path)
            return
        except (OSError, HTTPException, BadZipFile, EOFError):
            pending.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def source_paths(study, entries):
    paths = sorted(name for name in entries
                   if len(name.split("/")) == 2
                   and name.split("/")[0] in study.classes
                   and Path(name).suffix.lower() in EXTENSIONS)
    labels = np.array([study.classes.index(name.split("/")[0]) for name in paths], dtype=np.int64)
    if tuple(np.bincount(labels, minlength=2)) != study.counts:
        raise ValueError(f"Study {study.name} requires class counts {study.counts}")
    return paths, labels


def read_images(study, paths, read):
    images = np.empty((len(paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    inventory = hashlib.sha256()
    for index, path in enumerate(paths):
        data = read(path)
        inventory.update(f"{path} {hashlib.sha256(data).hexdigest()}\n".encode())
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images[index] = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    if inventory.hexdigest() != study.inventory_sha256:
        raise ValueError(f"Study {study.name} files differ from the shared image inventory")
    return images


def load_dataset(study, args):
    if args.data:
        root = args.data
        paths, labels = source_paths(study, (p.relative_to(root).as_posix()
                                             for p in root.glob("*/*") if p.is_file()))
        return paths, labels, read_images(study, paths, lambda p: (root / p).read_bytes())
    archive = args.archive or Path("data") / study.archive_name
    if not archive.is_file():
        download_archive(study, archive)
    prefix = study.member_prefix
    with ZipFile(archive) as zipped:
        paths, labels = source_paths(study, (p[len(prefix):] for p in zipped.namelist()
                                             if p.startswith(prefix)))
        return paths, labels, read_images(study, paths, lambda p: zipped.read(prefix + p))


def split_images(study, n):
    train, holdout = train_test_split(np.arange(n), test_size=0.30, random_state=SPLIT_SEED)
    val, test = train_test_split(holdout, test_size=0.66, random_state=SPLIT_SEED)
    if (len(train), len(val), len(test)) != study.split_sizes:
        raise ValueError(f"Unexpected split for Study {study.name}")
    return {"train": train, "validation": val, "test": test}


def build_extractors(weights):
    backbone = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    backbone.trainable = False

    legacy_input = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype="uint8", name="rgb")
    legacy_map = backbone(keras.layers.Rescaling(1 / 255)(legacy_input), training=False)
    reference = keras.layers.Flatten()(keras.layers.AveragePooling2D(pool_size=4)(legacy_map))
    reference_model = keras.Model(legacy_input, reference, name="reference_encoder")

    rgb = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype="uint8", name="rgb")
    normalized = keras.layers.Rescaling(1 / 127.5, offset=-1)(rgb)
    final_map = backbone(normalized, training=False)
    global_mean = keras.layers.GlobalAveragePooling2D()(final_map)
    # Shape: (batch, channels, spatial locations). Take the four largest values per channel.
    locations = keras.layers.Reshape((49, FEATURE_SIZE))(final_map)
    channels = keras.layers.Permute((2, 1))(locations)
    peaks, _ = keras.ops.top_k(channels, TOP_K, sorted=False)
    topk_mean = keras.ops.mean(peaks, axis=-1)
    fused = keras.layers.Add()([
        keras.layers.Rescaling(1 - TOP_K_WEIGHT)(global_mean),
        keras.layers.Rescaling(TOP_K_WEIGHT)(topk_mean),
    ])
    return (reference_model, keras.Model(rgb, (global_mean, fused), name="pooled_encoder"),
            keras.Model(rgb, fused, name="fusion_encoder"))


def extract_features(images, reference, pooled):
    features = {arm: np.empty((len(images), FEATURE_SIZE), dtype=np.float32) for arm in ARMS}
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            end = min(start + 32, len(images))
            batch = images[start:end]
            gap, fused = pooled(batch, training=False)
            for arm, tensor in zip(ARMS, (reference(batch, training=False), gap, fused), strict=True):
                features[arm][start:end] = keras.ops.convert_to_numpy(tensor)
            if end % 256 == 0 or end == len(images):
                print(f"Encoded {end}/{len(images)} images", flush=True)
    return features


def build_head(optimizer):
    head = keras.Sequential([
        keras.Input((FEATURE_SIZE,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation="softmax"),
    ], name="classifier")
    head.compile(optimizer=OPTIMIZERS[optimizer](learning_rate=LEARNING_RATE),
                 loss="binary_crossentropy")
    return head


def fit_head(study, features, targets, train, optimizer, seed):
    keras.utils.set_random_seed(seed)
    head = build_head(optimizer)
    rng = np.random.default_rng(seed)
    for _ in range(study.epochs):
        order = rng.permutation(train)
        for start in range(0, len(order), study.batch_size):
            indices = order[start:start + study.batch_size]
            head.train_on_batch(features[indices], targets[indices])
    updates = int(keras.ops.convert_to_numpy(head.optimizer.iterations))
    if updates != study.epochs * ceil(len(train) / study.batch_size):
        raise RuntimeError(f"Unexpected number of optimizer updates: {updates}")
    return head


def score(labels, probabilities):
    predicted = probabilities.argmax(axis=1)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, predicted, labels=(0, 1)).ravel()
    return {
        "accuracy": metrics.accuracy_score(labels, predicted),
        "precision": metrics.precision_score(labels, predicted, zero_division=0),
        "recall": metrics.recall_score(labels, predicted, zero_division=0),
        "f1": metrics.f1_score(labels, predicted, zero_division=0),
        "mcc": metrics.matthews_corrcoef(labels, predicted),
        "kappa": metrics.cohen_kappa_score(labels, predicted),
        "auc": metrics.roc_auc_score(labels, probabilities[:, 1]),
        "ap": metrics.average_precision_score(labels, probabilities[:, 1]),
        "defect_recall": float(tn / (tn + fp)),
        "normal_correct": int(tp), "defect_correct": int(tn),
        "missed_defects": int(fp), "false_alarms": int(fn), "n": len(labels),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(study, runs, out):
    comparison = []
    for optimizer, paper in study.paper.items():
        for metric, original in zip(SCORES, paper, strict=True):
            values = {arm: [r[metric] for r in runs if r["optimizer"] == optimizer
                           and r["arm"] == arm and r["split"] == "test"] for arm in ARMS}
            improved = mean(values["global_topk"])
            comparison.append({
                "study": study.name, "optimizer": optimizer, "metric": metric,
                "global_topk_mean": improved, "paper": original,
                "delta_vs_paper": improved - original,
                "global_topk_sd": stdev(values["global_topk"]),
                "matched_reference_mean": mean(values["reference"]),
                "matched_global_average_mean": mean(values["global_average"]),
            })
    write_csv(out / "paper_comparison.csv", comparison)


def train(study, args):
    if keras.backend.backend() != "torch":
        raise ValueError("This experiment uses the Keras PyTorch backend. Set KERAS_BACKEND=torch.")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    print(f"Study {study.name}: loading and verifying images", flush=True)
    paths, labels, images = load_dataset(study, args)
    splits = split_images(study, len(labels))
    write_csv(args.out / "split.csv", [{"path": paths[i], "label": int(labels[i]), "split": name}
                                      for name, ids in splits.items() for i in ids])
    weights = str(args.weights) if args.weights else "imagenet"
    if args.weights and hashlib.sha256(args.weights.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("The weights file must contain the pinned Keras MobileNetV2 ImageNet weights.")
    reference, pooled, fusion_encoder = build_extractors(weights)
    features = extract_features(images, reference, pooled)
    targets = keras.utils.to_categorical(labels, num_classes=2).astype(np.float32)
    del images

    runs, predictions = [], []
    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            for arm in ARMS:
                head = fit_head(study, features[arm], targets, splits["train"], optimizer, seed)
                identity = {"optimizer": optimizer, "seed": seed, "arm": arm}
                for name, ids in splits.items():
                    with torch.inference_mode():
                        prob = keras.ops.convert_to_numpy(head(features[arm][ids], training=False))
                    runs.append({**identity, "split": name, **score(labels[ids], prob)})
                    predictions.extend({**identity, "split": name, "path": paths[i],
                                        "label": int(labels[i]), "p_defect": float(p[0]),
                                        "p_normal": float(p[1])}
                                       for i, p in zip(ids, prob, strict=True))
                if arm == "global_topk":
                    model = keras.Model(fusion_encoder.input, head(fusion_encoder.output))
                    model.save(args.out / f"global_topk_{optimizer}_seed{seed}.keras")
                write_csv(args.out / "metrics.csv", runs)
                print(f"Fit {optimizer} seed {seed} {arm} | test accuracy "
                      f"{100 * runs[-1]['accuracy']:.2f}%", flush=True)

    write_csv(args.out / "predictions.csv", predictions)
    summarize(study, runs, args.out)
    print(f"Completed Study {study.name}. Evidence: {args.out}", flush=True)


def predict(args):
    rgb = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f"Could not decode {args.image}")
    rgb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))
    model = keras.models.load_model(args.model, compile=False)
    with torch.inference_mode():
        p = keras.ops.convert_to_numpy(model(rgb[None], training=False))[0]
    print(json.dumps({"label": ("defect", "normal")[int(p.argmax())],
                      "p_defect": float(p[0]), "p_normal": float(p[1])}, indent=2))


def main(study):
    parser = argparse.ArgumentParser(description=f"MobileNetV2 global and Top-k pooling, Study {study.name}")
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("train")
    source = experiment.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Path to the study's downloaded ZIP")
    source.add_argument("--data", type=Path, help="Directory containing the two class folders")
    experiment.add_argument("--weights", type=Path, help="Local Keras no-top ImageNet weights")
    experiment.add_argument("--out", type=Path, default=Path("runs") / f"study_{study.name}")
    experiment.add_argument("--threads", type=int, default=4)
    inference = commands.add_parser("predict")
    inference.add_argument("--image", type=Path, required=True)
    inference.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(study, args)
    else:
        predict(args)


if __name__ == "__main__":
    main(STUDY)
