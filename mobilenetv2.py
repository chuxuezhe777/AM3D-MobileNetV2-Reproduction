"""Reproduce the AM3d MobileNetV2 image classifier on the original cylinders."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras


# Keep the original LabelBinarizer's alphabetical order but report defect metrics.
CLASSES = ("badcylinder", "goodcylinder")
DEFECT = CLASSES.index("badcylinder")
SIZE = 224
OPTIMIZERS = {"adam": keras.optimizers.Adam, "sgd": keras.optimizers.SGD,
              "rmsprop": keras.optimizers.RMSprop}


def read_image(path):
    # imdecode also handles Windows paths containing non-ASCII characters.
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def load_split(root, seed):
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        files = sorted(p for p in (root / name).glob("*")
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise ValueError(f"Expected labeled images under {root / name}")
        paths.extend(files)
        labels.extend([label] * len(files))
    labels = np.asarray(labels)
    train, remaining = train_test_split(np.arange(len(labels)), test_size=0.30,
                                        random_state=seed)
    val, test = train_test_split(remaining, test_size=0.66, random_state=seed)
    split = {"train": train, "val": val, "test": test}
    for name, ids in split.items():
        if len(np.unique(labels[ids])) != 2:
            raise ValueError(f"{name} must contain examples of both classes.")
    return paths, labels, split


def build_extractor(weights):
    base = keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3),
                                         include_top=False, weights=weights)
    base.trainable = False
    rgb = keras.Input((SIZE, SIZE, 3), dtype="uint8", name="rgb")
    x = keras.layers.Rescaling(1.0 / 255)(rgb)
    x = base(x, training=False)
    x = keras.layers.AveragePooling2D(pool_size=4)(x)
    features = keras.layers.Flatten()(x)
    return keras.Model(rgb, features, name="frozen_features")


def extract_features(extractor, paths, batch_size):
    # The fixed backbone and pooling have no fitted statistics on AMdataset.
    vectors = []
    for start in range(0, len(paths), batch_size):
        images = np.stack([read_image(p) for p in paths[start:start + batch_size]])
        vectors.append(extractor.predict_on_batch(images))
    return np.concatenate(vectors)


def build_head(feature_dim, optimizer, learning_rate):
    head = keras.Sequential([
        keras.Input((feature_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation="softmax"),
    ], name="classifier")
    head.compile(optimizer=OPTIMIZERS[optimizer](learning_rate=learning_rate),
                 loss="binary_crossentropy", metrics=["accuracy"])
    return head


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def score(labels, probabilities):
    actual = labels == DEFECT
    predicted = probabilities.argmax(axis=1) == DEFECT
    confidence = probabilities[:, DEFECT]
    tn, fp, fn, tp = metrics.confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    return {
        "n": len(actual), "accuracy": metrics.accuracy_score(actual, predicted),
        "precision_defect": metrics.precision_score(actual, predicted, zero_division=0),
        "recall_defect": metrics.recall_score(actual, predicted, zero_division=0),
        "f1_defect": metrics.f1_score(actual, predicted, zero_division=0),
        "specificity": float(tn / (tn + fp)), "auc_roc": metrics.roc_auc_score(actual, confidence),
        "ap_defect": metrics.average_precision_score(actual, confidence),
        "mcc": metrics.matthews_corrcoef(actual, predicted),
        "kappa": metrics.cohen_kappa_score(actual, predicted),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def save_plot(directory, optimizer, history, test):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(epochs, history[key], label="Train", color="#186676")
        ax.plot(epochs, history[f"val_{key}"], label="Validation", color="#ad4f26")
        ax.set(xlabel="Epoch", ylabel=key.capitalize())
        ax.legend(frameon=False)
    cm = np.array([[test["tn"], test["fp"]], [test["fn"], test["tp"]]])
    metrics.ConfusionMatrixDisplay(cm, display_labels=["Normal", "Defect"]).plot(
        ax=axes[2], cmap="Blues", colorbar=False, values_format="d")
    axes[2].set_title("Test set")
    fig.suptitle(f"MobileNetV2  /  {optimizer}")
    fig.savefig(directory / "evaluation.png", dpi=160)
    plt.close(fig)


def train(args):
    keras.utils.set_random_seed(args.seed)
    paths, labels, split = load_split(args.data, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    save_csv(args.out / "split.csv", [
        {"split": name, "path": paths[i].relative_to(args.data).as_posix(),
         "label": CLASSES[labels[i]]} for name, ids in split.items() for i in ids])
    extractor = build_extractor(args.weights)
    print(f"Extracting fixed ImageNet features for {len(paths)} images.", flush=True)
    x = extract_features(extractor, paths, args.batch_size)
    y = keras.utils.to_categorical(labels, num_classes=2)
    tr, va, te = (split[name] for name in ("train", "val", "test"))
    save_json(args.out / "config.json", {
        "python": platform.python_version(), "backend": keras.backend.backend(),
        "backend_version": version(keras.backend.backend()),
        "keras": keras.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
        "weights": Path(args.weights).name,
        "weights_sha256": (hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
                           if Path(args.weights).is_file() else None),
        "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "optimizer": args.optimizer, "output_classes": CLASSES, "metrics_positive": CLASSES[DEFECT],
        "counts": {name: {c: int((labels[ids] == j).sum()) for j, c in enumerate(CLASSES)}
                   for name, ids in split.items()},
    })
    summary = []
    for name in args.optimizer:
        # Fresh, identically initialized heads make the optimizer comparison paired.
        keras.utils.set_random_seed(args.seed)
        head = build_head(x.shape[1], name, args.lr)
        print(f"Training {name} on {len(tr)} images.", flush=True)
        history = head.fit(x[tr], y[tr], validation_data=(x[va], y[va]),
                           batch_size=args.batch_size, epochs=args.epochs, verbose=2).history
        probabilities = head.predict_on_batch(x[te])
        test = score(labels[te], probabilities)
        directory = args.out / name
        directory.mkdir(exist_ok=True)
        complete = keras.Model(extractor.input, head(extractor.output), name="mobilenetv2_defects")
        complete.save(directory / "model.keras")
        save_json(directory / "test_metrics.json", test)
        save_csv(directory / "history.csv", [
            {"epoch": i + 1, **{k: v[i] for k, v in history.items()}} for i in range(args.epochs)])
        save_csv(directory / "predictions.csv", [
            {"path": paths[idx].relative_to(args.data).as_posix(), "actual": CLASSES[labels[idx]],
             "predicted": CLASSES[int(np.argmax(p))], "p_defect": float(p[DEFECT])}
            for idx, p in zip(te, probabilities)])
        save_plot(directory, name, history, test)
        summary.append({"optimizer": name, **test})
        print(f"{name}: accuracy={test['accuracy']:.4f}, defect recall={test['recall_defect']:.4f}")
    save_csv(args.out / "comparison.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("train")
    run.add_argument("--data", type=Path, default=Path("AMdataset/dataset"))
    run.add_argument("--out", type=Path, default=Path("runs"))
    run.add_argument("--weights", default="imagenet", help="imagenet or local Keras no-top .h5")
    run.add_argument("--optimizer", nargs="+", choices=OPTIMIZERS, default=["adam"])
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--epochs", type=int, default=50)
    run.add_argument("--batch-size", type=int, default=30)
    run.add_argument("--lr", type=float, default=0.001)
    infer = commands.add_parser("predict")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        model = keras.models.load_model(args.model, compile=False)
        p = model.predict_on_batch(read_image(args.image)[None])[0]
        print(json.dumps({"class": CLASSES[int(np.argmax(p))], "p_defect": float(p[DEFECT])}))


if __name__ == "__main__":
    main()
