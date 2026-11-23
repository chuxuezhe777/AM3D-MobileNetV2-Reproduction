"""Compare clean and corruption-augmented MobileNetV2 training on Study One."""

import argparse
import csv
import hashlib
from math import ceil
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np

from mobilenetv2 import (CLASSES, SIZE, build_extractor, build_head, keras,
                        read_image, save_csv, save_json, score)


SEEDS = (42, 43, 44)
EPOCHS = 30
BATCH = 30
LR = 0.001
ARMS = ("clean", "augmented")
VIEWS = ("clean", "blur9_sigma2.0", "jpeg_quality15", "resize56", "rotate10_reflect")
CONDITIONS = ("clean", "brightness0.5", "brightness1.5", *VIEWS[1:])
RULE = ("Adopt augmentation if mean validation F1 improves by at least 0.01, "
        "clean validation F1 is maintained, and worst condition recall is maintained.")


def transform(image, condition):
    if condition == "clean":
        return image
    if condition in ("brightness0.5", "brightness1.5"):
        factor = 0.5 if condition == "brightness0.5" else 1.5
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if condition == "blur9_sigma2.0":
        return cv2.GaussianBlur(image, (9, 9), 2.0)
    if condition == "jpeg_quality15":
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 15])
        if not ok:
            raise ValueError("OpenCV could not encode the JPEG test view.")
        return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if condition == "resize56":
        return cv2.resize(cv2.resize(image, (56, 56)), (SIZE, SIZE))
    if condition == "rotate10_reflect":
        matrix = cv2.getRotationMatrix2D(((SIZE - 1) / 2, (SIZE - 1) / 2), 10, 1)
        return cv2.warpAffine(image, matrix, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT_101)
    raise ValueError(f"Unknown condition: {condition}")


def read_split(manifest, data):
    groups = {part: [] for part in ("train", "val", "test")}
    seen = set()
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            path = Path(row["path"])
            if (len(path.parts) != 2 or path.parts[0] != row["label"]
                    or path.parts[0] not in CLASSES or path in seen):
                raise ValueError(f"Invalid or duplicate manifest path: {path}")
            if not (data / path).is_file():
                raise FileNotFoundError(data / path)
            groups[row["split"]].append((path, CLASSES.index(row["label"])))
            seen.add(path)
    if tuple(map(len, groups.values())) != (189, 27, 55):
        raise ValueError("Expected the fixed Study One split of 189 / 27 / 55 images.")
    return groups


def features(extractor, data, rows, conditions):
    result = {c: [] for c in conditions}
    for start in range(0, len(rows), 24):
        images = [read_image(data / path) for path, _ in rows[start:start + 24]]
        for condition in conditions:
            batch = np.stack([transform(im, condition) for im in images])
            result[condition].append(extractor.predict_on_batch(batch))
    return {c: np.concatenate(batches) for c, batches in result.items()}


def fit_head(x, y, orders, views, seed, arm):
    keras.utils.set_random_seed(seed)
    head = build_head(x.shape[-1], "adam", LR)
    initial = head.get_weights()
    history = []
    for epoch, order in enumerate(orders):
        head.reset_metrics()
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH]
            chosen = views[epoch, start:start + BATCH] if arm == "augmented" else 0
            logs = head.train_on_batch(x[chosen, ids], y[ids], return_dict=True)
        history.append({"arm": arm, "seed": seed, "epoch": epoch + 1,
                        **{key: float(value) for key, value in logs.items()}})
    updates = int(keras.ops.convert_to_numpy(head.optimizer.iterations))
    if updates != EPOCHS * ceil(len(y) / BATCH):
        raise RuntimeError(f"Unexpected optimizer updates for {arm}, seed {seed}: {updates}")
    return head, initial, history, updates


def evaluate(heads, encoded, rows, part):
    labels = np.array([label for _, label in rows])
    summaries, predictions = [], []
    for (arm, seed), head in heads.items():
        for condition, vectors in encoded.items():
            p = head.predict_on_batch(vectors)
            tags = {"split": part, "arm": arm, "seed": seed, "condition": condition}
            summaries.append({**tags, **score(labels, p)})
            predictions.extend(
                {**tags, "path": path.as_posix(), "actual": CLASSES[label],
                 "predicted": CLASSES[int(prob.argmax())], "p_defect": float(prob[0])}
                for (path, label), prob in zip(rows, p, strict=True))
    return summaries, predictions


def select(rows):
    by_arm = {}
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = {
            "mean_f1": mean(r["f1_defect"] for r in subset),
            "clean_f1": mean(r["f1_defect"] for r in subset if r["condition"] == "clean"),
            "worst_recall": min(mean(r["recall_defect"] for r in subset if r["condition"] == c)
                                for c in CONDITIONS),
        }
    clean, aug = (by_arm[arm] for arm in ARMS)
    adopt = (aug["mean_f1"] >= clean["mean_f1"] + 0.01
             and aug["clean_f1"] >= clean["clean_f1"]
             and aug["worst_recall"] >= clean["worst_recall"])
    return {"selected_arm": "augmented" if adopt else "clean", "rule": RULE, "validation": by_arm}


def summarize(metrics):
    for part in ("val", "test"):
        for arm in ARMS:
            subset = [r for r in metrics if r["split"] == part and r["arm"] == arm]
            for condition in (*CONDITIONS, "seven_condition_mean"):
                # Average conditions within each seed before estimating the across-seed SD.
                by_seed = [[r for r in subset if r["seed"] == seed
                            and (r["condition"] == condition or condition == "seven_condition_mean")]
                           for seed in SEEDS]
                accuracies = [mean(r["accuracy"] for r in rows) for rows in by_seed]
                yield {"split": part, "arm": arm, "condition": condition,
                       "accuracy_mean": mean(accuracies), "accuracy_sd": stdev(accuracies),
                       "recall_mean": mean(r["recall_defect"] for rows in by_seed for r in rows),
                       "f1_mean": mean(r["f1_defect"] for rows in by_seed for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("AMdataset/dataset"))
    parser.add_argument("--split", type=Path, default=Path("results/split.csv"))
    parser.add_argument("--weights", default="imagenet")
    parser.add_argument("--out", type=Path, default=Path("robust-runs"))
    args = parser.parse_args()
    if keras.backend.backend() == "torch":
        import torch
        torch.set_num_threads(4)
    cv2.setNumThreads(1)
    groups = read_split(args.split, args.data)
    args.out.mkdir(parents=True, exist_ok=False)
    keras.utils.set_random_seed(SEEDS[0])
    extractor = build_extractor(args.weights)
    save_json(args.out / "protocol.json", {
        "source": "https://github.com/AHJ8273528/IDAT7215-EfficientNetB0/tree/c48cdc9ae66d9281d839a1ac55468215927a0c15",
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "weights_sha256": (hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
                           if Path(args.weights).is_file() else None),
        "keras": keras.__version__, "backend": keras.backend.backend(), "opencv": cv2.__version__,
        "seeds": SEEDS, "epochs": EPOCHS, "batch_size": BATCH, "optimizer": "Adam",
        "learning_rate": LR, "training_views": VIEWS, "evaluation_conditions": CONDITIONS,
        "images_per_epoch": len(groups["train"]), "selection": RULE,
        "model_output_classes": CLASSES,
    })
    print("Extracting training features.", flush=True)
    by_view = features(extractor, args.data, groups["train"], VIEWS)
    x = np.stack([by_view[view] for view in VIEWS])
    y = keras.utils.to_categorical([label for _, label in groups["train"]], 2)
    heads, histories, fits = {}, [], []
    for seed in SEEDS:
        order_rng = np.random.default_rng(seed)
        orders = np.stack([order_rng.permutation(len(y)) for _ in range(EPOCHS)])
        views = np.random.default_rng(seed + 10000).integers(
            0, len(VIEWS), size=orders.shape, dtype=np.int32)
        for arm in ARMS:
            head, initial, history, updates = fit_head(x, y, orders, views, seed, arm)
            if arm == "clean":
                reference = initial
            else:
                for same, current in zip(reference, initial, strict=True):
                    np.testing.assert_array_equal(same, current)
            heads[arm, seed] = head
            histories.extend(history)
            fits.append({"arm": arm, "seed": seed, "updates": updates,
                         "initial_weights_sha256": hashlib.sha256(
                             b"".join(a.tobytes() for a in initial)).hexdigest(),
                         "order_sha256": hashlib.sha256(orders.tobytes()).hexdigest()})
            print(f"FIT {arm} seed {seed}: {updates} updates", flush=True)
    save_csv(args.out / "history.csv", histories)
    save_csv(args.out / "fits.csv", fits)

    print("Evaluating validation conditions.", flush=True)
    val_x = features(extractor, args.data, groups["val"], CONDITIONS)
    val_metrics, val_preds = evaluate(heads, val_x, groups["val"], "val")
    selection = select(val_metrics)
    save_json(args.out / "selection.json", selection)
    head = heads[selection["selected_arm"], SEEDS[0]]
    head.save_weights(args.out / "selected_head.weights.h5")
    print(f"SELECTION {selection['selected_arm']}. Extracting test features.", flush=True)
    test_x = features(extractor, args.data, groups["test"], CONDITIONS)
    test_metrics, test_preds = evaluate(heads, test_x, groups["test"], "test")
    save_csv(args.out / "metrics.csv", val_metrics + test_metrics)
    save_csv(args.out / "predictions.csv", val_preds + test_preds)
    save_csv(args.out / "summary.csv", list(summarize(val_metrics + test_metrics)))

    model = keras.Model(extractor.input, head(extractor.output), name="mobilenetv2_robustness")
    save_path = args.out / "selected_seed42.keras"
    model.save(save_path)
    loaded = keras.models.load_model(save_path, compile=False)
    images = np.stack([read_image(args.data / path) for path, _ in groups["test"][:4]])
    direct = model.predict_on_batch(images)
    cached = head.predict_on_batch(test_x["clean"][:4])
    reloaded = loaded.predict_on_batch(images)
    np.testing.assert_allclose(direct, cached, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(reloaded, direct, atol=1e-6, rtol=1e-5)
    save_json(args.out / "model_checks.json", {
        "seed": SEEDS[0], "arm": selection["selected_arm"], "parameters": model.count_params(),
        "head_parameters": head.count_params(), "checked_original_images": len(images),
        "cached_direct_max_error": float(np.max(np.abs(cached - direct))),
        "reload_max_error": float(np.max(np.abs(reloaded - direct))),
        "inference": "Single RGB view, no test-time augmentation and no rejection",
    })
    print(f"DONE {save_path}", flush=True)


if __name__ == "__main__":
    main()
