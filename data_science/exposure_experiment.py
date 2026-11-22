"""Reproducible recovery of simulated exposure error on grouped synthetic scenes."""
import argparse
import io
import json
import platform

import numpy as np
from PIL import Image
from photo_advisor.processor import process_image
from photo_advisor.types import Recipe


def scene_samples(rng, count):
    rows = []
    for scene_id in range(count):
        corners = rng.uniform(.12, .58, (4, 4, 3))
        image = Image.fromarray((corners * 255).astype("uint8")).resize((64, 64), Image.Resampling.BILINEAR)
        clean = np.asarray(image, dtype=float) / 255
        for shift in (-.8, -.4, .4, .8):
            linear = ((clean + .055) / 1.055) ** 2.4
            degraded = 1.055 * (linear * 2**shift) ** (1 / 2.4) - .055
            pixels = np.uint8(np.rint(255 * np.clip(degraded, 0, 1)))
            luma = ((pixels / 255 + .055) / 1.055) ** 2.4 @ [.2126, .7152, .0722]
            features = np.log2(np.maximum(np.quantile(luma, [.1, .5, .9]), 1e-6))
            rows.append((scene_id, features, -shift, pixels, clean))
    return rows


def fit(features, targets, penalty):
    center = features.mean(axis=0)
    design = features - center
    weights = np.linalg.solve(design.T @ design + penalty * np.eye(design.shape[1]),
                              design.T @ (targets - targets.mean()))
    return lambda x: np.clip((x - center) @ weights + targets.mean(), -1.5, 1.5)


def experiment(seed):
    rng = np.random.default_rng(seed)
    rows = scene_samples(rng, 48)
    groups = rng.permutation(48)
    splits = dict(train=groups[:30], validation=groups[30:39], test=groups[39:])
    features = np.array([row[1] for row in rows])
    targets = np.array([row[2] for row in rows])
    masks = {name: np.isin([row[0] for row in rows], ids) for name, ids in splits.items()}
    errors = {}
    for penalty in (.01, .1, 1., 10.):
        model = fit(features[masks["train"]], targets[masks["train"]], penalty)
        errors[penalty] = float(np.mean(abs(model(features[masks["validation"]]) - targets[masks["validation"]])))
    penalty = min(errors, key=errors.get)
    model = fit(features[masks["train"]], targets[masks["train"]], penalty)
    edited, original, fallbacks = [], [], 0
    for index in np.flatnonzero(masks["test"]):
        _, x, _, pixels, clean = rows[index]
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        recipe = Recipe("pipeline_v1", float(model(x)), 1., 1.)
        result = process_image(buffer.getvalue(), recipe, min_dimension_threshold=1)
        fallbacks += result.status != "success"
        restored = np.asarray(Image.open(io.BytesIO(result.output_image_bytes))) / 255
        edited.append(np.mean(abs(clean - restored)))
        original.append(np.mean(abs(clean - pixels / 255)))
    return dict(dataset="synthetic-exposure-v1", seed=seed, numpy=np.__version__, python=platform.python_version(),
                scene_ids={k: v.tolist() for k, v in splits.items()}, validation_mae_ev=errors,
                penalty=penalty, test_images=len(original), test_noop_mae=float(np.mean(original)),
                test_edited_mae=float(np.mean(edited)), test_fallbacks=fallbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    print(json.dumps(experiment(parser.parse_args().seed), indent=2))
