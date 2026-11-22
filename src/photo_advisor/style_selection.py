"""Build the bounded, unranked set shown by the mobile API."""
from itertools import combinations

import numpy as np
from PIL import Image, ImageOps
from skimage.color import deltaE_ciede2000, rgb2lab

from .processor import process_image
from .types import Recipe
from .validator import validate_and_clamp_recipe


PRESETS = (
    ("style_warm", "Ấm hơn", {"temperature": .8}),
    ("style_cool", "Mát hơn", {"temperature": -.8}),
    ("style_light", "Sáng hơn", {"exposure": .55}),
    ("style_muted", "Màu dịu", {"saturation": .8, "contrast": .88}),
    ("style_contrast", "Tương phản cao hơn", {"contrast": 1.2}),
)


def lab_thumbnail(path):
    with Image.open(path) as image:
        tile = ImageOps.pad(image.convert("RGB"), (256, 256),
                            method=Image.Resampling.LANCZOS, color=(128, 128, 128))
    return rgb2lab(np.asarray(tile, dtype=np.float32) / 255)


def diverse_indices(tiles, limit):
    if len(tiles) < 2:
        return list(range(len(tiles)))
    distances = np.zeros((len(tiles), len(tiles)))
    pairs = list(combinations(range(len(tiles)), 2))
    for i, j in pairs:
        distances[i, j] = distances[j, i] = deltaE_ciede2000(tiles[i], tiles[j]).mean()
    first, second = max(pairs, key=lambda pair: distances[pair])
    if distances[first, second] < 3:
        return [int(distances.mean(axis=1).argmax())]
    chosen = [first, second]
    while len(chosen) < min(limit, len(tiles)):
        nearest = distances[:, chosen].min(axis=1)
        index = int(nearest.argmax())
        if nearest[index] < 3:
            break
        chosen.append(index)
    return sorted(chosen)


def build_styles(payload, goal, output, records, limit=4):
    if limit not in (4, 5):
        raise ValueError("Style count must be 4 or 5")
    original = next(c for c in records if c["source_type"] == "noop")
    with Image.open(output / original["preview_filename"]) as image:
        width, height = image.size
    pool = [dict(c, style_id="style_rule", style_label=c["source_name"])
            for c in records if c["source_type"] == "rule" and c["status"] == "success"]
    for key, label, changes in PRESETS:
        if goal == "silhouette" and key == "style_light":
            continue
        params = dict(exposure=0, contrast=1, saturation=1)
        params.update(changes)
        recipe = Recipe(pipeline_version="pipeline_v1", **params).to_dict()
        validation = validate_and_clamp_recipe(recipe, width, height, goal=goal)
        if not validation.is_valid:
            continue
        result = process_image(payload, validation.effective_recipe)
        if result.status != "success":
            continue
        filename = key + ".png"
        (output / filename).write_bytes(result.output_image_bytes)
        pool.append(dict(style_id=key, style_label=label, source_type="rule", source_name=label,
                         requested_recipe=recipe, effective_recipe=result.effective_recipe.to_dict(),
                         status="success", preview_filename=filename, export_eligible=result.export_eligible,
                         output_sha256=result.output_sha256))
    seen = {original.get("output_sha256")}
    unique = []
    for item in sorted(pool, key=lambda c: c["style_id"]):
        if item["output_sha256"] not in seen:
            seen.add(item["output_sha256"])
            unique.append(item)
    tiles = [lab_thumbnail(output / c["preview_filename"]) for c in unique]
    selected = diverse_indices(tiles, 2 if goal in ("memorial", "silhouette") else limit)
    return [original] + [unique[i] for i in selected]
