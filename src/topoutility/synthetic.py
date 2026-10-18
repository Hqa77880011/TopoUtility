"""Procedural branching structures and independent downstream preference tasks."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def line(mask, start, end):
    points = np.rint(np.linspace(start, end, int(np.linalg.norm(np.subtract(end, start))) * 3 + 2)).astype(int)
    points = np.clip(points, 0, np.array(mask.shape) - 1)
    mask[points[:, 0], points[:, 1]] = True


def geometry(rng, size=64):
    root = np.array([size - 6, size // 2])
    ends = np.array([[8, 10], [9, size - 11], [size // 2, 7], [size // 2 + 2, size - 8]])
    ends += rng.integers(-2, 3, ends.shape)
    mask = np.zeros((size, size), bool)
    line(mask, root, [7, size // 2])
    for endpoint in ends:
        junction = [min(size - 8, endpoint[0] + rng.integers(8, 14)), size // 2]
        line(mask, junction, endpoint)
    mask = ndimage.binary_dilation(mask, iterations=1)
    return mask, root, ends


def edit(mask, root, ends, kind, rng):
    result = mask.copy()
    pixels = np.argwhere(mask)
    if kind == 0:
        y, x = pixels[int(rng.integers(len(pixels)))]
        radius = int(rng.integers(2, 5))
        result[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = False
    elif kind == 1:
        indices = rng.choice(len(ends), 2, replace=False)
        line(result, ends[indices[0]], ends[indices[1]])
    elif kind == 2:
        start = pixels[int(rng.integers(len(pixels)))]
        line(result, start, start + rng.integers(-14, 15, 2))
    elif kind == 3:
        y, x = len(mask) // 2, len(mask) // 2
        width = int(rng.integers(6, 15))
        corners = [[y, x], [y - 8, x + width], [y + 7, x + width], [y, x]]
        for a, b in zip(corners[:-1], corners[1:]):
            line(result, a, b)
    elif kind == 4:
        y, x = ends[int(rng.integers(len(ends)))]
        result[y - 4:y + 5, x - 4:x + 5] = False
    elif kind == 5:
        endpoint = ends[int(rng.integers(len(ends)))]
        y, x = np.rint((endpoint + [endpoint[0] + 10, len(mask) // 2]) / 2).astype(int)
        result[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = False
    else:
        raise ValueError("Unknown geometric edit")
    return result


def distances(mask, root):
    result = np.full(mask.shape, -1, int)
    if not mask[tuple(root)]:
        return result
    queue = deque([tuple(root)])
    result[tuple(root)] = 0
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < len(mask) and 0 <= xx < mask.shape[1] and mask[yy, xx] and result[yy, xx] < 0:
                result[yy, xx] = result[y, x] + 1
                queue.append((yy, xx))
    return result


def downstream_cost(candidate, truth, root, ends, task):
    """Exact reachability/path distortion, independent of differentiable signatures."""
    reference = distances(truth, root)[ends[:, 0], ends[:, 1]]
    observed = distances(candidate, root)[ends[:, 0], ends[:, 1]]
    valid = reference >= 0
    missed = observed[valid] < 0
    distortion = np.ones(valid.sum())
    available = ~missed
    distortion[available] = np.abs(np.log((observed[valid][available] + 1) /
                                         (reference[valid][available] + 1)))
    extra_area = max(0, candidate.sum() - truth.sum()) / max(1, truth.sum())
    fragments = max(0, ndimage.label(candidate)[1] - ndimage.label(truth)[1])
    return float(np.exp(task[0]) * missed.mean() + np.exp(task[1]) * distortion.mean() +
                 0.2 * extra_area + 0.05 * fragments)


def generate(output, objects, utility_objects, size, seed):
    segmentation = {key: [] for key in ("x", "y", "task", "group", "split", "path_pairs")}
    pairs = {key: [] for key in ("target", "a", "b", "task", "preferred_a", "group", "split")}
    for group in range(utility_objects):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, group]))
        truth, root, ends = geometry(rng, size)
        for repeat in range(10):
            task = rng.uniform(-1, 1, 2)
            a, b = truth.copy(), truth.copy()
            for candidate in (a, b):
                for _ in range(int(rng.integers(1, 4))):
                    candidate[:] = edit(candidate, root, ends, int(rng.integers(6)), rng)
            ca, cb = (downstream_cost(candidate, truth, root, ends, task) for candidate in (a, b))
            if abs(ca - cb) < 1e-9:
                continue
            for key, value in dict(target=truth, a=a, b=b, task=task, preferred_a=int(ca < cb),
                                   group=group, split=int(group >= utility_objects * 2 / 3)).items():
                pairs[key].append(value)
    for group in range(objects):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, group]))
        truth, root, ends = geometry(rng, size)
        texture = ndimage.gaussian_filter(rng.normal(size=truth.shape), 2)
        image = 0.15 + rng.uniform(0.45, 0.7) * ndimage.gaussian_filter(truth.astype(float), 0.65)
        image += 0.06 * texture + rng.normal(0, 0.07, truth.shape)
        values = dict(x=np.clip(image, 0, 1).astype(np.float32)[None], y=truth.astype(np.float32),
                      task=rng.uniform(-1, 1, 2).astype(np.float32), group=10000 + group,
                      split=0 if group < objects * 0.6 else 1 if group < objects * 0.8 else 2,
                      path_pairs=np.column_stack([np.tile(root, (len(ends), 1)), ends]))
        for key, value in values.items():
            segmentation[key].append(value)
    np.savez_compressed(output / "segmentation.npz", **{k: np.asarray(v) for k, v in segmentation.items()})
    np.savez_compressed(output / "utility_masks.npz", **{k: np.asarray(v) for k, v in pairs.items()})


def extract(source, output):
    import torch
    from .core import signatures

    arrays = dict(np.load(source, allow_pickle=False))
    with torch.no_grad():
        target = torch.tensor(arrays["target"].astype(np.float32))[:, None]
        a = signatures(torch.tensor(arrays["a"].astype(np.float32))[:, None], target).numpy()
        b = signatures(torch.tensor(arrays["b"].astype(np.float32))[:, None], target).numpy()
    for split, name in ((0, "utility_train"), (1, "utility_validation")):
        selected = arrays["split"] == split
        if not selected.any():
            raise ValueError("Each independent utility partition needs nontied preference pairs")
        np.savez_compressed(output / (name + ".npz"), signature_a=a[selected], signature_b=b[selected],
                            **{k: arrays[k][selected] for k in ("task", "preferred_a", "group")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extract-signatures", help="Preprocess generated utility_masks.npz with the exact shared extractor")
    parser.add_argument("--objects", type=int, default=60)
    parser.add_argument("--utility-objects", type=int, default=36)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(str(output))
    output.mkdir(parents=True, exist_ok=True)
    if args.extract_signatures:
        extract(args.extract_signatures, output)
    else:
        if args.objects < 10 or args.utility_objects < 12:
            parser.error("Use at least ten segmentation and twelve independent utility objects")
        generate(output, args.objects, args.utility_objects, 64, args.seed)
        (output / "source.json").write_text(json.dumps(dict(
            source="procedural branching masks and noisy rendered images",
            preference_truth="exact root-terminal reachability, path distortion and occupied-area cost",
            human_ratings=False, seed=args.seed, segmentation_objects=args.objects,
            utility_objects=args.utility_objects,
        ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
