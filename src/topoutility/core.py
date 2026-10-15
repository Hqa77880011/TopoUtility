import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from .vision import UNet, cldice_loss, dice_loss, segmentation_metrics, soft_skeleton


METHODS = ["topoutility", "dice", "cldice", "uniform_utility"]


def signatures(probability, target):
    sp, st = soft_skeleton(probability), soft_skeleton(target)
    kernel = torch.ones((1, 1, 3, 3), device=probability.device)
    kernel[..., 1, 1] = 0
    degree_p, degree_t = F.conv2d(sp, kernel, padding=1), F.conv2d(st, kernel, padding=1)
    endpoint_p = sp * torch.sigmoid(10 * (1.5 - degree_p))
    endpoint_t = st * torch.sigmoid(10 * (1.5 - degree_t))
    branch_p = sp * torch.sigmoid(5 * (degree_p - 2.5))
    branch_t = st * torch.sigmoid(5 * (degree_t - 2.5))
    dims = (-3, -2, -1)
    normalizer = st.sum(dims).clamp_min(1)
    gap = (st * (1 - probability)).sum(dims) / normalizer
    bridge = (sp * (1 - F.max_pool2d(target, 3, 1, 1))).sum(dims) / normalizer
    spur = (endpoint_p * (1 - F.max_pool2d(endpoint_t, 5, 1, 2))).sum(dims) / normalizer

    def euler(x):
        pixels = x.sum(dims)
        edge = (x[..., 1:, :] * x[..., :-1, :]).sum(dims) + (x[..., :, 1:] * x[..., :, :-1]).sum(
            dims
        )
        face = (x[..., 1:, 1:] * x[..., 1:, :-1] * x[..., :-1, 1:] * x[..., :-1, :-1]).sum(dims)
        return pixels - edge + face

    loop = F.relu(euler(target) - euler(probability)) / normalizer
    ending = (endpoint_t * (1 - F.max_pool2d(endpoint_p, 5, 1, 2))).sum(dims) / normalizer
    branch = ((branch_p - branch_t).abs()).sum(dims) / normalizer
    return torch.stack([gap, bridge, spur, loop, ending, branch], -1)


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        self.segmenter = UNet(cfg["channels"], cfg["width"])
        if method == "topoutility":
            payload = json.loads(Path(cfg["utility_path"]).read_text(encoding="utf-8"))
            if not payload["accepted"]:
                raise ValueError("Utility did not pass identification/validation")
            self.register_buffer("A", torch.tensor(payload["A"], dtype=torch.float32))
            self.register_buffer("b", torch.tensor(payload["b"], dtype=torch.float32))
        else:
            self.register_buffer("A", torch.zeros(6, cfg["task_dimensions"]))
            self.register_buffer("b", torch.zeros(6))

    def forward(self, x):
        return self.segmenter(x)


def loss(model, batch, cfg):
    y = batch["y"].float()
    if y.ndim == 3:
        y = y[:, None]
    logits = model(batch["x"])
    probability = logits.sigmoid()
    base = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(probability, y)
    if model.method == "dice":
        return base
    if model.method == "cldice":
        return base + cfg["utility_weight"] * cldice_loss(probability, y)
    weights = torch.softmax(batch["task"].float() @ model.A.T + model.b, -1)
    return base + cfg["utility_weight"] * (weights * signatures(probability, y)).sum(-1).mean()


def predict(model, batch, cfg, calibration=None):
    probability = model(batch["x"]).sigmoid()
    y = batch["y"].float()
    if y.ndim == 3:
        y = y[:, None]
    weights = torch.softmax(batch["task"].float() @ model.A.T + model.b, -1)
    return {
        "probability": probability,
        "signature": signatures(probability, y),
        "utility": (weights * signatures(probability, y)).sum(-1, keepdim=True),
    }


def metrics(prediction, data, cfg):
    result = segmentation_metrics(prediction["probability"], data["y"])
    if "failure" in data and len(np.unique(data["failure"])) == 2:
        result["functional_failure_auroc"] = float(
            roc_auc_score(data["failure"], prediction["utility"].ravel())
        )
    if "path_pairs" in data:
        failures = []
        for p, truth, pairs in zip(prediction["probability"], data["y"], data["path_pairs"]):
            plabel = ndimage.label(p.squeeze() >= 0.5)[0]
            tlabel = ndimage.label(truth.squeeze() > 0.5)[0]
            for r1, c1, r2, c2 in pairs.astype(int):
                if tlabel[r1, c1] and tlabel[r1, c1] == tlabel[r2, c2]:
                    failures.append(not plabel[r1, c1] or plabel[r1, c1] != plabel[r2, c2])
        result["downstream_path_failure"] = float(np.mean(failures)) if failures else None
    return result


def inference(model, batch, cfg):
    return model(batch["x"])
