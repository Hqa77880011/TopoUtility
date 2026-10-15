import numpy as np

import torch

from scipy import ndimage

from scipy.optimize import linear_sum_assignment


from skimage.morphology import skeletonize

from torch import nn

from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(4, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(4, cout),
        nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, channels=1, width=32):
        super().__init__()
        self.encoder1 = block(channels, width)
        self.encoder2 = block(width, width * 2)
        self.middle = block(width * 2, width * 4)
        self.decoder2 = block(width * 6, width * 2)
        self.decoder1 = block(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)

    def features(self, x):
        a = self.encoder1(x.float())
        b = self.encoder2(F.avg_pool2d(a, 2))
        c = self.middle(F.avg_pool2d(b, 2))
        d = self.decoder2(
            torch.cat(
                [F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1
            )
        )
        return self.decoder1(
            torch.cat(
                [F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1
            )
        )

    def forward(self, x):
        return self.head(self.features(x))


def soft_erode(x):
    vertical = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.minimum(vertical, horizontal)


def soft_skeleton(x, iterations=10):
    opened = F.max_pool2d(soft_erode(x), 3, 1, 1)
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - F.max_pool2d(soft_erode(x), 3, 1, 1))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def dice_loss(probability, target):
    dims = tuple(range(1, probability.ndim))
    return (
        1
        - (
            (2 * (probability * target).sum(dims) + 1e-6)
            / (probability.sum(dims) + target.sum(dims) + 1e-6)
        ).mean()
    )


def cldice_loss(probability, target):
    sp, st = soft_skeleton(probability), soft_skeleton(target)
    dims = tuple(range(1, probability.ndim))
    precision = ((sp * target).sum(dims) + 1e-6) / (sp.sum(dims) + 1e-6)
    recall = ((st * probability).sum(dims) + 1e-6) / (st.sum(dims) + 1e-6)
    return (1 - 2 * precision * recall / (precision + recall + 1e-6)).mean()


def segmentation_metrics(probability, target, instances=None):
    prediction = np.asarray(probability) >= 0.5
    truth = np.asarray(target) > 0.5
    rows = []
    for i, (p, y) in enumerate(zip(prediction, truth)):
        p, y = p.squeeze(), y.squeeze()
        intersection = (p & y).sum()
        union = (p | y).sum()
        dice = (2 * intersection + 1e-8) / (p.sum() + y.sum() + 1e-8)
        foreground_iou = (intersection + 1e-8) / (union + 1e-8)
        background_iou = ((~p & ~y).sum() + 1e-8) / ((~p | ~y).sum() + 1e-8)
        pb = p & ~ndimage.binary_erosion(p, iterations=3, border_value=0)
        yb = y & ~ndimage.binary_erosion(y, iterations=3, border_value=0)
        boundary = ((pb & yb).sum() + 1e-8) / ((pb | yb).sum() + 1e-8)
        ps, ys = skeletonize(p), skeletonize(y)
        precision = (ps & y).sum() / max(ps.sum(), 1)
        recall = (ys & p).sum() / max(ys.sum(), 1)
        cldice = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else float(not p.any() and not y.any())
        )
        psurf = p & ~ndimage.binary_erosion(p)
        ysurf = y & ~ndimage.binary_erosion(y)
        if psurf.any() and ysurf.any():
            distances = np.r_[
                ndimage.distance_transform_edt(~ysurf)[psurf],
                ndimage.distance_transform_edt(~psurf)[ysurf],
            ]
            hd95 = float(np.quantile(distances, 0.95))
        else:
            hd95 = float(np.hypot(*p.shape)) if p.any() != y.any() else 0.0
        gt_instances = (
            np.asarray(instances[i]).squeeze() if instances is not None else ndimage.label(y)[0]
        )
        pred_instances, _ = ndimage.label(p)
        gids = [g for g in np.unique(gt_instances) if g]
        pids = [g for g in np.unique(pred_instances) if g]
        iou = np.zeros((len(gids), len(pids)))
        small = []
        fragmented = []
        for a, gid in enumerate(gids):
            mask = gt_instances == gid
            if mask.sum() < 0.0025 * mask.size:
                small.append(a)
            coords = np.argwhere(mask)
            if len(coords) >= 16:
                eigen = np.linalg.eigvalsh(np.cov(coords.T))
                elongation = 1 - (eigen[0] + 1e-7) / (eigen[-1] + 1e-7)
                if elongation >= 0.75:
                    selected, count = ndimage.label(p & ndimage.binary_dilation(mask, iterations=3))
                    valid = sum((selected == j).sum() >= 4 for j in range(1, count + 1))
                    fragmented.append(valid > 1)
            for b, pid in enumerate(pids):
                other = pred_instances == pid
                iou[a, b] = (mask & other).sum() / max((mask | other).sum(), 1)
        matched = set()
        if iou.size:
            left, right = linear_sum_assignment(-iou)
            matched = {a for a, b in zip(left, right) if iou[a, b] >= 0.5}
        rows.append(
            {
                "dice": float(dice),
                "miou": float((foreground_iou + background_iou) / 2),
                "foreground_iou": float(foreground_iou),
                "boundary_iou": float(boundary),
                "hd95_pixels": hd95,
                "cldice": float(cldice),
                "small_hits": sum(a in matched for a in small),
                "small_total": len(small),
                "fragmented": sum(fragmented),
                "elongated_total": len(fragmented),
            }
        )
    result = {
        k: float(np.mean([row[k] for row in rows]))
        for k in ["dice", "miou", "foreground_iou", "boundary_iou", "hd95_pixels", "cldice"]
    }
    small_total = sum(row["small_total"] for row in rows)
    elongated = sum(row["elongated_total"] for row in rows)
    result["small_object_recall"] = (
        sum(row["small_hits"] for row in rows) / small_total if small_total else None
    )
    result["fragmentation_rate"] = (
        sum(row["fragmented"] for row in rows) / elongated if elongated else None
    )
    return result
