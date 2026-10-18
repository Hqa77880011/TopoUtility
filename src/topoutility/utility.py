import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import helmert
from scipy.stats import spearmanr


def identification(difference, tasks, weights=None):
    """Check task-by-edit directions after removing the softmax gauge."""
    contrast = helmert(6, full=False).T
    if weights is not None:
        difference = weights * (difference - np.sum(weights * difference, axis=1, keepdims=True))
    q = np.column_stack([tasks, np.ones(len(tasks))])
    design = np.einsum("ni,nj->nij", difference @ contrast, q).reshape(len(q), -1)
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    if rank != design.shape[1] or condition is None or condition > 1e6:
        raise ValueError("Task-by-edit design is confounded; vary tasks and independent edit directions")
    return rank, condition


def fit(signatures_a, signatures_b, tasks, labels, epochs, seed):
    torch.manual_seed(seed)
    q = torch.tensor(tasks, dtype=torch.float32)
    difference = torch.tensor(signatures_b - signatures_a, dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.float32)
    A = torch.zeros((6, q.shape[1]), requires_grad=True)
    b = torch.zeros(6, requires_grad=True)
    optimizer = torch.optim.Adam([A, b], lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad()
        weights = torch.softmax(q @ A.T + b, -1)
        logits = (weights * difference).sum(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()
    return A.detach().numpy(), b.detach().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train = dict(np.load(args.train, allow_pickle=False))
    validation = dict(np.load(args.validation, allow_pickle=False))
    train_groups = set(map(str, train["group"].tolist()))
    validation_groups = set(map(str, validation["group"].tolist()))
    if train_groups & validation_groups:
        raise ValueError("Utility fitting and validation source objects overlap")
    X = train["signature_b"] - train["signature_a"]
    # Normalized utilities have five independent directions: sum(weights)=1.
    contrast_design = X @ helmert(6, full=False).T
    singular = np.linalg.svd(contrast_design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(contrast_design))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    if rank < 5 or condition is None or condition > 1e6:
        raise ValueError(
            "Edit design is rank-deficient or ill-conditioned; group confounded edit types before fitting"
        )
    task_rank, task_condition = identification(X, train["task"])
    A, b = fit(
        train["signature_a"],
        train["signature_b"],
        train["task"],
        train["preferred_a"],
        args.epochs,
        args.seed,
    )
    logits = validation["task"] @ A.T + b
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    difference = np.sum(weights * (validation["signature_b"] - validation["signature_a"]), axis=-1)
    agreement = float(np.mean((difference >= 0) == validation["preferred_a"]))
    train_logits = train["task"] @ A.T + b
    train_weights = np.exp(train_logits - train_logits.max(-1, keepdims=True))
    train_weights /= train_weights.sum(-1, keepdims=True)
    local_rank, local_condition = identification(X, train["task"], train_weights)
    rng = np.random.default_rng(args.seed)
    boot = []
    for replicate in range(args.bootstrap):
        groups = np.unique(train["group"])
        sampled = rng.choice(groups, len(groups), replace=True)
        indices = np.concatenate([np.flatnonzero(train["group"] == group) for group in sampled])
        fitted_A, fitted_b = fit(
            train["signature_a"][indices],
            train["signature_b"][indices],
            train["task"][indices],
            train["preferred_a"][indices],
            args.epochs,
            args.seed + replicate + 1,
        )
        z = np.mean(train["task"], 0) @ fitted_A.T + fitted_b
        weight = np.exp(z - z.max())
        boot.append(weight / weight.sum())
    payload = {
        "accepted": agreement > 0.5,
        "A": A.tolist(),
        "b": b.tolist(),
        "rank": rank,
        "condition_number": condition,
        "task_design_rank": task_rank,
        "task_design_condition": task_condition,
        "local_utility_rank": local_rank,
        "local_utility_condition": local_condition,
        "utility_groups": sorted(train_groups | validation_groups),
        "validation_pairwise_agreement": agreement,
        "weight_intervals95": np.quantile(np.asarray(boot), [0.025, 0.975], axis=0).tolist(),
    }
    if "severity_difference" in validation:
        payload["severity_spearman"] = float(
            spearmanr(difference, validation["severity_difference"]).statistic
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
