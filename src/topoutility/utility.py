import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


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
    X = train["signature_b"] - train["signature_a"]
    singular = np.linalg.svd(X, compute_uv=False)
    rank = int(np.linalg.matrix_rank(X))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    if rank < 5 or condition is None or condition > 1e6:
        raise ValueError(
            "Edit design is rank-deficient or ill-conditioned; group confounded edit types before fitting"
        )
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
    rng = np.random.default_rng(args.seed)
    boot = []
    for replicate in range(args.bootstrap):
        indices = rng.integers(len(X), size=len(X))
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
