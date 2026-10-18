# TopoUtility

Implementation of **Learning task-conditioned utilities for topological errors in thin-structure segmentation**.

Fits positive normalized task-dependent edit costs from independent preference pairs, checks identification and held-out agreement, then freezes them in the segmentation objective.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/TopoUtility.git
cd TopoUtility
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Use procedural branching structures, [DRIVE](https://drive.grand-challenge.org/) or registered crack data. Utility-training, segmentation-training and final-test parent objects must be disjoint.

Segmentation arrays are x, binary y, task descriptors and group. Optional path_pairs and independent failure labels enable functional evaluation. Utility files contain signature_a, signature_b, task, preferred_a, source-object group and optional severity_difference. Utility train/validation objects must be disjoint; their groups are stored in the artifact and excluded from segmentation fitting and testing. Bootstrap resampling uses whole source groups.

```bash
python -m topoutility prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Fit the utility first

```bash
python -m topoutility.utility --train data/utility_train.npz --validation data/utility_validation.npz --output runs/utility.json
```

The command checks rank and conditioning in the five-dimensional zero-sum edit tangent, the task-by-edit augmented design (5 * (task_dimensions + 1) directions), and the fitted local softmax utility Jacobian. It refuses a confounded design. Task coordinates must vary independently of the intercept; remove a redundant one-hot coordinate when necessary. An accepted artifact is required by the full segmentation method; uniform and Dice controls do not load it.



## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m topoutility train --config configs/default.json --data data/prepared --output runs/main
python -m topoutility suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. topoutility uses a separately accepted utility artifact; dice and cldice are region/topology controls; uniform_utility sets equal edit weights.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m topoutility evaluate --checkpoint runs/main/topoutility/model.pt --data data/prepared/test.npz --output results/topoutility
python -m topoutility plot --inputs runs/main/topoutility results/topoutility --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Stage 1 reports held-out pairwise agreement, rank/conditioning, bootstrap weight intervals and optional severity correlation. Stage 2 reports Dice/clDice, boundary/topology metrics, functional failure AUROC and GT-connected path failure.

## Implementation conventions

The six differentiable signatures are normalized proxies for gap, bridge, spur, loop, endpoint and branch error. The loop proxy uses an expected cubical Euler expression. Extractor settings remain fixed across fitting and segmentation; task-specific weights are never fitted to final segmentation labels.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m topoutility.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 1 --target mask
```

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.

## Registered compact reproduction setting

Procedural branching structures provide noisy segmentation images and separate edited mask pairs. Preferences follow exact root-terminal reachability, route distortion and occupied-area costs, independently of fitted utilities and differentiable signatures. Utility and segmentation object IDs are disjoint. Signature preprocessing calls the exact shared core.signatures function; the existing identification and held-out acceptance checks remain active.

Generate the declared input process, preprocess it, then use the existing method entry points:

```bash
python -m topoutility.synthetic --output data/generated --objects 60 --utility-objects 36 --seed 17
python -m topoutility.synthetic --extract-signatures data/generated/utility_masks.npz --output data/generated/utility
python -m topoutility prepare --source data/generated/segmentation.npz --output data/reproduce --seed 17
python -m topoutility.utility --train data/generated/utility/utility_train.npz --validation data/generated/utility/utility_validation.npz --epochs 300 --bootstrap 50 --output runs/reproduce_utility.json
python -m topoutility suite --config configs/reproduce.json --data data/reproduce --output runs/reproduce
python -m topoutility evaluate --checkpoint runs/reproduce/topoutility/model.pt --data data/reproduce/test.npz --output results/reproduce
```

`source.json` records the data-generating assumptions and seed. The configuration records `input_source`. All inputs for this selected setting are generated locally using installed package dependencies.
