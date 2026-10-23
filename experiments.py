"""
ELEC 475: Lab 2 Post-Lab - Experiment registry.
One place that defines every training run, so run_all.py (execution), make_report.py (tables),
utils/visualization.py (ablation plot) and commands.txt all agree on names and settings.
"""

SEEDS = (42, 43, 44)

# Rungs of the mandatory ablation ladder (Tables 7.3a and 7.3b)
# (key, label, cifar tag, tiny tag, conv depth, normalization/regularization text, augmentation text)
LADDER = [
    ("lenet",    "1. Base LeNet-5",         "c10_lenet",        "t200_lenet",        2, "None", "None"),
    ("modern",   "2. Modernized LeNet",     "c10_lenet_modern", "t200_lenet_modern", 2, "None (ReLU + Overlapping MaxPool)", "None"),
    ("alex_none","3. AlexNet (No Norm)",    "c10_alex_none",    "t200_alex_none",    5, "Dropout (p=0.5)", "None"),
    ("alex_lrn", "4. AlexNet + LRN",        "c10_alex_lrn",     "t200_alex_lrn",     5, "LRN + Dropout (p=0.5)", "None"),
    ("alex_bn",  "5. AlexNet + BatchNorm",  "c10_alex_bn",      "t200_alex_bn",      5, "BatchNorm2d + Dropout (p=0.5)", "None"),
    ("alex_aug", "6. AlexNet + BN + Aug",   "c10_alex_bn_aug15","t200_alex_bn_aug",  5, "BatchNorm + Dropout (p=0.5)", "Pure PyTorch Aug"),
    ("zoo",      "7. PyTorch Zoo AlexNet",  "c10_zoo",          "t200_zoo",          5, "PyTorch Zoo (AdaptiveAvgPool2d)", "None"),
]

# Models that also get extra seeds (CIFAR-10 only) for the Mean +/- Std table
MULTI_SEED_TAGS = [
    ("c10_lenet",         "Base LeNet-5 (10 ep, no aug)"),
    ("c10_lenet_modern",  "Modernized LeNet (10 ep, no aug)"),
    ("c10_alex_none",     "AlexNet No Norm (10 ep, no aug)"),
    ("c10_alex_lrn",      "AlexNet + LRN (10 ep, no aug)"),
    ("c10_alex_bn",       "AlexNet + BatchNorm (10 ep, no aug)"),
    ("c10_alex_bn_noaug15", "AlexNet + BN (15 ep, no aug)"),
    ("c10_alex_bn_aug15", "AlexNet + BN + Aug (15 ep)"),
]


def _run(tag, model, dataset, epochs, augment, norm="bn", seed=42, lr=1e-3, bs=128,
         ckpt=None, plot=None, stage="core", label=None, reference=None):
    return {
        "tag": tag, "model": model, "norm": norm, "dataset": dataset, "epochs": epochs,
        "augment": augment, "seed": seed, "lr": lr, "bs": bs, "ckpt": ckpt, "plot": plot,
        "stage": stage, "label": label or tag, "reference": reference,
    }


def build_runs(smoke=False):
    """Return the ordered list of run dicts. smoke=True shortens every run to 2 epochs."""
    R = []
    # ---- Stage "core": CIFAR-10 mandatory runs (seed 42) -------------------------------
    R += [
        _run("c10_lenet", "lenet", "cifar10", 10, False, ckpt="checkpoints/lenet_baseline.pth", plot="plots/loss_lenet.png"),
        _run("c10_lenet_modern", "lenet_modern", "cifar10", 10, False, ckpt="checkpoints/lenet_modern.pth", plot="plots/loss_lenet_modern.png"),
        _run("c10_alex_none", "alexnet", "cifar10", 10, False, norm="none", ckpt="checkpoints/alexnet_none.pth", plot="plots/loss_alexnet_none.png"),
        _run("c10_alex_lrn", "alexnet", "cifar10", 10, False, norm="lrn", ckpt="checkpoints/alexnet_lrn.pth", plot="plots/loss_alexnet_lrn.png"),
        _run("c10_alex_bn", "alexnet", "cifar10", 10, False, norm="bn", ckpt="checkpoints/alexnet_bn.pth", plot="plots/loss_alexnet_bn.png"),
        _run("c10_alex_bn_noaug15", "alexnet", "cifar10", 15, False, norm="bn", ckpt="checkpoints/alexnet_no_aug.pth", plot="plots/loss_no_aug.png"),
        _run("c10_alex_bn_aug15", "alexnet", "cifar10", 15, True, norm="bn", ckpt="checkpoints/alexnet_with_aug.pth", plot="plots/loss_with_aug.png"),
        _run("c10_zoo", "zoo_alexnet", "cifar10", 10, False, plot="plots/loss_zoo_alexnet.png"),
    ]
    # ---- Stage "tiny": Tiny ImageNet-200 (seed 42, 10 epochs) ----------------------------
    R += [
        _run("t200_lenet", "lenet", "tiny_imagenet", 5, False, stage="tiny", plot="plots/loss_t200_lenet.png"),
        _run("t200_lenet_modern", "lenet_modern", "tiny_imagenet", 5, False, stage="tiny", plot="plots/loss_t200_lenet_modern.png"),
        _run("t200_alex_none", "alexnet", "tiny_imagenet", 5, False, norm="none", stage="tiny", plot="plots/loss_t200_alex_none.png"),
        _run("t200_alex_lrn", "alexnet", "tiny_imagenet", 5, False, norm="lrn", stage="tiny", plot="plots/loss_t200_alex_lrn.png"),
        _run("t200_alex_bn", "alexnet", "tiny_imagenet", 5, False, norm="bn", stage="tiny", plot="plots/loss_t200_alex_bn.png"),
        # This is the handout's Step 4 command (augmentation left at its default = on)
        _run("t200_alex_bn_aug", "alexnet", "tiny_imagenet", 5, True, norm="bn", stage="tiny",
             ckpt="checkpoints/alexnet_tiny_imagenet.pth", plot="plots/loss_tiny_imagenet.png"),
        _run("t200_zoo", "zoo_alexnet", "tiny_imagenet", 5, False, lr=1e-4, stage="tiny", plot="plots/loss_t200_zoo.png"),
    ]
    # ---- Stage "seeds": extra seeds on CIFAR-10 for Mean +/- Std --------------------------
    base = {r["tag"]: r for r in R}
    for tag, _ in MULTI_SEED_TAGS:
        for s in SEEDS:
            if s == 42:
                continue
            r = dict(base[tag])
            r.update(tag=f"{tag}_s{s}", seed=s, ckpt=None, plot=None, stage="seeds")
            R.append(r)
    # ---- Stage "extended": custom exploration, CIFAR-10, seed 42 ---------------------------
    ref = "c10_alex_bn"
    R += [
        _run("x_lr1e-4", "alexnet", "cifar10", 10, False, lr=1e-4, stage="extended", reference=ref, label="AlexNet+BN, lr 1e-4"),
        _run("x_lr1e-2", "alexnet", "cifar10", 10, False, lr=1e-2, stage="extended", reference=ref, label="AlexNet+BN, lr 1e-2"),
        _run("x_bs64", "alexnet", "cifar10", 10, False, bs=64, stage="extended", reference=ref, label="AlexNet+BN, batch 64"),
        _run("x_bs256", "alexnet", "cifar10", 10, False, bs=256, stage="extended", reference=ref, label="AlexNet+BN, batch 256"),
        _run("x_none_aug", "alexnet", "cifar10", 10, True, norm="none", stage="extended", reference="c10_alex_none",
             label="AlexNet No Norm + Aug"),
        _run("x_modern_aug", "lenet_modern", "cifar10", 10, True, stage="extended", reference="c10_lenet_modern",
             label="Modernized LeNet + Aug"),
    ]
    if smoke:
        for r in R:
            r["epochs"] = 2
    return R


def train_cmd(r, results_dir="results", python="python"):
    """Return the train.py argument list (strings) for a run dict."""
    a = [python, "train.py", "--model", r["model"]]
    if r["model"] == "alexnet":
        a += ["--norm", r["norm"]]
    a += ["--dataset", r["dataset"], "--epochs", str(r["epochs"])]
    if r["seed"] != 42:
        a += ["--seed", str(r["seed"])]
    if r["lr"] != 1e-3:
        a += ["--lr", str(r["lr"])]
    if r["bs"] != 128:
        a += ["--batch-size", str(r["bs"])]
    if not r["augment"]:
        a += ["--no-augment"]
    if r["ckpt"]:
        a += ["--save-model", r["ckpt"]]
    if r["plot"]:
        a += ["--plot-file", r["plot"]]
    a += ["--results-json", f"{results_dir}/{r['tag']}.json"]
    return a
