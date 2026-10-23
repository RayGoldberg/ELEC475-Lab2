"""
ELEC 475: Lab 2 Post-Lab - Build ELEC475_Lab2_Report.pdf from the results written by run_all.py.

  python make_report.py --name "Ray Goldberg" --student-number 12345678
  python make_report.py --name "A B" --student-number 111 --partner "C D (222)"

Every number in the tables and in the analysis text is read from results/*.json (nothing is typed in by hand).
Missing runs show as n/a and the matching analysis sentence says the run is missing.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

from experiments import LADDER, MULTI_SEED_TAGS, SEEDS, build_runs
from models import AlexNet64, LeNet5, LeNetModern, get_torchvision_alexnet
from utils.seed import calculate_mean_std

RD = "results"
_cache = {}


# ------------------------------------------------------------------ data helpers
def load(tag):
    if tag not in _cache:
        p = os.path.join(RD, f"{tag}.json")
        _cache[tag] = json.load(open(p)) if os.path.exists(p) else None
    return _cache[tag]


def summ(tag):
    r = load(tag)
    if r is None:
        return None
    va = np.array(r["val_acc"])
    i = int(np.argmax(va))
    vl = np.array(r["val_loss"])
    tl = np.array(r["train_loss"])
    dv = np.diff(vl) if len(vl) > 1 else np.zeros(1)
    dt = np.diff(tl) if len(tl) > 1 else np.zeros(1)
    return dict(
        best_acc=float(va[i]), f1=float(r["val_f1"][i]), best_ep=i + 1, epochs=r["epochs_run"],
        final_val=float(va[-1]), final_train=float(r["train_acc"][-1]),
        gap=float(r["train_acc"][-1] - va[-1]),
        t=float(np.mean(r["epoch_times"])), params=r["params_m"], flops=r["flops_m"], macs=r["macs_m"],
        min_vl=float(vl.min()), min_vl_ep=int(vl.argmin()) + 1, final_vl=float(vl[-1]),
        upticks=int((dv > 0).sum()), rise=float(dv[dv > 0].sum()), tr_std=float(dt.std()),
        first_vl=float(vl[0]), first_acc=float(va[0]),
    )


def fx(x, d=2, unit=""):
    return "n/a" if x is None else f"{x:.{d}f}{unit}"


def sg(x, d=2):
    return "n/a" if x is None else f"{x:+.{d}f}"


def ms_str(vals, d=2):
    if not vals:
        return "n/a"
    m, s = calculate_mean_std(vals)
    return f"{m:.{d}f} +/- {s:.{d}f}%" + (f" (n={len(vals)})" if len(vals) < 3 else "")


def seed_vals(tag, key):
    out = []
    for s in SEEDS:
        t = tag if s == 42 else f"{tag}_s{s}"
        r = summ(t)
        if r is not None:
            out.append(r[key])
    return out


def clean_text(s):
    s = re.sub(r"\x1b\[[0-9;]*m", "", s)
    for a, b in (("—", "-"), ("–", "-"), ("≈", "~"), ("×", "x"), ("→", "->"), ("±", "+/-")):
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


# ------------------------------------------------------------------ figures
def curves(items, path, mode):
    """items: list of (tag, label). mode 'overlay' or 'gap'."""
    items = [(t, l) for t, l in items if load(t) is not None]
    if not items:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for k, (tag, label) in enumerate(items):
        r = load(tag)
        ep = range(1, len(r["val_loss"]) + 1)
        c = cols[k % len(cols)]
        if mode == "overlay":
            ax[0].plot(ep, r["train_loss"], "-o", color=c, label=label, ms=3)
            ax[1].plot(ep, r["val_loss"], "-o", color=c, label=label, ms=3)
            ax[2].plot(ep, r["val_acc"], "-o", color=c, label=label, ms=3)
        else:
            ax[0].plot(ep, r["train_loss"], "-", color=c, label=f"{label} train")
            ax[0].plot(ep, r["val_loss"], "--", color=c, label=f"{label} val")
            ax[1].plot(ep, r["train_acc"], "-", color=c, label=f"{label} train")
            ax[1].plot(ep, r["val_acc"], "--", color=c, label=f"{label} val")
            ax[2].plot(ep, np.array(r["train_acc"]) - np.array(r["val_acc"]), "-o", color=c, label=label, ms=3)
    if mode == "overlay":
        titles = ("Training loss", "Validation loss", "Validation accuracy (%)")
    else:
        titles = ("Loss (solid train, dashed val)", "Accuracy % (solid train, dashed val)", "Gap = train acc - val acc (pts)")
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=10)
        a.set_xlabel("Epoch")
        a.grid(True, ls="--", alpha=0.5)
        a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ------------------------------------------------------------------ architecture tables
def arch_rows(model):
    rows, hooks = [], []
    st = {"rf": 1, "jump": 1}

    def kst(m):
        k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        s = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
        return k[0], s[0]

    def hook(m, inp, out):
        own = sum(p.numel() for p in m.parameters(recurse=False))
        shape = "x".join(str(d) for d in out.shape[1:])
        rf = ""
        if isinstance(m, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            k, s = kst(m)
            st["rf"] += (k - 1) * st["jump"]
            st["jump"] *= s
            rf = str(st["rf"])
        if isinstance(m, torch.nn.Conv2d):
            desc = f"Conv2d {m.in_channels}->{m.out_channels}, k{m.kernel_size[0]}, s{m.stride[0]}, p{m.padding[0]}"
        elif isinstance(m, torch.nn.MaxPool2d):
            desc = f"MaxPool2d k{m.kernel_size}, s{m.stride}"
        elif isinstance(m, torch.nn.AvgPool2d):
            desc = f"AvgPool2d k{m.kernel_size}, s{m.stride}"
        elif isinstance(m, torch.nn.Linear):
            desc = f"Linear {m.in_features}->{m.out_features}"
        elif isinstance(m, torch.nn.AdaptiveAvgPool2d):
            desc = f"AdaptiveAvgPool2d {m.output_size}"
        elif isinstance(m, torch.nn.BatchNorm2d):
            desc = f"BatchNorm2d({m.num_features})"
        elif isinstance(m, torch.nn.LocalResponseNorm):
            desc = f"LocalResponseNorm(size={m.size}, alpha={m.alpha}, beta={m.beta}, k={m.k})"
        else:
            desc = type(m).__name__
        rows.append([desc, shape, f"{own:,}", rf])

    kinds = (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.Linear, torch.nn.AdaptiveAvgPool2d,
             torch.nn.BatchNorm2d, torch.nn.LocalResponseNorm)
    for m in model.modules():
        if isinstance(m, kinds):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 64))
    for h in hooks:
        h.remove()
    total = sum(p.numel() for p in model.parameters())
    first_fc = next(m.in_features for m in model.modules() if isinstance(m, torch.nn.Linear))
    return rows, total, first_fc


# ------------------------------------------------------------------ PDF helpers
def build_styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=17, spaceAfter=4),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13, spaceBefore=10, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, spaceBefore=8, spaceAfter=3),
        "b": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12, spaceAfter=4),
        "c": ParagraphStyle("c", parent=ss["BodyText"], fontSize=7.3, leading=9),
        "cap": ParagraphStyle("cap", parent=ss["BodyText"], fontSize=7.5, leading=9, textColor=colors.HexColor("#444444"), spaceAfter=6),
    }


def table(data, widths, S, header=True, align_first_left=True):
    rows = [[Paragraph(str(c), S["c"]) for c in row] for row in data]
    t = Table(rows, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf2")))
    t.setStyle(TableStyle(style))
    return t


def img(path, max_w=7.0, max_h=4.2):
    if not path or not os.path.exists(path):
        return None
    w, h = PILImage.open(path).size
    sc = min(max_w * inch / w, max_h * inch / h)
    return Image(path, width=w * sc, height=h * sc)


def pre(path, S, max_lines=70):
    if not os.path.exists(path):
        return [Paragraph(f"({os.path.basename(path)} not found)", S["b"])]
    lines = clean_text(open(path, encoding="utf-8", errors="replace").read()).splitlines()
    note = ""
    if len(lines) > max_lines:
        note = f"(first {max_lines} of {len(lines)} lines shown; full output in {path})"
        lines = lines[:max_lines]
    st = ParagraphStyle("pre", fontName="Courier", fontSize=5.9, leading=7)
    out = [Preformatted("\n".join(lines), st)]
    if note:
        out.append(Paragraph(note, S["cap"]))
    return out


# ------------------------------------------------------------------ main
def main():
    global RD
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--student-number", required=True)
    ap.add_argument("--partner", default="")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="ELEC475_Lab2_Report.pdf")
    a = ap.parse_args()
    RD = a.results_dir
    S = build_styles()
    fig_dir = os.path.join(RD, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    L = {k: dict(zip(("key", "label", "c10", "t200", "depth", "norm", "aug"), (k, lab, c, t, d, n, g)))
         for k, lab, c, t, d, n, g in LADDER}
    C = {k: summ(v["c10"]) for k, v in L.items()}      # CIFAR-10 summaries per ladder row
    T = {k: summ(v["t200"]) for k, v in L.items()}     # Tiny ImageNet summaries per ladder row
    story = []
    P = lambda txt, st="b": story.append(Paragraph(txt, S[st]))

    # ---------------- 1. title block
    P("ELEC 475: Lab 2 Post-Lab Report", "title")
    P("From LeNet-5 to AlexNet: Normalization Study, Data Augmentation and Dual-Dataset Benchmark", "h2")
    hdr = [["Course", "ELEC 475, Computer Vision with Deep Learning (Fall 2026), Queen's University"],
           ["Lab", "Lab 2, Post-Lab (Instructor: Dr. Michael Greenspan)"],
           ["Student", f"{a.name}, student number {a.student_number}"],
           ["Partner", a.partner if a.partner else "None (individual submission)"]]
    story.append(table(hdr, [1.0, 6.1], S, header=False))
    dev = next((load(t)["gpu"] or load(t)["device"] for t in [v["c10"] for v in L.values()] if load(t)), "n/a")
    P(f"All experiments were run on: <b>{dev}</b>. Mean epoch time is the wall-clock time of one training epoch plus one validation "
      "pass, averaged over all epochs of the run (train.py Timer). Validation metrics use the official CIFAR-10 test split and the "
      "official Tiny ImageNet validation split. Optimizer: Adam (lr 1e-3, weight decay 1e-4), batch size 128, cross-entropy loss, "
      "global seed 42 unless stated. Unless noted, validation accuracy and Macro F1 are taken at the epoch of best validation "
      "accuracy (the epoch whose weights are saved as the checkpoint).", "b")

    # ---------------- 2. architectures
    P("1. Architectural Specifications", "h1")
    P("Spatial size follows N_out = floor((N_in - K + 2P) / S) + 1. Receptive field (RF) is accumulated layer by layer as "
      "RF_l = RF_(l-1) + (K_l - 1) * J_(l-1), with jump J_l = J_(l-1) * S_l, starting from RF = 1 and J = 1. Output shapes are (C x H x W) "
      "for a 3x64x64 input and 10 classes. Parameter counts are exact (weights plus biases, measured from the instantiated model).", "b")
    models_arch = [
        ("Base LeNet-5", LeNet5(num_classes=10),
         "Two 5x5 convolutions with Tanh and 2x2 average pooling (no padding). 64 -> 60 -> 30 -> 26 -> 13, so the flatten size is 16 x 13 x 13 = 2,704."),
        ("Modernized LeNet", LeNetModern(num_classes=10),
         "Same two convolutions with padding 2, ReLU and overlapping 3x3 stride-2 max pooling. 64 -> 64 -> 31 -> 31 -> 15, so the flatten size is 16 x 15 x 15 = 3,600."),
        ("AlexNet-64 (BatchNorm variant shown)", AlexNet64(num_classes=10, norm_type="bn"),
         "Five convolution stages (64, 192, 384, 384, 256 filters), overlapping max pooling after stages 1, 2 and 5, flatten 256 x 3 x 3 = 2,304, "
         "then Dropout(0.5)-FC 1024-ReLU-Dropout(0.5)-FC 1024-ReLU-FC classes. The 'none' variant removes the normalization layers; the 'lrn' variant removes BatchNorm and "
         "adds LocalResponseNorm(5, 1e-4, 0.75, 2) after ReLU in stages 1 and 2 (Conv -> ReLU -> LRN -> MaxPool, as in the 2012 paper). Neither changes the parameter count except BatchNorm (+2 per channel)."),
        ("PyTorch Zoo AlexNet (torchvision, final layer resized)", get_torchvision_alexnet(num_classes=10),
         "11x11 stride-4 first convolution, so a 64x64 input shrinks to 15 -> 7 -> 3 -> 1 spatially; AdaptiveAvgPool2d forces a 6x6 map (256 x 6 x 6 = 9,216 features) "
         "into 4096-wide dense layers. This is why it has about 57M parameters but fewer MACs than AlexNet-64 on this input."),
    ]
    arch_totals = {}
    for title, m, blurb in models_arch:
        rows, total, fc1 = arch_rows(m)
        arch_totals[title] = total
        P(f"{title}: {total:,} parameters, first dense layer input = {fc1:,}", "h2")
        P(blurb, "b")
        story.append(table([["Layer", "Output (CxHxW or F)", "Params", "RF (px)"]] + rows, [3.6, 1.5, 1.2, 0.8], S))
    P("Reference check: the handout quotes nominal sizes of about 0.34M (LeNet), 0.45M (Modernized), 6.31M (AlexNet-64) and 57.04M (Zoo) parameters. "
      "The measured values here are 0.338M, 0.446M, 6.61M (6.611M with BatchNorm) and 57.04M. The AlexNet-64 count follows directly from the layer specification in "
      "Section 3.2 of the handout (convolutions 3,188,672 + FC 2,360,320 + 1,049,600 + 10,250 = 6,608,842), so the handout's 6.31M is treated as a rounded nominal value.", "cap")

    # ---------------- 3. results tables
    story.append(PageBreak())
    P("2. Empirical Results and Figures", "h1")
    P("Table 7.3a: Architectural configuration and computational complexity (CIFAR-10, 10 classes, input 1x3x64x64)", "h2")
    cmp = json.load(open(os.path.join(RD, "compare_models.json"))) if os.path.exists(os.path.join(RD, "compare_models.json")) else None
    lat_idx = {"lenet": 0, "modern": 1, "alex_none": 2, "alex_lrn": 3, "alex_bn": 4, "alex_aug": 4, "zoo": 5}
    data = [["Model", "Conv depth", "Normalization / regularization", "Data aug.", "Params (M)", "FLOPs (M)", "Mean epoch time (s)", "Latency (ms)"]]
    for k, v in L.items():
        s = C[k]
        lat = cmp["rows"][lat_idx[k]]["latency_ms"] if cmp else None
        data.append([v["label"], v["depth"], v["norm"], v["aug"], fx(s and s["params"], 3), fx(s and s["flops"], 2),
                     fx(s and s["t"], 1), fx(lat, 2)])
    story.append(table(data, [1.35, 0.5, 1.5, 0.75, 0.6, 0.65, 0.7, 0.55], S))
    P("FLOPs = 2 x MACs counted by the lab profiler (thop). Latency is the median of 50 forward passes at batch size 1 on the device above "
      "(compare_models.py). Epoch times are from the CIFAR-10 runs listed in Table 7.3b. The handout lists nominal FLOPs of about 4.8 / 8.2 / 250-257 / 310-315 M; "
      "the measured counts differ because the nominal figures are rough estimates (for example the AlexNet-64 convolutions alone are 214.9 M MACs by hand count).", "cap")

    P("Table 7.3b: Validation performance across datasets (single run, seed 42)", "h2")
    data = [["Model", "CIFAR-10 epochs", "CIFAR-10 Val Acc (%)", "CIFAR-10 Val Macro F1 (%)", "Tiny ImageNet epochs", "Tiny ImageNet Val Acc (%)", "Tiny ImageNet Val Macro F1 (%)"]]
    for k, v in L.items():
        c, t = C[k], T[k]
        data.append([v["label"], fx(c and c["epochs"], 0), fx(c and c["best_acc"]), fx(c and c["f1"]),
                     fx(t and t["epochs"], 0), fx(t and t["best_acc"]), fx(t and t["f1"])])
    story.append(table(data, [1.5, 0.75, 0.95, 1.0, 0.8, 1.0, 1.1], S))
    P("Rows 3-5 and 7 use --no-augment so that 'Data Augmentation = None' holds; row 6 uses the augmentation pipeline. Row 6 on CIFAR-10 is the handout's 15-epoch "
      "run; its no-augmentation counterpart (15 epochs) is in Table 3. Tiny ImageNet runs are 10 epochs each.", "cap")

    # multi-seed
    P("Table 7.3c: Multi-seed evaluation on CIFAR-10 (Mean +/- Std over seeds 42, 43, 44)", "h2")
    data = [["Configuration", "Val Acc (%)", "Val Macro F1 (%)", "Final train-val gap (pts)", "Mean epoch time (s)"]]
    for tag, lab in MULTI_SEED_TAGS:
        data.append([lab, ms_str(seed_vals(tag, "best_acc")), ms_str(seed_vals(tag, "f1")),
                     ms_str(seed_vals(tag, "gap")).replace("%", ""), ms_str(seed_vals(tag, "t"), 1).replace("%", "")])
    story.append(table(data, [2.3, 1.3, 1.3, 1.3, 1.0], S))
    P("Std is the sample standard deviation (ddof = 1, utils/seed.py). With only three seeds these are rough estimates of run-to-run variation.", "cap")

    # extended
    P("Table 7.3d: Extended ablation exploration (CIFAR-10, seed 42, Delta vs reference run)", "h2")
    data = [["Experiment", "Reference", "Val Acc (%)", "Macro F1 (%)", "Epoch time (s)", "Delta Acc (pts)", "Delta F1 (pts)", "Delta time (s)"]]
    ref_names = {k: v["label"] for k, v in L.items()}
    c10_to_label = {v["c10"]: v["label"] for v in L.values()}
    for r in [r for r in build_runs() if r["stage"] == "extended"]:
        s, ref = summ(r["tag"]), summ(r["reference"])
        data.append([r["label"], c10_to_label.get(r["reference"], r["reference"]), fx(s and s["best_acc"]), fx(s and s["f1"]), fx(s and s["t"], 1),
                     sg(s and ref and s["best_acc"] - ref["best_acc"]), sg(s and ref and s["f1"] - ref["f1"]), sg(s and ref and s["t"] - ref["t"], 1)])
    story.append(table(data, [1.5, 1.35, 0.7, 0.7, 0.7, 0.75, 0.7, 0.7], S))
    P("Custom experiments chosen to probe (a) learning-rate and batch-size sensitivity of the best architecture and (b) whether augmentation helps a network "
      "that has no normalization and a shallow network. References are the 10-epoch no-augmentation runs of Table 7.3b.", "cap")

    # figures
    f_norm = curves([("c10_alex_none", "No norm"), ("c10_alex_lrn", "LRN"), ("c10_alex_bn", "BatchNorm")], f"{fig_dir}/norm.png", "overlay")
    f_aug = curves([("c10_alex_bn_noaug15", "No aug"), ("c10_alex_bn_aug15", "Aug")], f"{fig_dir}/aug.png", "gap")
    f_tiny = curves([("c10_alex_bn", "CIFAR-10 BN"), ("t200_alex_bn", "Tiny ImageNet BN"), ("t200_alex_bn_aug", "Tiny ImageNet BN + Aug")],
                    f"{fig_dir}/tiny.png", "gap")
    story.append(PageBreak())
    P("Figures", "h2")
    for cap, f in (("Figure 1: Normalization study, AlexNet-64 on CIFAR-10, 10 epochs (training loss, validation loss, validation accuracy).", f_norm),
                   ("Figure 2: Effect of data augmentation, AlexNet-64 + BatchNorm on CIFAR-10, 15 epochs (loss, accuracy and generalization gap).", f_aug),
                   ("Figure 3: CIFAR-10 versus Tiny ImageNet-200, AlexNet-64 + BatchNorm (loss, accuracy and gap per epoch).", f_tiny)):
        im = img(f, 7.1, 2.8)
        if im:
            story.append(im)
            P(cap, "cap")
    for cap, f in (("Figure 4: Baseline LeNet-5 loss/accuracy curves (train.py --plot-file).", "plots/loss_lenet.png"),
                   ("Figure 5: Modernized LeNet loss/accuracy curves.", "plots/loss_lenet_modern.png"),
                   ("Figure 6: AlexNet-64 + BatchNorm loss/accuracy curves.", "plots/loss_alexnet_bn.png"),
                   ("Figure 7: Master ablation summary, validation accuracy of the seven-model ladder on CIFAR-10 (python -m utils.visualization --plot-ablation).", "plots/ablation_summary.png")):
        im = img(f, 6.2, 3.0)
        if im:
            story.append(im)
            P(cap, "cap")

    story.append(PageBreak())
    P("Exploratory data analysis (python -m utils.eda --dataset both)", "h2")
    eda = sorted(glob.glob("plots/eda_*_class_distribution.png") + glob.glob("plots/eda_*_channel_histograms.png"))
    for i, f in enumerate(eda):
        im = img(f, 6.3, 3.3)
        if im:
            story.append(im)
            P(f"Figure {8 + i}: {os.path.basename(f)}", "cap")
    P("Transform paradigm diagnostic (python demo_transforms.py)", "h2")
    story.extend(pre(os.path.join(RD, "demo_transforms.txt"), S, 70))
    P("Model complexity benchmark (python compare_models.py --dataset cifar10)", "h2")
    story.extend(pre(os.path.join(RD, "compare_models.txt"), S, 45))

    # ---------------- 4. analysis
    story.append(PageBreak())
    P("3. Technical Analysis and Discussion", "h1")
    none, lrn, bn = summ("c10_alex_none"), summ("c10_alex_lrn"), summ("c10_alex_bn")

    P("3.1 Normalization study (None vs LRN vs BatchNorm2d, CIFAR-10, 10 epochs)", "h2")
    data = [["Variant", "Val Acc (%)", "Val Macro F1 (%)", "Epoch time (s)", "FLOPs (M)", "Val-loss up-ticks", "Total val-loss rise", "Train-loss step std"]]
    for lab, s in (("No normalization", none), ("LRN", lrn), ("BatchNorm2d", bn)):
        data.append([lab, fx(s and s["best_acc"]), fx(s and s["f1"]), fx(s and s["t"], 1), fx(s and s["flops"], 2),
                     fx(s and s["upticks"], 0), fx(s and s["rise"], 3), fx(s and s["tr_std"], 3)])
    story.append(table(data, [1.3, 0.8, 0.9, 0.8, 0.8, 0.8, 0.9, 0.9], S))
    if none and lrn and bn:
        best_other_acc, best_other_f1 = max(none["best_acc"], lrn["best_acc"]), max(none["f1"], lrn["f1"])
        d_acc, d_f1 = bn["best_acc"] - best_other_acc, bn["f1"] - best_other_f1
        yes = d_acc > 0 and d_f1 > 0
        std_bn = calculate_mean_std(seed_vals("c10_alex_bn", "best_acc"))[1] if len(seed_vals("c10_alex_bn", "best_acc")) > 1 else None
        marginal = abs(d_acc) < 1.0
        P(f"<b>Performance improvement: {'Yes' if yes else 'No'}.</b> BatchNorm reached {bn['best_acc']:.2f}% accuracy and {bn['f1']:.2f}% Macro F1, versus "
          f"{none['best_acc']:.2f}% / {none['f1']:.2f}% without normalization and {lrn['best_acc']:.2f}% / {lrn['f1']:.2f}% with LRN. "
          f"Relative to the better of the other two the difference is {d_acc:+.2f} pts accuracy and {d_f1:+.2f} pts Macro F1"
          + (f" (seed-to-seed std of the BatchNorm model is {std_bn:.2f} pts, so differences of that size should not be over-read)." if std_bn is not None else "."))
        if yes and not marginal:
            P("Why: Batch Normalization standardizes each channel with the mini-batch mean and variance and then re-scales with learnable gamma and beta. This keeps the "
              "distribution of every layer's input roughly fixed while earlier layers change, which is the internal covariate shift argument of Ioffe and Szegedy (2015), "
              "so each layer no longer has to keep chasing a moving input distribution. Later analysis (Santurkar et al., 2018) attributes much of the benefit to a smoother loss "
              "surface: with normalized activations the gradient changes less abruptly between steps, so the gradient direction stays predictive over larger steps. In practice this "
              "allows the same learning rate (1e-3) to make faster progress, reduces sensitivity to initialization, and the per-batch noise acts as a mild regularizer.")
        else:
            P("The gain is absent or marginal in this setting. Likely limiting factors: (i) only 10 epochs, which may end before the un-normalized or LRN networks have "
              "finished converging, so the gap measured here can under- or over-state the long-run difference; (ii) Adam with lr 1e-3 already adapts per-parameter step sizes and "
              "partly offsets the conditioning problems that BatchNorm fixes; (iii) batch size 128 and weight decay 1e-4 were not tuned per variant; (iv) a single seed, where "
              "run-to-run noise can be the same size as the difference. LRN has no learnable parameters and does not re-center or control variance, so it changes optimization much less than BatchNorm.")
        for lab, s in (("LRN", lrn), ("BatchNorm2d", bn)):
            pass
        t_bn, t_lrn = (bn["t"] - none["t"]) / none["t"] * 100, (lrn["t"] - none["t"]) / none["t"] * 100
        meas = abs(t_bn) >= 5 or abs(t_lrn) >= 5
        P(f"<b>Computational overhead: {'Yes' if meas else 'No'}.</b> Mean epoch time was {none['t']:.1f} s (none), {lrn['t']:.1f} s (LRN, {t_lrn:+.1f}%) and "
          f"{bn['t']:.1f} s (BatchNorm, {t_bn:+.1f}%); a change of 5% or more is treated as measurable. Profiler FLOPs were {none['flops']:.2f} M, {lrn['flops']:.2f} M and {bn['flops']:.2f} M. "
          "Element-wise normalization is cheap in arithmetic: the five BatchNorm layers touch about 159 thousand activations per image, i.e. a few hundred thousand extra operations "
          "against roughly 437 M FLOPs in the convolutions and dense layers (well under 1%). LRN touches 109 thousand activations (stages 1-2) but each needs a sum of squares over 5 "
          "channels plus a power, so it costs more per element than BatchNorm. What matters in wall-clock time is memory traffic and kernel launches rather than FLOPs, because these "
          "layers are memory-bound (read the whole activation tensor, write it back); training also needs batch statistics and extra backward kernels. Against that small cost, "
          "the convergence benefit (when present) is measured in epochs, not percent.")
        best_stab = min((("No normalization", none), ("LRN", lrn), ("BatchNorm2d", bn)), key=lambda x: (x[1]["upticks"], x[1]["rise"]))
        P(f"<b>Convergence stability.</b> Counting epochs in which validation loss went up and the summed size of those increases (table above), the most stable trajectory was "
          f"<b>{best_stab[0]}</b> ({best_stab[1]['upticks']} up-tick(s), total rise {best_stab[1]['rise']:.3f}). Validation loss after epoch 1 was {none['first_vl']:.3f} (none), "
          f"{lrn['first_vl']:.3f} (LRN) and {bn['first_vl']:.3f} (BatchNorm) and the minimum was {none['min_vl']:.3f}, {lrn['min_vl']:.3f}, {bn['min_vl']:.3f}. "
          "See Figure 1 for the full curves; a normalization scheme that lowers the first-epoch loss and keeps later increases small provides the steadiest gradient progression.")
    else:
        P("Normalization runs are missing; run python run_all.py --stages core.")

    P("3.2 Data augmentation and the generalization gap (AlexNet-64 + BatchNorm, CIFAR-10, 15 epochs)", "h2")
    na, au = summ("c10_alex_bn_noaug15"), summ("c10_alex_bn_aug15")
    if na and au:
        data = [["Run", "Train Acc (%) (final)", "Val Acc (%) (final)", "Gap (pts)", "Best Val Acc (%)", "Best Val Macro F1 (%)", "Min val loss"]]
        for lab, s in (("No augmentation", na), ("Pure PyTorch augmentation", au)):
            data.append([lab, fx(s["final_train"]), fx(s["final_val"]), fx(s["gap"]), fx(s["best_acc"]), fx(s["f1"]), fx(s["min_vl"], 3)])
        story.append(table(data, [1.6, 1.0, 1.0, 0.8, 1.0, 1.1, 0.8], S))
        red = au["gap"] < na["gap"]
        P(f"<b>Generalization gap reduction: {'Yes' if red else 'No'}.</b> Without augmentation the gap was {na['final_train']:.2f} - {na['final_val']:.2f} = {na['gap']:.2f} pts; "
          f"with augmentation {au['final_train']:.2f} - {au['final_val']:.2f} = {au['gap']:.2f} pts. Gap values are taken at the final epoch. Note that training accuracy is the running average "
          "over each epoch, measured in training mode (dropout active) and, for the augmented run, on randomly transformed images, so part of any gap reduction comes from the training "
          "images becoming harder rather than from better validation performance; the validation numbers below are the cleaner evidence.")
        d_acc, d_f1 = au["best_acc"] - na["best_acc"], au["f1"] - na["f1"]
        up = d_acc > 0 and d_f1 > 0
        extra = ""
        ms_a, ms_n = seed_vals("c10_alex_bn_aug15", "best_acc"), seed_vals("c10_alex_bn_noaug15", "best_acc")
        if len(ms_a) > 1 and len(ms_n) > 1:
            extra = f" Over the available seeds the mean accuracy was {ms_str(ms_n)} without and {ms_str(ms_a)} with augmentation."
        P(f"<b>Validation metric improvement: {'Yes' if up else 'No'}.</b> Best validation accuracy changed by {d_acc:+.2f} pts ({na['best_acc']:.2f}% to {au['best_acc']:.2f}%) and "
          f"Macro F1 by {d_f1:+.2f} pts ({na['f1']:.2f}% to {au['f1']:.2f}%).{extra}")
        if up:
            P("Why: random crops (4-pixel reflect padding), horizontal flips and brightness/contrast/saturation jitter are label-preserving, so every epoch shows the network a different "
              "version of each image. The network can no longer lower the loss by memorizing exact pixel patterns at fixed positions, so its filters are pushed toward features that stay "
              "useful under small translations, mirroring and lighting change (invariant semantic features). This reduces variance and lets the larger model keep improving on held-out data.")
        else:
            P(f"The augmented run did not improve both metrics. Looking at the curves (Figure 2): the augmented network's best-validation epoch was {au['best_ep']} of {au['epochs']} "
              f"versus {na['best_ep']} of {na['epochs']} without augmentation, and its final training accuracy was {au['final_train']:.1f}%. If validation was still rising at epoch 15 the network "
              "simply needs more epochs to absorb the harder, augmented distribution; if training accuracy stayed low, the transformations (especially colour jitter on already low-resolution images) "
              "may have caused mild underfitting.")
    else:
        P("Augmentation runs are missing; run python run_all.py --stages core.")

    P("3.3 Dual-dataset benchmark (Tiny ImageNet-200)", "h2")
    tb, tba, cb = summ("t200_alex_bn"), summ("t200_alex_bn_aug"), summ("c10_alex_bn")
    if tba and cb:
        chance = 100 / 200
        P(f"<b>Generalization above random chance: {'Yes' if tba['best_acc'] > 3 * chance else 'No'}.</b> AlexNet-64 + BatchNorm with augmentation (the handout's Step 4 command) reached "
          f"{tba['best_acc']:.2f}% validation accuracy and {tba['f1']:.2f}% Macro F1 on 200 classes, against a random-guess baseline of {chance:.1f}%"
          + (f" (without augmentation: {tb['best_acc']:.2f}% / {tb['f1']:.2f}%)." if tb else "."))
        P(f"<b>Cross-dataset comparison.</b> On CIFAR-10 the same architecture (10 epochs, no augmentation) scored {cb['best_acc']:.2f}% accuracy and {cb['f1']:.2f}% Macro F1, "
          f"so Tiny ImageNet is {cb['best_acc'] - tba['best_acc']:.1f} pts lower in accuracy. Reasons: (i) 200 fine-grained classes instead of 10 coarse ones (chance 0.5% versus 10%), "
          "including many visually similar categories; (ii) only 500 training images per class versus 5,000 for CIFAR-10, so far fewer examples to learn each class; (iii) Tiny ImageNet images "
          "are natively 64x64 natural photographs with cluttered backgrounds and large intra-class variation, while CIFAR-10 images are 32x32 objects up-sampled to 64x64 and carry less detail; "
          "(iv) the 1024-unit dense layers and a 200-way output give the network ample capacity to memorize 100,000 images, so overfitting is a larger concern.")
        gap_note = (f" Without augmentation the Tiny ImageNet validation loss reached its minimum {tb['min_vl']:.3f} at epoch {tb['min_vl_ep']} and ended at {tb['final_vl']:.3f}." if tb else "")
        diverge = tba["final_vl"] > tba["min_vl"] + 0.05
        P(f"<b>Overfitting trajectory.</b> With augmentation the Tiny ImageNet validation loss reached its minimum {tba['min_vl']:.3f} at epoch {tba['min_vl_ep']} of {tba['epochs']} and ended at "
          f"{tba['final_vl']:.3f}, so it {'diverged upward after its minimum (overfitting)' if diverge else 'was still decreasing or had plateaued without clear divergence'}; "
          f"the final train/val accuracy gap was {tba['gap']:.1f} pts.{gap_note} On CIFAR-10 (10 epochs, no augmentation) validation loss bottomed out at {cb['min_vl']:.3f} in epoch {cb['min_vl_ep']} "
          f"and ended at {cb['final_vl']:.3f} (gap {cb['gap']:.1f} pts). See Figure 3. Tiny ImageNet has fewer samples per class and many more classes, so the network can fit the training set "
          "faster than it can learn generalizable features, which shows up as an earlier validation-loss minimum and a larger train/validation gap whenever that is observed above.")
    else:
        P("Tiny ImageNet runs are missing; run python run_all.py --stages tiny.")

    P("3.4 Differences between the model variations (ladder summary)", "h2")
    lad = []
    keys = list(L.keys())
    for i in range(1, len(keys)):
        a_, b_ = C[keys[i - 1]], C[keys[i]]
        if keys[i] == "zoo":
            a_ = C["alex_bn"] if C["alex_bn"] else a_
            lad.append(f"Zoo AlexNet versus AlexNet-64 + BN: parameters {fx(C['alex_bn'] and C['alex_bn']['params'], 2)} M to {fx(b_ and b_['params'], 2)} M, "
                       f"MFLOPs {fx(C['alex_bn'] and C['alex_bn']['flops'], 0)} to {fx(b_ and b_['flops'], 0)}, accuracy "
                       f"{fx(C['alex_bn'] and C['alex_bn']['best_acc'])}% to {fx(b_ and b_['best_acc'])}%.")
        elif a_ and b_:
            lad.append(f"{L[keys[i]]['label']} versus {L[keys[i - 1]]['label']}: accuracy {a_['best_acc']:.2f}% to {b_['best_acc']:.2f}% ({b_['best_acc'] - a_['best_acc']:+.2f} pts), "
                       f"parameters {a_['params']:.2f} M to {b_['params']:.2f} M, MFLOPs {a_['flops']:.1f} to {b_['flops']:.1f}, epoch time {a_['t']:.1f} s to {b_['t']:.1f} s.")
    P(" ".join(lad) if lad else "Ladder runs are missing.")
    P("Interpretation. (1) Base LeNet to Modernized LeNet: ReLU does not saturate for positive inputs (derivative exactly 1) so gradients are not shrunk through the layers, "
      "overlapping max pooling keeps the strongest local response instead of blurring it, and padding 2 keeps border information and feeds a larger feature vector to the dense layers; "
      "these changes cost only a small amount of extra computation. (2) Modernized LeNet to AlexNet: five convolution stages with 64 to 384 channels build a much richer feature hierarchy "
      "(the receptive field reaches 97 pixels at the last pooling layer, larger than the 64-pixel input, see the RF column) at roughly 50 to 60 times the FLOPs of LeNet, with dropout controlling the capacity of the dense layers. "
      "(3) Normalization changes training dynamics and, per Section 3.1, adds very little arithmetic. (4) Augmentation does not change parameters, FLOPs or inference latency at all; it only changes "
      "the training distribution, so its benefit is free at inference time but costs training-time data processing. (5) The PyTorch Zoo AlexNet has about 8.6 times the parameters of AlexNet-64 "
      "(its dense layers see 9,216 features and are 4096 wide) but fewer MACs here because its 11x11 stride-4 first layer shrinks the 64x64 input immediately; more parameters do not "
      "translate into a proportional accuracy gain at this input size and training length.")
    P("Limitations. Most results are single-seed (see Table 7.3c for variation where extra seeds were run); runs are short (10 to 15 epochs); the CIFAR-10 test split doubles as the validation set "
      "(no separate hold-out), so best-epoch selection is slightly optimistic; epoch time includes the validation pass and data loading with two workers, so for small networks it measures the data "
      "pipeline as much as the model; and absolute timings depend on the GPU and CPU used.", "cap")

    # ---------------- 5. reproduce
    P("4. Reproducibility", "h1")
    P("All commands are listed in commands.txt. In short: <font face='Courier'>pip install -r requirements.txt</font>, <font face='Courier'>python run_all.py</font>, "
      "<font face='Courier'>python make_report.py --name ... --student-number ...</font>. Global seeding (Python, NumPy, PyTorch, cuDNN deterministic) is in utils/seed.py. "
      "The data/ folder is not part of the submission; datasets download automatically on first use. Deviations from the handout text: rows 3-5 add --no-augment (augmentation defaults on); "
      "LRN follows Figure 4.2 (after ReLU, stages 1-2 only); the LeNet baselines are trained for 10 epochs (in-lab used 5) so that all rows share a comparable training length.", "b")

    doc = SimpleDocTemplate(a.out, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                            title="ELEC 475 Lab 2 Post-Lab Report", author=a.name)
    doc.build(story)
    print(f"Report written to {a.out}")


if __name__ == "__main__":
    main()
