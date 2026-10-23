"""
ELEC 475: Lab 2 Post-Lab - Run every experiment in order and log the results.

  python run_all.py                     # everything (GPU strongly recommended)
  python run_all.py --stages core tiny  # only the mandatory handout runs
  python run_all.py --smoke             # 2-epoch dry run to check the whole pipeline
  python run_all.py --write-commands    # only (re)write commands.txt

Finished runs are skipped (results/<tag>.json exists), so it is safe to re-run after a disconnect.
Stages in order: core, tiny, tools, seeds, extended, plots.
"""
import argparse
import os
import subprocess
import sys
import time

import torch

from experiments import build_runs, train_cmd

ALL_STAGES = ["core", "tiny", "tools", "seeds", "extended", "plots"]
SHOW = ("Device", "Model ", "Dataset", "Train batches", "Training completed", "Best Val", "EarlyStopping", "Error", "Traceback")


def run_logged(cmd, log_path, quiet=False):
    """Run cmd, write full output to log_path, echo only the useful lines. Returns exit code."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            log.write(line)
            s = line.rstrip()
            if quiet:
                continue
            if (" | " in s and s.strip()[:1].isdigit()) or s.startswith(SHOW) or "Error" in s:
                print("   ", s, flush=True)
        p.wait()
    return p.returncode


def write_commands(runs, path="commands.txt"):
    lines = [
        "ELEC 475 Lab 2 - exact commands (run from the project root, Python 3.10+, pip install -r requirements.txt)",
        "",
        "# One command that runs everything below, skips finished runs, and builds the report inputs:",
        "python run_all.py",
        "python make_report.py --name \"<full name>\" --student-number <student_number>",
        "",
        "# Note: rows 3-5 of the ablation ladder are trained with --no-augment so that 'Data Augmentation = None'",
        "# (augmentation is ON by default in train.py once implemented). Extra seeds and the extended study use",
        "# --seed / --lr / --batch-size as shown.",
        "",
    ]
    sections = [("core", "CIFAR-10 mandatory runs (seed 42)"), ("tiny", "Tiny ImageNet-200 runs (seed 42)"),
                ("seeds", "Extra seeds for Mean +/- Std (CIFAR-10)"), ("extended", "Extended ablation exploration (CIFAR-10)")]
    for stage, title in sections:
        lines.append(f"# {title}")
        for r in runs:
            if r["stage"] == stage:
                lines.append(" ".join(train_cmd(r)))
        lines.append("")
    lines += [
        "# Tooling outputs used in the report",
        "python demo_transforms.py",
        "python compare_models.py --dataset cifar10 --device cuda",
        "python -m utils.eda --dataset both",
        "python -m utils.visualization --plot-ablation",
        "",
        "# Evaluate a saved checkpoint",
        "python evaluate.py --checkpoint checkpoints/alexnet_bn.pth --model alexnet --norm bn --dataset cifar10",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--smoke", action="store_true", help="2 epochs per run (pipeline check only)")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--write-commands", action="store_true")
    args = ap.parse_args()

    runs = build_runs(smoke=args.smoke)
    if args.write_commands:
        write_commands(build_runs(smoke=False))
        return
    write_commands(build_runs(smoke=False))

    rd = args.results_dir
    if args.smoke and rd == "results":
        rd = "results_smoke"  # never mix 2-epoch dry-run output with the real results
        print("Smoke mode: writing to results_smoke/ (delete it afterwards)")
    os.makedirs(f"{rd}/logs", exist_ok=True)
    for d in ("checkpoints", "plots"):
        os.makedirs(d, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    if dev == "cpu" and not args.smoke:
        print("WARNING: no GPU detected. The full set of runs will take many hours on CPU. Use Colab (GPU runtime).")

    t_all = time.time()
    failures = []

    def do_train(r):
        out = f"{rd}/{r['tag']}.json"
        if os.path.exists(out) and not args.force:
            print(f"[skip] {r['tag']} (done)")
            return
        print(f"[run ] {r['tag']}: {' '.join(train_cmd(r, rd)[1:])}", flush=True)
        t0 = time.time()
        rc = run_logged(train_cmd(r, rd, python=sys.executable), f"{rd}/logs/{r['tag']}.log")
        print(f"[done] {r['tag']} rc={rc} in {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            failures.append(r["tag"])

    for stage in ("core", "tiny"):
        if stage in args.stages:
            print(f"\n===== STAGE: {stage} =====")
            for r in runs:
                if r["stage"] == stage:
                    do_train(r)

    if "tools" in args.stages:
        print("\n===== STAGE: tools =====")
        rc = run_logged([sys.executable, "demo_transforms.py"], f"{rd}/demo_transforms.txt", quiet=True)
        print(f"[tool] demo_transforms.py rc={rc}")
        rc = run_logged([sys.executable, "compare_models.py", "--dataset", "cifar10", "--device", dev,
                         "--json", f"{rd}/compare_models.json"], f"{rd}/compare_models.txt", quiet=True)
        print(f"[tool] compare_models.py rc={rc}")
        if rc != 0:
            failures.append("compare_models")
        if not args.smoke:
            rc = run_logged([sys.executable, "-m", "utils.eda", "--dataset", "both"], f"{rd}/logs/eda.log", quiet=True)
            print(f"[tool] utils.eda rc={rc}")
            if rc != 0:
                failures.append("eda")

    for stage in ("seeds", "extended"):
        if stage in args.stages:
            print(f"\n===== STAGE: {stage} =====")
            for r in runs:
                if r["stage"] == stage:
                    do_train(r)

    if "plots" in args.stages:
        print("\n===== STAGE: plots =====")
        rc = run_logged([sys.executable, "-m", "utils.visualization", "--plot-ablation", "--results-dir", rd],
                        f"{rd}/logs/ablation_plot.log", quiet=True)
        print(f"[tool] utils.visualization rc={rc}")

    print(f"\nAll requested stages finished in {(time.time() - t_all) / 60:.1f} min.")
    if failures:
        print("FAILED:", ", ".join(failures), "(see results/logs/<tag>.log)")
        sys.exit(1)


if __name__ == "__main__":
    main()
