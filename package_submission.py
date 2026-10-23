"""
ELEC 475: Lab 2 Post-Lab - Build <student_number>.zip in the layout required by the handout (no data/ folder).

  python package_submission.py --student-number 12345678            (or 11111111_22222222 for pairs)
Needs: ELEC475_Lab2_Report.pdf (python make_report.py ...), github_repo.txt (one line, your repo URL), commands.txt (python run_all.py --write-commands).
"""
import argparse
import os
import shutil
import sys
import zipfile

REQUIRED = [
    "models/__init__.py", "models/lenet.py", "models/lenet_modern.py", "models/alexnet.py", "models/zoo.py",
    "utils/__init__.py", "utils/dataset.py", "utils/eda.py", "utils/transforms.py", "utils/metrics.py",
    "utils/visualization.py", "utils/profiler.py", "utils/seed.py", "configs/config.py",
    "checkpoints/lenet_baseline.pth", "checkpoints/lenet_modern.pth", "checkpoints/alexnet_bn.pth",
    "plots/loss_lenet.png", "plots/loss_lenet_modern.png", "plots/loss_alexnet_bn.png", "plots/ablation_summary.png",
    "demo_transforms.py", "compare_models.py", "train.py", "evaluate.py", "github_repo.txt", "commands.txt",
    "ELEC475_Lab2_Report.pdf",
]
EXTRA_FILES = ["run_all.py", "make_report.py", "experiments.py", "package_submission.py", "requirements.txt", "README.md", "prelab_verify.py", ".gitignore"]
KEEP_CKPT = {"lenet_baseline.pth", "lenet_modern.pth", "alexnet_bn.pth"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-number", required=True)
    a = ap.parse_args()
    num = a.student_number
    if os.path.exists("github_repo.txt") is False:
        print("MISSING github_repo.txt. Create it with your repository URL on one line.")
    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing:
        print("MISSING required files:\n  " + "\n  ".join(missing))
        sys.exit(1)
    url = open("github_repo.txt").read().strip()
    if not url.startswith("http") or "github" not in url.lower():
        print(f"github_repo.txt does not look like a GitHub URL: '{url}'")
        sys.exit(1)

    stage = os.path.join(".submission_tmp", num)
    shutil.rmtree(".submission_tmp", ignore_errors=True)
    os.makedirs(stage)
    for d in ("models", "utils", "configs", "plots", "results"):
        if os.path.isdir(d):
            shutil.copytree(d, os.path.join(stage, d), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "figs"))
    os.makedirs(os.path.join(stage, "checkpoints"))
    for f in KEEP_CKPT:
        shutil.copy(os.path.join("checkpoints", f), os.path.join(stage, "checkpoints", f))
    for f in REQUIRED + EXTRA_FILES:
        if os.path.isfile(f) and "/" not in f:
            shutil.copy(f, os.path.join(stage, f))

    out = f"{num}.zip"
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(stage):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, ".submission_tmp"))
    shutil.rmtree(".submission_tmp", ignore_errors=True)
    mb = os.path.getsize(out) / 1e6
    print(f"\nCreated {out} ({mb:.1f} MB). Contents check:")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        bad = [n for n in names if "/data/" in n or "__pycache__" in n or n.endswith(".zip")]
        for f in REQUIRED:
            print(("  ok  " if f"{num}/{f}" in names else "  MISSING  ") + f)
        if bad:
            print("  WARNING unexpected entries:", bad[:5])
    print("\nUpload this single file to OnQ before October 22, 2026 23:59 ET.")


if __name__ == "__main__":
    main()
