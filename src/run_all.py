"""Regenerate every result and figure in order. One command:

    python src/run_all.py            # everything (about 2.5 hours on an RTX 3050 laptop)
    python src/run_all.py --quick    # skips the slow steps (personalisation, GRU) for a fast check
    python src/run_all.py --only forecast_baselines plot_baselines

Prerequisites: the datasets are downloaded to MBZUAI_DATA (default D:\\MBZUAI_data), see README and
src/download_datasets.py. All seeds are fixed in the scripts; results land in ../results and ../figures.
Each step's stdout is saved to ../results/logs/<step>.log and the wall time is recorded.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOGS = os.path.join(RES, "logs")

# (script, slow?)  Order matters: later steps read earlier outputs.
STEPS = [
    ("cgmacros_loader", False),         # builds D:\MBZUAI_data\processed\cgmacros_all.parquet
    ("eda_cgmacros", False),
    ("sensor_interpolation_check", False),
    ("sensor_offset_check", False),
    ("sensor_offset_by_day", False),
    ("forecast_baselines", False),
    ("plot_baselines", False),
    ("external_loaders", False),        # writes results/external_cohort_summary.json
    ("forecast_external", False),
    ("plot_external", False),
    ("forecast_ladder", False),
    ("plot_ladder", False),
    ("forecast_microbiome", False),
    ("microbiome_control_check", False),
    ("forecast_meals_rich", False),
    ("forecast_personal", True),        # about 22 min
    ("plot_personal", False),
    ("forecast_personal_meals", True),  # about 33 min
    ("forecast_gru", True),             # about 9 min on the GPU
    ("export_tables", False),
    ("derived_numbers", False),         # ratios and percentages quoted in the text
    ("build_paper", False),             # strips the {{tags}} from paper/paper_tagged.md
    ("audit_numbers", False),           # checks the manuscript numbers against results/
    ("locate_numbers", False),          # stricter check: each number against the file its tag names
    ("build_docs", False),              # Word and PDF versions (needs Microsoft Edge)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip slow steps")
    ap.add_argument("--only", nargs="*", help="run only these steps")
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE")
    timings = {}
    for name, slow in STEPS:
        if args.only and name not in args.only:
            continue
        if args.quick and slow and not args.only:
            print(f"-- skip (slow) {name}")
            continue
        if not os.path.exists(os.path.join(HERE, f"{name}.py")):
            print(f"-- skip {name} (script not in this repository)")
            continue
        t0 = time.time()
        print(f"-- running {name} ...", flush=True)
        with open(os.path.join(LOGS, f"{name}.log"), "w", encoding="utf-8") as log:
            rc = subprocess.run([sys.executable, os.path.join(HERE, f"{name}.py")], cwd=HERE, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        timings[name] = dict(seconds=round(time.time() - t0, 1), returncode=rc)
        print(f"   done in {timings[name]['seconds']} s (exit {rc})", flush=True)
        if rc != 0:
            print(f"!! {name} failed, see results/logs/{name}.log")
            break
    json.dump(timings, open(os.path.join(RES, "run_all_timings.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
