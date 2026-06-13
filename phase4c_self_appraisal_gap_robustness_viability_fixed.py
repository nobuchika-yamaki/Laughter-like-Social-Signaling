#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness runner for Phase 4c calibrated self-appraisal-gap model.
Runs a predefined sensitivity battery and summarizes whether key signs persist.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

SETTINGS = [
    ("default", {}),
    ("threshold_low", {"--perturb-signal-threshold": "-0.035"}),
    ("threshold_high", {"--perturb-signal-threshold": "0.035"}),
    ("bias_low", {"--perturb-signal-bias": "-0.10"}),
    ("bias_high", {"--perturb-signal-bias": "0.10"}),
    ("signal_lr_low", {"--perturb-signal-lr-mult": "0.70"}),
    ("signal_lr_high", {"--perturb-signal-lr-mult": "1.30"}),
    ("listener_lr_low", {"--perturb-listener-lr-mult": "0.70"}),
    ("listener_lr_high", {"--perturb-listener-lr-mult": "1.30"}),
    ("social_low", {"--perturb-social-scale-mult": "0.75"}),
    ("social_high", {"--perturb-social-scale-mult": "1.25"}),
    ("risk_low", {"--perturb-event-risk-mult": "0.85"}),
    ("risk_high", {"--perturb-event-risk-mult": "1.12"}),
    ("listener_effect_low", {"--perturb-listener-effect-mult": "0.75"}),
    ("listener_effect_high", {"--perturb-listener-effect-mult": "1.25"}),
    ("independent_seed", {"--base-seed": "9100"}),
]

KEYS = [
    "emergent_function_score",
    "selected_signal_rate",
    "benign_selectivity",
    "danger_signal_rate",
    "danger_suppression",
    "relief_association",
    "safe_surprise_association",
    "self_appraisal_gap_association",
    "social_recovery_score",
    "receiver_recovery_score",
    "cross_agent_spread",
    "viable_fraction",
    "safe_context_fraction",
    "danger_context_fraction",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    ap.add_argument("--core-script", default="phase4c_self_appraisal_gap_multiagent_core_viability_fixed.py")
    ap.add_argument("--outdir", default="phase4c_self_appraisal_gap_robustness")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only-full-condition", action="store_true", default=True)
    return ap.parse_args()


def run_one(args, name, flags):
    out = Path(args.outdir).expanduser() / name
    out.mkdir(parents=True, exist_ok=True)
    core = Path(args.core_script).expanduser()
    cmd = [args.python, str(core), "--mode", args.mode, "--outdir", str(out), "--resume"]
    if args.only_full_condition:
        cmd += ["--conditions", "full,no_self_appraisal,random_signal,label_rule_signal"]
    for k, v in flags.items():
        cmd += [k, v]
    print("[run]", name, " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    ep = pd.read_csv(out / "episode_results.csv")
    full = ep[ep["condition"] == "full"]
    row = {"setting": name, "n_episodes": len(full)}
    for k in KEYS:
        row[k] = float(full[k].mean()) if k in full.columns and len(full) else np.nan
    mc_path = out / "model_comparison.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        cm = mc[(mc["condition"] == "full") & (mc["model"] == "controlled")]
        if len(cm):
            for c in ["coef_relief", "coef_safe_surprise", "coef_self_appraisal_gap", "pseudo_r2"]:
                row[f"controlled_{c}"] = float(cm.iloc[0].get(c, np.nan))
    return row


def main():
    args = parse_args()
    Path(args.outdir).expanduser().mkdir(parents=True, exist_ok=True)
    rows = []
    for name, flags in SETTINGS:
        try:
            row = run_one(args, name, flags)
            rows.append(row)
        except subprocess.CalledProcessError as e:
            print(f"[error] {name}: {e}", flush=True)
            rows.append({"setting": name, "error": str(e)})
    df = pd.DataFrame(rows)
    outdir = Path(args.outdir).expanduser()
    df.to_csv(outdir / "robustness_summary.csv", index=False)

    # sign-stability summary
    lines = []
    lines.append("Phase 4c robustness summary")
    lines.append("===========================")
    ok = df.dropna(subset=["self_appraisal_gap_association", "receiver_recovery_score", "danger_suppression", "selected_signal_rate"], how="any")
    lines.append(f"Completed settings: {len(ok)} / {len(SETTINGS)}")
    if len(ok):
        lines.append(f"self_appraisal_gap_association positive: {(ok['self_appraisal_gap_association'] > 0).sum()} / {len(ok)}")
        lines.append(f"receiver_recovery_score positive: {(ok['receiver_recovery_score'] > 0).sum()} / {len(ok)}")
        lines.append(f"danger_suppression > 0.95: {(ok['danger_suppression'] > 0.95).sum()} / {len(ok)}")
        lines.append(f"selected_signal_rate > 0: {(ok['selected_signal_rate'] > 0).sum()} / {len(ok)}")
        if "controlled_coef_self_appraisal_gap" in ok.columns:
            lines.append(f"controlled coef_self_appraisal_gap positive: {(ok['controlled_coef_self_appraisal_gap'] > 0).sum()} / {len(ok)}")
    (outdir / "robustness_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("[done]", outdir / "robustness_summary.csv")


if __name__ == "__main__":
    main()
