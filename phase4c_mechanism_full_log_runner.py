#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4c mechanism full-log runner.

Purpose
-------
Generate the step-level logs needed for mechanism analyses that cannot be
performed from episode-level CSVs alone.

This script does not change the Phase 4c theoretical model. It imports the fixed
Phase 4c core script and reruns selected conditions while logging:
- full step/agent rows;
- selected anonymous channel per episode, discovered from the first half only;
- held-out selected-channel signal indicator;
- deterministic signal probabilities for every anonymous channel;
- counterfactual probabilities with past self-threat appraisal removed or swapped.

Strict constraints preserved from Phase 4c
-----------------------------------------
- No LAUGH/HUMOR action label.
- No benign-violation variable used by the controller.
- No laughter reward, no prompt, no LMM/API.
- Relief, SafeSurprise, and SelfAppraisalGap remain analysis variables.
- Counterfactual quantities are recorded for mechanism analysis only; they are not
  fed back into the simulated controller.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9


def load_core(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"core script not found: {p}")
    spec = importlib.util.spec_from_file_location("phase4c_core", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def deterministic_probs(core, cfg, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Signal probabilities without exploration or random noise."""
    risk_inhibition = 0.55 * features[3] + 0.42 * features[4]
    logits = weights @ features + (cfg.signal_bias + cfg.perturb_signal_bias) - risk_inhibition
    return sigmoid_np(logits)


def counterfactual_feature_variants(features: np.ndarray, actual_risk: float) -> Dict[str, np.ndarray]:
    """Create counterfactual variants that alter only past self-appraisal terms.

    Feature indices follow Phase 4c feature_vector:
    [0] intercept, [1] q, [2] pe, [3] actual_risk, [4] damage,
    [5] last_appraisal/past_threat, [6] relief_raw, ...
    """
    variants = {}

    f0 = features.copy()
    f0[5] = 0.0
    f0[6] = 0.0
    variants["no_past"] = f0

    flow = features.copy()
    low = 0.05
    flow[5] = low
    flow[6] = max(0.0, low - actual_risk)
    variants["low_past"] = flow

    fhigh = features.copy()
    high = 0.85
    fhigh[5] = high
    fhigh[6] = max(0.0, high - actual_risk)
    variants["high_past"] = fhigh

    # Same current state, but remove the reflective relief term while retaining
    # the stored past appraisal. This separates stored appraisal from relief.
    fnorelief = features.copy()
    fnorelief[6] = 0.0
    variants["no_relief"] = fnorelief

    return variants


def run_episode_mechanism(core, cfg, condition: str, risk_regime: str, seed: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    rng = core.make_rng(seed)
    agents = [core.AgentState(rng, cfg) for _ in range(cfg.n_agents)]
    n_features = 14
    signal_weights = core.make_initial_signal_weights(rng, cfg, n_features)
    pred_int = np.full(len(core.EVENT_NAMES), 0.35, dtype=float)
    pred_risk = np.full(len(core.EVENT_NAMES), 0.20, dtype=float)

    rows = []
    terminated = False

    for t in range(cfg.steps):
        if all((a.integrity <= 0.05 or a.energy <= 0.05) for a in agents):
            terminated = True
            break
        i = int(rng.integers(0, cfg.n_agents))
        a = agents[i]
        if a.integrity <= 0.05 or a.energy <= 0.05:
            continue

        event = core.sample_event(rng, risk_regime)
        out = core.event_outcome(event, risk_regime, rng, cfg)
        ei = core.EVENT_INDEX[event]
        pred_before = float(pred_int[ei])
        q_before = core.compute_q(a, out["actual_risk"], out["damage"], cfg, condition)
        feats = core.feature_vector(a, out, pred_before, q_before, condition)

        # Mechanism-only deterministic probabilities and counterfactual probabilities.
        probs_det = deterministic_probs(core, cfg, feats, signal_weights)
        cf_probs = {}
        for name, fcf in counterfactual_feature_variants(feats, out["actual_risk"]).items():
            cf_probs[name] = deterministic_probs(core, cfg, fcf, signal_weights)

        channel, probs_noisy = core.select_signal(a, feats, signal_weights, rng, cfg, condition, event)
        emitted = channel >= 0
        if emitted:
            a.refractory = cfg.refractory_steps
        elif a.refractory > 0:
            a.refractory -= 1

        own_tension_before = a.social_tension
        own_explore_before = a.exploration_drive
        receiver_recovery, spread, n_heard = core.apply_signal_to_listeners(agents, i, channel, out, cfg, condition, rng)
        core.update_body(a, out, q_before, event, cfg, condition, emitted)
        q_after = core.compute_q(a, out["actual_risk"], out["damage"], cfg, condition)
        own_recovery = (own_tension_before - a.social_tension) + 0.5 * (a.exploration_drive - own_explore_before)
        total_benefit = own_recovery + 0.8 * receiver_recovery + 0.003 * spread
        core.update_signal_learning(a, feats, signal_weights, channel, total_benefit, out, cfg, condition)
        if channel >= 0 and condition != "private_signal":
            for j, d in core.distances(agents, i):
                if d <= cfg.signal_radius:
                    core.update_listener_learning(agents[j], channel, receiver_recovery, out, cfg, condition)

        lr = cfg.predictor_lr
        pred_int[ei] = (1 - lr) * pred_int[ei] + lr * out["intensity"]
        pred_risk[ei] = (1 - lr) * pred_risk[ei] + lr * out["actual_risk"]

        av = core.analysis_variables(a, out, pred_before, q_before, q_after, cfg, condition)
        row = {
            "episode_id": f"{condition}|{risk_regime}|{seed}",
            "t": int(t),
            "agent": int(i),
            "condition": condition,
            "risk_regime": risk_regime,
            "seed": int(seed),
            "event": event,
            "selected_signal": int(channel),
            "signal_emitted": float(emitted),
            "receiver_recovery": float(receiver_recovery),
            "cross_agent_spread": float(spread),
            "n_heard": float(n_heard),
            "own_recovery": float(own_recovery),
            "q_before": float(q_before),
            "q_after": float(q_after),
            "actual_risk": float(out["actual_risk"]),
            "damage": float(out["damage"]),
            "initial_appraisal": float(out["initial_appraisal"]),
            "integrity": float(a.integrity),
            "energy": float(a.energy),
            "social_tension": float(a.social_tension),
            "social_sync": float(a.social_sync),
            "exploration_drive": float(a.exploration_drive),
            "pred_intensity_before": float(pred_before),
            "prob_noise_max": float(np.max(probs_noisy)) if len(probs_noisy) else 0.0,
        }
        row.update(av)
        for ch in range(cfg.n_signals):
            row[f"prob_ch{ch}"] = float(probs_det[ch])
            row[f"cf_no_past_prob_ch{ch}"] = float(cf_probs["no_past"][ch])
            row[f"cf_low_past_prob_ch{ch}"] = float(cf_probs["low_past"][ch])
            row[f"cf_high_past_prob_ch{ch}"] = float(cf_probs["high_past"][ch])
            row[f"cf_no_relief_prob_ch{ch}"] = float(cf_probs["no_relief"][ch])
        rows.append(row)

        a.last_appraisal = out["initial_appraisal"]
        a.last_actual_risk = out["actual_risk"]
        a.last_q = q_after
        a.x = float(np.clip(a.x + rng.normal(0, 0.18), 0, 5))
        a.y = float(np.clip(a.y + rng.normal(0, 0.18), 0, 5))

    df = pd.DataFrame(rows)
    if df.empty:
        ep = core.empty_episode_summary(condition, risk_regime, seed, terminated)
        return ep, df

    split_t = df["t"].max() * cfg.discovery_fraction
    first = df[df["t"] <= split_t]
    selected_channel = core.choose_channel_discovery(first, cfg)
    df["selected_channel"] = int(selected_channel)
    if selected_channel >= 0:
        df["selected_channel_signal"] = (df["selected_signal"] == selected_channel).astype(float)
        df["actual_prob_selected"] = df[f"prob_ch{selected_channel}"]
        df["cf_no_past_prob_selected"] = df[f"cf_no_past_prob_ch{selected_channel}"]
        df["cf_low_past_prob_selected"] = df[f"cf_low_past_prob_ch{selected_channel}"]
        df["cf_high_past_prob_selected"] = df[f"cf_high_past_prob_ch{selected_channel}"]
        df["cf_no_relief_prob_selected"] = df[f"cf_no_relief_prob_ch{selected_channel}"]
        df["cf_high_minus_low_prob_selected"] = df["cf_high_past_prob_selected"] - df["cf_low_past_prob_selected"]
        df["cf_actual_minus_no_past_prob_selected"] = df["actual_prob_selected"] - df["cf_no_past_prob_selected"]
        df["cf_actual_minus_no_relief_prob_selected"] = df["actual_prob_selected"] - df["cf_no_relief_prob_selected"]
    else:
        df["selected_channel_signal"] = 0.0
        for c in ["actual_prob_selected", "cf_no_past_prob_selected", "cf_low_past_prob_selected", "cf_high_past_prob_selected", "cf_no_relief_prob_selected", "cf_high_minus_low_prob_selected", "cf_actual_minus_no_past_prob_selected", "cf_actual_minus_no_relief_prob_selected"]:
            df[c] = 0.0

    ep = core.summarize_episode(df, condition, risk_regime, seed, terminated, agents)
    ep["selected_channel"] = int(selected_channel)
    # additional mechanism summary at episode level
    test = df[df["t"] > split_t].copy()
    if not test.empty:
        sig = test["selected_channel_signal"] > 0.5
        ep["cf_high_minus_low_prob_selected"] = float(test["cf_high_minus_low_prob_selected"].mean())
        ep["cf_actual_minus_no_past_prob_selected"] = float(test["cf_actual_minus_no_past_prob_selected"].mean())
        ep["cf_actual_minus_no_relief_prob_selected"] = float(test["cf_actual_minus_no_relief_prob_selected"].mean())
        ep["mean_actual_prob_selected"] = float(test["actual_prob_selected"].mean())
        ep["future_rows"] = float(len(test))
    else:
        ep["cf_high_minus_low_prob_selected"] = 0.0
        ep["cf_actual_minus_no_past_prob_selected"] = 0.0
        ep["cf_actual_minus_no_relief_prob_selected"] = 0.0
        ep["mean_actual_prob_selected"] = 0.0
        ep["future_rows"] = 0.0
    return ep, df


def mode_params(mode: str) -> Tuple[int, int]:
    if mode == "smoke":
        return 3, 350
    if mode == "quick":
        return 10, 1000
    if mode == "full":
        return 50, 2500
    raise ValueError(mode)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-script", default="~/Desktop/phase4c_self_appraisal_gap_multiagent_core_viability_fixed.py")
    ap.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--base-seed", type=int, default=9000)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--conditions", default="full,no_self_appraisal,random_signal,label_rule_signal,no_listener_learning,no_signal_learning,private_signal")
    ap.add_argument("--risk-regimes", default="mild,moderate,harsh")
    ap.add_argument("--compress", action="store_true", help="also write gzipped full log")
    return ap.parse_args()


def main():
    args = parse_args()
    core = load_core(args.core_script)
    n_seeds, default_steps = mode_params(args.mode)
    cfg = core.Config(steps=args.steps if args.steps is not None else default_steps, base_seed=args.base_seed)
    outdir = Path(args.outdir or f"phase4c_mechanism_logs_{args.mode}").expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    conds = [x.strip() for x in args.conditions.split(",") if x.strip()]
    risks = [x.strip() for x in args.risk_regimes.split(",") if x.strip()]

    ep_rows = []
    log_parts = []
    total = len(conds) * len(risks) * n_seeds
    k = 0
    for condition in conds:
        for risk in risks:
            for s in range(n_seeds):
                seed = cfg.base_seed + 100000 * core.CONDITIONS.index(condition) + 1000 * core.RISK_REGIMES.index(risk) + s
                ep, df = run_episode_mechanism(core, cfg, condition, risk, seed)
                ep_rows.append(ep)
                if not df.empty:
                    log_parts.append(df)
                k += 1
                print(f"[{k}/{total}] {condition} {risk} seed={s} selected_rate={ep.get('selected_signal_rate',0):.4f} self_gap={ep.get('self_appraisal_gap_association',0):.4f} viable={ep.get('viable_fraction',0):.3f}", flush=True)
                pd.DataFrame(ep_rows).to_csv(outdir / "episode_results_mechanism.csv", index=False)

    ep_df = pd.DataFrame(ep_rows)
    ep_df.to_csv(outdir / "episode_results_mechanism.csv", index=False)
    step_df = pd.concat(log_parts, ignore_index=True) if log_parts else pd.DataFrame()
    step_df.to_csv(outdir / "step_logs_mechanism.csv", index=False)
    if args.compress and not step_df.empty:
        step_df.to_csv(outdir / "step_logs_mechanism.csv.gz", index=False, compression="gzip")

    # summaries
    num_cols = ep_df.select_dtypes(include=[np.number]).columns.tolist()
    ep_df.groupby("condition")[num_cols].agg(["mean", "std", "sem"]).to_csv(outdir / "condition_summary_mechanism.csv")
    ep_df.groupby(["condition", "risk_regime"])[num_cols].agg(["mean", "std", "sem"]).to_csv(outdir / "condition_by_risk_summary_mechanism.csv")
    report = [
        "Phase 4c mechanism full-log run",
        "================================",
        "Core script: " + str(Path(args.core_script).expanduser()),
        "Configuration:", json.dumps(asdict(cfg), indent=2), "",
        "Conditions: " + ", ".join(conds),
        "Risk regimes: " + ", ".join(risks),
        f"Rows written: {len(step_df)}",
        "Generated files:",
        "- step_logs_mechanism.csv",
        "- episode_results_mechanism.csv",
        "- condition_summary_mechanism.csv",
        "- condition_by_risk_summary_mechanism.csv",
    ]
    (outdir / "mechanism_log_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"[done] outputs written to {outdir}")


if __name__ == "__main__":
    main()
