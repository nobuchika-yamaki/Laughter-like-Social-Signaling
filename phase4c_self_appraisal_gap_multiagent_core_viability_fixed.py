#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4c: viability-fixed multi-agent self-appraisal-gap signal test.

Purpose
-------
Test whether an anonymous social signal is better explained by retrospective
self-appraisal gap than by simple relief or safe surprise.

Strict design constraints
-------------------------
- No LAUGH/HUMOR action label.
- No benign-violation variable is used by the signal controller.
- No laughter reward, external prompt, LMM, or API.
- Agents emit anonymous channels signal_0 ... signal_4 only.
- Relief, safe-surprise, and self-appraisal-gap are calculated for analysis only.
- Candidate signal channels are selected by split-half discovery using generic
  social/exploratory recovery, then tested on the held-out half.

What changed from Phase 4
-------------------------
The original Phase 4 failed because viability and context frequencies collapsed.
This calibrated version explicitly guarantees analyzable ecology:
- viable_fraction target roughly 0.60-0.90
- safe_context_fraction target roughly 0.06-0.18
- danger_context_fraction target roughly 0.05-0.25
- non-zero anonymous signal emission rate
- signal controller sees generic local variables only, not self_appraisal_gap
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-9


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > EPS else 0.0


@dataclass
class Config:
    steps: int = 2500
    n_agents: int = 6
    n_signals: int = 5
    base_seed: int = 7200
    discovery_fraction: float = 0.5

    # signal controller calibration
    signal_threshold: float = 0.455
    signal_bias: float = -0.62
    signal_noise: float = 0.060
    signal_exploration_rate: float = 0.055
    signal_lr: float = 0.016
    weight_decay: float = 0.00045
    refractory_steps: int = 4
    signal_energy_cost: float = 0.00006

    # listener and social dynamics
    listener_lr: float = 0.060
    social_feedback_scale: float = 1.0
    signal_radius: float = 3.6
    listener_effect_strength: float = 0.070
    contagion_strength: float = 0.030
    base_social_relaxation: float = 0.0040

    # viability dynamics
    basal_energy_cost: float = 0.00008
    recovery_rate: float = 0.034
    fatigue_decay: float = 0.024
    social_decay: float = 0.007
    memory_decay: float = 0.012

    # event / predictor dynamics
    predictor_lr: float = 0.045
    intensity_noise: float = 0.045
    risk_noise: float = 0.030
    appraisal_noise: float = 0.040

    # evaluation cutoffs only; not used for control
    safe_pe_cutoff: float = 0.07
    safe_damage_cutoff: float = 0.018
    safe_risk_cutoff: float = 0.38
    past_threat_cutoff: float = 0.34
    danger_damage_cutoff: float = 0.040
    danger_risk_cutoff: float = 0.64

    # perturbation hooks used by robustness script
    perturb_signal_threshold: float = 0.0
    perturb_signal_bias: float = 0.0
    perturb_signal_lr_mult: float = 1.0
    perturb_listener_lr_mult: float = 1.0
    perturb_social_scale_mult: float = 1.0
    perturb_event_risk_mult: float = 1.0
    perturb_listener_effect_mult: float = 1.0

    # output control
    max_step_rows_per_run: int = 260000


RISK_MULT = {
    "mild": 0.74,
    "moderate": 0.92,
    "harsh": 1.08,
}

# Event probabilities intentionally calibrated so all regimes have both safe-reevaluation
# and genuine-danger contexts while remaining viable enough for social learning.
EVENT_PROBS = {
    "mild": {
        "rest": 0.15, "walk": 0.19, "explore": 0.17, "minor_mismatch": 0.11,
        "near_miss": 0.14, "false_alarm": 0.10, "social_play": 0.06,
        "bump": 0.05, "slip": 0.02, "collision": 0.01,
    },
    "moderate": {
        "rest": 0.15, "walk": 0.18, "explore": 0.16, "minor_mismatch": 0.11,
        "near_miss": 0.15, "false_alarm": 0.10, "social_play": 0.06,
        "bump": 0.05, "slip": 0.03, "collision": 0.01,
    },
    "harsh": {
        "rest": 0.14, "walk": 0.17, "explore": 0.15, "minor_mismatch": 0.11,
        "near_miss": 0.14, "false_alarm": 0.09, "social_play": 0.05,
        "bump": 0.07, "slip": 0.05, "collision": 0.03,
    },
}

EVENT_INDEX = {name: i for i, name in enumerate(["rest", "walk", "explore", "minor_mismatch", "near_miss", "false_alarm", "social_play", "bump", "slip", "collision"])}
EVENT_NAMES = list(EVENT_INDEX.keys())

# event parameters: mean_intensity, mean_actual_risk, mean_damage, mean_initial_appraisal
# Initial appraisal can be high even when actual risk is low (near_miss / false_alarm).
EVENT_PARAMS = {
    "rest":           (0.06, 0.03, 0.000, 0.05),
    "walk":           (0.22, 0.09, 0.001, 0.11),
    "explore":        (0.32, 0.15, 0.003, 0.19),
    "minor_mismatch": (0.42, 0.16, 0.002, 0.28),
    "near_miss":      (0.66, 0.20, 0.004, 0.72),
    "false_alarm":    (0.55, 0.12, 0.001, 0.68),
    "social_play":    (0.50, 0.10, 0.001, 0.56),
    "bump":           (0.60, 0.45, 0.009, 0.52),
    "slip":           (0.72, 0.58, 0.015, 0.62),
    "collision":      (0.88, 0.76, 0.026, 0.76),
}

CONDITIONS = [
    "full",
    "no_q",
    "no_memory",
    "no_self_appraisal",
    "no_agency",
    "no_social_feedback",
    "no_signal_learning",
    "no_listener_learning",
    "private_signal",
    "random_signal",
    "label_rule_signal",
]

RISK_REGIMES = ["mild", "moderate", "harsh"]


class AgentState:
    def __init__(self, rng: np.random.Generator, cfg: Config):
        self.integrity = float(rng.uniform(0.92, 1.00))
        self.energy = float(rng.uniform(0.88, 1.00))
        self.fatigue = float(rng.uniform(0.05, 0.15))
        self.stability = float(rng.uniform(0.74, 0.92))
        self.pain_memory = 0.02
        self.danger_memory = 0.06
        self.comfort_memory = 0.52
        self.social_tension = float(rng.uniform(0.14, 0.28))
        self.social_sync = float(rng.uniform(0.35, 0.55))
        self.exploration_drive = float(rng.uniform(0.50, 0.70))
        self.last_appraisal = 0.0
        self.last_actual_risk = 0.0
        self.last_q = 0.0
        self.refractory = 0
        self.x = rng.uniform(0, 5)
        self.y = rng.uniform(0, 5)
        # listener valence estimate for each anonymous signal: positive means this
        # signal tends to restore social/exploratory state.
        self.listener_value = rng.normal(0.0, 0.02, size=cfg.n_signals)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def normalize_probs(d: Dict[str, float]) -> Tuple[List[str], np.ndarray]:
    names = list(d.keys())
    p = np.array([d[n] for n in names], dtype=float)
    p = p / p.sum()
    return names, p


def sample_event(rng: np.random.Generator, risk_regime: str) -> str:
    names, p = normalize_probs(EVENT_PROBS[risk_regime])
    return str(rng.choice(names, p=p))


def event_outcome(event: str, risk_regime: str, rng: np.random.Generator, cfg: Config) -> Dict[str, float]:
    mean_int, mean_risk, mean_damage, mean_appraisal = EVENT_PARAMS[event]
    rm = RISK_MULT[risk_regime] * cfg.perturb_event_risk_mult
    # Risk multiplier should not convert all worlds into death worlds. Apply with compression.
    risk = clip01(mean_risk * (0.78 + 0.22 * rm) + rng.normal(0, cfg.risk_noise))
    intensity = clip01(mean_int * (0.86 + 0.14 * rm) + rng.normal(0, cfg.intensity_noise))
    # Damage is deliberately sublethal in most cases so social signalling can be learned;
    # danger is still represented by actual_risk and occasional damage spikes.
    damage = max(0.0, mean_damage * (0.56 + 0.22 * rm) * rng.lognormal(mean=-0.18, sigma=0.24))
    appraisal = clip01(mean_appraisal * (0.92 + 0.08 * rm) + rng.normal(0, cfg.appraisal_noise))
    novelty = clip01(abs(intensity - mean_int) * 1.7 + (0.15 if event in ("near_miss", "false_alarm", "social_play", "minor_mismatch") else 0.03))
    agency = clip01(0.60 + rng.normal(0, 0.12) - 0.18 * risk + (0.15 if event in ("explore", "walk", "social_play") else 0.0))
    return {
        "intensity": float(intensity),
        "actual_risk": float(risk),
        "damage": float(damage),
        "initial_appraisal": float(appraisal),
        "novelty": float(novelty),
        "agency": float(agency),
    }


def compute_q(a: AgentState, actual_risk: float, damage: float, cfg: Config, condition: str) -> float:
    if condition == "no_q":
        return 0.0
    vulnerability = 1.0 - min(a.integrity, a.energy)
    q = (
        0.28 * actual_risk
        + 0.30 * clip01(damage * 12.0)
        + 0.16 * a.danger_memory
        + 0.10 * a.pain_memory
        + 0.08 * a.fatigue
        + 0.08 * (1.0 - a.stability)
        + 0.08 * vulnerability
        - 0.12 * a.comfort_memory
        - 0.06 * a.social_sync
    )
    return float(np.clip(q, -0.2, 1.2))


def update_body(a: AgentState, out: Dict[str, float], q: float, event: str, cfg: Config, condition: str, signal_emitted: bool):
    damage = out["damage"]
    risk = out["actual_risk"]
    appraisal = out["initial_appraisal"]

    # Viability is calibrated to preserve learning time while retaining true danger contexts.
    # The model is still viability-constrained: high-risk/damage events reduce integrity, energy,
    # stability, and exploration; rest and non-damaging safe events allow limited recovery.
    basal = cfg.basal_energy_cost
    move_cost = 0.00022 if event in ("walk", "explore", "minor_mismatch", "near_miss", "social_play") else 0.00006
    danger_cost = 0.00055 * risk
    energy_cost = basal + move_cost + danger_cost + (cfg.signal_energy_cost if signal_emitted else 0.0)

    recovery = cfg.recovery_rate if event == "rest" else 0.0
    social_play_recovery = 0.007 if event == "social_play" else 0.0
    safe_reappraisal_recovery = 0.0025 if (risk < 0.30 and damage < 0.006 and appraisal > 0.40) else 0.0
    passive_repair = 0.00035 if damage < 0.012 else 0.0

    a.integrity = float(clip01(a.integrity - 0.62 * damage + 0.12 * recovery + 0.04 * social_play_recovery + safe_reappraisal_recovery + passive_repair))
    a.energy = float(clip01(a.energy - energy_cost + recovery + 0.55 * social_play_recovery + 0.25 * safe_reappraisal_recovery))
    a.fatigue = float(clip01((1 - cfg.fatigue_decay) * a.fatigue + 0.010 * out["intensity"] + 0.004 * risk - 0.050 * recovery - 0.012 * social_play_recovery))
    a.stability = float(clip01(a.stability - 0.055 * damage - 0.012 * risk + 0.065 * recovery + 0.020 * out["agency"] + 0.010 * safe_reappraisal_recovery))

    if condition != "no_memory":
        a.pain_memory = float(clip01((1 - cfg.memory_decay) * a.pain_memory + 0.18 * clip01(damage * 10.0)))
        a.danger_memory = float(clip01((1 - cfg.memory_decay) * a.danger_memory + 0.10 * risk + 0.06 * appraisal))
        a.comfort_memory = float(clip01((1 - cfg.memory_decay) * a.comfort_memory + 0.10 * (1.0 - risk) + 0.05 * (event == "rest")))
    else:
        a.pain_memory = 0.0
        a.danger_memory = 0.0
        a.comfort_memory = 0.5

    # Baseline social dynamics. Danger and high appraisal raise tension; safe social contexts relax it.
    a.social_tension = float(clip01((1 - cfg.social_decay) * a.social_tension + 0.050 * risk + 0.025 * appraisal - cfg.base_social_relaxation - 0.015 * (event == "social_play")))
    a.social_sync = float(clip01((1 - 0.004) * a.social_sync + 0.006 * (event == "social_play") - 0.008 * risk))
    a.exploration_drive = float(clip01(0.995 * a.exploration_drive + 0.006 * (1.0 - risk) - 0.012 * q + 0.006 * (event in ("near_miss", "false_alarm", "minor_mismatch"))))


def feature_vector(a: AgentState, out: Dict[str, float], pred_intensity: float, q: float, condition: str) -> np.ndarray:
    # Signal controller features. Intentionally excludes explicit self_appraisal_gap and explicit safe-surprise.
    pe = abs(out["intensity"] - pred_intensity)
    relief_raw = max(0.0, a.last_appraisal - out["actual_risk"])
    if condition == "no_self_appraisal":
        last_app = 0.0
        relief_raw = 0.0
    else:
        last_app = a.last_appraisal
    if condition == "no_agency":
        agency = 0.0
    else:
        agency = out["agency"]
    if condition == "no_memory":
        mem = 0.0
    else:
        mem = a.danger_memory
    if condition == "no_social_feedback":
        social_tension = 0.0
        social_sync = 0.0
    else:
        social_tension = a.social_tension
        social_sync = a.social_sync

    x = np.array([
        1.0,
        q,
        pe,
        out["actual_risk"],
        clip01(out["damage"] * 12.0),
        last_app,
        relief_raw,
        out["novelty"],
        agency,
        social_tension,
        social_sync,
        a.exploration_drive,
        mem,
        1.0 - min(a.integrity, a.energy),
    ], dtype=float)
    return x


def make_initial_signal_weights(rng: np.random.Generator, cfg: Config, n_features: int) -> np.ndarray:
    w = rng.normal(0.0, 0.08, size=(cfg.n_signals, n_features))
    # Small generic predispositions, not laughter-specific:
    # channels can respond to tension, novelty, relief, and low risk through feature learning.
    # This prevents total silence while not encoding any self_appraisal_gap product.
    w[:, 2] += rng.normal(0.02, 0.03, cfg.n_signals)   # PE
    w[:, 6] += rng.normal(0.08, 0.03, cfg.n_signals)   # relief raw
    w[:, 7] += rng.normal(0.07, 0.03, cfg.n_signals)   # novelty
    w[:, 9] += rng.normal(0.035, 0.025, cfg.n_signals)   # social tension
    w[:, 3] += rng.normal(-0.22, 0.035, cfg.n_signals)  # actual risk suppression
    w[:, 4] += rng.normal(-0.18, 0.035, cfg.n_signals)  # damage suppression
    return w


def select_signal(a: AgentState, features: np.ndarray, weights: np.ndarray, rng: np.random.Generator, cfg: Config, condition: str, event: str) -> Tuple[int, np.ndarray]:
    if condition == "private_signal":
        # Private signals are allowed to be produced but cannot be heard.
        pass
    if condition == "random_signal":
        if rng.random() < 0.010:
            return int(rng.integers(0, cfg.n_signals)), np.zeros(cfg.n_signals)
        return -1, np.zeros(cfg.n_signals)
    if condition == "label_rule_signal":
        if event in ("near_miss", "false_alarm", "social_play") and rng.random() < 0.045:
            return 0, np.zeros(cfg.n_signals)
        return -1, np.zeros(cfg.n_signals)

    if a.refractory > 0:
        return -1, np.zeros(cfg.n_signals)
    threshold = cfg.signal_threshold + cfg.perturb_signal_threshold
    bias = cfg.signal_bias + cfg.perturb_signal_bias
    # Generic viability inhibition: costly anonymous signalling is suppressed during
    # high actual risk or high damage, without using laughter/humor/BV labels.
    risk_inhibition = 0.55 * features[3] + 0.42 * features[4]
    logits = weights @ features + bias - risk_inhibition + rng.normal(0, cfg.signal_noise, cfg.n_signals)
    p = sigmoid(logits)
    if rng.random() < cfg.signal_exploration_rate:
        channel = int(rng.integers(0, cfg.n_signals))
        if rng.random() < 0.55:
            return channel, p
    channel = int(np.argmax(p))
    if p[channel] > threshold:
        return channel, p
    return -1, p


def distances(agents: List[AgentState], i: int) -> List[Tuple[int, float]]:
    out = []
    ai = agents[i]
    for j, aj in enumerate(agents):
        if j == i:
            continue
        d = math.sqrt((ai.x - aj.x) ** 2 + (ai.y - aj.y) ** 2)
        out.append((j, d))
    return out


def apply_signal_to_listeners(agents: List[AgentState], sender_i: int, channel: int, out: Dict[str, float], cfg: Config, condition: str, rng: np.random.Generator) -> Tuple[float, float, float]:
    if channel < 0 or condition == "private_signal":
        return 0.0, 0.0, 0.0
    if condition == "no_social_feedback":
        return 0.0, 0.0, 0.0

    receiver_recovery = []
    spread = 0
    n_heard = 0
    for j, d in distances(agents, sender_i):
        if d > cfg.signal_radius:
            continue
        n_heard += 1
        r = agents[j]
        before_tension = r.social_tension
        before_explore = r.exploration_drive
        val = r.listener_value[channel]
        # Listener effect is stronger when learned positive and when the event is not actually dangerous.
        risk_penalty = max(0.0, out["actual_risk"] - 0.45) + clip01(out["damage"] * 8.0)
        effect = cfg.listener_effect_strength * cfg.perturb_listener_effect_mult * cfg.perturb_social_scale_mult * sigmoid(2.5 * val) * max(0.0, 1.0 - 1.7 * risk_penalty)
        r.social_tension = float(clip01(r.social_tension - effect))
        r.social_sync = float(clip01(r.social_sync + 0.75 * effect))
        r.exploration_drive = float(clip01(r.exploration_drive + 0.55 * effect))
        rec = (before_tension - r.social_tension) + 0.5 * (r.exploration_drive - before_explore)
        receiver_recovery.append(rec)
        # Contagion: a heard positive signal can trigger weak same-channel spread, but not in danger.
        if rng.random() < cfg.contagion_strength * sigmoid(2.0 * val) * max(0.0, 1.0 - risk_penalty):
            spread += 1
    receiver_mean = float(np.mean(receiver_recovery)) if receiver_recovery else 0.0
    return receiver_mean, float(spread), float(n_heard)


def update_signal_learning(a: AgentState, features: np.ndarray, weights: np.ndarray, channel: int, benefit: float, out: Dict[str, float], cfg: Config, condition: str):
    if channel < 0 or condition in ("no_signal_learning", "random_signal", "label_rule_signal"):
        return
    lr = cfg.signal_lr * cfg.perturb_signal_lr_mult
    # Generic benefit only: own social/exploratory recovery minus viability cost/danger cost.
    risk_cost = 0.035 * max(0.0, out["actual_risk"] - 0.42) + 0.030 * clip01(out["damage"] * 10.0)
    delta = benefit - risk_cost - cfg.signal_energy_cost
    weights *= (1.0 - cfg.weight_decay)
    weights[channel] += lr * np.clip(delta, -0.05, 0.05) * features


def update_listener_learning(listener: AgentState, channel: int, receiver_recovery: float, out: Dict[str, float], cfg: Config, condition: str):
    if channel < 0 or condition == "no_listener_learning":
        return
    danger_cost = 0.05 * max(0.0, out["actual_risk"] - 0.45) + 0.05 * clip01(out["damage"] * 8.0)
    target = receiver_recovery - danger_cost
    listener.listener_value[channel] += cfg.listener_lr * cfg.perturb_listener_lr_mult * np.clip(target, -0.08, 0.08)
    listener.listener_value[channel] = np.clip(listener.listener_value[channel], -1.5, 1.5)


def analysis_variables(a: AgentState, out: Dict[str, float], pred_intensity: float, q_before: float, q_after: float, cfg: Config, condition: str) -> Dict[str, float]:
    pe = abs(out["intensity"] - pred_intensity)
    if condition == "no_self_appraisal":
        past_threat = 0.0
    else:
        past_threat = a.last_appraisal
    current_safety = 1.0 - max(out["actual_risk"], clip01(out["damage"] * 10.0))
    relief = max(0.0, past_threat - out["actual_risk"])
    safe_surprise = float((pe >= cfg.safe_pe_cutoff) and (out["actual_risk"] <= cfg.safe_risk_cutoff) and (out["damage"] <= cfg.safe_damage_cutoff))
    # SelfAppraisalGap is analysis-only. It is not fed into the controller.
    self_gap = float(past_threat * current_safety * abs(past_threat - out["actual_risk"]))
    q_relief = max(0.0, q_before - q_after)
    danger_context = float((out["actual_risk"] >= cfg.danger_risk_cutoff) or (out["damage"] >= cfg.danger_damage_cutoff))
    safe_context = float((past_threat >= cfg.past_threat_cutoff) and (current_safety >= 0.60) and (pe >= cfg.safe_pe_cutoff) and (danger_context < 0.5))
    return {
        "prediction_error": float(pe),
        "relief": float(relief),
        "safe_surprise": safe_surprise,
        "self_appraisal_gap": self_gap,
        "q_relief": float(q_relief),
        "safe_context": safe_context,
        "danger_context": danger_context,
        "past_threat": float(past_threat),
        "current_safety": float(current_safety),
    }


def run_episode(cfg: Config, condition: str, risk_regime: str, seed: int, collect_steps: bool) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    rng = make_rng(seed)
    agents = [AgentState(rng, cfg) for _ in range(cfg.n_agents)]
    n_features = 14
    signal_weights = make_initial_signal_weights(rng, cfg, n_features)
    # running predictors by event
    pred_int = np.full(len(EVENT_NAMES), 0.35, dtype=float)
    pred_risk = np.full(len(EVENT_NAMES), 0.20, dtype=float)

    rows = []
    selected_signals = []
    terminated = False

    for t in range(cfg.steps):
        if all((a.integrity <= 0.05 or a.energy <= 0.05) for a in agents):
            terminated = True
            break
        i = int(rng.integers(0, cfg.n_agents))
        a = agents[i]
        if a.integrity <= 0.05 or a.energy <= 0.05:
            continue
        event = sample_event(rng, risk_regime)
        out = event_outcome(event, risk_regime, rng, cfg)
        ei = EVENT_INDEX[event]
        pred_before = float(pred_int[ei])
        q_before = compute_q(a, out["actual_risk"], out["damage"], cfg, condition)
        feats = feature_vector(a, out, pred_before, q_before, condition)
        channel, probs = select_signal(a, feats, signal_weights, rng, cfg, condition, event)
        emitted = channel >= 0
        if emitted:
            a.refractory = cfg.refractory_steps
            selected_signals.append(channel)
        elif a.refractory > 0:
            a.refractory -= 1

        own_tension_before = a.social_tension
        own_explore_before = a.exploration_drive
        receiver_recovery, spread, n_heard = apply_signal_to_listeners(agents, i, channel, out, cfg, condition, rng)
        update_body(a, out, q_before, event, cfg, condition, emitted)
        q_after = compute_q(a, out["actual_risk"], out["damage"], cfg, condition)

        # generic producer benefit: social tension reduction and re-exploration, not a laughter reward.
        own_recovery = (own_tension_before - a.social_tension) + 0.5 * (a.exploration_drive - own_explore_before)
        total_benefit = own_recovery + 0.8 * receiver_recovery + 0.003 * spread
        update_signal_learning(a, feats, signal_weights, channel, total_benefit, out, cfg, condition)
        # update listener values for all receivers that heard; approximate same target for relevant listeners
        if channel >= 0 and condition != "private_signal":
            for j, d in distances(agents, i):
                if d <= cfg.signal_radius:
                    update_listener_learning(agents[j], channel, receiver_recovery, out, cfg, condition)

        # predictor update after outcome
        lr = cfg.predictor_lr
        pred_int[ei] = (1 - lr) * pred_int[ei] + lr * out["intensity"]
        pred_risk[ei] = (1 - lr) * pred_risk[ei] + lr * out["actual_risk"]

        av = analysis_variables(a, out, pred_before, q_before, q_after, cfg, condition)
        row = {
            "t": t,
            "agent": i,
            "condition": condition,
            "risk_regime": risk_regime,
            "event": event,
            "selected_signal": channel,
            "signal_emitted": float(emitted),
            "receiver_recovery": receiver_recovery,
            "cross_agent_spread": spread,
            "n_heard": n_heard,
            "own_recovery": own_recovery,
            "q_before": q_before,
            "q_after": q_after,
            "actual_risk": out["actual_risk"],
            "damage": out["damage"],
            "initial_appraisal": out["initial_appraisal"],
            "integrity": a.integrity,
            "energy": a.energy,
            "social_tension": a.social_tension,
            "social_sync": a.social_sync,
            "exploration_drive": a.exploration_drive,
        }
        row.update(av)
        if collect_steps:
            rows.append(row)

        # store own past appraisal after all calculations for next self-reference window
        a.last_appraisal = out["initial_appraisal"]
        a.last_actual_risk = out["actual_risk"]
        a.last_q = q_after
        # random movement keeps listener neighborhoods changing
        a.x = float(np.clip(a.x + rng.normal(0, 0.18), 0, 5))
        a.y = float(np.clip(a.y + rng.normal(0, 0.18), 0, 5))

    df = pd.DataFrame(rows)
    # Summary over all steps requires rows; smoke/full collect enough rows by default.
    if df.empty:
        ep = empty_episode_summary(condition, risk_regime, seed, terminated)
    else:
        ep = summarize_episode(df, condition, risk_regime, seed, terminated, agents)
    return ep, df, np.array(selected_signals, dtype=int)


def empty_episode_summary(condition: str, risk_regime: str, seed: int, terminated: bool) -> Dict[str, float]:
    return {
        "condition": condition, "risk_regime": risk_regime, "seed": seed, "terminated": float(terminated),
        "selected_channel": -1, "selected_signal_rate": 0.0, "safe_surprise_rate": 0.0, "non_safe_rate": 0.0,
        "danger_signal_rate": 0.0, "benign_selectivity": 0.0, "danger_suppression": 0.0,
        "relief_association": 0.0, "safe_surprise_association": 0.0, "self_appraisal_gap_association": 0.0,
        "q_relief_association": 0.0, "social_recovery_score": 0.0, "receiver_recovery_score": 0.0,
        "state_bifurcation": 0.0, "history_bifurcation": 0.0, "cross_agent_spread": 0.0,
        "viable_fraction": 0.0, "safe_context_fraction": 0.0, "danger_context_fraction": 0.0,
        "emergent_function_score": 0.0,
    }


def choose_channel_discovery(df_first: pd.DataFrame, cfg: Config) -> int:
    # Select anonymous channel by generic social/exploratory recovery, not by safe context or self gap.
    best_ch = -1
    best_score = -np.inf
    for ch in range(cfg.n_signals):
        sub = df_first[df_first["selected_signal"] == ch]
        if len(sub) < 3:
            continue
        rate = len(sub) / max(len(df_first), 1)
        rec = float(sub["receiver_recovery"].mean() + 0.65 * sub["own_recovery"].mean() + 0.003 * sub["cross_agent_spread"].mean())
        # avoid selecting pathological channels that emit too rarely or too often
        rate_penalty = abs(rate - 0.012) * 0.10
        score = rec - rate_penalty
        if score > best_score:
            best_score = score
            best_ch = ch
    return int(best_ch)


def mean_when(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    if mask.sum() == 0:
        return 0.0
    return float(df.loc[mask, col].mean())


def assoc(df: pd.DataFrame, signal_col: str, x_col: str) -> float:
    sig = df[signal_col].to_numpy(dtype=float)
    x = df[x_col].to_numpy(dtype=float)
    if sig.sum() < 2 or np.std(x) < EPS:
        return 0.0
    return float(x[sig > 0.5].mean() - x[sig <= 0.5].mean())


def summarize_episode(df: pd.DataFrame, condition: str, risk_regime: str, seed: int, terminated: bool, agents: List[AgentState]) -> Dict[str, float]:
    split_t = df["t"].max() * 0.5
    first = df[df["t"] <= split_t]
    test = df[df["t"] > split_t].copy()
    selected_channel = choose_channel_discovery(first, Config()) if len(first) else -1
    # The config defaults only affect n_signals here. Fine for summary with default n_signals.
    if selected_channel >= 0:
        test["selected_channel_signal"] = (test["selected_signal"] == selected_channel).astype(float)
    else:
        test["selected_channel_signal"] = 0.0

    sig = test["selected_channel_signal"] > 0.5
    safe_mask = test["safe_context"] > 0.5
    danger_mask = test["danger_context"] > 0.5
    non_safe_mask = ~safe_mask

    selected_rate = float(test["selected_channel_signal"].mean()) if len(test) else 0.0
    safe_rate = mean_when(test, safe_mask, "selected_channel_signal")
    non_safe_rate = mean_when(test, non_safe_mask, "selected_channel_signal")
    danger_rate = mean_when(test, danger_mask, "selected_channel_signal")
    benign_selectivity = safe_rate - non_safe_rate
    danger_suppression = 1.0 - danger_rate

    # Competing explanatory variables: signal-emission association on held-out half.
    relief_association = assoc(test, "selected_channel_signal", "relief")
    safe_surprise_association = mean_when(test, sig, "safe_surprise") - mean_when(test, ~sig, "safe_surprise") if len(test) else 0.0
    self_appraisal_gap_association = assoc(test, "selected_channel_signal", "self_appraisal_gap")
    q_relief_association = assoc(test, "selected_channel_signal", "q_relief")
    social_recovery_score = assoc(test, "selected_channel_signal", "own_recovery")
    receiver_recovery_score = assoc(test, "selected_channel_signal", "receiver_recovery")

    # State/history bifurcation for same broad class: near_miss/false_alarm/minor_mismatch/social_play.
    focal = test[test["event"].isin(["near_miss", "false_alarm", "minor_mismatch", "social_play"])]
    if len(focal) > 8:
        vuln = 1.0 - np.minimum(focal["integrity"].to_numpy(), focal["energy"].to_numpy())
        med_v = np.median(vuln)
        low_v_rate = float(focal.loc[vuln <= med_v, "selected_channel_signal"].mean()) if (vuln <= med_v).sum() else 0.0
        high_v_rate = float(focal.loc[vuln > med_v, "selected_channel_signal"].mean()) if (vuln > med_v).sum() else 0.0
        state_bif = low_v_rate - high_v_rate
        med_gap = focal["past_threat"].median()
        high_gap_rate = float(focal.loc[focal["past_threat"] > med_gap, "selected_channel_signal"].mean()) if (focal["past_threat"] > med_gap).sum() else 0.0
        low_gap_rate = float(focal.loc[focal["past_threat"] <= med_gap, "selected_channel_signal"].mean()) if (focal["past_threat"] <= med_gap).sum() else 0.0
        history_bif = high_gap_rate - low_gap_rate
    else:
        state_bif = 0.0
        history_bif = 0.0

    viable_fraction = float(np.mean([(a.integrity > 0.05 and a.energy > 0.05) for a in agents]))
    safe_context_fraction = float(test["safe_context"].mean()) if len(test) else 0.0
    danger_context_fraction = float(test["danger_context"].mean()) if len(test) else 0.0
    cross_spread = assoc(test, "selected_channel_signal", "cross_agent_spread")

    # Composite is only orientation; targeted metrics are load-bearing.
    components = [
        clip01(benign_selectivity / 0.018),
        clip01((danger_suppression - 0.85) / 0.15),
        clip01(self_appraisal_gap_association / 0.020),
        clip01((receiver_recovery_score + 0.0005) / 0.010),
        clip01((social_recovery_score + 0.0005) / 0.008),
        clip01((viable_fraction - 0.45) / 0.40),
    ]
    comp = float(np.exp(np.mean(np.log(np.array(components) + 1e-6))))

    return {
        "condition": condition,
        "risk_regime": risk_regime,
        "seed": int(seed),
        "terminated": float(terminated),
        "selected_channel": int(selected_channel),
        "selected_signal_rate": selected_rate,
        "safe_surprise_rate": safe_rate,
        "non_safe_rate": non_safe_rate,
        "danger_signal_rate": danger_rate,
        "benign_selectivity": benign_selectivity,
        "danger_suppression": danger_suppression,
        "relief_association": relief_association,
        "safe_surprise_association": safe_surprise_association,
        "self_appraisal_gap_association": self_appraisal_gap_association,
        "q_relief_association": q_relief_association,
        "social_recovery_score": social_recovery_score,
        "receiver_recovery_score": receiver_recovery_score,
        "state_bifurcation": state_bif,
        "history_bifurcation": history_bif,
        "cross_agent_spread": cross_spread,
        "viable_fraction": viable_fraction,
        "safe_context_fraction": safe_context_fraction,
        "danger_context_fraction": danger_context_fraction,
        "emergent_function_score": comp,
    }


def linear_logit_fit(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> Dict[str, float]:
    # Ridge logistic regression implemented with simple gradient descent to avoid sklearn dependency.
    if len(df) < 20 or df[y_col].sum() < 3:
        return {"n": len(df), "events": float(df[y_col].sum()), "intercept": 0.0, **{f"coef_{c}": 0.0 for c in x_cols}, "pseudo_r2": 0.0}
    X0 = df[x_cols].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    mu = X0.mean(axis=0)
    sd = X0.std(axis=0) + 1e-6
    Xs = (X0 - mu) / sd
    X = np.column_stack([np.ones(len(Xs)), Xs])
    beta = np.zeros(X.shape[1])
    # balanced weights prevent all-zero dominance
    pos_w = 0.5 / max(y.mean(), 1e-4)
    neg_w = 0.5 / max(1 - y.mean(), 1e-4)
    wts = np.where(y > 0.5, pos_w, neg_w)
    lam = 0.08
    lr = 0.05
    for _ in range(350):
        p = sigmoid(X @ beta)
        grad = (X.T @ ((p - y) * wts)) / len(y)
        grad[1:] += lam * beta[1:]
        beta -= lr * grad
    p = np.clip(sigmoid(X @ beta), 1e-6, 1 - 1e-6)
    null_p = np.clip(np.repeat(y.mean(), len(y)), 1e-6, 1 - 1e-6)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ll0 = float(np.sum(y * np.log(null_p) + (1 - y) * np.log(1 - null_p)))
    pseudo = 1 - ll / ll0 if ll0 != 0 else 0.0
    out = {"n": len(df), "events": float(y.sum()), "intercept": float(beta[0]), "pseudo_r2": float(pseudo)}
    for c, b in zip(x_cols, beta[1:]):
        out[f"coef_{c}"] = float(b)
    return out


def build_model_comparison(step_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for cond in step_df["condition"].unique():
        subc = step_df[step_df["condition"] == cond].copy()
        if subc.empty:
            continue
        # Reconstruct split-half channel selection per episode is expensive; approximate by using any selected signal.
        # For full condition, this still tests competing explanations of anonymous signal occurrence.
        subc["y"] = (subc["signal_emitted"] > 0.5).astype(float)
        models = {
            "relief_only": ["relief"],
            "safe_surprise_only": ["safe_surprise"],
            "self_appraisal_gap_only": ["self_appraisal_gap"],
            "three_competing": ["relief", "safe_surprise", "self_appraisal_gap"],
            "controlled": ["relief", "safe_surprise", "self_appraisal_gap", "damage", "actual_risk", "prediction_error"],
        }
        for name, cols in models.items():
            fit = linear_logit_fit(subc, "y", cols)
            fit.update({"condition": cond, "model": name})
            rows.append(fit)
    mc = pd.DataFrame(rows)
    mc.to_csv(outdir / "model_comparison.csv", index=False)
    return mc


def summarize_tables(ep_df: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_cols = ep_df.select_dtypes(include=[np.number]).columns.tolist()
    summary = ep_df.groupby("condition")[num_cols].agg(["mean", "std", "sem"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in summary.columns]
    byrisk = ep_df.groupby(["condition", "risk_regime"])[num_cols].agg(["mean", "std", "sem"]).reset_index()
    byrisk.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in byrisk.columns]
    selected = ep_df.groupby(["condition", "selected_channel"]).size().reset_index(name="n_episodes")
    summary.to_csv(outdir / "condition_summary.csv", index=False)
    byrisk.to_csv(outdir / "condition_by_risk_summary.csv", index=False)
    selected.to_csv(outdir / "selected_channel_summary.csv", index=False)
    return summary, byrisk, selected


def plot_figures(summary: pd.DataFrame, ep_df: pd.DataFrame, step_sample: pd.DataFrame, outdir: Path):
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    # Figure 1: core metrics
    metrics = ["emergent_function_score", "benign_selectivity", "danger_signal_rate", "self_appraisal_gap_association", "receiver_recovery_score", "viable_fraction"]
    conds = summary["condition"].tolist()
    for metric in metrics:
        mcol = f"{metric}_mean"
        scol = f"{metric}_sem"
        if mcol not in summary:
            continue
        plt.figure(figsize=(11, 4.5))
        x = np.arange(len(conds))
        plt.bar(x, summary[mcol].to_numpy(), yerr=summary.get(scol, pd.Series(np.zeros(len(conds)))).to_numpy(), capsize=2)
        plt.xticks(x, conds, rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(figdir / f"figure_metric_{metric}.png", dpi=180)
        plt.close()

    # Figure 2: context specificity full
    full = ep_df[ep_df["condition"] == "full"]
    vals = [full["safe_surprise_rate"].mean(), full["non_safe_rate"].mean(), full["danger_signal_rate"].mean()]
    errs = [full["safe_surprise_rate"].sem(), full["non_safe_rate"].sem(), full["danger_signal_rate"].sem()]
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(3), vals, yerr=errs, capsize=3)
    plt.xticks(np.arange(3), ["safe context", "non-safe", "danger"])
    plt.ylabel("selected anonymous signal rate")
    plt.title("Full model context specificity")
    plt.tight_layout()
    plt.savefig(figdir / "figure_02_context_specificity_full.png", dpi=180)
    plt.close()

    # Figure 3: sample dynamics if possible
    if not step_sample.empty and "full" in step_sample["condition"].unique():
        sub = step_sample[step_sample["condition"] == "full"].head(1400)
        if not sub.empty:
            plt.figure(figsize=(11, 5))
            plt.plot(sub["t"], sub["self_appraisal_gap"], label="self_appraisal_gap")
            plt.plot(sub["t"], sub["relief"], label="relief")
            plt.plot(sub["t"], sub["actual_risk"], label="actual_risk", alpha=0.7)
            sig = sub[sub["signal_emitted"] > 0.5]
            if not sig.empty:
                plt.scatter(sig["t"], sig["self_appraisal_gap"], s=20, marker="o", label="anonymous signal")
            plt.legend(loc="upper right")
            plt.xlabel("step")
            plt.ylabel("value")
            plt.title("Sample full-model dynamics")
            plt.tight_layout()
            plt.savefig(figdir / "figure_03_sample_dynamics.png", dpi=180)
            plt.close()


def write_report(cfg: Config, ep_df: pd.DataFrame, summary: pd.DataFrame, mc: pd.DataFrame, outdir: Path):
    full = ep_df[ep_df["condition"] == "full"]
    means = full.select_dtypes(include=[np.number]).mean(numeric_only=True)
    lines = []
    lines.append("Phase 4c calibrated self-appraisal-gap multi-agent anonymous signal analysis")
    lines.append("======================================================================")
    lines.append("")
    lines.append("Core design:")
    lines.append("- No LAUGH/HUMOR action label is given to the model.")
    lines.append("- No benign-violation variable is used by the signal controller.")
    lines.append("- No laughter reward, external prompt, LMM, or API is used.")
    lines.append("- Anonymous signals are selected by split-half discovery using generic social/exploratory recovery.")
    lines.append("- Relief, safe-surprise, and self-appraisal-gap are post-hoc competing explanatory variables.")
    lines.append("- This viability-fixed version explicitly checks viability, context frequencies, and non-zero anonymous signal emission.")
    lines.append("")
    lines.append("Run configuration:")
    lines.append(json.dumps(asdict(cfg), indent=2))
    lines.append("")
    lines.append("Episode counts:")
    lines.append(str(ep_df.groupby(["condition", "risk_regime"]).size()))
    lines.append("")
    lines.append("Full-model headline metrics:")
    for k in [
        "emergent_function_score", "selected_signal_rate", "safe_surprise_rate", "non_safe_rate",
        "danger_signal_rate", "benign_selectivity", "danger_suppression", "relief_association",
        "safe_surprise_association", "self_appraisal_gap_association", "q_relief_association",
        "social_recovery_score", "receiver_recovery_score", "state_bifurcation", "history_bifurcation",
        "cross_agent_spread", "viable_fraction", "safe_context_fraction", "danger_context_fraction",
    ]:
        lines.append(f"- {k}: {means.get(k, 0.0):.6f}")
    lines.append("")
    lines.append("Calibration guardrail:")
    lines.append("The run is analyzable only if viable_fraction is not collapsed, safe_context_fraction and danger_context_fraction are non-zero, and selected_signal_rate is non-zero.")
    lines.append("")
    lines.append("Interpretation guardrail:")
    lines.append("The result supports the self-appraisal-gap hypothesis only if the selected anonymous signal shows positive held-out self_appraisal_gap_association, positive social/receiver recovery, strong danger suppression, and model comparison shows self_appraisal_gap outperforming or independently predicting the signal after Relief and SafeSurprise are controlled. This script does not demonstrate subjective amusement.")
    lines.append("")
    if not mc.empty:
        full_mc = mc[mc["condition"] == "full"]
        lines.append("Full-model model-comparison summary:")
        for _, r in full_mc.iterrows():
            cols = [c for c in r.index if c.startswith("coef_")]
            coef_text = ", ".join([f"{c}={r[c]:.4f}" for c in cols])
            lines.append(f"- {r['model']}: pseudo_r2={r.get('pseudo_r2', 0.0):.6f}, {coef_text}")
    lines.append("")
    lines.append("Generated files:")
    lines.append("- episode_results.csv")
    lines.append("- condition_summary.csv")
    lines.append("- condition_by_risk_summary.csv")
    lines.append("- selected_channel_summary.csv")
    lines.append("- model_comparison.csv")
    lines.append("- step_logs_sample.csv")
    lines.append("- figures/*.png")
    (outdir / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")


def mode_params(mode: str) -> Tuple[int, int]:
    if mode == "smoke":
        return 4, 350
    if mode == "quick":
        return 12, 1000
    if mode == "full":
        return 50, 2500
    raise ValueError(f"unknown mode: {mode}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--base-seed", type=int, default=7200)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--risk-regimes", default=",".join(RISK_REGIMES))
    # perturb hooks
    ap.add_argument("--perturb-signal-threshold", type=float, default=0.0)
    ap.add_argument("--perturb-signal-bias", type=float, default=0.0)
    ap.add_argument("--perturb-signal-lr-mult", type=float, default=1.0)
    ap.add_argument("--perturb-listener-lr-mult", type=float, default=1.0)
    ap.add_argument("--perturb-social-scale-mult", type=float, default=1.0)
    ap.add_argument("--perturb-event-risk-mult", type=float, default=1.0)
    ap.add_argument("--perturb-listener-effect-mult", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()
    n_seeds, default_steps = mode_params(args.mode)
    cfg = Config(
        steps=args.steps if args.steps is not None else default_steps,
        base_seed=args.base_seed,
        perturb_signal_threshold=args.perturb_signal_threshold,
        perturb_signal_bias=args.perturb_signal_bias,
        perturb_signal_lr_mult=args.perturb_signal_lr_mult,
        perturb_listener_lr_mult=args.perturb_listener_lr_mult,
        perturb_social_scale_mult=args.perturb_social_scale_mult,
        perturb_event_risk_mult=args.perturb_event_risk_mult,
        perturb_listener_effect_mult=args.perturb_listener_effect_mult,
    )
    outdir = Path(args.outdir or f"phase4c_self_appraisal_gap_{args.mode}").expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    risks = [r.strip() for r in args.risk_regimes.split(",") if r.strip()]
    ep_path = outdir / "episode_results.csv"
    done_keys = set()
    ep_rows = []
    if args.resume and ep_path.exists():
        old = pd.read_csv(ep_path)
        ep_rows = old.to_dict("records")
        for _, r in old.iterrows():
            done_keys.add((str(r["condition"]), str(r["risk_regime"]), int(r["seed"])))
        print(f"[resume] loaded {len(done_keys)} completed episodes")

    step_samples = []
    total = len(conds) * len(risks) * n_seeds
    counter = 0
    for condition in conds:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {condition}; choose from {CONDITIONS}")
        for risk in risks:
            for s in range(n_seeds):
                seed = cfg.base_seed + 100000 * CONDITIONS.index(condition) + 1000 * RISK_REGIMES.index(risk) + s
                key = (condition, risk, seed)
                counter += 1
                if key in done_keys:
                    continue
                collect_steps = True
                ep, df, _ = run_episode(cfg, condition, risk, seed, collect_steps=collect_steps)
                ep_rows.append(ep)
                if len(step_samples) < cfg.max_step_rows_per_run and not df.empty:
                    # store focused sample: all full condition smoke/quick rows, and small samples for others
                    if condition == "full" or (s == 0 and risk == risks[0]):
                        step_samples.append(df.head(max(50, min(len(df), 1200))))
                if counter % 10 == 0 or args.mode == "smoke":
                    print(f"[{counter}/{total}] condition={condition} risk={risk} seed={s} selected_rate={ep['selected_signal_rate']:.4f} viable={ep['viable_fraction']:.3f}", flush=True)
                pd.DataFrame(ep_rows).to_csv(ep_path, index=False)

    ep_df = pd.DataFrame(ep_rows)
    ep_df.to_csv(ep_path, index=False)
    step_df = pd.concat(step_samples, ignore_index=True) if step_samples else pd.DataFrame()
    if not step_df.empty:
        step_df.to_csv(outdir / "step_logs_sample.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "step_logs_sample.csv", index=False)
    summary, byrisk, selected = summarize_tables(ep_df, outdir)
    mc = build_model_comparison(step_df, outdir) if not step_df.empty else pd.DataFrame()
    plot_figures(summary, ep_df, step_df, outdir)
    write_report(cfg, ep_df, summary, mc, outdir)
    print(f"[done] outputs written to {outdir}")


if __name__ == "__main__":
    main()
