Simulation codes of "Beyond Relief and Surprise: Retrospective Self-Appraisal Gap as a Mechanism for Laughter-like Social Signaling".

Purpose
-------
This version fixes the Phase 4b failure in which the self-appraisal-gap
hypothesis showed a positive signal but the population collapsed
(viable_fraction = 0). The revision keeps the same hypothesis-comparison
logic while making the ecology analyzable.

Strict constraints retained
---------------------------
- No LAUGH or HUMOR action label.
- No benign-violation variable in the signal controller.
- No laughter reward.
- No external prompt.
- No LMM/API.
- Anonymous channels only: signal_0 ... signal_4.
- Relief, SafeSurprise, and SelfAppraisalGap are computed only for post-hoc
  competing-model analysis.
- Candidate channels are selected by split-half discovery using generic
  social/exploratory recovery, not by the self-appraisal-gap variable.

Main fixes from Phase 4b
-----------------------
- Reduced basal energetic cost and signal energetic cost.
- Increased rest/recovery and safe-event recovery.
- Reduced event damage while preserving actual-risk danger contexts.
- Added generic viability inhibition to anonymous signalling under high
  actual risk or damage. This is not a laughter gate; it is a general
  cost-sensitive communication constraint.
- Rebalanced risk regimes so safe-context and danger-context rows are both
  non-zero.

Basic command
-------------
cd ~/Desktop
python3 -u phase4c_self_appraisal_gap_multiagent_core_viability_fixed.py \
  --mode full \
  --outdir ~/Desktop/phase4c_self_appraisal_gap_viability_fixed_full \
  --resume \
  2>&1 | tee ~/Desktop/phase4c_self_appraisal_gap_full.log

Robustness command
------------------
python3 -u phase4c_self_appraisal_gap_robustness_viability_fixed.py \
  --mode quick \
  --core-script ~/Desktop/phase4c_self_appraisal_gap_multiagent_core_viability_fixed.py \
  --outdir ~/Desktop/phase4c_self_appraisal_gap_viability_fixed_robustness \
  2>&1 | tee ~/Desktop/phase4c_self_appraisal_gap_robustness.log

Primary output files
--------------------
- episode_results.csv
- condition_summary.csv
- condition_by_risk_summary.csv
- selected_channel_summary.csv
- model_comparison.csv
- step_logs_sample.csv
- summary_report.txt
- figures/*.png

Success criteria
----------------
The self-appraisal-gap hypothesis is supported only if:
1. selected_signal_rate > 0
2. safe_context_fraction > 0 and danger_context_fraction > 0
3. viable_fraction is not collapsed
4. self_appraisal_gap_association > 0
5. receiver_recovery_score > 0 and social_recovery_score > 0
6. danger_suppression is high
7. the controlled model retains coef_self_appraisal_gap > 0 after Relief,
   SafeSurprise, damage, actual risk, and prediction error are included
8. no_self_appraisal, random_signal, and label_rule_signal do not reproduce
   the same profile
9. robustness runs preserve the signs of the key effects

Interpretation
--------------
This model does not demonstrate subjective amusement. It tests whether an
anonymous social regulatory signal is better explained by the retrospective
self-appraisal gap than by simple relief or safe surprise.
