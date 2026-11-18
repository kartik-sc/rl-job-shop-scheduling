# RAJS — Finalisation Report: Next Steps and Architecture
**Branch:** `feature/kartik-risk-layer`  
**Date:** 2026-10-30  
**Status:** Phase 1 audit complete. Experiments not yet started.

---

## 1. Current Implementation Audit

### 1.1 Repository layout
```
backend/risk/           ← All risk-layer code (Layer 1 + Layer 2)
  cmapss_pipeline.py    ← Data loading, constant-feature removal, engine split, scaling
  anomaly_models.py     ← OC-SVM + IForest, scoring functions, AnomalyBundle
  train_anomaly.py      ← Training entry-point, saves models/anomaly_bundle.pkl
  calibration.py        ← CalibrationBundle, isotonic fit, risk/RUL inference
  train_calibration.py  ← Calibration entry-point, evaluation, plots
  diagnostic.py         ← Fit-split diagnostic analysis (Layer 1 evaluation)
  feature_analysis.py   ← Sensor ranking, temporal features, FD001 vs FD003
  rul_error_analysis.py ← Binned RUL error and bias analysis

backend/env/            ← Gymnasium environment (stub only — not yet implemented)
backend/baselines/      ← SPT/LPT/FIFO policies (stub only)
backend/evaluation/     ← Multi-seed evaluation (stub only)
backend/api/main.py     ← FastAPI with single /health endpoint (stub only)
frontend/               ← Next.js + D3 (not scaffolded)
shared/schemas/         ← event_log.schema.json (risk field: [0,1])
```

### 1.2 Installed packages (relevant)
- scikit-learn 1.9.1, numpy 2.4.6, pandas 3.1.0, scipy 1.17.1
- gymnasium 1.4.0, stable-baselines3 2.9.2, sb3-contrib 2.3
- job-shop-lib 1.7.2 (provides job-shop instances and solvers)
- torch 2.15.0 (CPU), fastapi 0.162, uvicorn 0.54
- matplotlib 3.11.2

### 1.3 Test suite
114 tests passing across 5 files:
- `test_cmapss_pipeline.py` — 33 tests (parsing, RUL, splits, scaler, leakage)
- `test_anomaly_models.py` — 17 tests (score direction, shape, alignment, bundle)
- `test_diagnostic.py` — 26 tests (separation, Spearman, LOO threshold, EW)
- `test_calibration.py` — 38 tests (isotonic direction, range, leakage, interface)
- `test_smoke.py` — 1 test (API health)

---

## 2. Completed Work and Existing Results

### Layer 1 — Anomaly detection (complete)
- Data: FD001 + FD003, 200 engines, 120/40/40 split (SPLIT_SEED=42), stable JSON split.
- Features: 18 retained after dropping 6 constant columns.
- Healthy window: cycles 1–30.
- Models: OC-SVM (RBF, nu=0.05, gamma=scale) + IForest (200 trees, seed=42), both trained on fit-split healthy cycles only.
- Both saved in `models/anomaly_bundle.pkl`.
- Both models score higher for more degraded engines on 100% of fit engines.

### Layer 1 diagnostics (complete)
- Fit-split analysis on 120 engines: Spearman ρ, separation, early-warning (LOO).
- IForest marginally better on normalised separation (Cohen's d 8.65 vs 7.95, p=0.007).
- OC-SVM marginally better on lead time (62.5 vs 56.0 cycles median, not significant).
- Both achieve 100% directional success and 100% positive per-engine Spearman ρ.
- Recommendation: retain both for calibration.

### Layer 2 — Calibration (complete, baseline only)
- Method: isotonic regression (PAVA, `increasing=False`) mapping anomaly score → expected RUL.
- Fitted on 40 calibration engines only. Test engines untouched.
- Outputs: `risk_score ∈ [0,1]`, `rul_estimate` (cycles).
- Artifacts: `models/calibration_ocsvm.pkl`, `models/calibration_iforest.pkl`.
- Both calibrations achieve 100% directional success on calibration split.

### RUL error analysis (complete)
Key numbers for the **calibration split** (40 engines):

| Model | Overall MAE | Overall RMSE | RUL 0–20 MAE | RUL 0–20 mean error |
|---|---|---|---|---|
| OC-SVM  | 49.0 cycles | 60.8 cycles | **16.1 cycles** | +15.6 (overestimate) |
| IForest | 55.5 cycles | 65.9 cycles | **52.1 cycles** | +51.9 (overestimate) |

Root causes identified (verified from data, not hypothetical):
1. Isotonic regression predicts conditional mean RUL within each score bin — the bin contains a mix of RUL values, pulling predictions up.
2. IForest's score range (0.37–0.71, span 0.34) is far more compressed than OC-SVM's (−12.4 to +14.1, span 26.5). IForest's near-failure scores overlap 88.6% with its early-life scores vs 37.4% for OC-SVM.
3. The isotonic floor for IForest FD001 engines is 25.8 cycles — it cannot physically predict lower than that for FD001 engines, causing 100% overestimation at RUL ≤ 20 for FD001/IForest.

### Feature analysis (complete)
Performed on fit split only. Key findings:

**Raw feature ranking (composite Spearman ρ + Cohen's d):**

| Rank | Feature | Composite | Notes |
|---|---|---|---|
| 1 | s11 | 1.000 | Strong in both datasets |
| 2 | s4  | 0.936 | Strong in both |
| 3 | s17 | 0.814 | Strong in both |
| 4 | s2  | 0.777 | Strong in both |
| 5 | s3  | 0.761 | Strong in both |
| 6–7 | s8, s13 | ~0.713 | Near-duplicate pair (ρ correlation high) |
| 8 | s9 | 0.628 | Good overall |
| 9 | s10 | 0.608 | FD003-specific (Cohen's d ≈ 2.0 in FD003, ≈ 0 in FD001) |
| 16–18 | s6, op1, op2 | < 0.06 | Near-zero degradation signal |

**Dataset-specific divergence (confirmed):**
- s10: Cohen's d = 2.0 (FD003) vs ~0 (FD001) — FD003-only sensor
- s12, s15, s7, s20, s21: stronger in FD001, weak in FD003
- s11, s4, s17: consistent across both datasets

**Temporal features:** Rolling means of top sensors dominate (s11_rm10: median ρ = 0.942 vs raw s11: 0.794). Rolling means with w=10–20 are the most promising temporal additions.

**Correlation matrix:** No pairs exceed |ρ| = 0.90 by the formal threshold, but s8 and s13 track together very closely in both datasets (~0.88–0.92 range).

---

## 3. Unresolved Technical Problems

### P1 — IForest near-failure RUL overestimation (critical)
IForest MAE at RUL ≤ 20 is 52 cycles — three times worse than OC-SVM. The root cause is score compression: IForest's narrow score range means near-failure cycles cannot be separated from mid-life cycles in the calibration mapping. The isotonic floor for FD001/IForest engines is physically 25.8 cycles. This is a fundamental limitation of the current single-combined IForest model.

### P2 — Score saturation prevents late-stage RUL estimation (IForest)
IForest's score saturates before RUL = 0. Even at RUL = 5, the median score is 0.658 — indistinguishable from scores at RUL = 60–80. There is no calibration method that can recover fine-grained RUL from a flat score signal. The fix must address the signal, not the calibration.

### P3 — Joint FD001+FD003 model may be suboptimal
The two datasets have fundamentally different degradation signatures for several sensors. Training one model on both means the healthy-region boundary is a compromise between two distinct healthy distributions, potentially making it less sensitive to the degradation patterns of each.

### P4 — No scheduler, environment, or frontend implementation
`backend/env/`, `backend/baselines/`, `backend/evaluation/`, and `frontend/` are all empty stubs. The risk module cannot be demonstrated end-to-end without these.

### P5 — Risk score denominator (max_lifetime=494) is from fit-split
The normalisation `risk = 1 - rul_est / 494` means any engine expected to live longer than 494 cycles would produce a negative risk score before clipping. FD003 has engines with lifetimes up to 525 cycles. This is handled by `clamp(0, 1)` but means the risk scale is not uniformly meaningful across all engines.

### P6 — Calibration evaluations are all in-calibration
Every calibration evaluation shown so far uses the same 40 engines used to fit the isotonic regressor. The numbers are therefore optimistic. The true performance on unseen engines is unknown until the held-out test split is evaluated.

### P7 — No online inference interface
The `CalibrationBundle.predict(score)` interface works for pre-computed scores, but there is no end-to-end function that takes raw sensor readings and returns risk + RUL. This gap must be filled before scheduler integration.

---

## 4. Dataset-Specific Model Experiment Plan

### Rationale
Sensor s10 is irrelevant for FD001 but is a meaningful degradation indicator for FD003 (Cohen's d ≈ 2.0). Sensors s12, s15, s7, s20, s21 are meaningful for FD001 but not FD003. The combined model trains on the union of healthy distributions, which is a compromise. Separate models can learn each dataset's healthy region precisely.

### Configurations to compare

**Config A — Current baseline (already trained)**
- One OC-SVM + one IForest, trained on FD001+FD003 combined, 18 features.
- Artifacts: `models/anomaly_bundle.pkl`

**Config B — Separate models, same 18 features**
- Train separate OC-SVM and IForest for FD001 and FD003 using only their respective fit-split engines.
- FD001 fit: 58 engines × 30 healthy cycles = 1,740 rows.
- FD003 fit: 62 engines × 30 healthy cycles = 1,860 rows.
- Save as `models/anomaly_bundle_fd001.pkl` and `models/anomaly_bundle_fd003.pkl`.
- Evaluate on their respective calibration engines (20 FD001 cal, 20 FD003 cal).

**Config C — Separate models, dataset-specific feature subsets**
- FD001 subset (proposed): s11, s4, s17, s2, s3, s8, s9, s12, s15, s21, s20 (11 features — retains FD001-specific sensors, drops op1, op2, s6, s10)
- FD003 subset (proposed): s11, s4, s17, s2, s3, s8, s9, s10, s13, s14 (10 features — adds s10, drops FD001-specific sensors and noise)
- Justification: based on per-dataset Cohen's d and Spearman ρ from feature analysis. Features dropped only if both composite score < 0.3 **and** the per-dataset Cohen's d < 0.3.

**Important caveat on feature selection:** Cohen's d and Spearman ρ are marginal measures — a sensor with low individual signal can still contribute useful information when combined with others. The proposed subsets should be treated as hypotheses to test, not established improvements.

### Evaluation protocol for B and C
- Use the same leakage-free per-engine Spearman ρ, Cohen's d, directional success, LOO early-warning as in the existing diagnostic.
- Evaluate on their respective calibration engines (separately).
- Report metrics separately for FD001 and FD003 and compare against Config A performance on the same engines.
- Do NOT claim improvement if the calibration sample (20 engines per dataset) is too small to give reliable estimates — explicitly flag uncertainty.
- Keep all artifacts separate from baseline.

---

## 5. Sensor Selection Recommendations

### For Config C (proposed initial subsets)

**FD001 proposed subset (11 features):**
`s11, s4, s17, s2, s3, s8, s9, s12, s15, s20, s21`

Rationale: Keeps the universally strong top-5 plus FD001-specific sensors (s12, s15, s20, s21 all have Cohen's d > 0.8 in FD001 vs < 0.3 in FD003). Drops op1, op2, s6 (near-zero composite < 0.06), s10 (near-zero in FD001), s13 (near-duplicate of s8 in FD001). Retains s7 in the sensitivity check — s7 has Cohen's d ~3.2 in FD001 but negative Spearman ρ (decreasing as engine degrades); it may hurt OC-SVM's boundary but is worth testing.

**FD003 proposed subset (10 features):**
`s11, s4, s17, s2, s3, s8, s9, s10, s13, s14`

Rationale: Keeps universally strong sensors plus s10 (Cohen's d = 2.0 in FD003 vs ~0 in FD001), adds s13 and s14 which have Cohen's d > 2.0 in FD003. Drops FD001-specific sensors (s12, s15, s20, s21 all below 0.3 composite in FD003), drops op1, op2, s6, s7.

**Do not include s8 and s13 together in a final subset.** They are near-duplicates — their correlation is ~0.88–0.92. For any final model, keep whichever has slightly better individual signal in that dataset (s8 for FD001, s13 for FD003), not both.

### Revised final suggestion after sensitivity testing

Run the experiments first. If Config C outperforms Config B on calibration metrics, proceed with the subsets. If not, use Config B (separate models, 18 features) as the production config. Do not force feature reduction if it doesn't improve measurable performance.

---

## 6. Temporal Feature Experiment Plan

### What to test
A single experiment: retrain anomaly models on rolling-mean augmented features for the top sensors, using fit-split engines only.

**Proposed temporal feature set (augment, do not replace):**
Add `s11_rm10, s4_rm10, s17_rm10, s3_rm10, s2_rm10` to the existing 18 features (or whichever subset is selected from Config B/C).

Rolling window: w=10 (strongest composite score, good balance of smoothing vs responsiveness).

Rationale: Median Spearman ρ for s11_rm10 is 0.942 vs 0.794 for raw s11. This improvement in monotonicity directly addresses IForest's score compression problem — a smoother, more monotone feature would give the isotonic calibration a cleaner score-to-RUL mapping with less overlap between early-life and near-failure scores at the same score level.

### Implementation requirements
1. Rolling features must be computed **per engine** with a **causal window** (only past cycles).
2. The window computation at training time (healthy cycles 1–30) will have many NaN values for early cycles since w=10 means the first 9 cycles have incomplete windows. Handle with `min_periods=1` (mean of available cycles).
3. At inference time, maintain a rolling buffer per engine. The buffer is reset when a new engine observation sequence begins.
4. Scaler must be refitted on the same fit-split data with temporal features included.
5. Save as a separate bundle — do not overwrite baseline.

### Leakage verification checklist
- [ ] Rolling window uses only past cycles (pandas `rolling` with `min_periods` and no `center=True`)
- [ ] Window is computed per engine (groupby engine_id before rolling)
- [ ] Scaler fitted on fit-split rows only, including their temporal features
- [ ] Healthy cycles (1–30) used for model training use temporal features computed from those cycles only (no future cycles visible)
- [ ] Calibration engines' temporal features computed independently of fit engines

---

## 7. Calibration Improvement Plan

### Current baseline behaviour (verified)
- OC-SVM: RUL ≤ 20 MAE = 16 cycles, 91% overestimation rate
- IForest: RUL ≤ 20 MAE = 52 cycles, 97% overestimation rate, FD001 floor = 25.8 cycles

### Approaches ranked by risk/reward

**Approach 1 — Per-dataset calibration (low risk, moderate reward)**

Fit separate isotonic regressors for FD001 and FD003 calibration engines. The 20 FD003 calibration engines reach higher scores (max score up to 0.711 for IForest) while FD001 engines top out at 0.672. A FD003-only calibrator can map 0.695–0.711 directly to RUL < 10 without FD001 engines polluting that score region with higher RUL values.

Required change: split the 40 calibration engines by dataset (20 each) and fit two CalibrationBundles. The CalibrationBundle dataclass stores `cal_engine_ids` — just filter them by dataset. No interface change required. Artifacts stored separately.

Statistical concern: 20 engines × ~240 cycles = ~4,800 rows per dataset. The near-failure region (RUL ≤ 20) has ~170 rows per dataset. This is thin. Report the sample sizes explicitly with the results and flag any instability in the isotonic mapping near the tails.

**Approach 2 — Endpoint anchoring (very low risk, moderate reward for IForest)**

Add synthetic anchor observations: one (score_at_fit_p99, 0) and one (score_at_fit_p1, max_fit_lifetime). The anchor score thresholds are derived from fit-split healthy score statistics (already in `healthy_window_score_stats.csv`). This directly fixes IForest's isotonic floor problem.

For IForest: the fit-split p99 score is 0.617. Adding an anchor at (0.617, 0) tells the isotonic regressor there exist near-zero RUL observations at that score level, pulling the curve down. The anchor must be placed carefully — too aggressive and it creates an artificial cliff that produces near-zero predictions well before actual failure.

Required change: a `anchor_at_zero` parameter in `fit_calibration()` that prepends synthetic (score, 0) pairs before isotonic fitting. Fully backward-compatible.

Statistical concern: the anchor artificially introduces data. Document this clearly. The anchor value is a design choice and its effect on calibration uncertainty is not estimable from 40 engines.

**Approach 3 — Piecewise isotonic calibration (medium risk, potentially high reward)**

Fit separate isotonic regressors for the "bulk" (RUL > 50) and "critical" (RUL ≤ 50) zones. The critical-zone regressor is fitted on near-failure calibration cycles only, without dilution from early-life observations.

Required change: modify `CalibrationBundle` to hold two isotonic regressors plus a boundary score. The `predict_rul` method routes to the appropriate regressor based on whether the input score is above or below the boundary. The boundary must be defined by score (not RUL) to avoid look-ahead leakage.

Statistical concern: the critical-zone regressor has only ~840 observations (all RUL ≤ 50 across 40 cal engines). For IForest, these 840 observations span a very narrow score range (0.53–0.71) with massive overlap. A piecewise fit on such data risks overfitting the tails. This approach is most defensible if combined with per-dataset calibration (which gives separate critical-zone calibrators with less score overlap).

**Approach 4 — Quantile-based calibration (medium risk, changes semantics)**

Fit isotonic regression on the 25th percentile of RUL rather than the mean. This produces a conservative (lower) RUL estimate — appropriate for a safety-critical scheduler that should err on the side of earlier maintenance. The 25th-percentile estimate would be systematically lower than the mean estimate, directly reducing the overestimation problem.

Required change: sklearn's `IsotonicRegression` does not support quantile loss. Would need a custom PAVA implementation with pinball loss or an approximation. Non-trivial to implement robustly. Also changes the interpretation of the output — the RUL estimate is no longer "expected remaining life" but "conservative lower bound". This is arguably more appropriate for the scheduler but must be documented clearly.

### Recommended sequence
1. Implement Approach 1 (per-dataset calibration) first — lowest risk, directly addressable with existing infrastructure.
2. Add Approach 2 (endpoint anchoring) on top of per-dataset calibration for IForest.
3. Evaluate on calibration split and compare against baseline.
4. Only implement Approach 3 (piecewise) if 1+2 leave meaningful near-failure errors.
5. Do not implement Approach 4 in this phase — the quantile change is a semantic decision requiring sign-off on what "RUL estimate" means to the scheduler.

---

## 8. Recommended Final Model Architecture

Based on all existing evidence, the recommended production architecture is:

```
┌──────────────────────────────────────────────────────────────┐
│                  RAJS Risk Module                           │
│                                                             │
│  Input: raw sensor readings for one engine at one cycle    │
│                                                             │
│  1. Preprocessing                                           │
│     - Apply existing StandardScaler (fit-split statistics) │
│     - Compute rolling means (w=10) per engine buffer       │
│     - Select dataset-appropriate feature subset            │
│                                                             │
│  2. Anomaly detection (dataset-specific)                   │
│     - FD001 engine → FD001 OC-SVM                          │
│     - FD003 engine → FD003 OC-SVM                          │
│     (keep IForest as secondary output for comparison)      │
│                                                             │
│  3. Calibration (dataset-specific, with anchoring)         │
│     - FD001 engine → FD001 isotonic calibrator             │
│     - FD003 engine → FD003 isotonic calibrator             │
│                                                             │
│  4. Outputs                                                 │
│     - risk_score ∈ [0, 1]  (degradation risk, not prob)   │
│     - rul_estimate (cycles, with documented uncertainty)   │
│                                                             │
│  5. Scheduler interface                                     │
│     - risk_update event per engine per scheduling step     │
│     - Fields: engine_id, t, risk, rul_estimate, dataset    │
└──────────────────────────────────────────────────────────────┘
```

**Why OC-SVM as the primary model:**
OC-SVM has substantially better near-failure RUL accuracy (MAE 16 vs 52 cycles), lower variance in Spearman ρ across datasets, and wider score range enabling better isotonic calibration. The IForest advantages (marginally better Cohen's d, lower FD001 false-alarm rate) are modest and partially attributable to the score compression problem that feature and model changes are intended to fix. After the experiments are run, this choice should be confirmed or revised.

**Why dataset-specific models:**
s10 is informative for FD003 but noise for FD001. The combined healthy distribution learned by the current model is a compromise. Even if the improvement is small, separate models carry no additional complexity at inference time (the dataset tag is known in advance for C-MAPSS data).

**Why rolling means (w=10):**
The feature analysis shows a 0.15+ improvement in median Spearman ρ for the top sensors when using 10-cycle rolling means. This directly addresses IForest's score compression by making the score signal more monotone — even if we use OC-SVM primarily, the smoother features reduce cycle-to-cycle noise in the OC-SVM scores.

**This is a recommendation, not a conclusion.** The experiments in Phase 2 must either confirm or revise this architecture.

---

## 9. Scheduler Integration Requirements

### Current state
The scheduler environment (`backend/env/`) is an empty stub. `job-shop-lib 1.7.2` is installed and provides job-shop problem instances and a CP solver (`ORToolsSolver`). Gymnasium 1.4.0 is available for the RL environment. No RL policy is yet implemented.

### What the scheduler needs from the risk module
Based on the event log schema (`shared/schemas/event_log.schema.json`):
- `risk_update` events require: `t` (simulation time), `machine` (integer ID), `risk` (float, [0,1])
- The schema also supports `maintenance_start`, `maintenance_end`, `breakdown` events
- Policies referenced: `blind_ppo`, `threshold_spt`, `full_ppo`, `spt`, `lpt`, `fifo`

### Required interface (not yet implemented)
```python
class RiskEngine:
    """Single inference interface for the scheduler."""

    def __init__(self, anomaly_bundle, calibration_bundle):
        ...

    def update(self, engine_id: str, cycle: int, sensor_obs: dict) -> RiskUpdate:
        """Process one cycle of observations for one engine.
        
        Returns RiskUpdate(risk_score, rul_estimate, anomaly_score)
        """
        ...

    def reset_engine(self, engine_id: str) -> None:
        """Reset the rolling buffer when an engine restarts."""
        ...
```

This interface must:
1. Accept raw (unscaled) sensor readings as a dict keyed by sensor name.
2. Apply the StandardScaler internally.
3. Compute rolling-mean features from an internal per-engine buffer.
4. Run the anomaly detector.
5. Run the calibrator.
6. Return risk_score, rul_estimate.

### Missing pieces for integration
- [ ] `RiskEngine` class (above)
- [ ] Gymnasium `JobShopEnv` with machine health state
- [ ] Reward function that incorporates risk (breakdown cost vs maintenance cost)
- [ ] Baseline policies (SPT, LPT, threshold+SPT)
- [ ] RL training loop (PPO via SB3)
- [ ] Multi-seed evaluation harness
- [ ] FastAPI endpoints for `/engines`, `/risk/{engine_id}`, `/schedule`

### Integration constraints
- Risk scores must remain in [0,1] at all times — the `CalibrationBundle.predict_risk()` already enforces this via clamp.
- RUL estimates should be bounded below at 0 — already enforced.
- The scheduler should NOT depend on anomaly scores directly — only on `risk_score` and optionally `rul_estimate`.
- Engine IDs must be consistent between the risk module and the scheduler. The current C-MAPSS IDs (`FD001_042`) may need mapping to integer machine IDs for the event log schema.

---

## 10. Online Demonstration Functionality

### What is currently available for a demo
- Pre-computed anomaly scores for 160 engines (all non-test engines).
- Calibrated risk scores and RUL estimates for the 40 calibration engines.
- 7 diagnostic plots, 7 calibration plots, 10 feature analysis plots, 7 RUL error plots.
- All CSVs needed to drive a static dashboard.

### What must be built for a live demonstration

**A. Engine health dashboard (feasible now)**
- Display engine ID, dataset, current risk score, estimated RUL, historical risk trajectory.
- Show sensor trends for s11, s4, s17 (top features) with degradation markers.
- Indicate health status: healthy (risk < 0.3), degrading (0.3–0.7), critical (> 0.7).
- Data source: pre-computed CSVs for the 40 calibration engines.
- This can be built with the existing calibration output data without any new backend work.

**B. Model comparison panel (feasible now)**
- Side-by-side OC-SVM vs IForest risk trajectories for the same engine.
- Highlight where they diverge (typically FD001 near failure).
- Show the effect of the calibration mapping (raw score vs risk score).
- Data source: `results/cal_scores_ocsvm.csv`, `results/cal_scores_iforest.csv`.

**C. Scheduling integration (requires env implementation)**
- Show a simple job-shop schedule before and after incorporating risk-aware maintenance.
- Use threshold+SPT as the risk-aware baseline (easier than RL for a demo).
- Demonstrate that machines with high risk are scheduled for maintenance before breakdown.
- This requires implementing `backend/env/` — the largest outstanding engineering task.

**D. Demonstration scenarios (feasible with existing data)**
The calibration data already contains representative scenarios:
- Healthy engine: any engine in first 30 cycles (risk ~0.67).
- Gradual degradation: any FD001 engine from cycle 1 to end (clean trajectory).
- Near failure: FD003 engines with lifetime > 300 cycles show clear late-stage acceleration.
- FD003_055 (525 cycles, longest-lived engine) is the most visually compelling for a gradual degradation story.

**E. Explainability (feasible with existing analysis)**
- Show top-3 contributing sensors from the feature analysis (s11, s4, s17).
- Display their current values vs healthy baseline.
- Show how rolling mean tracks smooth degradation vs noisy raw sensor.
- This is display-only — does not require a SHAP or attribution implementation.

### Frontend scaffolding (not yet started)
The `frontend/` directory is empty. To scaffold: `npx create-next-app@latest .` inside `frontend/`. The dashboard should use D3 (already referenced in the README) for time-series charts.

---

## 11. Research and Validation Gaps

### Methodology gaps

**G1 — Healthy window validity**
Cycles 1–30 are assumed undegraded. For FD003 (combined fault modes), degradation may begin before cycle 30. If so, both models are trained on partially degraded data, compressing their healthy-region boundary. Mitigation: check mean sensor values in cycles 1–10 vs 21–30 for FD003 fit engines. If there is a trend, reduce the window to cycles 1–20. This analysis has not been performed.

**G2 — Engine lifetime distribution and calibration bias**
The calibration engines include FD003 engines with lifetimes up to 525 cycles. The `max_lifetime=494` normalisation was derived from the fit split, which may not represent the full lifetime range. Any cal or test engine that exceeds 494 cycles would produce a risk score above 1.0 before clamping — effectively a broken risk scale for long-lived FD003 engines.

**G3 — Calibration metrics are in-sample**
All calibration evaluation (MAE, RMSE, Spearman ρ) uses the same 40 engines used to fit the isotonic regressor. This provides an optimistic estimate. The true performance will not be known until the held-out test split is evaluated. With only 40 calibration engines, there is limited statistical power to distinguish calibration approaches — the test split (40 engines) must be evaluated to reach a reliable conclusion.

**G4 — LOO early-warning is within fit split only**
The early-warning analysis (LOO threshold) was performed within the 120 fit engines. The actual operational threshold would be set using the fit-split data, but applied to calibration and test engines — this transfer has not been tested. The fit-split LOO results are an optimistic estimate of real early-warning performance.

**G5 — No validation against published C-MAPSS benchmarks**
The project does not compare against published RUL estimation benchmarks (e.g. RMSE < 20 on FD001 test set is a standard target in the PHM literature). Our current OC-SVM MAE of ~16 cycles at RUL ≤ 20 is reasonable but not benchmarked.

### Scientific defensibility of current methodology
- Engine-level splitting: correctly implemented, verified by tests. No leakage.
- Healthy-cycle selection: technically correct but assumption of undegraded cycles 1–30 is unvalidated for FD003.
- Isotonic calibration: appropriate for monotone signal-to-target mapping. The conditional-mean interpretation is correct and the output is correctly not labelled as a probability.
- Temporal features: the causal window requirement has been specified and the implementation uses `groupby → rolling` which is correct. Verified in feature_analysis.py.
- Evaluation metrics: per-engine rather than per-cycle, which is correct for comparing models with different engine populations.

---

## 12. Prioritised Implementation Roadmap

### Phase 2 — Controlled experiments (immediate next steps)

**Week 1: Dataset-specific anomaly models**
1. Implement `train_anomaly_dataset_specific.py` — trains Config B and C models using fit-split-only FD001/FD003 engines.
2. Evaluate each model on its calibration subset using the existing `diagnostic.py` analysis functions.
3. Compare vs Config A (existing baseline) on the same calibration engines.
4. Save artifacts as `models/anomaly_fd001.pkl`, `models/anomaly_fd003.pkl`.
5. Run all existing tests to confirm no regressions.

**Week 2: Calibration improvements**
1. Implement per-dataset calibration (Approach 1) in `calibration.py` — add `dataset_filter` parameter to `fit_calibration()`.
2. Implement endpoint anchoring (Approach 2) — add `anchor_at_zero_score` parameter.
3. Evaluate both on their calibration splits with the same binned RUL error analysis.
4. Compare against baseline calibration numbers.

**Week 3: Temporal features**
1. Add `build_temporal_features()` to pipeline (it exists in `feature_analysis.py` — refactor into `cmapss_pipeline.py`).
2. Retrain one anomaly model (OC-SVM, Config B, dataset-specific) with rolling-mean augmented features.
3. Evaluate and compare.

### Phase 3 — Model selection

After Phase 2 experiments, select the final architecture based on:
1. Near-failure RUL MAE (RUL ≤ 20) — primary metric.
2. Risk Spearman ρ on calibration split.
3. False-alarm rate.
4. Complexity and inference time.

**Decision gate:** Do not access the test split until a single final configuration is selected and this decision is documented.

### Phase 4 — Integration

5. Implement `RiskEngine` class for online inference.
6. Implement `backend/env/JobShopEnv` with machine health state.
7. Implement baseline policies (SPT, threshold+SPT).
8. Wire risk updates into the event log schema.
9. Build FastAPI endpoints for the dashboard.

### Phase 5 — Test split evaluation and finalisation

10. Run final evaluation on 40 held-out test engines.
11. Compare against calibration-split results to check for generalisation.
12. Scaffold frontend and build the interactive dashboard.
13. Prepare demonstration scenarios.

---

## Summary Answers

**Should we use separate FD001 and FD003 models?**  
Yes, with high confidence. The sensor signature divergence is verified and structurally justified (FD001 = single fault mode, FD003 = combined fault mode). Even if the performance gain is modest, separate models eliminate noise from irrelevant sensors and allow dataset-specific calibration to address the IForest floor problem.

**Which sensors should we initially test for each dataset?**  
FD001: `s11, s4, s17, s2, s3, s8, s9, s12, s15, s20, s21` (11 features, removes noise, retains FD001-specific signals)  
FD003: `s11, s4, s17, s2, s3, s8, s9, s10, s13, s14` (10 features, adds s10 which is FD003-only)  
Start with Config B (18 features, separate models) first to isolate the contribution of model separation from feature selection.

**Should we retain both OC-SVM and Isolation Forest?**  
Yes, through the experiments. OC-SVM is currently the better calibration model (MAE 16 vs 52 cycles near failure). IForest's score compression may be fixable with temporal features and separate models. Retain both until the experiments show a clear winner. If IForest continues to underperform after the fixes, select OC-SVM as the production model.

**Should we use rolling temporal features?**  
Yes, rolling means with w=10 for the top sensors (s11, s4, s17, s2, s3). The evidence from the feature analysis is strong (median ρ 0.94 vs 0.79 for s11). Implement as a separate experiment to isolate the contribution. Use causal windows, compute per engine, and include in the scaler fit.

**Which calibration approach should we pursue?**  
Approach 1 (per-dataset) first, then Approach 2 (endpoint anchoring for IForest) on top. Evaluate before implementing piecewise calibration (Approach 3). Do not implement quantile calibration (Approach 4) without further discussion on the semantic implications for the scheduler.

**What should we implement next?**  
In order: (1) dataset-specific anomaly models (Config B), (2) per-dataset calibration with anchoring, (3) temporal feature augmentation, (4) RiskEngine online inference interface.

**What is essential for a convincing online demonstration?**  
Minimum viable demo: (1) engine health dashboard showing risk trajectory and sensor trends for a pre-computed calibration engine, (2) model comparison panel (OC-SVM vs IForest), (3) one demonstration scenario (long-lived FD003 engine from health to failure). The scheduler integration (required for the full scheduling comparison) is the largest remaining engineering effort and is not prerequisite for a compelling risk-layer demo.

---

*All 114 existing tests pass. No baseline artifacts have been modified. Test split not accessed.*
