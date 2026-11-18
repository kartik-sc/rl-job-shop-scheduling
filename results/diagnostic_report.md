# OC-SVM vs Isolation Forest — Diagnostic Report
## RAJS Risk Layer · Fit-Split Analysis · 2026-10-29

---

## 1. Implementation Verification

### Confirmed configuration (from bundle + code inspection)

| Parameter | Value |
|---|---|
| Datasets | FD001 + FD003 (C-MAPSS) |
| Total engines | 200 |
| Fit / Cal / Test split | 120 / 40 / 40 |
| FD001 fit engines | 58 |
| FD003 fit engines | 62 |
| Features (after constant removal) | 18 (op1, op2, s2–s4, s6–s15, s17, s20, s21) |
| Dropped constant cols | op3, s1, s5, s16, s18, s19 |
| Healthy training window | cycles 1–30 |
| OC-SVM config | RBF kernel, nu=0.05, gamma=scale |
| IForest config | 200 estimators, contamination=auto, random_state=42 |
| SPLIT_SEED | 42 |

### Score convention
Both models use **higher score = more anomalous**:
- OC-SVM: `-decision_function(X)` — sklearn decision_function is positive for inliers; negated so positives indicate anomalies.
- IForest: `-score_samples(X)` — sklearn score_samples returns negative average isolation path length (less negative = more normal); negated so higher = more anomalous.

The negation is consistent and verified. The two models' raw score scales are **not comparable by magnitude** — OC-SVM scores are centred around 0 with range roughly [-12, +20], while IForest scores sit in a narrower positive band around [0.38, 0.65]. Raw separation differences must not be used to judge which model is "stronger".

### No retraining performed
The existing `models/anomaly_bundle.pkl` was loaded directly. Data was re-preprocessed via `build_pipeline` with the same `SPLIT_SEED=42` to reconstruct the scaled DataFrame and verify alignment with the bundle's split. No calibration or test engines were used at any stage.

---

## 2. Scored Population

- **27,055 total scored rows** across 120 fit engines
- RUL verified: every engine's last cycle has RUL = 0 ✓
- Cycle monotonicity: every engine's RUL decrements by exactly -1/cycle ✓
- No calibration or test engines appear in the scored data ✓

---

## 3. Healthy vs Late-Life Separation

**Window definitions:**
- Healthy window: cycle ≤ 30 (exactly 30 rows per engine)
- Late-life window: RUL < 30 (final 30 cycles by true RUL)

### Overall (120 engines)

| Model | Healthy mean | Late-life mean | Sep (late−healthy) | Norm sep (Cohen's d) | Directional success |
|---|---|---|---|---|---|
| OC-SVM | ≈ −4.84 | ≈ +11.76 | **16.60** | 7.95 | **100%** |
| IForest | ≈ 0.44 | ≈ 0.64 | **0.19** | 8.65 | **100%** |

**Key observations:**
- Both models show 100% directional success — every single fit engine has a higher late-life mean score than its healthy-window mean. This is a strong result.
- Raw separation magnitudes differ massively (OC-SVM ~16.6 vs IForest ~0.19) because the score scales are incomparable.
- **Normalised separation (Cohen's d-like pooled effect size) is the right comparison**: IForest (8.65) slightly edges OC-SVM (7.95), though bootstrap analysis shows this is borderline (see §7).
- OC-SVM has somewhat higher IQR on normalised separation (4.53 vs 3.60), suggesting slightly more variability across engines.

### FD001 vs FD003

| Model | Metric | FD001 mean | FD003 mean |
|---|---|---|---|
| OC-SVM | norm_sep | 7.65 | 8.24 |
| IForest | norm_sep | 8.06 | 9.20 |
| OC-SVM | dir_success | 100% | 100% |
| IForest | dir_success | 100% | 100% |

Both models perform slightly better on FD003 (combined fault mode) than FD001 (single fault mode). IForest shows a larger FD001→FD003 gap (8.06→9.20) than OC-SVM (7.65→8.24). This may reflect IForest's higher sensitivity to the richer degradation patterns in FD003.

---

## 4. Spearman Correlation with Degradation Stage

Degradation stage = −RUL (larger = closer to failure).
Positive Spearman ρ = score increases monotonically as engine degrades.

**Undefined cases: 0 for both models** — no engine had a constant score sequence.

### Overall (120 engines)

| Model | Mean ρ | Median ρ | Std | Q25 | Q75 | % Positive |
|---|---|---|---|---|---|---|
| OC-SVM | 0.750 | 0.781 | 0.119 | 0.693 | 0.835 | **100%** |
| IForest | 0.771 | 0.828 | 0.168 | 0.740 | 0.883 | **100%** |

Both models have 100% positive per-engine Spearman correlation — scores increase as every fit engine approaches failure. This is strong and unambiguous.

IForest has a slightly higher median ρ (0.828 vs 0.781) and Q75 (0.883 vs 0.835), suggesting more consistently monotone degradation signals on the better-behaved engines. However, IForest also has higher std (0.168 vs 0.119) and a lower Q25 (0.740 vs 0.693), indicating more engine-to-engine variability — some IForest engine scores are less monotone than the worst OC-SVM engines.

### FD001 vs FD003

| Model | FD001 mean ρ | FD001 median ρ | FD003 mean ρ | FD003 median ρ |
|---|---|---|---|---|
| OC-SVM | 0.752 | 0.780 | 0.749 | 0.788 |
| IForest | 0.792 | 0.833 | 0.752 | 0.822 |

OC-SVM is nearly identical across datasets (std 0.115 FD001, 0.122 FD003). IForest performs better on FD001 in mean ρ terms but shows higher variance on FD003 (std 0.197 vs 0.129 on FD001). This may make OC-SVM more predictable as a downstream risk signal.

**Pooled Spearman warning:** Pooled cycle-level ρ (OC-SVM: 0.631, IForest: 0.554) is lower than per-engine medians because between-engine lifetime differences dilute the within-engine monotone signal. Pooled figures must not be used for model selection — they are reported only for reference.

**Note:** Spearman ρ measures monotonic association between score and degradation stage. It does not imply causal relationship or calibrated failure probability.

---

## 5. Early-Warning Analysis

**Threshold procedure:** Leave-one-engine-out (LOO) within the fit split. For each target engine, the threshold = 95th percentile of healthy-window (cycle ≤ 30) scores pooled from the other 119 fit engines. The target engine's own healthy data never contributes to its threshold.

**Warning criterion:** A warning is triggered when at least **3 consecutive cycles** exceed the threshold (persistence requirement to suppress single-cycle noise spikes).

### Overall (120 engines)

| Model | % Warned | Lead time mean | Lead time median | Lead std | Mean FA rate | Missed |
|---|---|---|---|---|---|---|
| OC-SVM | **100%** | 84.7 cycles | 62.5 cycles | 76.1 | 5.06% | 0 |
| IForest | **100%** | 80.5 cycles | 56.0 cycles | 80.5 | 5.11% | 0 |

Both models warn on all 120 engines — zero missed detections. This is a strong result, though it partially reflects that C-MAPSS engines have long lifetimes and the 95th-percentile threshold is permissive enough to trigger eventually on every engine.

The high standard deviation of lead times (OC-SVM: 76.1, IForest: 80.5) relative to the mean indicates wide spread. Some engines are warned very early (>200 cycles before failure); others receive warnings only a few dozen cycles out. This spread is driven mostly by the FD003 engines (see below).

### FD001 vs FD003

| Model | Dataset | Lead mean | Lead median | Lead std | FA rate (mean) |
|---|---|---|---|---|---|
| OC-SVM | FD001 | 60.8 | 60.0 | 22.0 | 2.76% |
| OC-SVM | FD003 | 107.1 | 74.5 | 98.9 | 7.20% |
| IForest | FD001 | 52.6 | 52.5 | 21.0 | 0.17% |
| IForest | FD003 | 106.7 | 63.5 | 103.9 | 9.73% |

**FD001 is dramatically different from FD003:**
- FD001 lead times are tightly clustered (~50–60 cycles, std ~21), consistent behaviour.
- FD003 lead times are dispersed (std ~99–104), driven by FD003's longer and more variable engine lifetimes and combined fault modes that trigger the detectors very early on some engines.
- IForest has remarkably low false-alarm rates on FD001 (median = 0%), compared to OC-SVM (median ≈ 1.7%). On FD003, IForest has higher false alarm rates (mean 9.7% vs OC-SVM 7.2%).

**Important caveat:** High lead times on FD003 do not necessarily mean better predictive power. A warning issued 200 cycles before failure on a 300-cycle engine may simply reflect early onset of degradation in FD003's combined fault modes, not superior detector sensitivity. The persistence filter (3 cycles) helps reduce false spikes but does not eliminate structural false alarms from engines with early-onset degradation.

---

## 6. Statistical Comparison (Paired Bootstrap)

All comparisons are engine-level (n=120 engines) paired bootstrap with 5,000 resamples. "Positive diff = OC-SVM better".

| Metric | OC-SVM | IForest | Diff (OC-SVM − IF) | 95% CI | p-value |
|---|---|---|---|---|---|
| sep (raw) | 16.60 | 0.19 | +16.40 | [+16.00, +16.80] | 0.000 |
| norm_sep | 7.95 | 8.65 | −0.70 | [−1.20, −0.19] | **0.007** |
| rho | 0.750 | 0.771 | −0.021 | [−0.036, −0.003] | **0.022** |
| lead_time | 84.7 | 80.5 | +4.2 | [−0.14, +7.93] | 0.057 |

**Interpretation:**
- **Raw separation** is meaningless for comparison (different scales) — OC-SVM's larger raw numbers are an artefact of score range, not better performance.
- **Normalised separation**: IForest is statistically significantly better (p=0.007, CI does not contain 0). Effect size is small (−0.70 on a normalised scale where both models score ~8).
- **Spearman ρ**: IForest is statistically significantly better (p=0.022), but the mean difference is small (−0.021).
- **Lead time**: OC-SVM has a slightly longer mean lead time (+4.2 cycles), but the 95% CI just crosses zero (p=0.057). Not conclusive.

**These are paired engine-level bootstrap confidence intervals, not cycle-level statistics.** The differences are statistically significant at α=0.05 for norm_sep and rho, but the effect sizes are modest.

---

## 7. Model Assessment and Recommendation

### Criteria (pre-specified)

| Criterion | OC-SVM | IForest | Winner |
|---|---|---|---|
| Reliable healthy/late-life separation | 100% dir success, norm_sep 7.95 | 100% dir success, norm_sep 8.65 | **IForest** (marginally) |
| Consistent degradation association | Median ρ 0.781, std 0.119 | Median ρ 0.828, std 0.168 | Mixed |
| Useful early-warning (lead time) | 62.5 cycles median | 56.0 cycles median | **OC-SVM** (marginally) |
| Acceptable false-alarm behaviour | FA rate 5.1% overall, 2.8% FD001 | FA rate 5.1% overall, 0.2% FD001 | **IForest on FD001** |
| Reproducibility | Deterministic (no RNG) | Seeded (random_state=42) | Both good |
| Downstream calibration suitability | Smooth, wide range, easier to calibrate | Narrow range, more stable percentile position | Open question |

### Summary

The evidence is **mixed but leans slightly toward IForest** on the two most fundamental quality metrics (normalised separation effect size and Spearman ρ). However:

- IForest's advantage is statistically significant but small in absolute terms — both models perform strongly on all 120 fit engines.
- IForest shows notably lower false-alarm rates on FD001 (0.2% vs 2.8% median) — a meaningful practical advantage if FD001-type engines are common in deployment.
- OC-SVM has slightly longer early-warning lead times and lower variance in Spearman ρ, making it more predictable.
- IForest's higher variance in both ρ and lead time across FD003 is a concern for reliability.

**Recommendation: retain both models for the calibration-stage comparison.** Neither model's advantage is decisive enough to eliminate the other before calibration. The downstream risk signal depends on how well scores can be mapped to failure probabilities — this can only be assessed on the held-out calibration split (40 engines), which must not be touched until after this diagnostic is reviewed.

If forced to choose one for prototyping, **IForest is the marginal preference** based on better normalised separation and lower false-alarm rates on FD001.

---

## 8. Assumptions and Limitations

1. **Healthy window at cycle ≤ 30 assumes early cycles are undegraded.** C-MAPSS does not guarantee this — FD003's combined fault modes may begin earlier. If early cycles already contain mild degradation, both models are trained on partially degraded data, which would compress the healthy–late-life separation.

2. **Late-life window = final 30 cycles (RUL < 30).** For engines with very long lifetimes (FD003 has engines with 200+ cycles), this window may be representative. For shorter engines, it captures a larger fraction of the lifetime.

3. **The LOO threshold approach is leakage-free but not cross-validated.** A more robust early-warning estimate would use K-fold at the engine level and evaluate on genuinely held-out data. Current results are all within the fit split.

4. **No hyperparameter tuning was performed.** The diagnostic uses the existing bundle. Any observed weaknesses (e.g. IForest's FD003 false alarm rate) should be diagnosed as tuning candidates for a later, separately evaluated experiment.

5. **Pooled correlation is misleading.** OC-SVM's pooled ρ (0.631) exceeds IForest's (0.554) — the opposite of the per-engine conclusion. This inversion illustrates exactly why per-engine analysis is required and pooled figures are purely descriptive.

---

## 9. Files Created / Modified

### New code
| File | Description |
|---|---|
| `backend/risk/diagnostic.py` | Full diagnostic analysis module |
| `tests/test_diagnostic.py` | 26 diagnostic tests |

### Generated outputs (git-ignored)
| Path | Contents |
|---|---|
| `results/all_fit_scores.csv` | 27,055 rows: engine_id, dataset, cycle, RUL, both model scores, window flags |
| `results/per_engine_separation.csv` | 120 rows: healthy/late-life means, sep, norm_sep, dir_success per engine |
| `results/per_engine_spearman.csv` | 120 rows: Spearman ρ per engine, both models |
| `results/per_engine_early_warning.csv` | 240 rows: warned, lead time, FA rate per engine × model |
| `results/bootstrap_comparison.csv` | Paired bootstrap results for 4 metrics |
| `results/healthy_window_score_stats.csv` | Score distribution stats in the healthy window |
| `results/summary_separation_{overall,FD001,FD003}.csv` | Aggregate separation by scope |
| `results/summary_spearman_{overall,FD001,FD003}.csv` | Aggregate Spearman by scope |
| `results/summary_early_warning_{overall,FD001,FD003}.csv` | Aggregate EW by scope |
| `plots/diagnostic/FD001_score_trajectories.png` | Score vs RUL for sample FD001 engines |
| `plots/diagnostic/FD003_score_trajectories.png` | Score vs RUL for sample FD003 engines |
| `plots/diagnostic/score_distributions.png` | Violin: healthy vs late-life score distributions |
| `plots/diagnostic/separation_distributions.png` | Per-engine separation by dataset |
| `plots/diagnostic/spearman_distributions.png` | Per-engine Spearman ρ by dataset |
| `plots/diagnostic/model_comparison.png` | OC-SVM vs IForest scatter (sep, ρ) |
| `plots/diagnostic/early_warning_summary.png` | Lead time and FA rate boxplots |

### Test results
- Pre-existing suite: **50/50 passed** (no regressions)
- New diagnostic tests: **26/26 passed**
- **Total: 76/76 passed**

---

## 10. Stopping Point

This report covers only fit-split engines. **Calibration and test sets have not been touched.** The next step — if you approve these results — is conformal calibration of the anomaly scores on the 40-engine calibration split to produce valid coverage guarantees for the downstream risk signal. That step requires a separate experiment.
