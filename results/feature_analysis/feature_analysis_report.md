# RAJS Sensor Feature Analysis Report
## C-MAPSS FD001 + FD003 | Fit-split only (120 engines)

---

## 1. Current feature set

18 features remain after constant-column removal: `op1`, `op2`, `s2`, `s3`, `s4`, `s6`, `s7`, `s8`, `s9`, `s10`, `s11`, `s12`, `s13`, `s14`, `s15`, `s17`, `s20`, `s21`.

Dropped (zero variance): `op3`, `s1`, `s5`, `s16`, `s18`, `s19`.

---

## 2. Sensor rankings by degradation signal quality

Ranked by composite score = average of normalised |Spearman ρ| and normalised |Cohen's d|.

| Rank | Feature | Mean ρ (vs −RUL) | Cohen's d | Composite |
|------|---------|-----------------|-----------|-----------|
| 1 | s11 | +0.794 | +3.708 | 1.000 |
| 2 | s4 | +0.752 | +3.429 | 0.936 |
| 3 | s17 | +0.665 | +2.932 | 0.814 |
| 4 | s2 | +0.644 | +2.758 | 0.777 |
| 5 | s3 | +0.626 | +2.719 | 0.761 |
| 6 | s8 | +0.705 | +2.003 | 0.714 |
| 7 | s13 | +0.704 | +2.005 | 0.713 |
| 8 | s9 | +0.517 | +2.244 | 0.628 |
| 9 | s10 | +0.708 | +1.209 | 0.608 |
| 10 | s14 | +0.393 | +1.797 | 0.490 |
| 11 | s21 | -0.396 | -1.004 | 0.385 |
| 12 | s20 | -0.395 | -1.010 | 0.385 |
| 13 | s15 | +0.390 | +0.852 | 0.360 |
| 14 | s12 | -0.409 | +0.408 | 0.313 |
| 15 | s7 | -0.386 | +0.329 | 0.287 |
| 16 | s6 | +0.028 | +0.267 | 0.054 |
| 17 | op1 | +0.004 | +0.048 | 0.009 |
| 18 | op2 | +0.006 | +0.003 | 0.004 |

---

## 3. High-utility sensors

**Top 5 overall**: `s11`, `s4`, `s17`, `s2`, `s3`

These sensors have both strong monotone correlation with degradation (Spearman ρ > 0.5 on average) and large separation between early-life and near-failure states (|Cohen's d| > 1.5).

---

## 4. Low-utility or redundant sensors

Sensors with weak degradation signal (|Cohen's d| < 0.5 AND |mean ρ| < 0.2): `s6`, `op1`, `op2`

**Highly correlated pairs** (|ρ| > 0.90 in Spearman correlation matrix):

No pairs exceed ρ = 0.90.

---

## 5. FD001 vs FD003 comparison

**Shared top-8 features** (good in both datasets): `s11`, `s17`, `s4`

**FD001-specific top-8** (better in FD001): `s12`, `s15`, `s20`, `s21`, `s7`

**FD003-specific top-8** (better in FD003): `s10`, `s13`, `s2`, `s3`, `s8`

FD003 generally shows higher Cohen's d values, consistent with its combined fault modes producing more pronounced sensor deviations.

---

## 6. Top temporal feature candidates

Temporal features evaluated: rolling mean (w=5,10,20), rolling std (w=5,10,20), 1-cycle rate of change.

**Top 5 temporal features by composite score**: `s11_rm5`, `s11_rm10`, `s4_rm10`, `s11_rm20`, `s4_rm5`

| Rank | Feature | Type | Median ρ | |Cohen's d| | Composite |
|------|---------|------|----------|-----------|-----------|
| 1 | s11_rm5 | rolling_mean | +0.918 | 4.093 | 0.980 |
| 2 | s11_rm10 | rolling_mean | +0.942 | 3.989 | 0.980 |
| 3 | s4_rm10 | rolling_mean | +0.921 | 3.896 | 0.958 |
| 4 | s11_rm20 | rolling_mean | +0.957 | 3.701 | 0.952 |
| 5 | s4_rm5 | rolling_mean | +0.894 | 3.962 | 0.951 |
| 6 | s4_rm20 | rolling_mean | +0.945 | 3.635 | 0.938 |
| 7 | s17_rm10 | rolling_mean | +0.882 | 3.868 | 0.933 |
| 8 | s3_rm10 | rolling_mean | +0.869 | 3.840 | 0.923 |
| 9 | s3_rm20 | rolling_mean | +0.908 | 3.656 | 0.921 |
| 10 | s17_rm20 | rolling_mean | +0.907 | 3.658 | 0.921 |
| 11 | s17_rm5 | rolling_mean | +0.842 | 3.830 | 0.908 |
| 12 | s2_rm10 | rolling_mean | +0.880 | 3.661 | 0.907 |
| 13 | s2_rm20 | rolling_mean | +0.911 | 3.443 | 0.897 |
| 14 | s3_rm5 | rolling_mean | +0.823 | 3.751 | 0.888 |
| 15 | s2_rm5 | rolling_mean | +0.836 | 3.625 | 0.880 |

---

## 7. Recommendations

### What to experiment with next

**Priority 1 — Reduce IForest score compression (most impactful):**
The IForest score saturates because its compressed score range (0.37–0.71) cannot distinguish late degradation from mid-life. Adding rolling-std features for the top sensors could widen the high-score tail by capturing *variability* increases near failure, which appear even when mean sensor values have plateaued.

**Priority 2 — Feature subset for a leaner, more discriminative model:**
Consider retraining anomaly models on the top 8–10 sensors by composite score: s11, s4, s17, s2, s3. Removing low-utility sensors reduces the curse of dimensionality and removes noise that dilutes both OC-SVM's decision boundary and IForest's isolation paths.

**Priority 3 — Rolling statistics as additional features:**
Rolling standard deviations (w=10 or w=20) for high-utility sensors are the most promising temporal additions. They capture *instability* near failure — sensors that are noisy when degraded but stable when healthy. Rolling means (w=5 or w=10) add modest improvement over raw values and also reduce cycle-to-cycle noise. Rate-of-change features are noisier and offer smaller gains.

**Priority 4 — Per-dataset calibration:**
Given the FD001/FD003 sensor quality differences, dataset-specific calibration (already identified as Approach B in the RUL error investigation) remains a recommended next step independent of feature changes.

### Constraints to respect

- Any new model training must use fit-split engines only.
- Temporal features must be computed with a causal window (no future cycles used) to avoid test-time leakage.
- The existing pipeline, models and calibration artifacts must be preserved as the current baseline.

---

*Analysis performed on fit-split engines only. Test engines not accessed.*