"""Phase 3 pre-implementation inspection — read-only, no artifacts modified."""
import pickle, pathlib
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = pathlib.Path(__file__).parent.parent.parent
AE   = ROOT / "results" / "anomaly_experiments"
EXP  = ROOT / "models" / "experiments"

# ── 1. Calibration data density by RUL bin ────────────────────────────────
print("=" * 70)
print("1. CALIBRATION DATA DENSITY BY RUL BIN")
print("=" * 70)
for cfg in ("A", "D"):
    for ds in ("fd001", "fd003"):
        fname = AE / f"scores_{cfg}_{ds}.csv"
        if not fname.exists():
            continue
        df = pd.read_csv(fname)
        total = len(df)
        n_eng = df["engine_id"].nunique()
        lifetimes = df.groupby("engine_id")["cycle"].max()
        print(f"\nConfig {cfg} {ds.upper()} ({n_eng} engines, "
              f"lifetime median={lifetimes.median():.0f} max={lifetimes.max():.0f})")
        rc = "rul" if "rul" in df.columns else "rul_true"
        for lo, hi in [(0,5),(6,10),(11,20),(21,50),(51,100),(101,200),(201,9999)]:
            n = ((df[rc] >= lo) & (df[rc] <= hi)).sum()
            print(f"  RUL {lo:3d}-{hi:4d}: {n:5d} rows  ({n/total*100:.1f}%)")

# ── 2. Score distributions at key RUL bands (baseline A vs D_temporal) ───
print("\n" + "=" * 70)
print("2. SCORE DISTRIBUTION AT RUL BANDS: A_shared vs D_temporal")
print("=" * 70)
for cfg in ("A", "D"):
    for ds in ("fd001", "fd003"):
        fname = AE / f"scores_{cfg}_{ds}.csv"
        if not fname.exists():
            continue
        df = pd.read_csv(fname)
        df["_rul"] = df["rul"] if "rul" in df.columns else df["rul_true"]
        print(f"\nConfig {cfg} {ds.upper()} — OC-SVM scores:")
        for lo, hi in [(0,5),(11,20),(21,50),(101,200)]:
            sub = df[(df["_rul"] >= lo) & (df["_rul"] <= hi)]["ocsvm_score"]
            if len(sub): print(f"  RUL {lo:3d}-{hi:3d}: mean={sub.mean():.4f}  "
                               f"p25={sub.quantile(.25):.4f}  p75={sub.quantile(.75):.4f}  "
                               f"max={sub.max():.4f}  n={len(sub)}")
        print(f"Config {cfg} {ds.upper()} — IForest scores:")
        for lo, hi in [(0,5),(11,20),(21,50),(101,200)]:
            sub = df[(df["_rul"] >= lo) & (df["_rul"] <= hi)]["iforest_score"]
            if len(sub): print(f"  RUL {lo:3d}-{hi:3d}: mean={sub.mean():.4f}  "
                               f"p25={sub.quantile(.25):.4f}  p75={sub.quantile(.75):.4f}  "
                               f"max={sub.max():.4f}  n={len(sub)}")

# ── 3. Isotonic regressor inspection: what does it predict near failure? ──
print("\n" + "=" * 70)
print("3. EXISTING ISOTONIC REGRESSOR INSPECTION (baseline calibrators)")
print("=" * 70)
for mname, path in [
    ("ocsvm",   ROOT / "models" / "calibration_ocsvm.pkl"),
    ("iforest", ROOT / "models" / "calibration_iforest.pkl"),
]:
    if not path.exists(): continue
    with open(path, "rb") as fh:
        cb = pickle.load(fh)
    iso = cb.iso_reg
    print(f"\n{mname.upper()}:")
    print(f"  score_min={cb.score_min:.4f}  score_max={cb.score_max:.4f}  "
          f"max_lifetime={cb.max_lifetime:.0f}")
    print(f"  isotonic knots: {len(iso.X_thresholds_)}")
    # What does the regressor predict at the top 5% of scores?
    top_scores = np.linspace(cb.score_min + 0.95*(cb.score_max - cb.score_min),
                              cb.score_max, 10)
    preds = iso.predict(top_scores)
    print(f"  Predictions at top 5% of score range:")
    for s, p in zip(top_scores, preds):
        print(f"    score={s:.4f}  → predicted_RUL={p:.1f}")
    # Effective floor
    all_s = np.linspace(cb.score_min, cb.score_max, 10000)
    all_p = iso.predict(all_s)
    floor_score = all_s[all_p <= 5].min() if (all_p <= 5).any() else np.nan
    pct_leq20   = (all_p <= 20).mean()
    print(f"  Min predicted RUL: {all_p.min():.1f}")
    print(f"  Score that first achieves RUL≤5: {floor_score:.4f}")
    print(f"  Fraction of score range mapping to RUL≤20: {pct_leq20:.1%}")

# ── 4. Trajectory dominance: do long-lived engines swamp the calibration? ─
print("\n" + "=" * 70)
print("4. TRAJECTORY DOMINANCE IN CALIBRATION DATA")
print("=" * 70)
for cfg in ("A", "D"):
    df_all = []
    for ds in ("fd001", "fd003"):
        f = AE / f"scores_{cfg}_{ds}.csv"
        if f.exists(): df_all.append(pd.read_csv(f))
    if not df_all: continue
    df = pd.concat(df_all, ignore_index=True)
    rc = "rul" if "rul" in df.columns else "rul_true"
    lifetimes = df.groupby("engine_id")["cycle"].max().sort_values(ascending=False)
    total_rows = len(df)
    top5 = lifetimes.head(5).index.tolist()
    top5_rows = df[df["engine_id"].isin(top5)]
    nf_rows   = df[df[rc] <= 20]
    nf_top5   = nf_rows[nf_rows["engine_id"].isin(top5)]
    print(f"\nConfig {cfg}:")
    print(f"  Total rows: {total_rows}  near-failure (RUL≤20): {len(nf_rows)} "
          f"({len(nf_rows)/total_rows*100:.1f}%)")
    print(f"  Top-5 longest engines: {lifetimes.head(5).values.tolist()}")
    print(f"  Top-5 rows: {len(top5_rows)} ({len(top5_rows)/total_rows*100:.1f}% of total)")
    print(f"  Near-failure rows from top-5 engines: {len(nf_top5)} "
          f"({len(nf_top5)/max(len(nf_rows),1)*100:.1f}% of all NF rows)")

# ── 5. Per-engine lifetime histogram for calibration set ─────────────────
print("\n" + "=" * 70)
print("5. CALIBRATION ENGINE LIFETIME DISTRIBUTION")
print("=" * 70)
for cfg in ("A",):
    for ds in ("fd001", "fd003"):
        f = AE / f"scores_{cfg}_{ds}.csv"
        if not f.exists(): continue
        df = pd.read_csv(f)
        lifetimes = df.groupby("engine_id")["cycle"].max().sort_values()
        print(f"\n{ds.upper()} calibration engine lifetimes:")
        for cutoff, label in [(100,"≤100"),(200,"101-200"),(300,"201-300"),(9999,">300")]:
            lo = 0 if label=="≤100" else int(label.split("-")[0])
            hi = cutoff
            count = ((lifetimes >= lo) & (lifetimes <= hi)).sum()
            print(f"  {label}: {count} engines")
        print(f"  Shortest 5: {lifetimes.head(5).values.tolist()}")
        print(f"  Longest  5: {lifetimes.tail(5).values.tolist()}")

# ── 6. Isotonic regression: where does it have knots near failure? ────────
print("\n" + "=" * 70)
print("6. ISOTONIC KNOT DENSITY NEAR FAILURE (baseline calibrators)")
print("=" * 70)
for mname, path in [
    ("ocsvm",   ROOT / "models" / "calibration_ocsvm.pkl"),
    ("iforest", ROOT / "models" / "calibration_iforest.pkl"),
]:
    if not path.exists(): continue
    with open(path, "rb") as fh:
        cb = pickle.load(fh)
    # X_thresholds_ are the distinct score values used as knots
    thresholds  = cb.iso_reg.X_thresholds_
    y_thresholds = cb.iso_reg.y_thresholds_
    # Find knots where predicted RUL ≤ 50
    near_fail_mask = y_thresholds <= 50
    print(f"\n{mname.upper()}: {near_fail_mask.sum()} of {len(thresholds)} knots predict RUL≤50")
    if near_fail_mask.any():
        idx = np.where(near_fail_mask)[0]
        print(f"  First knot in NF zone: score={thresholds[idx[0]]:.4f}  "
              f"pred_RUL={y_thresholds[idx[0]]:.1f}")
        print(f"  Last  knot in NF zone: score={thresholds[idx[-1]]:.4f}  "
              f"pred_RUL={y_thresholds[idx[-1]]:.1f}")

# ── 7. Score overlap for D_temporal (the Phase 3 input) ──────────────────
print("\n" + "=" * 70)
print("7. D_TEMPORAL SCORE STATISTICS (the Phase 3 input signal)")
print("=" * 70)
for ds in ("fd001", "fd003"):
    f = AE / f"scores_D_{ds}.csv"
    if not f.exists(): continue
    df = pd.read_csv(f)
    for model in ("ocsvm", "iforest"):
        col = f"{model}_score"
        el  = df[df["rul_true"] >  100][col]
        nf  = df[df["rul_true"] <= 20][col]
        print(f"\nD_temporal {ds.upper()} {model.upper()}:")
        print(f"  Early-life (RUL>100) range: [{el.min():.4f}, {el.max():.4f}]  "
              f"mean={el.mean():.4f}")
        print(f"  Near-fail  (RUL≤20)  range: [{nf.min():.4f}, {nf.max():.4f}]  "
              f"mean={nf.mean():.4f}")
        overlap = ((nf >= el.min()) & (nf <= el.max())).mean()
        print(f"  Overlap fraction: {overlap:.1%}  "
              f"Gap: {nf.mean()-el.mean():.4f}")
        # Distribution of scores exactly at RUL=0,1,2,...,5
        print(f"  Score at RUL 0-2: {df[df['rul_true']<=2][col].mean():.4f}")
        print(f"  Score at RUL 3-5: {df[(df['rul_true']>=3)&(df['rul_true']<=5)][col].mean():.4f}")
