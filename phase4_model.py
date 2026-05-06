import pandas as pd
import numpy as np
from pathlib import Path
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading TLE data...")
with open(DATA_DIR / "spacetrack_raw.txt") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

satellites = []
tle_meta = {}
i = 0
while i < len(lines) - 1:
    line1, line2 = lines[i], lines[i+1]
    if line1.startswith("1 ") and line2.startswith("2 "):
        try:
            norad_id = int(line1[2:7])
            sat = Satrec.twoline2rv(line1, line2)
            inc = float(line2[8:16])
            ecc = float("0." + line2[26:33])
            mean_motion = float(line2[52:63])
            mu = 398600.4418
            Re = 6378.137
            n_rad = mean_motion * 2 * np.pi / 86400
            a = (mu / n_rad**2) ** (1/3)
            alt_km = a - Re
            orbit_type = "LEO" if alt_km < 2000 else ("MEO" if alt_km < 35000 else "GEO")
            bstar = sat.bstar
            rcs_est = max(0.001, 1.0 / (abs(bstar) * 10000 + 0.001))
            satellites.append({"norad_id": norad_id, "sat": sat})
            tle_meta[norad_id] = {
                "inclination": inc, "eccentricity": ecc,
                "altitude_km": round(alt_km, 1), "orbit_type": orbit_type,
                "rcs_est": round(rcs_est, 4)
            }
        except Exception:
            pass
        i += 2
    else:
        i += 1

print(f"Loaded {len(satellites)} objects")

# Use all LEO objects only — most conjunctions happen in LEO
leo_sats = [s for s in satellites if tle_meta.get(s["norad_id"], {}).get("orbit_type") == "LEO"]
print(f"LEO objects (highest conjunction risk): {len(leo_sats)}")

# Screen at 500km threshold to generate enough training data
THRESHOLD_KM = 500.0
HIGH_RISK_KM = 50.0
print(f"Propagating {len(leo_sats)} LEO objects across 30 time steps...")
print(f"Screening threshold: {THRESHOLD_KM} km | High risk label: < {HIGH_RISK_KM} km")

base_time = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
time_steps = [base_time + timedelta(hours=h*4) for h in range(30)]

all_conjunctions = []

for t_idx, t in enumerate(time_steps):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
    positions = []
    for obj in leo_sats:
        try:
            e, r, v = obj["sat"].sgp4(jd, fr)
            if e == 0:
                positions.append({
                    "norad_id": obj["norad_id"],
                    "x": r[0], "y": r[1], "z": r[2],
                    "vx": v[0], "vy": v[1], "vz": v[2],
                })
        except Exception:
            pass

    coords = np.array([[p["x"], p["y"], p["z"]] for p in positions])
    vels   = np.array([[p["vx"], p["vy"], p["vz"]] for p in positions])
    ids    = [p["norad_id"] for p in positions]
    n = len(coords)

    for i in range(n):
        for j in range(i+1, n):
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist < THRESHOLD_KM:
                rel_vel = np.linalg.norm(vels[i] - vels[j])
                na, nb = ids[i], ids[j]
                ma = tle_meta.get(na, {})
                mb = tle_meta.get(nb, {})
                if not ma or not mb:
                    continue
                alt_a = np.linalg.norm(coords[i]) - 6378.137
                alt_b = np.linalg.norm(coords[j]) - 6378.137
                all_conjunctions.append({
                    "norad_a":         na,
                    "norad_b":         nb,
                    "miss_dist_km":    round(dist, 4),
                    "rel_vel_kms":     round(rel_vel, 4),
                    "avg_altitude_km": round((alt_a + alt_b) / 2, 1),
                    "alt_diff_km":     round(abs(alt_a - alt_b), 1),
                    "combined_rcs":    round(ma["rcs_est"] + mb["rcs_est"], 4),
                    "same_orbit_type": 1,
                    "inc_diff":        round(abs(ma["inclination"] - mb["inclination"]), 4),
                    "high_risk":       1 if dist < HIGH_RISK_KM else 0,
                    "time_step":       t_idx,
                })

    if (t_idx + 1) % 10 == 0:
        print(f"  Step {t_idx+1}/30 — {len(all_conjunctions)} events so far")

df = pd.DataFrame(all_conjunctions)
print(f"\nTotal conjunction events: {len(df)}")
print(f"High risk (< {HIGH_RISK_KM} km):  {df['high_risk'].sum()}")
print(f"Low risk:                 {(df['high_risk']==0).sum()}")
df.to_csv(OUT_DIR / "full_conjunction_dataset.csv", index=False)

# ── Train model ───────────────────────────────────────────────────────────────
FEATURES = ["miss_dist_km", "rel_vel_kms", "avg_altitude_km",
            "alt_diff_km", "combined_rcs", "same_orbit_type", "inc_diff"]

X = df[FEATURES]
y = df["high_risk"]

print(f"\nTraining Random Forest on {len(df)} samples ({y.sum()} high risk)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, max_depth=6,
                              class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n── Model Evaluation ──────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

print("\n── Feature Importance ────────────────────────────")
for feat, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<22} {bar} {imp:.4f}")

with open(OUT_DIR / "conjunction_model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("\nModel saved: data/processed/conjunction_model.pkl")
print("Phase 4 complete. Ready for Phase 5: Visualization.")