import pandas as pd
import numpy as np
from pathlib import Path
from sgp4.api import Satrec, jday
from datetime import datetime, timezone

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load positions and conjunctions ──────────────────────────────────────────
print("Loading data...")
df_pos = pd.read_csv(OUT_DIR / "positions.csv")
df_conj = pd.read_csv(OUT_DIR / "conjunctions.csv")

# ── Load TLE data for additional features ────────────────────────────────────
with open(DATA_DIR / "spacetrack_raw.txt") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

tle_dict = {}
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
            bstar = sat.bstar

            # Altitude estimate from mean motion
            mu = 398600.4418
            Re = 6378.137
            n_rad = mean_motion * 2 * np.pi / 86400
            a = (mu / n_rad**2) ** (1/3)
            alt_km = a - Re

            # Orbit type classification
            if alt_km < 2000:
                orbit_type = "LEO"
            elif alt_km < 35000:
                orbit_type = "MEO"
            else:
                orbit_type = "GEO"

            # Radar cross section estimate from bstar (drag term)
            # Higher bstar = smaller object = more debris-like
            rcs_estimate = max(0.001, 1.0 / (abs(bstar) * 10000 + 0.001))

            tle_dict[norad_id] = {
                "norad_id":   norad_id,
                "inclination": inc,
                "eccentricity": ecc,
                "mean_motion": mean_motion,
                "altitude_km": round(alt_km, 1),
                "orbit_type":  orbit_type,
                "rcs_est":     round(rcs_estimate, 4),
                "bstar":       bstar,
            }
        except Exception:
            pass
        i += 2
    else:
        i += 1

df_tle = pd.DataFrame(tle_dict.values())
tle_path = OUT_DIR / "tle_features.csv"
df_tle.to_csv(tle_path, index=False)
print(f"TLE features built for {len(df_tle)} objects")
print(f"\nOrbit type distribution:")
print(df_tle["orbit_type"].value_counts().to_string())

# ── Build conjunction feature set ─────────────────────────────────────────────
print("\nBuilding conjunction feature set...")

def get_features(norad, df_pos, tle_dict):
    pos = df_pos[df_pos["norad_id"] == norad]
    if pos.empty or norad not in tle_dict:
        return None
    p = pos.iloc[0]
    t = tle_dict[norad]
    altitude = np.sqrt(p.x**2 + p.y**2 + p.z**2) - 6378.137
    speed = np.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
    return {
        "altitude_km":   round(altitude, 1),
        "speed_kms":     round(speed, 4),
        "inclination":   t["inclination"],
        "eccentricity":  t["eccentricity"],
        "orbit_type":    t["orbit_type"],
        "rcs_est":       t["rcs_est"],
    }

records = []
for _, row in df_conj.iterrows():
    fa = get_features(row["norad_a"], df_pos, tle_dict)
    fb = get_features(row["norad_b"], df_pos, tle_dict)
    if fa is None or fb is None:
        continue

    same_orbit = 1 if fa["orbit_type"] == fb["orbit_type"] else 0
    alt_diff = abs(fa["altitude_km"] - fb["altitude_km"])
    combined_rcs = fa["rcs_est"] + fb["rcs_est"]
    avg_alt = (fa["altitude_km"] + fb["altitude_km"]) / 2

    # Risk label: miss_dist < 10km AND rel_vel > 5 km/s = high risk
    risk = 1 if (row["miss_dist_km"] < 10 and row["rel_vel_kms"] > 5) else 0

    records.append({
        "norad_a":         row["norad_a"],
        "norad_b":         row["norad_b"],
        "miss_dist_km":    row["miss_dist_km"],
        "rel_vel_kms":     row["rel_vel_kms"],
        "avg_altitude_km": round(avg_alt, 1),
        "alt_diff_km":     round(alt_diff, 1),
        "combined_rcs":    round(combined_rcs, 4),
        "same_orbit_type": same_orbit,
        "inc_a":           fa["inclination"],
        "inc_b":           fb["inclination"],
        "inc_diff":        round(abs(fa["inclination"] - fb["inclination"]), 4),
        "orbit_type_a":    fa["orbit_type"],
        "orbit_type_b":    fb["orbit_type"],
        "high_risk":       risk,
    })

df_features = pd.DataFrame(records)
feat_path = OUT_DIR / "conjunction_features.csv"
df_features.to_csv(feat_path, index=False)

print(f"Conjunction feature set saved: {feat_path}")
print(f"\nFeature set preview:")
print(df_features.to_string(index=False))
print("\nPhase 3 complete. Ready for Phase 4: Model Training.")