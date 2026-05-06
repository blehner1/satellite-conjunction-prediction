import pandas as pd
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load TLE data ─────────────────────────────────────────────────────────────
print("Loading TLE data...")
with open(DATA_DIR / "spacetrack_raw.txt") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

satellites = []
i = 0
while i < len(lines) - 1:
    line1, line2 = lines[i], lines[i+1]
    if line1.startswith("1 ") and line2.startswith("2 "):
        try:
            sat = Satrec.twoline2rv(line1, line2)
            satellites.append({"norad_id": int(line1[2:7]), "sat": sat})
        except Exception:
            pass
        i += 2
    else:
        i += 1

print(f"Loaded {len(satellites)} satellite objects")

# ── Propagate positions at a single epoch ────────────────────────────────────
# We compute positions at NOW and check pairwise distances
TARGET_TIME = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
jd, fr = jday(TARGET_TIME.year, TARGET_TIME.month, TARGET_TIME.day,
              TARGET_TIME.hour, TARGET_TIME.minute, TARGET_TIME.second)

print(f"Propagating orbits to {TARGET_TIME.strftime('%Y-%m-%d %H:%M UTC')}...")

positions = []
for obj in satellites:
    try:
        e, r, v = obj["sat"].sgp4(jd, fr)
        if e == 0:  # 0 = no error
            positions.append({
                "norad_id": obj["norad_id"],
                "x": r[0], "y": r[1], "z": r[2],      # km, ECI frame
                "vx": v[0], "vy": v[1], "vz": v[2],   # km/s
            })
    except Exception:
        pass

df_pos = pd.DataFrame(positions)
print(f"Successfully propagated: {len(df_pos)} objects")

pos_path = OUT_DIR / "positions.csv"
df_pos.to_csv(pos_path, index=False)
print(f"Positions saved: {pos_path}")

# ── Compute pairwise conjunction screening ────────────────────────────────────
# Flag any pair of objects within 10 km of each other
THRESHOLD_KM = 10.0
print(f"\nScreening all pairs for close approaches (threshold: {THRESHOLD_KM} km)...")
print("(This may take a minute for 2,000 objects...)")

coords = df_pos[["x","y","z"]].values
vels   = df_pos[["vx","vy","vz"]].values
ids    = df_pos["norad_id"].values

conjunctions = []
n = len(coords)

for i in range(n):
    for j in range(i+1, n):
        diff = coords[i] - coords[j]
        dist = np.linalg.norm(diff)
        if dist < THRESHOLD_KM:
            rel_vel = np.linalg.norm(vels[i] - vels[j])
            conjunctions.append({
                "norad_a":      ids[i],
                "norad_b":      ids[j],
                "miss_dist_km": round(dist, 4),
                "rel_vel_kms":  round(rel_vel, 4),
            })

df_conj = pd.DataFrame(conjunctions)

if df_conj.empty:
    print("No conjunctions found within threshold — widening to 50 km for output...")
    THRESHOLD_KM = 50.0
    conjunctions = []
    for i in range(n):
        for j in range(i+1, n):
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist < THRESHOLD_KM:
                rel_vel = np.linalg.norm(vels[i] - vels[j])
                conjunctions.append({
                    "norad_a":      ids[i],
                    "norad_b":      ids[j],
                    "miss_dist_km": round(dist, 4),
                    "rel_vel_kms":  round(rel_vel, 4),
                })
    df_conj = pd.DataFrame(conjunctions)

conj_path = OUT_DIR / "conjunctions.csv"
df_conj.to_csv(conj_path, index=False)

print(f"\nConjunction events found: {len(df_conj)}")
if not df_conj.empty:
    print(f"Closest approach:        {df_conj['miss_dist_km'].min()} km")
    print(f"Highest relative vel:    {df_conj['rel_vel_kms'].max()} km/s")
    print(f"\nTop 10 closest pairs:")
    print(df_conj.sort_values("miss_dist_km").head(10).to_string(index=False))

print(f"\nConjunctions saved: {conj_path}")
print("Phase 2 complete. Ready for Phase 3: Feature Engineering.")