import pandas as pd
import numpy as np
from pathlib import Path
from sgp4.api import Satrec, jday
from datetime import datetime, timezone
import plotly.graph_objects as go
import webbrowser

DATA_DIR = Path("data/raw")
OUT_DIR  = Path("data/processed")

print("Loading data...")
df_full = pd.read_csv(OUT_DIR / "full_conjunction_dataset.csv")
df_risk = df_full[df_full["high_risk"] == 1].copy()
print(f"High risk conjunction events: {len(df_risk)}")
print("Rebuilding positions...")
with open(DATA_DIR / "spacetrack_raw.txt") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

TARGET_TIME = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
jd, fr = jday(TARGET_TIME.year, TARGET_TIME.month, TARGET_TIME.day,
              TARGET_TIME.hour, TARGET_TIME.minute, TARGET_TIME.second)

positions = []
i = 0
while i < len(lines) - 1:
    line1, line2 = lines[i], lines[i+1]
    if line1.startswith("1 ") and line2.startswith("2 "):
        try:
            norad_id = int(line1[2:7])
            sat = Satrec.twoline2rv(line1, line2)
            e, r, v = sat.sgp4(jd, fr)
            if e == 0:
                alt = np.linalg.norm(r) - 6378.137
                orbit_type = "LEO" if alt < 2000 else ("MEO" if alt < 35000 else "GEO")
                positions.append({
                    "norad_id": norad_id,
                    "x": r[0], "y": r[1], "z": r[2],
                    "alt_km": round(alt, 1),
                    "orbit_type": orbit_type,
                })
        except Exception:
            pass
        i += 2
    else:
        i += 1

df_all = pd.DataFrame(positions)
print(f"Positions ready: {len(df_all)} objects")

# ── Earth sphere ──────────────────────────────────────────────────────────────
Re = 6378.137
u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 80)
ex = Re * np.outer(np.cos(u), np.sin(v))
ey = Re * np.outer(np.sin(u), np.sin(v))
ez = Re * np.outer(np.ones(np.size(u)), np.cos(v))
ecolor = np.abs(np.outer(np.cos(u), np.cos(v)))

traces = []
traces.append(go.Surface(
    x=ex, y=ey, z=ez,
    surfacecolor=ecolor,
    colorscale=[
        [0.0, "#0a2a4a"],
        [0.3, "#1a5276"],
        [0.6, "#1e8449"],
        [1.0, "#27ae60"],
    ],
    showscale=False,
    opacity=1.0,
    hoverinfo="skip",
    name="Earth",
    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
))

# ── Satellites by orbit type ──────────────────────────────────────────────────
orbit_styles = {
    "LEO": dict(color="#4fc3f7", size=2, opacity=0.7),
    "MEO": dict(color="#fff176", size=3, opacity=0.8),
    "GEO": dict(color="#ef9a9a", size=3, opacity=0.8),
}

for orbit, style in orbit_styles.items():
    subset = df_all[df_all["orbit_type"] == orbit]
    traces.append(go.Scatter3d(
        x=subset["x"], y=subset["y"], z=subset["z"],
        mode="markers",
        marker=dict(size=style["size"], color=style["color"],
                    opacity=style["opacity"]),
        name=f"{orbit} ({len(subset)} objects)",
        hovertemplate="NORAD: %{customdata}<br>Alt: %{text} km<extra></extra>",
        customdata=subset["norad_id"],
        text=subset["alt_km"],
    ))

# ── High risk conjunction lines ───────────────────────────────────────────────
high_risk_pairs = df_risk[["norad_a","norad_b","miss_dist_km","rel_vel_kms"]]\
    .drop_duplicates(subset=["norad_a","norad_b"])

pos_lookup = df_all.set_index("norad_id")[["x","y","z"]].to_dict("index")

conj_x, conj_y, conj_z = [], [], []
for _, row in high_risk_pairs.iterrows():
    a, b = int(row["norad_a"]), int(row["norad_b"])
    if a in pos_lookup and b in pos_lookup:
        pa, pb = pos_lookup[a], pos_lookup[b]
        conj_x += [pa["x"], pb["x"], None]
        conj_y += [pa["y"], pb["y"], None]
        conj_z += [pa["z"], pb["z"], None]

if conj_x:
    traces.append(go.Scatter3d(
        x=conj_x, y=conj_y, z=conj_z,
        mode="lines",
        line=dict(color="#ff1744", width=4),
        name=f"High risk conjunctions ({len(high_risk_pairs)})",
        hoverinfo="skip",
        opacity=0.9,
    ))

risk_norads = set(high_risk_pairs["norad_a"].tolist() +
                  high_risk_pairs["norad_b"].tolist())
risk_pos = df_all[df_all["norad_id"].isin(risk_norads)]
if not risk_pos.empty:
    traces.append(go.Scatter3d(
        x=risk_pos["x"], y=risk_pos["y"], z=risk_pos["z"],
        mode="markers",
        marker=dict(size=6, color="#ff1744", symbol="diamond",
                    line=dict(color="white", width=1)),
        name=f"High risk objects ({len(risk_pos)})",
        hovertemplate="NORAD: %{customdata}<br>Alt: %{text} km<extra></extra>",
        customdata=risk_pos["norad_id"],
        text=risk_pos["alt_km"],
    ))

# ── Layout ────────────────────────────────────────────────────────────────────
fig = go.Figure(data=traces)
fig.update_layout(
    title=dict(
        text="Satellite Conjunction Risk — Space Domain Awareness Dashboard<br>"
             f"<sub>{len(df_all)} tracked objects | "
             f"{len(high_risk_pairs)} high-risk pairs flagged | "
             f"Epoch: {TARGET_TIME.strftime('%Y-%m-%d %H:%M UTC')}</sub>",
        font=dict(size=16, color="white"),
        x=0.5,
    ),
    paper_bgcolor="#0d1117",
    scene=dict(
        bgcolor="#0d1117",
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title=""),
        zaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title=""),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)),
    ),
    legend=dict(
        font=dict(color="white", size=12),
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        x=0.01, y=0.99,
    ),
    margin=dict(l=0, r=0, t=80, b=0),
    height=800,
)

out_path = OUT_DIR / "conjunction_dashboard.html"
fig.write_html(str(out_path))
print(f"Dashboard saved: {out_path}")
webbrowser.open(str(out_path.resolve()))
print("Phase 5 complete.")