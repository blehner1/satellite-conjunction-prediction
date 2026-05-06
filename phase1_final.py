import requests, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from config import USERNAME, PASSWORD

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def parse_tle_text(raw_text, source):
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    records = []
    i = 0
    while i < len(lines) - 2:
        line1, line2 = lines[i], lines[i+1]
        if line1.startswith("1 ") and line2.startswith("2 "):
            try:
                norad_id = int(line1[2:7])
                epoch_str = line1[18:32].strip()
                yr2 = int(epoch_str[:2])
                year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
                day_of_year = float(epoch_str[2:])
                epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + pd.Timedelta(days=day_of_year - 1)
                records.append({"line1": line1, "line2": line2,
                                 "norad_id": norad_id, "epoch": epoch, "source": source})
            except (ValueError, IndexError):
                pass
            i += 2
        else:
            i += 1
    return pd.DataFrame(records)

print("Logging in to Space-Track...")
session = requests.Session()
session.post("https://www.space-track.org/ajaxauth/login",
    data={"identity": USERNAME, "password": PASSWORD}, timeout=30)
print("Login OK.")

print("Downloading TLE data (2,000 objects)...")
resp = session.get(
    "https://www.space-track.org/basicspacedata/query/class/gp/EPOCH/%3Enow-30/orderby/NORAD_CAT_ID/limit/2000/format/tle",
    timeout=60)

raw_path = DATA_DIR / "spacetrack_raw.txt"
with open(raw_path, "w") as f:
    f.write(resp.text)
print(f"Raw TLE saved: {raw_path}")

df = parse_tle_text(resp.text, "spacetrack")
csv_path = DATA_DIR / "spacetrack_parsed.csv"
df.to_csv(csv_path, index=False)

print(f"Total objects: {len(df)}")
print(f"Epoch range:   {df['epoch'].min().date()} to {df['epoch'].max().date()}")
print("Phase 1 complete. Ready for Phase 2.")