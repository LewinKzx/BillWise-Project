"""Turn survey rows / web form inputs into the numeric features the model uses."""
import math

import pandas as pd

from appliances import physics_kwh

# Survey column -> internal name
COLUMN_MAP = {
    "Occupants": "occupants",
    "No. of air-conditioners": "n_ac",
    "Typical AC horsepower": "ac_hp",
    "Inverter AC?": "inverter",
    "AC hours/day (each)": "ac_hours",
    "AC temperature (°C)": "ac_temp",
    "Electric water heater?": "heater",
    "Water heater type": "heater_type",
    "Hot shower minutes/person/day": "shower_min",
    "Refrigerators": "fridges",
    "PC hours/day": "pc_hours",
    "Total fan hours/day": "fan_hours",
    "No. of lights": "lights",
    "Light hours/day": "light_hours",
    "Washing machine loads/week": "washer_loads",
    "All housemates home this month?": "all_home",
    "Oct 2026 kWh": "kwh_oct",
    "Sep 2026 kWh": "kwh_sep",
    "Aug 2026 kWh": "kwh_aug",
}

BASE_FEATURES = [
    "occupants", "n_ac", "ac_hp", "inverter", "ac_hours", "ac_temp", "heater",
    "heater_instant", "shower_min", "fridges", "pc_hours", "fan_hours", "lights",
    "light_hours", "washer_loads", "all_home", "physics_kwh",
]
HISTORY_FEATURES = BASE_FEATURES + ["prev_kwh"]

DEFAULT_HP = 1.5   # used when the respondent answers "Don't know"


def _num(value, default=0.0):
    try:
        v = float(str(value).replace("HP", "").strip())
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _yes(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"yes", "y", "true", "1", "on"}


def clean_household(raw: dict) -> dict:
    """Works for both a dataset row (already renamed) and a web form submission."""
    n_ac = int(_num(raw.get("n_ac")))
    h = {
        "occupants": max(1.0, _num(raw.get("occupants"), 1)),
        "n_ac": n_ac,
        "ac_hp": _num(raw.get("ac_hp"), DEFAULT_HP) if n_ac else 0.0,
        "inverter": 1.0 if (n_ac and _yes(raw.get("inverter"))) else 0.0,
        "ac_hours": _num(raw.get("ac_hours")) if n_ac else 0.0,
        "ac_temp": _num(raw.get("ac_temp"), 24) if n_ac else 24.0,
        "heater": 1.0 if _yes(raw.get("heater")) else 0.0,
        "shower_min": _num(raw.get("shower_min")),
        "fridges": _num(raw.get("fridges"), 1),
        "pc_hours": _num(raw.get("pc_hours")),
        "fan_hours": _num(raw.get("fan_hours")),
        "lights": _num(raw.get("lights")),
        "light_hours": _num(raw.get("light_hours")),
        "washer_loads": _num(raw.get("washer_loads")),
        "all_home": 1.0 if _yes(raw.get("all_home", "Yes")) else 0.0,
    }
    heater_type = str(raw.get("heater_type", "Instant")).lower()
    h["heater_instant"] = 1.0 if (h["heater"] and "tank" not in heater_type) else 0.0
    if not h["heater"]:
        h["shower_min"] = 0.0
    h["physics_kwh"] = physics_kwh(h)
    return h


def load_dataset(path: str, sheet: str) -> pd.DataFrame:
    """Read the survey/synthetic Excel file into a clean feature table."""
    df = pd.read_excel(path, sheet_name=sheet).rename(columns=COLUMN_MAP)
    rows = []
    for _, r in df.iterrows():
        h = clean_household(r.to_dict())
        h["kwh_oct"] = _num(r.get("kwh_oct"), float("nan"))
        h["kwh_sep"] = _num(r.get("kwh_sep"), float("nan"))
        h["kwh_aug"] = _num(r.get("kwh_aug"), float("nan"))
        rows.append(h)
    out = pd.DataFrame(rows)
    return out.dropna(subset=["kwh_oct"]).reset_index(drop=True)
