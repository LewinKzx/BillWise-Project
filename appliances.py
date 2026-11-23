"""Engineering (physics) estimate of appliance energy use.

Used for three things:
  1. A baseline prediction to compare the ML model against.
  2. An input feature ('physics_kwh') for the ML model.
  3. Splitting the predicted bill into appliances and costing recommendations.

Energy formula (TNB, tnb.com.my/residential/energy-savings-at-home):
    kWh = power (W) x hours x 30 days / 1000
Sources for the main figures:
    AC 0.78 kW/HP  - Panasonic Malaysia 1.5 HP non-inverter spec (1,160 W input)
    Inverter -35%  - Malaysian manufacturers report 30-50% savings (e.g. Midea Malaysia)
    Water heater   - Panasonic Malaysia instant heater spec (3.6 kW)
Other wattages and adjustments are modelling assumptions.
"""

DAYS_PER_MONTH = 30

AC_KW_PER_HP = 0.78          # Panasonic Malaysia 1.5 HP spec: 1,160 W / 1.5 HP
INVERTER_FACTOR = 0.65       # inverter ACs use ~35% less
TEMP_FACTOR_PER_C = 0.03     # ~3% more energy per °C below 24°C
HEATER_KW = {"instant": 3.6, "tank": 2.5}
FRIDGE_KWH_DAY = 1.1
PC_KW = 0.25
FAN_KW = 0.055
LIGHT_KW = 0.012             # LED bulb/tube
WASHER_KWH_LOAD = 0.5
BASE_KWH_DAY = 0.8           # router, chargers, standby
BASE_KWH_PER_PERSON = 0.25   # kettle, rice cooker, iron, phone, etc.

LABELS = {
    "ac": "Air-conditioners",
    "water_heater": "Water heater",
    "fridge": "Refrigerator",
    "pc": "Desktop / gaming PC",
    "fan": "Fans",
    "lights": "Lights",
    "washer": "Washing machine",
    "other": "Other & standby",
}


def appliance_breakdown(h: dict, days: int = DAYS_PER_MONTH) -> dict:
    """Monthly kWh per appliance for one household (dict of cleaned inputs)."""
    temp_f = 1 + TEMP_FACTOR_PER_C * (24 - h["ac_temp"]) if h["n_ac"] > 0 else 1
    inv_f = INVERTER_FACTOR if h["inverter"] else 1.0
    ac = h["n_ac"] * h["ac_hp"] * AC_KW_PER_HP * h["ac_hours"] * inv_f * max(temp_f, 0.7)
    heater_kw = HEATER_KW["instant"] if h["heater_instant"] else HEATER_KW["tank"]
    heater = h["heater"] * h["occupants"] * h["shower_min"] / 60 * heater_kw
    daily = {
        "ac": ac,
        "water_heater": heater,
        "fridge": h["fridges"] * FRIDGE_KWH_DAY,
        "pc": h["pc_hours"] * PC_KW,
        "fan": h["fan_hours"] * FAN_KW,
        "lights": h["lights"] * h["light_hours"] * LIGHT_KW,
        "washer": h["washer_loads"] / 7 * WASHER_KWH_LOAD,
        "other": BASE_KWH_DAY + BASE_KWH_PER_PERSON * h["occupants"],
    }
    occupancy = 1.0 if h["all_home"] else 0.92
    return {k: v * days * occupancy for k, v in daily.items()}


def physics_kwh(h: dict) -> float:
    return sum(appliance_breakdown(h).values())
