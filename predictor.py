"""Prediction, appliance breakdown, recommendations and extras."""
import copy
import os

import joblib
import numpy as np
import pandas as pd

import config
from appliances import LABELS, appliance_breakdown
from features import clean_household
from tariff import calculate_bill

_bundle = None


def load_model():
    global _bundle
    if _bundle is None:
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError("Model not found. Run `python train.py` first.")
        _bundle = joblib.load(config.MODEL_PATH)
    return _bundle


def _predict_kwh(h: dict, prev_kwh):
    bundle = load_model()
    use_b = prev_kwh is not None and prev_kwh > 0
    m = bundle["b"] if use_b else bundle["a"]
    row = dict(h, prev_kwh=prev_kwh or 0)
    X = pd.DataFrame([[row[f] for f in m["features"]]], columns=m["features"])
    kwh = max(30.0, float(m["model"].predict(X)[0]))
    lo = max(0.0, kwh + m["interval"][0])
    hi = kwh + m["interval"][1]
    return kwh, lo, hi, ("B" if use_b else "A"), m["algorithm"]


def _breakdown(h: dict, total_kwh: float, total_rm: float):
    """Physics split, scaled so appliances add up to the ML prediction."""
    parts = appliance_breakdown(h)
    phys_total = sum(parts.values()) or 1
    scale = total_kwh / phys_total
    items = []
    for key, kwh in parts.items():
        share = kwh / phys_total
        if kwh <= 0:
            continue
        items.append({"key": key, "name": LABELS[key], "kwh": round(kwh * scale, 1),
                      "share": round(share * 100, 1), "cost_rm": round(share * total_rm, 2)})
    items.sort(key=lambda d: -d["kwh"])
    return items, scale


def _what_if(h: dict, changes: dict):
    new = copy.deepcopy(h)
    new.update(changes)
    return clean_household(new)


def _recommendations(h: dict, raw: dict, kwh: float, scale: float):
    """Rule-based tips; savings = physics change x calibration scale, priced on the real tariff."""
    phys_now = sum(appliance_breakdown(h).values())
    bill_now = calculate_bill(kwh)["total"]
    tips = []

    def add(tip_id, title, detail, changes, kind="habit"):
        new_h = _what_if(raw, changes)
        saved = (phys_now - sum(appliance_breakdown(new_h).values())) * scale
        if saved < 3:
            return
        rm = bill_now - calculate_bill(kwh - saved)["total"]
        tips.append({"id": tip_id, "title": title, "detail": detail, "kwh_saved": round(saved, 1),
                     "rm_saved": round(rm, 2), "co2_saved": round(saved * config.GRID_EMISSION_FACTOR, 1),
                     "type": kind, "changes": changes})

    if h["n_ac"] > 0:
        if h["ac_temp"] < 25:
            add("ac_temp", "Set the air-con to 25°C",
                f"You set {int(h['ac_temp'])}°C. Each degree warmer saves about 3% of AC energy. "
                "Use a fan together with the AC to stay comfortable.",
                {"ac_temp": 25})
        if h["ac_hours"] > 4:
            new_hours = max(h["ac_hours"] - 2, 1)
            add("ac_hours", "Cut 2 hours of air-con per unit",
                "Use the AC timer/sleep mode so it turns off in the early morning, "
                "and switch off when leaving the room.",
                {"ac_hours": new_hours})
        if not h["inverter"]:
            add("inverter", "Choose inverter air-cons",
                "Non-inverter units use about 35% more power. Ask your landlord about "
                "replacing old units (long-term saving).",
                {"inverter": "Yes"}, kind="upgrade")
    if h["heater"] and h["shower_min"] > 6:
        add("shower", "Shorter hot showers",
            f"Cutting from {int(h['shower_min'])} to 5 minutes per person matters because "
            "an instant heater draws ~3.6 kW.",
            {"shower_min": 5})
    if h["pc_hours"] > 4:
        new_pc = h["pc_hours"] - 2
        add("pc", "Put PCs to sleep when idle",
            "Enable sleep after 15 minutes and switch off overnight instead of leaving games/downloads running.",
            {"pc_hours": new_pc})
    if h["light_hours"] > 6:
        new_l = h["light_hours"] - 2
        add("lights", "Switch off unused lights",
            "Turn off lights in empty rooms and use daylight during the day.",
            {"light_hours": new_l})
    if h["fridges"] > 1:
        add("fridge", "Share one fridge",
            "A second fridge adds ~33 kWh a month. Combine items into one if possible.",
            {"fridges": h["fridges"] - 1}, kind="upgrade")

    tips.sort(key=lambda t: -t["rm_saved"])
    return tips


def _savings_plan(h: dict, raw: dict, kwh: float, scale: float, tips: list, applied: list):
    """Combine the tips the user clicked 'Try it' on and compute the bill after ALL of them.

    The household's ML correction factor (scale) stays fixed, because the household
    itself hasn't changed, only its habits. Combined changes are simulated together,
    so overlapping tips (e.g. fewer AC hours + inverter) aren't double-counted.
    """
    chosen = [t for t in tips if t["id"] in set(applied or [])]
    if not chosen:
        return None
    changes = {}
    for t in chosen:
        changes.update(t["changes"])
    new_h = _what_if(raw, changes)
    saved = (sum(appliance_breakdown(h).values()) - sum(appliance_breakdown(new_h).values())) * scale
    new_kwh = max(0.0, kwh - saved)
    bill_now, bill_new = calculate_bill(kwh)["total"], calculate_bill(new_kwh)["total"]
    return {"applied": [t["id"] for t in chosen], "kwh": round(new_kwh), "bill": bill_new,
            "kwh_saved": round(saved, 1), "rm_saved": round(bill_now - bill_new, 2),
            "co2_saved": round(saved * config.GRID_EMISSION_FACTOR, 1),
            "per_person": round(bill_new / h["occupants"], 2),
            "tier": _tier_alert(new_kwh)}


def _tier_alert(kwh: float):
    t = config.EXEMPTION_THRESHOLD
    if kwh > t:
        over = kwh - t
        saving = calculate_bill(kwh)["total"] - calculate_bill(t)["total"]
        return {"level": "danger",
                "message": f"You are {over:.0f} kWh above {t} kWh, so you pay the RM10 retail charge "
                           f"and 8% SST. Cutting {over:.0f} kWh (~{over / 30:.1f} kWh/day) "
                           f"would save about RM{saving:.2f}."}
    if kwh > t - 60:
        return {"level": "warning",
                "message": f"Only {t - kwh:.0f} kWh left before {t} kWh, after which retail charge "
                           "and SST apply."}
    return {"level": "ok", "message": f"Below {t} kWh, so no retail charge or SST."}


def predict(raw: dict) -> dict:
    """raw = the household's ORIGINAL habits (+ optional 'accepted_changes').

    The ML model predicts the original household, which fixes its correction factor.
    Accepted changes are then applied on top with the factor kept the same, because
    the household hasn't changed, only its habits. Everything shown (bill, breakdown,
    tips) is for the household AFTER accepted changes.
    """
    h0 = clean_household(raw)
    prev = raw.get("prev_kwh")
    prev = float(prev) if prev not in (None, "", "0") else None
    kwh0, lo0, hi0, model_used, algo = _predict_kwh(h0, prev)
    scale = kwh0 / max(h0["physics_kwh"], 1)

    accepted = raw.get("accepted_changes") or {}
    if accepted:
        raw = {**raw, **accepted}
        h = clean_household(raw)
        kwh = max(0.0, h["physics_kwh"] * scale)
    else:
        h, kwh = h0, kwh0
    lo = max(0.0, kwh + (lo0 - kwh0))
    hi = kwh + (hi0 - kwh0)

    bill = calculate_bill(kwh)
    breakdown, _ = _breakdown(h, kwh, bill["total"])
    tips = _recommendations(h, raw, kwh, scale)
    occupants = int(h["occupants"])

    budget = raw.get("budget")
    budget_info = None
    if budget not in (None, ""):
        b = float(budget)
        diff = bill["total"] - b
        fit = next((k for k in np.arange(kwh, 0, -5) if calculate_bill(k)["total"] <= b), 0)
        budget_info = {"budget": b, "over": diff > 0, "difference": round(abs(diff), 2),
                       "max_kwh": round(float(fit)), "daily_kwh": round(float(fit) / 30, 1)}

    accepted_info = None
    if accepted:
        bill0 = calculate_bill(kwh0)["total"]
        accepted_info = {"original_bill": bill0, "original_kwh": round(kwh0),
                         "rm_saved": round(bill0 - bill["total"], 2),
                         "kwh_saved": round(kwh0 - kwh, 1)}

    return {
        "kwh": round(kwh), "kwh_low": round(lo), "kwh_high": round(hi),
        "bill": bill,
        "bill_low": calculate_bill(lo)["total"], "bill_high": calculate_bill(hi)["total"],
        "per_person": round(bill["total"] / occupants, 2),
        "daily_kwh": round(kwh / 30, 1), "daily_rm": round(bill["total"] / 30, 2),
        "co2_kg": round(kwh * config.GRID_EMISSION_FACTOR),
        "baseline_kwh": round(h["physics_kwh"]),
        "breakdown": breakdown,
        "top_appliance": breakdown[0] if breakdown else None,
        "recommendations": tips,
        "plan": _savings_plan(h, raw, kwh, scale, tips, raw.get("applied")),
        "accepted": accepted_info,
        "tier": _tier_alert(kwh),
        "budget": budget_info,
        "model": {"used": model_used, "algorithm": algo},
    }
