"""Convert monthly kWh to an estimated TNB bill (RM).

Same formula as the 'Bill Check' sheet in the dataset; matches TNB's official
domestic bill comparison table to within RM0.20.
"""
import config


def eei_rate(kwh: float) -> float:
    for upper, rate in config.EEI_TABLE:
        if kwh <= upper:
            return rate
    return 0.0


def calculate_bill(kwh: float) -> dict:
    kwh = max(0.0, float(kwh))
    base_rate = config.ENERGY_RATE + config.CAPACITY_RATE + config.NETWORK_RATE
    eei = eei_rate(kwh)
    over = kwh > config.EXEMPTION_THRESHOLD

    usage = kwh * base_rate
    incentive = kwh * eei                                  # negative number
    retail = config.RETAIL_CHARGE if over else 0.0
    afa = kwh * config.AFA_RATE if over else 0.0
    subtotal = usage + incentive + retail + afa
    kwtbb = subtotal * config.KWTBB_RATE if kwh > config.KWTBB_THRESHOLD else 0.0
    sst = 0.0
    if over:
        taxable = (kwh - config.EXEMPTION_THRESHOLD) * (base_rate + eei) + retail
        sst = taxable * config.SST_RATE

    total = subtotal + kwtbb + sst
    return {
        "kwh": round(kwh, 1),
        "usage_charge": round(usage, 2),
        "eei_rebate": round(incentive, 2),
        "retail_charge": round(retail, 2),
        "afa": round(afa, 2),
        "kwtbb": round(kwtbb, 2),
        "sst": round(sst, 2),
        "total": round(total, 2),
    }


if __name__ == "__main__":
    # Quick self-check against TNB's published table
    tnb = {100: 19.45, 300: 65.80, 600: 216.00, 800: 345.90, 1000: 471.20}
    for k, expected in tnb.items():
        print(f"{k:>5} kWh  ours RM{calculate_bill(k)['total']:>7.2f}  TNB RM{expected:>7.2f}")
