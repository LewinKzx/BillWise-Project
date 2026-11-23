"""BillWise settings. Edit values here when TNB rates change."""

# ---- TNB domestic tariff (from 1 July 2025) ----
ENERGY_RATE = 0.2703      # RM/kWh (usage up to 1,500 kWh)
CAPACITY_RATE = 0.0455    # RM/kWh
NETWORK_RATE = 0.1285     # RM/kWh
RETAIL_CHARGE = 10.0      # RM/month, only above EXEMPTION_THRESHOLD
AFA_RATE = 0.0            # RM/kWh. Changes monthly; 0 matches TNB's comparison table
EXEMPTION_THRESHOLD = 600  # kWh. Set to 800 for the Sep-Dec 2026 relief period
KWTBB_RATE = 0.016        # Renewable energy fund levy
KWTBB_THRESHOLD = 300     # kWh, levy applies above this
SST_RATE = 0.08           # On usage above EXEMPTION_THRESHOLD + retail charge

# Energy Efficiency Incentive: (upper kWh limit, RM/kWh rebate). From TNB table 24-06-2025.
EEI_TABLE = [
    (200, -0.25), (250, -0.245), (300, -0.225), (350, -0.21), (400, -0.17),
    (450, -0.145), (500, -0.12), (550, -0.105), (600, -0.09), (650, -0.075),
    (700, -0.055), (750, -0.045), (800, -0.04), (850, -0.025), (900, -0.01),
    (1000, -0.005),
]

# ---- Environment ----
# Peninsular Malaysia grid emission factor (kg CO2e per kWh), 2024 provisional.
# Source: Suruhanjaya Tenaga, Grid Emission Factor (GEF) in Malaysia 2022-2024
# (Provisional), published 23 Feb 2026.
GRID_EMISSION_FACTOR = 0.740

# ---- Paths ----
DATA_PATH = "data/BillWise_Synthetic_Dataset_v3.xlsx"
DATA_SHEET = "Synthetic Data"
MODEL_PATH = "models/billwise_model.joblib"
REPORT_DIR = "reports"
