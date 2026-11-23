# BillWise Electricity Bill Predictor for Student Houses in Seri Iskandar

A machine learning web app that runs locally. It:

1. **Predicts the monthly TNB bill**, with a likely range, using the real TNB domestic tariff (EEI, KWTBB, retail charge, SST)
2. **Finds the appliances that cost the most**, with an RM and kWh breakdown
3. **Gives energy-saving recommendations**, each with RM, kWh and CO₂ saved, plus a "Try it" button

Extra features:
- **Try it → Accept selected**: click "Try it" on any tips to preview them, then press **Accept selected**. The changes are applied directly: the main bill, side cards, breakdown, bill details and the form inputs all update. A note in the bill card shows what you accepted and how much you save, with a **Reset** link to undo
- **Live updates**: results change as you edit your household inputs
- **Bill split** per housemate
- **600 kWh warning**: tells you when you're about to pay the retail charge and SST, and how much to cut
- **Budget checker**: tells you the maximum kWh you can use and stay within your budget
- **Carbon footprint** (SDG 13)
- **Two models**: Model A (appliance answers only) and Model B (also uses last month's kWh, more accurate)
- **"About the model" panel**: compares every algorithm against the no-ML baseline

---

## 1. Project structure

```
billwise/
├── data/BillWise-responses.xlsx   survey responses (Google Form, sheet "Form Responses 1")
├── config.py        tariff rates, emission factor, file paths
├── tariff.py        kWh → RM bill (matches TNB's official table)
├── appliances.py    physics estimate per appliance (wattage × hours × days)
├── models_ratio.py  correction-factor model wrapper (ML learns actual ÷ physics)
├── features.py      cleans survey/form answers into model features
├── train.py         trains & evaluates models, saves the best one
├── predictor.py     prediction, breakdown, recommendations, alerts
├── app.py           Flask web server (localhost)
├── templates/index.html
├── static/app.js, static/style.css
├── models/          saved model (created by train.py)
└── reports/         metrics.json + charts for your report (created by train.py)
```

## 2. Setup (one time)

You need **Python 3.10 or newer** and **VS Code** with the Python extension.

1. Unzip the project and open the `billwise` folder in VS Code (File → Open Folder).
2. Open a terminal in VS Code (Terminal → New Terminal).
3. Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   **Mac/Linux**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   You should see `(venv)` at the start of the terminal line.

4. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```

## 3. Train the model

```bash
python train.py
```

This takes a few seconds and prints a table like this:

```
=== Model A (no history): 52 households, 17 features ===
Physics baseline (no ML)   MAE  128.8 kWh
Linear Regression          MAE  106.0 ± 30.4  kWh  RMSE  139.2  R²  0.478  MAPE 24.5%
Ridge Regression           MAE  106.3 ± 30.6  kWh  RMSE  139.4  R²  0.477  MAPE 24.6%
Decision Tree              MAE  122.1 ± 35.8  kWh  RMSE  162.7  R²  0.305  MAPE 27.2%
Random Forest              MAE  111.1 ± 36.0  kWh  RMSE  144.8  R²  0.451  MAPE 25.0%
--> Best: Linear Regression
=== Model B (with last month) ===
Ridge Regression           MAE   44.1 ± 22.5  kWh  RMSE   60.6  R²  0.894  MAPE 9.3%
--> Best: Ridge Regression
```

It saves:
- `models/billwise_model.joblib`, the trained model the website uses
- `reports/metrics.json`, all the scores (use these in your report)
- `reports/model_a.png` and `reports/model_b.png`, predicted-vs-actual and feature-importance charts

## 4. Run the website

```bash
python app.py
```

Your browser opens **http://127.0.0.1:5000** automatically. If it doesn't, open that address yourself. Press `Ctrl + C` in the terminal to stop the app.

---

## 5. How the model is trained

**Goal:** predict monthly **kWh**, then convert kWh to RM with the exact TNB tariff formula (`tariff.py`). The tariff is a fixed rule, so ML is only needed for kWh.

**Model design: correction factor** (`models_ratio.py`)

```
predicted kWh = physics estimate × correction factor
                (wattage × hours)   (learned by ML)
```

- The **physics estimate** responds to every appliance input.
- The **ML model learns the correction factor** (actual kWh ÷ physics estimate) from household context: occupants, number and size of ACs, water heater type, whether everyone was home, and for Model B, last month's kWh compared with the estimate. It learns, for example, that people overestimate their AC hours.
- The model is trained on the log of the factor, and the factor is clipped to 0.3–2.5 so a prediction can never be absurd.

**Why this design?** A plain Random Forest on 52 houses didn't respond reliably to small changes such as shower minutes or AC temperature, and sometimes moved the wrong way. With the correction-factor design, appliance changes always move the bill in the right direction. It was also more accurate:

| Model A (no history) | MAE (kWh) |
|---|---|
| Physics baseline (no ML) | 128.8 |
| Correction factor, Linear Regression | **106.0** |

| Model B (+ last month) | MAE (kWh) |
|---|---|
| Correction factor, Ridge | **44.1** |

(In an earlier version on the previous dataset, a plain Random Forest predicting kWh directly scored 100.1 kWh for Model A and 45.7 kWh for Model B, against 84.7 and 35.0 for the correction-factor design.)

**Steps in `train.py`:**

1. **Load and clean** (`features.py`). Text answers become numbers (Yes → 1, "1.5 HP" → 1.5, "Don't know" → 1.5 HP, no AC → 0 hours).
2. **Physics estimate** for every house (`appliances.py`).
3. **Compare algorithms** (Linear, Ridge, Decision Tree, Random Forest) as the correction-factor learner, against the physics baseline.
4. **Repeated 5-fold cross-validation.** Train on 4/5 of the houses, test on 1/5, rotate 5 times, and repeat with 3 shuffles. With only 52 houses, this is more reliable than a single 80/20 split, which would test on about 10 houses.
5. **Metrics.** MAE (average kWh error, the main metric), RMSE, R² and MAPE (average % error).
6. **Pick the lowest MAE** and retrain it on all houses.
7. **Prediction range.** The 10th–90th percentile of cross-validated errors becomes the "likely RM… – RM…" range.

**How "Try it" and "Accept selected" work:** "Try it" previews tips in a green card. "Accept selected" applies them: the form inputs change to the new habits, and everything is recalculated. Behind the scenes the model still receives your *original* inputs plus the accepted changes, so your household's ML correction factor stays the same and the new bill equals the preview exactly. (Re-predicting from scratch with last month's kWh would cancel out the saving, because last month's bill reflects your old habits.) If you edit an accepted field by hand afterwards, that value becomes your real habit. Combined tips are simulated together, so overlapping savings aren't double-counted.

**Appliance breakdown:** the physics estimate per appliance, scaled by your correction factor so it adds up to the predicted total.



## 6. Using real survey data 

1. Download the Google Form responses as Excel and save it as `data/BillWise-responses.xlsx` (or change `DATA_PATH` / `DATA_SHEET` in `config.py`).
2. Keep the question wording the same. `COLUMN_MAP` in `features.py` links each form question to the model's inputs; if you reword a question, update it there too. `train.py` stops with a clear error listing any question it can't find.
3. Run `python train.py` again.


## 8. Updating the tariff

All rates are in `config.py`. Each month, update `AFA_RATE` from mytnb.com.my/tariff. For the Sep–Dec 2026 relief period, set `EXEMPTION_THRESHOLD = 800`. `GRID_EMISSION_FACTOR` is 0.740 kg CO₂e/kWh (Suruhanjaya Tenaga, 2024 provisional); update it when a newer figure is published.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `Model not found` | Run `python train.py` first |
| `python` not recognised | Use `py` (Windows) or `python3` (Mac) |
| `ModuleNotFoundError` | Activate the venv, then `pip install -r requirements.txt` |
| Port 5000 busy (Mac AirPlay) | Change `PORT = 5000` to `PORT = 5001` in `app.py` |
| Browser opens before the page is ready | Refresh the page once |
