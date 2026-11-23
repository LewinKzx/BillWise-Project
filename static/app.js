const form = document.getElementById("form");
const $ = (id) => document.getElementById(id);
const rm = (v) => (v < 0 ? "−RM" : "RM") + Math.abs(Number(v)).toLocaleString("en-MY", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// Tips being previewed with "Try it"
const applied = new Set();
// Accepted changes: field -> { from: original value, to: new value }
// The form shows the NEW values, but the model is given the ORIGINAL values plus
// these changes, so the household's ML correction factor stays the same.
const accepted = {};
const acceptedTitles = [];   // [{ title, fields: [...] }]
let last = null;             // last prediction response

function formData() {
  const data = {};
  new FormData(form).forEach((v, k) => (data[k] = v));
  const changes = {};
  for (const [field, c] of Object.entries(accepted)) {
    data[field] = c.from;
    changes[field] = c.to;
  }
  data.accepted_changes = changes;
  data.applied = [...applied];
  return data;
}

function updateSliderLabels() {
  document.querySelectorAll(".val").forEach((el) => {
    el.textContent = form.elements[el.dataset.for].value;
  });
}

let timer;
function schedule() {
  updateSliderLabels();
  clearTimeout(timer);
  timer = setTimeout(predict, 250);
}

async function predict() {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    $("error").classList.add("hidden");
    last = data;
    render(data);
  } catch (e) {
    $("error").textContent = e.message;
    $("error").classList.remove("hidden");
  }
}

// ---------- Accept / reset ----------
function acceptSelected() {
  if (!last) return;
  last.recommendations.filter((t) => applied.has(t.id)).forEach((t) => {
    for (const [field, value] of Object.entries(t.changes)) {
      const el = form.elements[field];
      const from = accepted[field] ? accepted[field].from : el.value;
      accepted[field] = { from, to: String(value) };
      el.value = value;   // show the new habit in the form
    }
    acceptedTitles.push({ title: t.title, fields: Object.keys(t.changes) });
  });
  applied.clear();
  updateSliderLabels();
  predict();
}

function resetAccepted() {
  for (const [field, c] of Object.entries(accepted)) {
    form.elements[field].value = c.from;
    delete accepted[field];
  }
  acceptedTitles.length = 0;
  updateSliderLabels();
  predict();
}

// If the user edits a field by hand, that becomes their real habit
form.addEventListener("input", (e) => {
  if (accepted[e.target.name]) {
    delete accepted[e.target.name];
    for (let i = acceptedTitles.length - 1; i >= 0; i--) {
      if (!acceptedTitles[i].fields.some((f) => accepted[f])) acceptedTitles.splice(i, 1);
    }
  }
  schedule();
});

// ---------- Rendering ----------
function render(d) {
  $("bill").textContent = rm(d.bill.total);
  $("billRange").textContent = `likely ${rm(d.bill_low)} – ${rm(d.bill_high)}`;
  $("kwh").textContent = `${d.kwh} kWh`;
  $("daily").textContent = `${d.daily_kwh} kWh/day · ${rm(d.daily_rm)}/day`;
  $("perPerson").textContent = rm(d.per_person);
  $("co2").textContent = d.co2_kg;

  const note = $("acceptedNote");
  if (d.accepted && acceptedTitles.length) {
    note.classList.remove("hidden");
    note.innerHTML = `
      <div>✓ ${acceptedTitles.map((t) => t.title).join(" · ")}</div>
      <div>Was ${rm(d.accepted.original_bill)} → saving ${rm(d.accepted.rm_saved)}/month
        (${rm(d.accepted.rm_saved * 12)}/year)
        <button type="button" class="link-light" id="resetAccepted">Reset</button></div>`;
    $("resetAccepted").onclick = resetAccepted;
  } else note.classList.add("hidden");

  const tier = $("tier");
  tier.className = "alert " + d.tier.level;
  tier.textContent = d.tier.message;

  const b = $("budget");
  if (d.budget) {
    b.className = "alert " + (d.budget.over ? "danger" : "ok");
    b.textContent = d.budget.over
      ? `Over your ${rm(d.budget.budget)} budget by ${rm(d.budget.difference)}. To stay within budget, use at most ${d.budget.max_kwh} kWh (${d.budget.daily_kwh} kWh/day).`
      : `Within your ${rm(d.budget.budget)} budget with ${rm(d.budget.difference)} to spare.`;
  } else b.className = "alert hidden";

  if (d.top_appliance) {
    $("top").innerHTML = `Biggest cost: <b>${d.top_appliance.name}</b>, about ${rm(d.top_appliance.cost_rm)} (${d.top_appliance.share}% of your bill).`;
  }
  const max = Math.max(...d.breakdown.map((i) => i.share));
  $("breakdown").innerHTML = d.breakdown
    .map((i) => `
      <div class="bar-row">
        <span class="bar-name">${i.name}</span>
        <div class="bar-track"><div class="bar" style="width:${(i.share / max) * 100}%"></div></div>
        <span class="bar-val">${rm(i.cost_rm)} · ${i.kwh} kWh</span>
      </div>`)
    .join("");

  // Recommendations (accepted ones disappear because their condition no longer applies)
  const ids = new Set(d.recommendations.map((t) => t.id));
  [...applied].forEach((id) => { if (!ids.has(id)) applied.delete(id); });

  $("tips").innerHTML = d.recommendations.length
    ? d.recommendations.map((t) => {
        const on = applied.has(t.id);
        return `
      <div class="tip ${on ? "on" : ""}">
        <div>
          <b>${t.title}</b> ${t.type === "upgrade" ? '<span class="tag">upgrade</span>' : ""}
          <p>${t.detail}</p>
        </div>
        <div class="tip-save">
          <span class="save">−${rm(t.rm_saved)}</span>
          <span class="sub">${t.kwh_saved} kWh · ${t.co2_saved} kg CO₂</span>
          <button type="button" class="${on ? "ghost" : ""}" data-id="${t.id}">${on ? "Undo" : "Try it"}</button>
        </div>
      </div>`;
      }).join("")
    : "<p>No more recommendations, your usage looks efficient. 👍</p>";

  document.querySelectorAll("#tips button").forEach((btn) => {
    btn.onclick = () => {
      const id = btn.dataset.id;
      applied.has(id) ? applied.delete(id) : applied.add(id);
      predict();
    };
  });

  const n = applied.size;
  $("acceptBar").classList.toggle("hidden", n === 0);
  $("acceptBarText").textContent = `${n} tip${n > 1 ? "s" : ""} selected`;

  // Preview card (only while trying tips)
  const plan = $("plan");
  if (d.plan) {
    plan.classList.remove("hidden");
    $("planBill").textContent = rm(d.plan.bill);
    $("planSave").textContent = `save ${rm(d.plan.rm_saved)}/month`;
    $("planSub").textContent =
      `${d.kwh} → ${d.plan.kwh} kWh · ${rm(d.plan.per_person)} per person · ` +
      `${d.plan.co2_saved} kg less CO₂ · ${rm(d.plan.rm_saved * 12)} saved per year`;
  } else plan.classList.add("hidden");

  const bill = d.bill;
  const rows = [
    ["Usage charge (energy + capacity + network)", bill.usage_charge],
    ["Energy Efficiency Incentive", bill.eei_rebate],
    ["Retail charge", bill.retail_charge],
    ["AFA (fuel adjustment)", bill.afa],
    ["KWTBB (1.6%)", bill.kwtbb],
    ["SST (8%)", bill.sst],
  ];
  $("billTable").innerHTML =
    rows.map(([n, v]) => `<tr><td>${n}</td><td>${rm(v)}</td></tr>`).join("") +
    `<tr class="total"><td>Total</td><td>${rm(bill.total)}</td></tr>`;
  $("modelNote").textContent =
    `Prediction by Model ${d.model.used} (${d.model.algorithm})` +
    (d.model.used === "A" ? ". Enter last month's kWh to use the more accurate Model B." : " using your last bill.") +
    ` Simple appliance calculation (no ML) would say ${d.baseline_kwh} kWh.`;
}

async function loadModelInfo() {
  const res = await fetch("/api/model-info");
  const m = await res.json();
  if (!res.ok) return ($("modelInfo").textContent = m.error);
  const table = (key, title) => {
    const r = m[key].results;
    return `<h3>${title} · best: ${m[key].algorithm}</h3>
      <table><tr><th>Method</th><th>MAE (kWh)</th><th>R²</th><th>MAPE</th></tr>
      ${Object.entries(r).map(([n, s]) => `<tr${n === m[key].algorithm ? ' class="total"' : ""}><td>${n}</td><td>${s.MAE_kWh.toFixed(1)}</td><td>${s.R2.toFixed(3)}</td><td>${s["MAPE_%"].toFixed(1)}%</td></tr>`).join("")}
      </table>`;
  };
  $("modelInfo").innerHTML =
    `<p>Trained on ${m.n_households} households with 5-fold cross-validation (repeated 3 times).</p>` +
    table("model_a", "Model A: appliance answers only") +
    table("model_b", "Model B: + last month's kWh");
}

$("planReset").onclick = () => { applied.clear(); predict(); };
$("acceptAll").onclick = acceptSelected;
$("acceptAll2").onclick = acceptSelected;
updateSliderLabels();
predict();
loadModelInfo();
