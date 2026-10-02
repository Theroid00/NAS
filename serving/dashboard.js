"use strict";

const colors = { ga: "#66b5ff", aging: "#64d8b8", random: "#f5bb6b" };
const names = { ga: "Genetic search", aging: "Aging evolution", random: "Random search" };
const el = (id) => document.getElementById(id);
const pct = (value) => value == null ? "—" : `${(value * 100).toFixed(2)}%`;
const seconds = (value) => value == null ? "—" : `${value.toFixed(1)} s`;
let report = null;
let model = null;

function cell(row, value) {
  const td = document.createElement("td");
  td.textContent = value;
  row.append(td);
}

function svg(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text != null) node.textContent = text;
  el("chart").append(node);
  return node;
}

function renderChart() {
  if (!report) return;
  el("chart").replaceChildren();
  const axis = el("axis").value;
  const points = report.runs.flatMap((run) => run.progress).filter((p) => p.best_accuracy != null);
  if (!points.length) return;
  const xmax = Math.max(...points.map((p) => p[axis]), 1);
  const low = Math.max(0, Math.min(...points.map((p) => p.best_accuracy)) - 0.03);
  const high = Math.min(1, Math.max(...points.map((p) => p.best_accuracy)) + 0.03);
  const x = (value) => 65 + value / xmax * 905;
  const y = (value) => 265 - (value - low) / (high - low || 1) * 235;
  for (let i = 0; i <= 4; i++) {
    const value = low + (high - low) * i / 4;
    svg("line", { x1: 65, x2: 970, y1: y(value), y2: y(value), stroke: "#30435c" });
    svg("text", { x: 55, y: y(value) + 4, fill: "#afbed2", "text-anchor": "end", "font-size": 12 }, pct(value));
    svg("text", { x: x(xmax * i / 4), y: 293, fill: "#afbed2", "text-anchor": "middle", "font-size": 12 },
      (xmax * i / 4).toFixed(axis === "evaluations" ? 0 : 1));
  }
  report.runs.forEach((run) => {
    const good = run.progress.filter((p) => p.best_accuracy != null);
    const path = svg("polyline", {
      points: good.map((p) => `${x(p[axis])},${y(p.best_accuracy)}`).join(" "),
      fill: "none", stroke: colors[run.method] || "#fff", "stroke-width": 2, opacity: 0.7,
    });
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${run.method} seed ${run.seed}`;
    path.append(title);
  });
}

function inspectWinner() {
  const run = report?.runs[Number(el("winner").value)];
  if (!run) return;
  el("architecture").textContent = JSON.stringify(run.architecture, null, 2);
  const full = run.full_training[0];
  el("detail").textContent = `${run.valid_trials} valid trials · ${run.failed_attempts} failed attempts · ` +
    `${run.num_params ?? "Unknown"} parameters` + (full ? ` · Full training seed ${full.seed}` : "");
  const metrics = full?.metrics || run.metrics;
  el("classes").replaceChildren();
  el("matrix").replaceChildren();
  if (!metrics) {
    el("detail").textContent += " · Class metrics unavailable";
    return;
  }
  metrics.class_ids.forEach((id, i) => {
    const row = document.createElement("tr");
    [model?.class_names?.[id] || `Class ${id}`, pct(metrics.per_class_recall[i]),
      pct(metrics.per_class_f1[i]), metrics.class_support[i]].forEach((value) => cell(row, value));
    el("classes").append(row);
  });
  const header = document.createElement("tr");
  cell(header, "Actual / predicted");
  metrics.class_ids.forEach((id) => cell(header, id));
  el("matrix").append(header);
  const max = Math.max(...metrics.confusion_matrix.flat(), 1);
  metrics.confusion_matrix.forEach((counts, i) => {
    const row = document.createElement("tr");
    cell(row, i);
    counts.forEach((count) => {
      cell(row, count);
      row.lastChild.style.background = `rgba(100,216,184,${0.05 + count / max * 0.4})`;
    });
    el("matrix").append(row);
  });
}

function renderReport() {
  el("status").textContent = `${report.dataset} · ${report.status}`;
  ["cards", "runs", "winner", "legend"].forEach((id) => el(id).replaceChildren());
  if (report.smoke) {
    el("notice").hidden = false;
    el("notice").textContent = "Smoke demonstration: these scores are synthetic and are not benchmark results.";
  }
  Object.entries(report.summary).forEach(([method, summary]) => {
    const card = document.createElement("article");
    card.className = "card";
    const heading = document.createElement("h3");
    heading.textContent = names[method] || method;
    const score = document.createElement("div");
    score.className = "score";
    score.textContent = pct(summary.proxy_accuracy?.mean);
    const detail = document.createElement("p");
    detail.className = "muted";
    detail.style.whiteSpace = "pre-line";
    detail.textContent = `Mean proxy accuracy · ${summary.searches_completed} ` +
      `${summary.searches_completed === 1 ? "search" : "searches"}\n` +
      `Full validation: ${pct(summary.full_validation_accuracy?.mean)}\n` +
      `Mean search time: ${seconds(summary.search_seconds?.mean)}`;
    card.append(heading, score, detail);
    el("cards").append(card);
    const legend = document.createElement("span");
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = colors[method];
    legend.append(dot, document.createTextNode(heading.textContent));
    el("legend").append(legend);
  });
  report.runs.forEach((run, i) => {
    const row = document.createElement("tr");
    const full = run.full_training[0];
    [`${run.method} / ${run.seed}`, pct(run.accuracy), pct(run.metrics?.balanced_accuracy),
      pct(run.metrics?.macro_f1), pct(full?.accuracy), pct(full?.metrics?.balanced_accuracy),
      pct(full?.metrics?.macro_f1), seconds(run.search_elapsed_s), run.num_params ?? "—"]
      .forEach((value) => cell(row, value));
    el("runs").append(row);
    const option = document.createElement("option");
    option.value = i;
    option.textContent = `${run.method} · seed ${run.seed}`;
    el("winner").append(option);
  });
  renderChart();
  inspectWinner();
}

async function loadReport() {
  try {
    const response = await fetch("/report");
    if (!response.ok) throw Error("No report loaded. Start the service with --report pointing to a generated results report.");
    report = await response.json();
    renderReport();
  } catch (error) {
    el("error").textContent = error.message;
    el("status").textContent = "Report unavailable";
  }
}

async function loadModel() {
  const response = await fetch("/model");
  if (!response.ok) {
    el("model-description").textContent = "No model loaded. Export a retrained winner and supply --artifact to enable predictions.";
    el("predict").disabled = true;
    return;
  }
  model = await response.json();
  el("model-description").textContent = `${model.dataset_name} · ${model.input_features} features · ` +
    `${model.num_params} parameters · Model ${model.model_id}`;
  el("features").textContent = model.feature_names.map((name, i) => `${i + 1}. ${name}`).join("\n");
  if (model.example_available) el("example").hidden = false;
  if (report) inspectWinner();
}

el("axis").addEventListener("change", renderChart);
el("winner").addEventListener("change", inspectWinner);
el("example").addEventListener("click", async () => {
  try {
    const response = await fetch("/example");
    if (!response.ok) throw Error("Example row unavailable");
    el("input").value = JSON.stringify((await response.json()).features);
  } catch (error) {
    el("prediction").textContent = error.message;
  }
});
el("predict").addEventListener("click", async () => {
  el("predict").disabled = true;
  try {
    const features = JSON.parse(el("input").value);
    const response = await fetch("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }),
    });
    el("prediction").textContent = JSON.stringify(await response.json(), null, 2);
  } catch (error) {
    el("prediction").textContent = `Could not predict: ${error.message}`;
  } finally {
    el("predict").disabled = !model;
  }
});
loadReport();
loadModel().catch((error) => {
  el("model-description").textContent = `Could not load model: ${error.message}`;
  el("predict").disabled = true;
});
