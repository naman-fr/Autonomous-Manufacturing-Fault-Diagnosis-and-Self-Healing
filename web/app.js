const state = {
  latest: null,
  signal: [],
};

const $ = (id) => document.getElementById(id);

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("backend offline");
    $("healthStatus").textContent = "Backend online";
    $("healthStatus").classList.add("ok");
  } catch {
    $("healthStatus").textContent = "Backend offline";
  }
}

async function runDemo() {
  const machineId = encodeURIComponent($("machineId").value || "PUMP-101");
  const response = await fetch(`/api/v1/demo?machine_id=${machineId}`);
  render(await response.json());
}

async function diagnoseCsv() {
  const file = $("csvFile").files[0];
  if (!file) {
    alert("Choose a CSV file first.");
    return;
  }
  const text = await file.text();
  const params = new URLSearchParams({
    machine_id: $("machineId").value || "PUMP-101",
    operator_notes: $("operatorNotes").value || "",
    force_human_review: $("forceReview").checked ? "true" : "false",
  });
  const response = await fetch(`/api/v1/diagnose/csv?${params.toString()}`, {
    method: "POST",
    headers: { "content-type": "text/csv" },
    body: text,
  });
  render(await response.json());
}

function render(payload) {
  if (payload.detail) {
    alert(payload.detail);
    return;
  }
  state.latest = payload;
  const report = payload.report;
  $("severity").textContent = report.detection.severity.toUpperCase();
  $("score").textContent = report.detection.anomaly_score.toFixed(2);
  $("confidence").textContent = report.detection.confidence.toFixed(2);
  $("latency").textContent = `${report.metrics.latency_ms.toFixed(0)} ms`;
  $("rootCause").textContent = titleCase(report.root_cause.label);
  $("jsonOutput").textContent = JSON.stringify(report, null, 2);

  renderList("evidence", report.detection.evidence);
  renderActions(report.actions);
  renderRag(report.rag_evidence);
  renderTimeline(payload.trace);
  drawSyntheticSignal(report.features);
}

function renderList(id, items) {
  const element = $(id);
  element.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    element.appendChild(li);
  });
}

function renderActions(actions) {
  const root = $("actions");
  root.innerHTML = "";
  actions.forEach((action) => {
    const item = document.createElement("div");
    item.className = "item";
    item.innerHTML = `<strong>${action.priority.toUpperCase()} · ${action.action}</strong><span>${action.rationale}</span>`;
    root.appendChild(item);
  });
}

function renderRag(items) {
  const root = $("rag");
  root.innerHTML = "";
  items.forEach((hit) => {
    const item = document.createElement("div");
    item.className = "item";
    item.innerHTML = `<strong>${hit.source}</strong><span>BM25 ${hit.bm25_score.toFixed(2)} · Rerank ${hit.rerank_score.toFixed(2)}</span><p>${hit.text}</p>`;
    root.appendChild(item);
  });
}

function renderTimeline(steps) {
  const root = $("timeline");
  root.innerHTML = "";
  steps.forEach((step, index) => {
    const item = document.createElement("div");
    item.className = "step";
    item.textContent = `${String(index + 1).padStart(2, "0")}  ${step}`;
    root.appendChild(item);
  });
}

function drawSyntheticSignal(features) {
  const canvas = $("waveform");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const samples = 500;
  const frequency = Math.max(1, features.dominant_frequency_hz / 30);
  const amplitude = Math.min(0.9, Math.max(0.1, features.rms * 2.5));
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#d9dee7";
  ctx.lineWidth = 1;
  for (let i = 0; i < 6; i += 1) {
    const y = (height / 5) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#006d77";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < samples; i += 1) {
    const x = (i / (samples - 1)) * width;
    const carrier = Math.sin((i / samples) * Math.PI * 2 * frequency);
    const impulse = i % 48 === 0 ? 0.55 : 0;
    const y = height / 2 - (carrier * amplitude + impulse) * height * 0.34;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  $("sampleCount").textContent = `Dominant ${features.dominant_frequency_hz.toFixed(1)} Hz`;
}

function titleCase(value) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

$("demoBtn").addEventListener("click", runDemo);
$("uploadBtn").addEventListener("click", diagnoseCsv);
$("copyBtn").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("jsonOutput").textContent);
});

checkHealth();
runDemo();

