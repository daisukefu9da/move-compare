// PoC: 動画 → ブラウザで骨格推定 → 人物をタップで選ぶ → Pyodide で今の解析（dance_compare）を動かす
import { PoseModel } from "./pose.js";
import { Tracker } from "./tracker.js";

const $ = (id) => document.getElementById(id);
const GROUPS = { overall: "総合", legs: "足", hips: "腰（体幹）", head: "頭", arms: "腕", timing: "タイミング" };
const SR = 22050;  // 音声のサンプリング周波数（dance_compare/media/audio.py と同じ）
const BONES = [[5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6]];

const perf = {};          // 計測結果（表示・コピー用）
const state = { tracks: {}, people: {}, video: null, pick: [] };

function setPerf(k, v) { perf[k] = v; renderPerf(); }
function renderPerf() {
  $("perfSec").classList.remove("hidden");
  $("perf").innerHTML = Object.entries(perf).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}
function status(msg) { $("status").textContent = msg; }
function fail(e) { console.error(e); $("err").textContent = "エラー: " + (e && e.stack || e); }
window.addEventListener("error", (e) => fail(e.error || e.message));
window.addEventListener("unhandledrejection", (e) => fail(e.reason));

// ---------------------------------------------------------------- 端末の情報
async function envInfo() {
  const rows = { "ブラウザ": navigator.userAgent, "CPU のコア数": navigator.hardwareConcurrency ?? "-",
                 "クロスオリジン分離（WASM の並列化）": String(self.crossOriginIsolated) };
  if (navigator.gpu) {
    try {
      const ad = await navigator.gpu.requestAdapter();
      const info = ad && (ad.info || (ad.requestAdapterInfo && await ad.requestAdapterInfo()));
      rows["WebGPU"] = ad ? `あり（${[info?.vendor, info?.architecture, info?.device].filter(Boolean).join(" ") || "詳細不明"}）` : "あり（アダプタ無し）";
    } catch (e) { rows["WebGPU"] = "エラー: " + e.message; }
  } else rows["WebGPU"] = "なし";
  $("env").innerHTML = Object.entries(rows).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  Object.assign(perf, { "端末": navigator.userAgent, "WebGPU": rows["WebGPU"] });
}

// ---------------------------------------------------------------- Pyodide（裏で先に読み込む）
let pyReady = null;
function loadPy() {
  pyReady ??= (async () => {
    const t0 = performance.now();
    const { loadPyodide } = await import("./vendor/pyodide/pyodide.mjs");
    const py = await loadPyodide({ indexURL: new URL("./vendor/pyodide/", location.href).href });
    await py.loadPackage("numpy");
    const zip = await (await fetch("./dance_compare.zip")).arrayBuffer();
    py.unpackArchive(zip, "zip", { extractDir: "/home/pyodide" });
    py.runPython("import sys; sys.path.insert(0, '/home/pyodide'); from dance_compare import web");
    setPerf("Pyodide + 解析コードの読み込み", `${((performance.now() - t0) / 1000).toFixed(1)} 秒`);
    return py;
  })();
  return pyReady;
}

// ---------------------------------------------------------------- 骨格推定
let model = null;
async function getModel() {
  if (model && model.backend === $("ep").value) return model;
  const t0 = performance.now();
  const ort = await import("./vendor/ort/ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = new URL("./vendor/ort/", location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  model = new PoseModel(ort);
  const ep = await model.load("./models/yolo11n-pose.onnx", $("ep").value);
  setPerf("推定の実行環境", ep === "webgpu" ? "WebGPU" : `WASM（${ort.env.wasm.numThreads} スレッド）`);
  setPerf("モデルの読み込み", `${((performance.now() - t0) / 1000).toFixed(1)} 秒`);
  return model;
}

async function openVideo(file) {
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.preload = "auto";
  v.setAttribute("playsinline", "");
  // iPhone では画面に置いて一度再生しないとコマが描けないことがあるので、見えない大きさで置く
  Object.assign(v.style, { position: "fixed", left: "0", top: "0", width: "2px", height: "2px", opacity: "0.01" });
  document.body.appendChild(v);
  v.src = URL.createObjectURL(file);
  await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(new Error("動画を開けません")); });
  try { await v.play(); v.pause(); } catch (e) { /* 自動再生できなくてもシークで読める端末が多い */ }
  return v;
}

function seek(v, t) {
  return new Promise((res) => {
    const done = () => { v.removeEventListener("seeked", done); res(); };
    v.addEventListener("seeked", done);
    v.currentTime = t;
  });
}

async function estimate(file, label) {
  const m = await getModel();
  const v = await openVideo(file);
  const fps = +$("fps").value;
  const w = v.videoWidth, h = v.videoHeight;
  const dur = Math.min(v.duration, +$("maxSec").value || v.duration);
  const n = Math.floor(dur * fps);
  const tr = new Tracker();
  const sum = { seek: 0, pre: 0, infer: 0, post: 0 };
  const pv = $("preview"), pctx = pv.getContext("2d");
  pv.classList.remove("hidden");
  pv.width = 360; pv.height = Math.round(360 * h / w);
  $("prog").classList.remove("hidden");
  let lock = null;
  try { lock = await navigator.wakeLock?.request("screen"); } catch (e) { /* 画面を消さない機能が無い端末 */ }
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    const s0 = performance.now();
    await seek(v, i / fps + 0.001);
    sum.seek += performance.now() - s0;
    const { dets, ms } = await m.detect(v, w, h);
    sum.pre += ms.pre; sum.infer += ms.infer; sum.post += ms.post;
    tr.update(i, dets);
    if (i % 10 === 0) {
      pctx.drawImage(v, 0, 0, pv.width, pv.height);
      drawPeople(pctx, dets.map((d) => ({ box: d.box, kp: d.kp })), pv.width / w);
      $("prog").value = (i + 1) / n;
      const el = (performance.now() - t0) / 1000;
      status(`${label}: ${i + 1}/${n} コマ（${el.toFixed(0)} 秒, 1 コマ ${(el * 1000 / (i + 1)).toFixed(0)}ms）`);
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  const total = (performance.now() - t0) / 1000;
  lock?.release?.();
  setPerf(`${label}: 動画`, `${w}×${h}, ${v.duration.toFixed(1)} 秒（うち ${dur.toFixed(1)} 秒を ${fps}fps で ${n} コマ）`);
  setPerf(`${label}: 骨格推定の合計`, `${total.toFixed(1)} 秒（動画 1 分あたり ${(total / dur * 60).toFixed(0)} 秒）`);
  setPerf(`${label}: 1 コマの内訳`, `シーク ${(sum.seek / n).toFixed(0)}ms / 前処理 ${(sum.pre / n).toFixed(0)}ms / 推定 ${(sum.infer / n).toFixed(0)}ms / 後処理 ${(sum.post / n).toFixed(1)}ms`);
  setPerf(`${label}: 追跡した ID の数`, Object.keys(tr.people).length);
  return { v, tr, json: tracksJson(tr, fps, w, h, n) };
}

function tracksJson(tr, fps, w, h, n) {
  const r = (x) => Math.round(x * 10) / 10, r3 = (x) => Math.round(x * 1000) / 1000;
  const people = {};
  for (const [id, p] of Object.entries(tr.people)) {
    people[id] = { frames: p.frames, kp: p.kp.map(r), sc: p.sc.map(r3), box: p.box.map(r) };
  }
  return JSON.stringify({ fps, width: w, height: h, n_frames: n, people });
}

function drawPeople(ctx, people, s, colors = {}) {
  ctx.lineWidth = 2;
  for (const p of people) {
    const c = colors[p.id] || "#3fb950";
    ctx.strokeStyle = c;
    if (p.box) ctx.strokeRect(p.box[0] * s, p.box[1] * s, (p.box[2] - p.box[0]) * s, (p.box[3] - p.box[1]) * s);
    if (p.kp) for (const [a, b] of BONES) {
      ctx.beginPath(); ctx.moveTo(p.kp[2 * a] * s, p.kp[2 * a + 1] * s); ctx.lineTo(p.kp[2 * b] * s, p.kp[2 * b + 1] * s); ctx.stroke();
    }
    if (p.label) { ctx.fillStyle = c; ctx.font = "bold 16px sans-serif"; ctx.fillText(p.label, p.box[0] * s + 4, p.box[1] * s + 18); }
  }
}

// ---------------------------------------------------------------- 音声（2動画モードの同期用）
async function decodeAudio(file) {
  if (file.size > 400e6) return null;  // 大きすぎる動画はメモリが足りなくなるので使わない
  try {
    const buf = await file.arrayBuffer();
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const audio = await ac.decodeAudioData(buf);
    ac.close?.();
    const off = new OfflineAudioContext(1, Math.ceil(audio.duration * SR), SR);
    const src = off.createBufferSource();
    src.buffer = audio; src.connect(off.destination); src.start();
    return (await off.startRendering()).getChannelData(0);
  } catch (e) {
    console.warn("音声を取り出せません", e);
    return null;
  }
}

// ---------------------------------------------------------------- 人物の選択（1動画モード）
async function showPick() {
  const { v, tr } = state.single;
  const fps = +$("fps").value, s = Math.min(1, 720 / v.videoWidth);
  const cv = $("pickCanvas"), ctx = cv.getContext("2d");
  cv.width = Math.round(v.videoWidth * s); cv.height = Math.round(v.videoHeight * s);
  const f = +$("pickT").value;
  await seek(v, f / fps + 0.001);
  ctx.drawImage(v, 0, 0, cv.width, cv.height);
  const at = tr.at(f);
  const labels = { [state.pick[0]]: "お手本", [state.pick[1]]: "比べる人" };
  const colors = { [state.pick[0]]: "#2f81f7", [state.pick[1]]: "#e5484d" };
  drawPeople(ctx, Object.entries(at).map(([id, box]) => ({ id, box, label: labels[id] || `ID ${id}` })), s, colors);
  state.pickBoxes = { at, s };
  $("pickMsg").textContent = state.pick.length === 0 ? "お手本の人をタップしてください"
    : state.pick.length === 1 ? "比べる人（レビュー対象）をタップしてください" : "「解析する」を押してください";
  $("analyze").disabled = state.pick.length < 2;
}

$("pickCanvas").addEventListener("click", (ev) => {
  if (state.pick.length >= 2) return;
  const cv = $("pickCanvas"), r = cv.getBoundingClientRect();
  const { at, s } = state.pickBoxes;
  const x = (ev.clientX - r.left) * cv.width / r.width / s, y = (ev.clientY - r.top) * cv.height / r.height / s;
  const hit = Object.entries(at).filter(([, b]) => b[0] <= x && x <= b[2] && b[1] <= y && y <= b[3])
    .sort((a, b) => (a[1][2] - a[1][0]) * (a[1][3] - a[1][1]) - (b[1][2] - b[1][0]) * (b[1][3] - b[1][1]));
  if (!hit.length || state.pick.includes(hit[0][0])) return;
  state.pick.push(hit[0][0]);
  showPick();
});
$("pickT").addEventListener("change", showPick);
$("repick").onclick = () => { state.pick = []; showPick(); };

// ---------------------------------------------------------------- 解析（Pyodide）
async function analyze(fn) {
  status("解析中…（初回は Python の読み込みを待ちます）");
  const py = await loadPy();
  const t0 = performance.now();
  const out = JSON.parse(fn(py));
  setPerf("解析（Pyodide）", `${((performance.now() - t0) / 1000).toFixed(1)} 秒`);
  showResult(out);
  status("完了");
}

function showResult(out) {
  $("resSec").classList.remove("hidden");
  const sc = out.scores;
  $("scores").innerHTML = Object.entries(GROUPS).filter(([k]) => k in sc)
    .map(([k, n]) => `<div class="score">${n}<b>${sc[k] == null ? "-" : Math.round(sc[k])}</b></div>`).join("");
  $("moments").innerHTML = "<h2 style='margin-top:12px'>直すところ</h2>" + (out.moments.length ? out.moments.map((m) =>
    `<p><b>${m.title}</b></p><ul>${m.lines.map((l) => `<li>${l}</li>`).join("")}</ul>`).join("") : "<p>大きなズレはありません</p>")
    + (out.tendencies.length ? `<h2>全体の傾向</h2><ul>${out.tendencies.map((l) => `<li>${l}</li>`).join("")}</ul>` : "");
  $("notes").innerHTML = out.notes.map((n) => `・${n}`).join("<br>");
  setPerf("指摘の数", out.n_findings);
  state.result = out;
}

$("analyze").onclick = () => {
  const { json } = state.single;
  const [ref, user] = state.pick.map(Number);
  analyze((py) => py.globals.get("web").run_single(json, ref, user)).catch(fail);
};

// ---------------------------------------------------------------- 画面の操作
function mode() { return document.querySelector("input[name=mode]:checked").value; }
function updateInputs() {
  const two = mode() === "two";
  $("rowB").classList.toggle("hidden", !two);
  $("labA").textContent = two ? "お手本" : "動画";
  $("run").disabled = !$("fileA").files[0] || (two && !$("fileB").files[0]);
}
document.querySelectorAll("input[name=mode]").forEach((e) => e.onchange = updateInputs);
$("fileA").onchange = $("fileB").onchange = updateInputs;

$("run").onclick = async () => {
  $("run").disabled = true; $("err").textContent = "";
  state.pick = [];
  loadPy().catch(fail);  // 骨格推定の間に裏で読み込む
  try {
    if (mode() === "single") {
      state.single = await estimate($("fileA").files[0], "動画");
      state.tracks = { single: state.single.json };
      $("pickSec").classList.remove("hidden");
      $("pickT").max = Math.max(0, JSON.parse(state.single.json).n_frames - 1);
      $("pickT").value = Math.min(+$("pickT").max, 3 * +$("fps").value);
      await showPick();
      status("骨格推定が終わりました。人物を選んでください");
    } else {
      const a = await estimate($("fileA").files[0], "お手本");
      const b = await estimate($("fileB").files[0], "レビュー対象");
      state.tracks = { ref: a.json, user: b.json };
      status("音声を取り出しています…");
      const t0 = performance.now();
      const [ya, yb] = [await decodeAudio($("fileA").files[0]), await decodeAudio($("fileB").files[0])];
      setPerf("音声の取り出し", ya && yb ? `${((performance.now() - t0) / 1000).toFixed(1)} 秒` : "できませんでした（動きだけで時間を合わせます）");
      await analyze((py) => {
        const toPy = (x) => x ? py.toPy(x) : null;
        return py.globals.get("web").run_two(a.json, b.json, toPy(ya), toPy(yb));
      });
    }
  } catch (e) { fail(e); }
  $("run").disabled = false;
};

$("copy").onclick = async () => {
  const text = Object.entries(perf).map(([k, v]) => `${k}: ${v}`).join("\n")
    + (state.result ? `\nスコア: ${JSON.stringify(state.result.scores)}` : "");
  try { await navigator.clipboard.writeText(text); $("copy").textContent = "コピーしました"; }
  catch (e) { prompt("コピーしてください", text); }
};
$("dl").onclick = () => {
  for (const [k, json] of Object.entries(state.tracks)) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    a.download = `tracks_${k}.json`; a.click();
  }
};

envInfo();
