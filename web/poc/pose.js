// 骨格推定（YOLO11n-pose の ONNX を ONNX Runtime Web で動かす）
// 入力: 動画のコマ（canvas に描いたもの）→ 出力: 人物ごとの枠・17 点の座標・信頼度（元の動画のピクセル座標）

const SIZE = 640;          // モデルの入力の大きさ（正方形, 余白は灰色）
const CONF = 0.25;         // 人物とみなす信頼度
const IOU = 0.7;           // 重なった候補をまとめるしきい値（Ultralytics の既定と同じ）

export class PoseModel {
  constructor(ort) {
    this.ort = ort;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = SIZE;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.input = new Float32Array(3 * SIZE * SIZE);
    this.backend = null;
  }

  // WebGPU が使えればそれで、無ければ WASM で読み込む
  async load(url, prefer = "webgpu") {
    const tries = prefer === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
    let lastErr;
    for (const ep of tries) {
      try {
        if (ep === "webgpu" && !navigator.gpu) throw new Error("WebGPU が使えません");
        this.session = await this.ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: "all" });
        this.backend = ep;
        return ep;
      } catch (e) { lastErr = e; }
    }
    throw lastErr;
  }

  // source（video / canvas）の今のコマを推定する
  async detect(source, w, h) {
    const r = Math.min(SIZE / w, SIZE / h);
    const nw = Math.round(w * r), nh = Math.round(h * r);
    const px = (SIZE - nw) / 2, py = (SIZE - nh) / 2;
    const ctx = this.ctx;
    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.drawImage(source, px, py, nw, nh);
    const t0 = performance.now();
    const img = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const n = SIZE * SIZE, inp = this.input;
    for (let i = 0; i < n; i++) {
      inp[i] = img[i * 4] / 255;
      inp[i + n] = img[i * 4 + 1] / 255;
      inp[i + 2 * n] = img[i * 4 + 2] / 255;
    }
    const t1 = performance.now();
    const tensor = new this.ort.Tensor("float32", inp, [1, 3, SIZE, SIZE]);
    const out = await this.session.run({ images: tensor });
    const t2 = performance.now();
    const o = out.output0;
    const data = o.data ?? (await o.getData());
    const dets = decode(data, o.dims[2], r, px, py);
    const t3 = performance.now();
    return { dets, ms: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2 } };
  }
}

// 出力 [1, 56, N]（cx, cy, w, h, conf, 17×(x, y, v)）→ 人物のリスト
function decode(d, N, r, px, py) {
  const cand = [];
  for (let i = 0; i < N; i++) {
    const c = d[4 * N + i];
    if (c < CONF) continue;
    const cx = d[i], cy = d[N + i], bw = d[2 * N + i], bh = d[3 * N + i];
    cand.push({ i, c, box: [(cx - bw / 2 - px) / r, (cy - bh / 2 - py) / r, (cx + bw / 2 - px) / r, (cy + bh / 2 - py) / r] });
  }
  cand.sort((a, b) => b.c - a.c);
  const keep = [];
  for (const a of cand) {
    if (keep.every((b) => iou(a.box, b.box) < IOU)) keep.push(a);
  }
  return keep.map(({ i, c, box }) => {
    const kp = new Float32Array(34), sc = new Float32Array(17);
    for (let k = 0; k < 17; k++) {
      kp[2 * k] = (d[(5 + 3 * k) * N + i] - px) / r;
      kp[2 * k + 1] = (d[(6 + 3 * k) * N + i] - py) / r;
      sc[k] = d[(7 + 3 * k) * N + i];
    }
    return { box, conf: c, kp, sc };
  });
}

export function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}
