// 人物の追跡（簡易版）: 前のコマの枠との重なりで同じ人物に同じ ID を付ける。
// 途切れた ID のつなぎ直しは Python 側（pose/tracking.py の stitch）が行うので、ここは単純でよい。
import { iou } from "./pose.js";

export class Tracker {
  constructor({ minIou = 0.3, maxLost = 90, newConf = 0.4 } = {}) {
    Object.assign(this, { minIou, maxLost, newConf });
    this.tracks = [];   // {id, box, vx, vy, lost}
    this.nextId = 1;
    this.people = {};   // id -> {frames, kp, sc, box}（TrackSet の JSON 形式）
  }

  update(frame, dets) {
    // 前のコマからの動きを足して予測した枠と比べる
    const pred = this.tracks.map((t) => [t.box[0] + t.vx, t.box[1] + t.vy, t.box[2] + t.vx, t.box[3] + t.vy]);
    const pairs = [];
    dets.forEach((d, j) => pred.forEach((p, k) => {
      const v = iou(d.box, p);
      if (v >= this.minIou) pairs.push([v, j, k]);
    }));
    pairs.sort((a, b) => b[0] - a[0]);
    const usedD = new Set(), usedT = new Set();
    for (const [, j, k] of pairs) {
      if (usedD.has(j) || usedT.has(k)) continue;
      usedD.add(j); usedT.add(k);
      const t = this.tracks[k], b = dets[j].box;
      t.vx = 0.5 * t.vx + 0.5 * ((b[0] + b[2]) - (t.box[0] + t.box[2])) / 2 / (t.lost + 1);
      t.vy = 0.5 * t.vy + 0.5 * ((b[1] + b[3]) - (t.box[1] + t.box[3])) / 2 / (t.lost + 1);
      t.box = b; t.lost = 0;
      this.record(t.id, frame, dets[j]);
    }
    this.tracks.forEach((t, k) => { if (!usedT.has(k)) t.lost++; });
    this.tracks = this.tracks.filter((t) => t.lost <= this.maxLost);
    dets.forEach((d, j) => {
      if (usedD.has(j) || d.conf < this.newConf) return;
      const t = { id: this.nextId++, box: d.box, vx: 0, vy: 0, lost: 0 };
      this.tracks.push(t);
      this.record(t.id, frame, d);
    });
  }

  record(id, frame, d) {
    const p = (this.people[id] ??= { frames: [], kp: [], sc: [], box: [] });
    p.frames.push(frame);
    p.kp.push(...d.kp);
    p.sc.push(...d.sc);
    p.box.push(...d.box);
  }

  // フレーム f に映っている人物 {id: box}
  at(f) {
    const out = {};
    for (const [id, p] of Object.entries(this.people)) {
      const i = p.frames.indexOf(f);
      if (i >= 0) out[id] = p.box.slice(4 * i, 4 * i + 4);
    }
    return out;
  }
}
