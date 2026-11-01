// PoC の画面を最初から最後まで操作して確かめる（Playwright, ヘッドレスの Chromium）
//   node web/e2e/poc_flow.js [URL] [動画]
// 既定: http://localhost:8600/ と .cache/e2e/two_people.webm（web/e2e/run.sh が用意する）
// 動画を選ぶ → 骨格推定（15 コマ/秒・最初の 12 秒）→ 左右の二人をタップ → 解析 → 結果・計測を表示し、
// 画面の写真を .cache/e2e/ に保存する。失敗したら終了コード 1。
const path = require("path");
const { chromium } = require("playwright");

const url = process.argv[2] || "http://localhost:8600/";
const video = path.resolve(process.argv[3] || ".cache/e2e/two_people.webm");
const out = path.resolve(".cache/e2e");

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 800, height: 1400 } });
  page.on("console", (m) => { if (m.type() === "error") console.log("ブラウザのエラー:", m.text().slice(0, 300)); });
  let ok = false;
  try {
    await page.goto(url);
    await page.setInputFiles("#fileA", video);
    await page.selectOption("#fps", "15");
    await page.fill("#maxSec", "12");
    await page.click("#run");
    // 骨格推定（GPU の無い環境では数分かかる）
    for (let i = 0; i < 60 && !(await page.isVisible("#pickSec")); i++) {
      await page.waitForTimeout(10000);
      const err = await page.textContent("#err");
      if (err) throw new Error(err);
      console.log(await page.textContent("#status"));
    }
    if (!(await page.isVisible("#pickSec"))) throw new Error("骨格推定が終わりませんでした");
    await page.waitForTimeout(1500);
    const bb = await (await page.$("#pickCanvas")).boundingBox();
    await page.mouse.click(bb.x + bb.width * 0.25, bb.y + bb.height * 0.5);
    await page.waitForTimeout(800);
    await page.mouse.click(bb.x + bb.width * 0.75, bb.y + bb.height * 0.5);
    await page.waitForTimeout(800);
    await page.click("#analyze", { timeout: 5000 });
    await page.waitForSelector("#resSec:not(.hidden)", { timeout: 120000 });
    console.log("\n--- 計測 ---\n" + (await page.innerText("#perf")));
    console.log("\n--- 結果 ---\n" + (await page.innerText("#resSec")));
    ok = !(await page.textContent("#err"));
  } catch (e) {
    console.log("失敗:", e.message);
  } finally {
    await page.screenshot({ path: path.join(out, "poc_flow.png"), fullPage: true });
    console.log("\n画面の写真:", path.join(out, "poc_flow.png"));
    await browser.close();
  }
  process.exit(ok ? 0 : 1);
})();
