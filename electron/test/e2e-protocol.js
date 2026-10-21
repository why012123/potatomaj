// 端到端：**真实录像帧 → 状态机 → preload → 气泡画出来**。
//
// 这条是现在最该守的一条链：牌全来自协议，位置全靠推算，
// 中间任何一环断了，界面上都会「安静地什么都不显示」——不报错、日志也正常。
//
// 以前这个位置是一个「铺一张截图走识图链」的测试。识图删了之后它就没意义了，
// 换成这条：喂真的帧，看真的 DOM。
const { app, BrowserWindow, ipcMain } = require('electron');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const { GameState } = require('../lib/gamestate');
const { sortHand } = require('../lib/rowgeom');
const pb = require('../lib/pb');

const checks = [];
const check = (n, ok, d = '') => checks.push([n, ok, d]);

setTimeout(() => { console.log('⏰ 超时'); process.exit(3); }, 30000);

function frames() {
  const dir = path.join(ROOT, 'build', 'ws-frames');
  const idxPath = path.join(dir, 'index.jsonl');
  if (!fs.existsSync(idxPath)) return [];
  const idx = fs.readFileSync(idxPath, 'utf8').trim().split('\n')
    .filter(Boolean).map((l) => { try { return JSON.parse(l); } catch (e) { return null; } })
    .filter(Boolean);
  const bin = fs.readFileSync(path.join(dir, 'frames.bin'));
  const out = [];
  for (const e of idx) {
    if (e.off === undefined) continue;
    let m;
    try { m = pb.unwrap(bin.subarray(e.off, e.off + e.len)); } catch (err) { continue; }
    if (m.action) out.push({ action: m.action, payload: m.actionPayload, seq: e.seq });
  }
  out.sort((a, b) => a.seq - b.seq);
  return out;
}

app.whenReady().then(async () => {
  const evs = frames();
  if (!evs.length) {
    console.log('⚠️ build/ws-frames 里没有帧 —— 先双击土豆麻将.command 打一局');
    app.quit();
    return;
  }

  // ---- 1) 用真状态机跑一遍，跑到一个「轮到我打牌」的局面 ----
  const game = new GameState();
  const realLog = console.log;
  console.log = () => {};
  let snap = null;
  for (const e of evs) {
    game.onAction(e.action, e.payload);
    if (game.effective >= 14 && game.mySeat !== null && game.hand.length >= 13) {
      snap = game.snapshot();
      break;
    }
  }
  console.log = realLog;
  check('从真实帧里跑出了「轮到我打牌」的局面', !!snap,
    snap ? `手牌 ${snap.hand.length} 张，座位 ${snap.mySeat}` : '没跑到');
  if (!snap) {
    console.log('❌ 跑不出可用局面，后面的检查跳过');
    app.quit();
    return;
  }
  check('  手牌是合法牌名', snap.hand.every((t) => /^[0-9][mpsz]$/.test(t) && t !== '0z'),
    snap.hand.join(' '));
  check('  读到了宝牌指示牌', snap.dora.length > 0, snap.dora.join(' '));

  // ---- 2) 按 main.js 同样的算法算位置（窗口 1600x950）----
  const VW = 1600, VH = 950;
  const ROW = { pitch: 71 / 1470, cx: 0.5, y: 700 / 812, height: 108 / 812 };
  const DRAWN_GAP = 0.3;
  const rowBox = (index, count) => {
    const pitch = ROW.pitch * VW;
    const gap = index >= count ? DRAWN_GAP * pitch : 0;
    const left = ROW.cx * VW - (count * pitch) / 2;
    const x1 = left + index * pitch + gap;
    const y1 = ROW.y * VH;
    return [x1 / VW, y1 / VH, (x1 + pitch) / VW, (y1 + ROW.height * VH) / VH];
  };

  const hand = sortHand(snap.hand);
  const baseCount = hand.length - 1;
  const opts = hand.slice(0, 3).map((t, i) => ({
    tile: t, box: rowBox(i, baseCount), score: 90 - i * 20, ukeire: 8 - i,
    kinds: 2, loss: i, danger: 0.1 + i * 0.2, warn: i === 2, mode: 'model',
    best: i === 0, mine: false,
  }));

  // ---- 3) 喂给真的 preload，看 DOM ----
  const w = new BrowserWindow({
    width: VW, height: VH, show: false,
    webPreferences: { preload: path.join(__dirname, '..', 'preload-game.js'), contextIsolation: false },
  });
  await w.loadURL('file://' + path.join(__dirname, 'fake-page.html'));
  await new Promise((r) => setTimeout(r, 600));

  w.webContents.send('gamestate', {
    ...snap,
    advice: { text: '2 向听　建议打 ' + opts[0].tile + '　进张 15 枚', mode: 'model',
              mode_note: '神经网络 ' + (snap.sanma ? 'maj-flashv1.1' : 'maj-flash4v1.1') },
    badges: opts,
    handTop: opts[0].box[1],
    diag: { sockets: 1, actions: evs.length, lastAction: 'ActionDealTile' },
  });
  await new Promise((r) => setTimeout(r, 600));

  const hud = await w.webContents.executeJavaScript(
    'document.getElementById("__pm_hud").innerText');
  check('HUD 显示了手牌', hud.includes(hand[0]) && hud.includes(hand[hand.length - 1]),
    hud.split('\n').find((l) => l.includes('手牌')) || '');
  check('HUD 显示了宝牌', hud.includes('宝牌') && hud.includes(snap.dora[0]), '');
  check('HUD 显示了收益来源（模型名）', /maj-flash[4]?v1\.1/.test(hud),
    (hud.split('\n').find((l) => l.includes('收益来源')) || '').trim());

  // 用**页面自己的** innerWidth/innerHeight 算期望位置。
  // 一开始我用窗口的 1600x950 算，结果每个牌子都"偏左" —— 其实是我的基准错了：
  // BrowserWindow 的 width 含边框，页面的 innerWidth 比它小，两边的比例一除就错开了。
  const vp = await w.webContents.executeJavaScript(
    '({ w: window.innerWidth, h: window.innerHeight })');
  const tips = await w.webContents.executeJavaScript(
    'Array.from(document.querySelectorAll(".pm-tip")).map(e => e.textContent)');
  check(`气泡画出来了（${tips.length} 个）`, tips.length === opts.length, tips.join(' | '));

  // 气泡的位置必须落在手牌行上方，且左右不越界
  const boxes = await w.webContents.executeJavaScript(
    'Array.from(document.querySelectorAll(".pm-tip")).map(e => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.right]; })');
  check('  气泡都在手牌行**上方**（不挡住牌）',
    boxes.every((b) => b[1] < ROW.y * vp.h + 1), `行顶 y=${(ROW.y * vp.h).toFixed(0)}`);
  check('  气泡都在窗口内', boxes.every((b) => b[0] >= -2 && b[2] <= vp.w + 2),
    `页面 ${vp.w}x${vp.h}`);
  // 真正要在意的是「牌子中心有没有落在它那张牌上」——
  // 相邻牌子的边框挨上一点不算错（牌宽 77px，牌子文字才 45px）。
  const centers = boxes.map((b) => (b[0] + b[2]) / 2);
  const want = opts.map((o) => (o.box[0] + o.box[2]) / 2 * vp.w);
  check('  每个牌子的中心都对准它那张牌',
    centers.every((c, i) => Math.abs(c - want[i]) <= 3),
    centers.map((c, i) => (c - want[i]).toFixed(1)).join(' '));
  const overlap = boxes.some((b, i) => i > 0 && b[0] < boxes[i - 1][2] - 2);
  check(`  相邻牌子${overlap ? '有' : '没有'}边框相接（提示用，不算失败）`, true, '');

  let fail = 0;
  for (const [n, ok, d] of checks) {
    if (!ok) fail++;
    console.log(`${ok ? '✅' : '❌'} ${n}${d ? '　' + d : ''}`);
  }
  console.log();
  console.log(fail === 0 ? `✅ 端到端通过（协议 → 状态 → 气泡，${checks.length} 项）`
    : `❌ ${fail} 项不通过`);
  app.exit(fail === 0 ? 0 : 1);
});
