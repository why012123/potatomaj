// 手牌位置**由排序下标推算** —— 这是弃用「逐巡识图定位」之后的整套几何。
//
// 背景：原来每巡截一次手牌条、靠识图给出每张牌的坐标。那条路会错也会漂
//（局间压暗、游戏面板压住牌顶、摸打时靠推测挪坐标）。
// 而位置本来就能算：雀魂的手牌是排序显示的，整行居中、每张等距 ——
// 只要「第一张在哪」和「间距」，后面每张都确定了。
//
// 这个测试里有一条**用真实截图核对过的**依据：手牌顺序。
// 那张图里从左到右是 `3p 3p 6p 7p 7p 8p 4s 6s 6s 7s`（逐张放大核对），
// 而 sortHand 必须原样给出这个顺序 —— 顺序错了所有气泡都会错位到别的牌上，
// 而且看上去还挺整齐，不容易发现。
const { RowGeometry, sortHand, tileKey } = require('../lib/rowgeom');

const checks = [];
const check = (n, ok, d = '') => checks.push([n, ok, d]);

// ---- 1) 排序：和真实截图一致 ----
const REAL_ORDER = ['3p', '3p', '6p', '7p', '7p', '8p', '4s', '6s', '6s', '7s'];
check('真实截图的顺序（本来就是排好的）原样通过',
  sortHand(REAL_ORDER).join(' ') === REAL_ORDER.join(' '), sortHand(REAL_ORDER).join(' '));
check('打乱之后能排回同一个顺序',
  sortHand(['7s', '4s', '3p', '6s', '7p', '6p', '3p', '6s', '8p', '7p']).join(' ') === REAL_ORDER.join(' '),
  sortHand(['7s', '4s', '3p', '6s', '7p', '6p', '3p', '6s', '8p', '7p']).join(' '));
check('花色顺序是 万→筒→索→字',
  sortHand(['1z', '1s', '1p', '1m']).join(' ') === '1m 1p 1s 1z',
  sortHand(['1z', '1s', '1p', '1m']).join(' '));
check('同一花色内从小到大',
  sortHand(['9p', '2p', '5p', '1p']).join(' ') === '1p 2p 5p 9p', '');
check('赤五按 5 排序（显示上就在 5 的位置）',
  sortHand(['6s', '0s', '4s', '5s']).join(' ') === '4s 0s 5s 6s',
  sortHand(['6s', '0s', '4s', '5s']).join(' '));

// ---- 2) 量几何 ----
const W = 1470, H = 812, PITCH = 71, N = 10, X0 = 400, Y = 700, FH = 108;
const g = new RowGeometry();
const items = [];
for (let i = 0; i < N; i++) {
  const x = X0 + i * PITCH;
  items.push({ id: '1p', box: [x / W, Y / H, (x + PITCH) / W, (Y + FH) / H] });
}
check('能从一次识图里量出间距和行顶', g.measure(items, W, H) === true, '');
check('  间距量对了', Math.abs(g.pitch - PITCH) < 1.5, `${g.pitch.toFixed(1)} vs ${PITCH}`);
check('  行顶量对了', Math.abs(g.y - Y) < 2, `${g.y.toFixed(1)} vs ${Y}`);
check('  整行中心量对了', Math.abs(g.cx - (X0 + N * PITCH / 2)) < 2,
  `${g.cx.toFixed(1)} vs ${X0 + N * PITCH / 2}`);

// ---- 3) 按下标算位置 ----
let evenOk = true, firstOk = true, matchOk = true;
for (let i = 0; i < N; i++) {
  const b = g.boxAt(i, N);
  const w = b[2] - b[0];
  if (Math.abs(w - PITCH) > 1) evenOk = false;
  if (i > 0) {
    const prev = g.boxAt(i - 1, N);
    if (Math.abs((b[0] - prev[0]) - PITCH) > 1) evenOk = false;
  }
  // 算出来的位置应该和当初量出来的框一一对应
  const want = X0 + i * PITCH;
  if (Math.abs(b[0] - want) > 2) matchOk = false;
}
check('每张牌宽度都等于间距', evenOk, '');
check('第 i 张的位置和实测框一一对应', matchOk, '');
check('第一张的左边缘 = 中心 - n*间距/2',
  Math.abs(g.boxAt(0, N)[0] - (g.cx - N * PITCH / 2)) < 0.01, '');

// ---- 4) 手牌变少时整行重新居中 ----
const b7 = g.boxAt(0, 7);
check('手牌变少 -> 整行重新居中（不是简单砍掉右边几张）',
  Math.abs((b7[0] + b7[2]) / 2 - (g.cx - 7 * PITCH / 2 + PITCH / 2)) < 0.01,
  `7 张时第一张 x=${b7[0].toFixed(1)}`);

// ---- 5) 摸到的那张：在整行右边单独挂着 ----
const drawn = g.boxAt(N, N, { drawnGap: 0.3 });
const last = g.boxAt(N - 1, N);
check('摸到的那张在整行右边、且隔开一点',
  drawn[0] > last[2], `最后一张到 ${last[2].toFixed(0)}，摸到的从 ${drawn[0].toFixed(0)} 开始`);
check('  隔开的距离 = 间距 × 0.3',
  Math.abs((drawn[0] - last[2]) - PITCH * 0.3) < 1,
  `${(drawn[0] - last[2]).toFixed(1)} vs ${(PITCH * 0.3).toFixed(1)}`);

// ---- 6) 存盘 / 读盘：下次启动连那一次识图都不需要 ----
{
  const saved = JSON.parse(JSON.stringify(g));
  const g3 = new RowGeometry();
  check('存下来的几何能恢复（同尺寸）', g3.restore(saved, W, H) === true, '');
  check('  恢复之后位置算得一样',
    JSON.stringify(g3.boxAt(3, N)) === JSON.stringify(g.boxAt(3, N)), '');
  const g4 = new RowGeometry();
  check('窗口尺寸变了就作废（会重新量）', g4.restore(saved, W + 200, H) === false, '');
  check('  作废之后不给位置', g4.boxAt(0, N) === null, '');
  const g5 = new RowGeometry();
  check('空数据不会误当成有效', g5.restore(null, W, H) === false, '');
}

// ---- 7) 量不到时不要瞎给位置 ----
const g2 = new RowGeometry();
check('没量到几何之前不返回位置', g2.boxAt(0, 13) === null, '');

let fail = 0;
for (const [n, ok, d] of checks) {
  if (!ok) fail++;
  console.log(`${ok ? '✅' : '❌'} ${n}${d ? '　' + d : ''}`);
}
console.log();
console.log(fail === 0 ? `✅ 位置推算 ${checks.length} 项全部通过` : `❌ ${fail} 项不通过`);
process.exit(fail === 0 ? 0 : 1);
