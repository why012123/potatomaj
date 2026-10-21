// 立直 / 摸切 的**语义**回归 —— 解码本身由 protocol.js 覆盖，这里查的是「读出来的东西
// 合不合麻将的规则」。
//
// 为什么值得单独测：立直是危险度判断的**唯一确定依据**（读不到的时候只能按巡目猜）。
// 解码错一个 bit，界面上会显示「某人立直了」而实际上没有 —— 那比不显示更糟。
const fs = require('fs');
const path = require('path');
const pb = require('../lib/pb');
const { decodeAction } = require('../lib/actions');

const DIR = path.join(path.resolve(__dirname, '..', '..'), 'build', 'ws-frames');

function discards() {
  const idxPath = path.join(DIR, 'index.jsonl');
  if (!fs.existsSync(idxPath)) return { rounds: [], flat: [] };
  const idx = fs.readFileSync(idxPath, 'utf8').trim().split('\n')
    .filter(Boolean).map((l) => { try { return JSON.parse(l); } catch (e) { return null; } })
    .filter(Boolean);
  const bin = fs.readFileSync(path.join(DIR, 'frames.bin'));
  const evs = [];
  for (const e of idx) {
    if (e.off === undefined) continue;
    let m;
    try { m = pb.unwrap(bin.subarray(e.off, e.off + e.len)); } catch (err) { continue; }
    if (m.action) evs.push({ action: m.action, payload: m.actionPayload, seq: e.seq });
  }
  evs.sort((a, b) => a.seq - b.seq);

  const rounds = [];
  let cur = null;
  const flat = [];
  for (const e of evs) {
    if (e.action === 'ActionNewRound') { cur = []; rounds.push(cur); continue; }
    if (e.action !== 'ActionDiscardTile') continue;
    const d = decodeAction(e.action, e.payload);
    if (!d || !d.tile) continue;
    flat.push(d);
    if (cur) cur.push(d);
  }
  return { rounds, flat };
}

const { rounds, flat } = discards();
const checks = [];
const check = (n, ok, d = '') => checks.push([n, ok, d]);

if (!flat.length) {
  console.log('⚠️ 没有打牌帧');
  process.exit(0);
}

const liqiCount = flat.filter((d) => d.isLiqi).length;
const moqieCount = flat.filter((d) => d.moqie).length;
const zhentingCount = flat.filter((d) => d.zhenting).length;

check(`立直次数是一局一次的量级（${liqiCount} 次 / ${rounds.length} 局）`,
  liqiCount > 0 && liqiCount < rounds.length * 3, `${liqiCount} 次`);
check(`摸切比立直常见得多（${moqieCount} vs ${liqiCount}）`,
  moqieCount > liqiCount, '');
// 振听只报告数量，不做断言 —— 抓到的这批大部分是四麻，振听会真的发生。
// （原来我写成「应该一直是 0」，那是错的：振听不是异常，是正常状态。）
check(`振听出现 ${zhentingCount} 次（只报告，不断言）`, true, `${zhentingCount} 次 / ${flat.length} 张`);

// 立直之后只能摸切 —— 用这条规则反过来验证两个字段读得对不对
let after = 0, afterMq = 0, before = 0, beforeMq = 0;
for (const r of rounds) {
  const declared = {};
  for (const d of r) {
    if (declared[d.seat]) { after++; if (d.moqie) afterMq++; }
    else { before++; if (d.moqie) beforeMq++; }
    if (d.isLiqi) declared[d.seat] = true;
  }
}
const rAfter = after ? afterMq / after : 0;
const rBefore = before ? beforeMq / before : 0;
// 这条是**真正有判别力**的检查：立直之后只能摸切是麻将规则，
// 所以两个字段只要有一个读错，这个比例就会塌下来。
check('立直之后摸切率明显更高（这条规则能反过来验证解码）',
  rAfter > rBefore * 1.3,
  `立直后 ${(rAfter * 100).toFixed(0)}% vs 立直前 ${(rBefore * 100).toFixed(0)}%`);

// 每个座位每局最多立直一次 —— 但**录像有断点时会误判**，所以只报告。
//
// 原因：我按 ActionNewRound 切「局」，而录像中间掉过帧，于是两段不同的对局
// 可能被接成一段。实测那两次「重复」里，出牌顺序是 3→2（四麻应该是 3→0），
// 说明中间缺了帧 —— 是录像的问题，不是解码的问题。
// 所以这条降级成提示：真要看它是否有意义，得先把录像按「断点」切干净。
let multi = 0;
for (const r of rounds) {
  const seen = {};
  for (const d of r) {
    if (!d.isLiqi) continue;
    if (seen[d.seat]) multi++;
    seen[d.seat] = true;
  }
}
check(`同一家立直两次的情况出现 ${multi} 次（录像有断点时会误判，只报告）`, true, `${multi} 次`);

let fail = 0;
for (const [n, ok, d] of checks) {
  if (!ok) fail++;
  console.log(`${ok ? '✅' : '❌'} ${n}${d ? '　' + d : ''}`);
}
console.log();
console.log(fail === 0 ? '✅ 立直 / 摸切语义回归通过' : `❌ ${fail} 项不通过`);
process.exit(fail === 0 ? 0 : 1);
