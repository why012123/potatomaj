// 协议解码的回归测试 —— 拿真实录下来的帧跑，不需要网络。
//
// 这个文件以前测的是我**手工试出来的**密钥表（牌河一套、立直一套……）。
// 现在全部换成 lib/actions.js 的正规解码（算法来自 shinkuan/Akagi，见 lib/liqi.js），
// 所以测试也跟着改成「用真实帧验证每条消息都解得出合法内容」。
//
// 为什么要留这个测试：解码是**唯一的数据来源**了 —— 手牌、宝牌、副露、牌河、
// 立直全靠它。它要是错，界面会安静地显示错误的东西。
const fs = require('fs');
const path = require('path');
const pb = require('../lib/pb');
const { decodeAction } = require('../lib/actions');
const { GameState } = require('../lib/gamestate');

const DIR = path.join(path.resolve(__dirname, '..', '..'), 'build', 'ws-frames');
const TILE_RE = /^[0-9][mpsz]$/;
const isTile = (t) => TILE_RE.test(t) && t !== '0z';

function frames() {
  const idxPath = path.join(DIR, 'index.jsonl');
  if (!fs.existsSync(idxPath)) return [];
  const idx = fs.readFileSync(idxPath, 'utf8').trim().split('\n')
    .filter(Boolean).map((l) => { try { return JSON.parse(l); } catch (e) { return null; } })
    .filter(Boolean);
  const bin = fs.readFileSync(path.join(DIR, 'frames.bin'));
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

const evs = frames();
const checks = [];
const check = (n, ok, d = '') => checks.push([n, ok, d]);

if (!evs.length) {
  console.log('⚠️ build/ws-frames 里没有帧 —— 先双击土豆麻将.command 打一局');
  process.exit(0);
}

// ---- 逐条消息：解得出、内容是合法的 ----
const byAction = {};
for (const e of evs) {
  const d = decodeAction(e.action, e.payload);
  (byAction[e.action] = byAction[e.action] || []).push({ d, payload: e.payload });
}

const nr = byAction.ActionNewRound || [];
const nrOk = nr.filter((x) => x.d && x.d.hand.length >= 13 && x.d.hand.every(isTile)).length;
check(`ActionNewRound 手牌全部合法（${nr.length} 条）`, nrOk === nr.length && nr.length > 0,
  `${nrOk}/${nr.length}`);
// scores 也是 packed repeated —— 以前一直读成空数组，界面上的点数就没更新过
const nrScores = nr.filter((x) => x.d && x.d.scores.length).length;
check('  ActionNewRound 点数解出来了（packed repeated）',
  nrScores === nr.length, `${nrScores}/${nr.length}`);

const dt = byAction.ActionDealTile || [];
const dtOk = dt.filter((x) => x.d).length;
const dtTile = dt.filter((x) => x.d && x.d.tile).length;
check(`ActionDealTile 全部解得出（${dt.length} 条）`, dtOk === dt.length, `${dtOk}/${dt.length}`);
check('  其中带牌名的（=自己摸的）不多不少', dtTile > 0 && dtTile < dt.length,
  `${dtTile} 条带牌名 / ${dt.length}`);

const ds = byAction.ActionDiscardTile || [];
const dsOk = ds.filter((x) => x.d && x.d.tile && isTile(x.d.tile) && x.d.seat <= 3).length;
check(`ActionDiscardTile 牌名和座位全部合法（${ds.length} 条）`, dsOk === ds.length, `${dsOk}/${ds.length}`);

// 打牌消息里带的「我可以吃/碰/杠/荣」操作列表（HUD 的鸣牌建议靠它）
const dsOps = ds.filter((x) => x.d && x.d.operations);
const opKinds = new Set();
for (const x of dsOps) for (const o of x.d.operations.ops) opKinds.add(o.kind);
check(`打牌消息里的鸣牌操作解出来了（${dsOps.length} 条 / 共 ${ds.length} 条打牌）`,
  dsOps.length > 0, [...opKinds].join(','));
check('  操作种类含吃/碰/荣（type 2/3/9）',
  ['chi', 'pon', 'ron'].every((k) => opKinds.has(k)), [...opKinds].join(','));

const cpg = byAction.ActionChiPengGang || [];
const cpgOk = cpg.filter((x) => x.d && x.d.tiles.length >= 3 && x.d.tiles.every(isTile)).length;
check(`ActionChiPengGang 副露的牌全部合法（${cpg.length} 条）`, cpgOk === cpg.length, `${cpgOk}/${cpg.length}`);
// froms 是 packed repeated —— 以前解成空数组，吃牌时只能猜哪张是别人打的
const cpgFroms = cpg.filter((x) => x.d && x.d.froms.length === x.d.tiles.length).length;
check('  副露来源 froms 全部解出来了（packed repeated）',
  cpgFroms === cpg.length, `${cpgFroms}/${cpg.length}`);

const bb = byAction.ActionBaBei || [];
check(`ActionBaBei 全部解得出（${bb.length} 条）`, bb.filter((x) => x.d).length === bb.length, '');

// ---- 整段跑一遍：每一局的手牌都得是 13 或 14 张 ----
const g = new GameState();
const rounds = [];
const logs = [];
const realLog = console.log;
console.log = () => {};
for (const e of evs) {
  const r = g.onAction(e.action, e.payload);
  if (r && r.kind === 'newround') rounds.push(r);
}
console.log = realLog;

const lens = rounds.map((r) => r.hand.length);
check(`整段录像跑出 ${rounds.length} 局`, rounds.length > 0, '');
check('每局手牌都是 13 或 14 张', lens.every((n) => n === 13 || n === 14),
  `张数分布 ${JSON.stringify([...new Set(lens)].sort())}`);
check('每局都读到了宝牌指示牌', rounds.every((r) => r.dora.length > 0),
  `${rounds.filter((r) => r.dora.length).length}/${rounds.length}`);
check('座位推断出来了', g.mySeat !== null && g.mySeat >= 0 && g.mySeat <= 3,
  g.mySeat === null ? '还是 null' : `${['东', '南', '西', '北'][g.mySeat]}家`);
check('牌桌类型认出来了（三麻/四麻）', typeof g.sanma === 'boolean', g.sanma ? '三麻' : '四麻');

let fail = 0;
for (const [n, ok, d] of checks) {
  if (!ok) fail++;
  console.log(`${ok ? '✅' : '❌'} ${n}${d ? '　' + d : ''}`);
}
console.log();
if (rounds.length) {
  console.log('前 3 局（全部来自协议，没有一句识图）：');
  for (const r of rounds.slice(0, 3)) {
    console.log(`  第${r.handNo}局  ${r.hand.join(' ')}   宝牌 ${r.dora.join(' ')}`);
  }
  console.log();
}
console.log(fail === 0 ? '✅ 协议解码回归通过' : `❌ ${fail} 项不通过`);
process.exit(fail === 0 ? 0 : 1);
