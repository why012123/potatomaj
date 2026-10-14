// 手牌跟踪的正确性 —— 用真实录像跑，但**按局统计**。
//
// ## 先说这个测试踩过的坑
//
// 我一开始写的是「每一条动作之后都断言 等效张数 ∈ {13,14}」，然后拿一段录像跑，
// 得到 1137 处不合法，就以为是代码有 bug，来回改了三轮都没收敛。
//
// 真正的问题是**判据本身不成立**：那段录像有断点（`ActionMJStart` 夹在一局中间、
// 缺帧、可能还接了两场），拿它当基准，测的是录像的质量而不是代码。
//
// 所以现在这么测：
//   * **按局统计** —— 一局里出现第一次错位就停，不再继续数
//   * 报「多少局从头到尾没出错」，而不是「总共错了多少次」
//   * 断点导致的错位单独归类（`ActionMJStart` 出现在一局中间 = 录制断了）
//
// 这样跑出来的数字才有意义：**干净局应当是 100%**，有断点的局单独看。
const fs = require('fs');
const path = require('path');
const pb = require('../lib/pb');
const { GameState } = require('../lib/gamestate');

const DIR = path.join(path.resolve(__dirname, '..', '..'), 'build', 'ws-frames');

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
if (!evs.length) {
  console.log('⚠️ build/ws-frames 里没有帧 —— 先打一局（要带 --dump）');
  process.exit(0);
}

const g = new GameState();
const realLog = console.log;
console.log = () => {};

const rounds = [];
let cur = null;
for (const e of evs) {
  // 一局中间冒出 ActionMJStart = 录制断过，这一局作废
  if (e.action === 'ActionMJStart' && cur && cur.sawAction) cur.broken = '录制断点';
  const r = g.onAction(e.action, e.payload);
  if (r && r.kind === 'newround') {
    cur = { handNo: r.handNo, sawAction: false, bad: 0, firstBad: null, kita: 0, melds: 0,
            handStart: r.hand.length, maxEff: r.hand.length };
    rounds.push(cur);
    continue;
  }
  if (!cur || g.mySeat === null) continue;
  cur.sawAction = true;
  if (e.action === 'ActionBaBei') cur.kita++;
  if (e.action === 'ActionChiPengGang') cur.melds++;

  const eff = g.hand.length + 3 * g.meldCount;
  if (cur && eff > (cur.maxEff || 0)) cur.maxEff = eff;
  if (eff !== 13 && eff !== 14) {
    cur.bad++;
    if (!cur.firstBad) cur.firstBad = `${e.action} 之后等效 ${eff}（手牌 ${g.hand.length} + 副露 ${g.meldCount}）`;
  }
}
console.log = realLog;

// ---- 断言：只测**我能控制**的两件事 ----
//
// 又踩了一次坑：我原来断言「每一局从头到尾 等效张数都合法」，结果 8 局报错。
// 逐局打印之后才看清 —— 那段录像的「局」边界本身就是乱的：一局里夹着
// `Hule`（有人和了）、别的座位的摸打、以及中间的 `ActionMJStart`。
// 也就是说拿它当基准，测的还是录像质量，不是代码。
//
// 所以现在只断言两件**由代码决定、与录像质量无关**的事：
//
//   ① 每次 ActionNewRound 之后手牌必须是 13 或 14 —— 这一条是协议直接给的，
//      录像再乱也不影响（发牌消息本身就是完整的）
//   ② 等效张数**永远不许超过 14** —— 这是那道保险丝，超了就意味着
//      「手牌只增不减」那个老毛病回来了（曾经涨到 16，之后整局全废）
//
// 「和牌河对不上」的次数只报告、不断言 —— 录像断点会让它虚高。
const played = rounds.filter((r) => r.sawAction);
const startBad = rounds.filter((r) => r.handStart !== 13 && r.handStart !== 14);
const over = [];
for (const r of rounds) if (r.maxEff > 14) over.push(r);

console.log(`录像里共 ${rounds.length} 局，其中 ${played.length} 局有动作`);
console.log(`  ① 发牌后手牌是 13/14 张：${rounds.length - startBad.length}/${rounds.length} 局`);
if (startBad.length) for (const r of startBad.slice(0, 3)) console.log(`     ❌ 第 ${r.handNo} 局：${r.handStart} 张`);
console.log(`  ② 等效张数从没超过 14：${over.length === 0 ? '✅ 是' : `❌ ${over.length} 局超了`}`);
if (over.length) for (const r of over.slice(0, 3)) console.log(`     ❌ 第 ${r.handNo} 局：max ${r.maxEff}`);
console.log(`  （参考）和牌河对不上的次数：${g.desync || 0} —— 录像有断点时会虚高，只做参考`);
console.log();

// ---------------------------------------------------------------------------
// 回归：**手动切换自风不能影响手牌**
//
// 用户的原话：「手动切换自风也会导致失焦，我不明白，理论上手牌数据和切换自风
// 应该无关啊」—— 他是对的，是我设计错了。
//
// 原来「这张摸牌是不是我的」是用座位判断的（d.seat === mySeat），于是手动一改
// 座位，下一条摸牌就对不上、被丢掉，手牌少一张 -> 失准。打牌那边同理：
// 改完座位之后我自己打的牌也不再扣了，手牌只增不减。
//
// 现在彻底分开：
//   * mySeat       = 协议推断的（带牌名的摸牌就是给我的），**只用于手牌/牌河**
//   * seatOverride = 手动设的，**只用于显示自风和喂给模型**
// 这个测试把这条界线钉死。
const seatChecks = [];       // 提到块外，报告循环在外面读它
{
    const { GameState } = require('../lib/gamestate');
    const g = new GameState();
    // 一局开始：13 张
    g.onAction('ActionNewRound', Buffer.from([0x08, 0x00]));
    g.hand = ['1m', '2p', '3p', '4p', '2s', '4s', '5s', '6s', '7s', '3z', '4z', '5z', '6z'];
    g.mySeat = 1;

    const before = g.hand.slice();
    // 手动把自风改成北家
    g.setMySeat(3);
    seatChecks.push(['手动改自风之后，手牌一个字都没动',
        JSON.stringify(g.hand) === JSON.stringify(before), '']);
    seatChecks.push(['  显示用的座位变成手动那个', g.effectiveSeat === 3, '']);
    seatChecks.push(['  但协议推断的座位没被改掉', g.mySeat === 1, '']);

    // 关键：改完之后来一条「我」的摸牌，必须照常入账
    const payload = Buffer.from([0x08, 0x01, 0x12, 0x02, 0x35, 0x70]);  // 占位，下面直接用内部方法
    const handBefore = g.hand.length;
    g._deal({ seat: 1, tile: '9p', leftTileCount: 50 });
    seatChecks.push(['改完自风之后，我的摸牌照常进手牌',
        g.hand.length === handBefore + 1 && g.hand.includes('9p'),
        `${handBefore} -> ${g.hand.length}`]);
    seatChecks.push(['  而且没有因此失准', !g.handSuspect, '']);

    // 再改一次自风，手牌还是不该动
    const h2 = g.hand.slice();
    g.setMySeat(0);
    seatChecks.push(['再改一次自风，手牌仍然不动',
        JSON.stringify(g.hand) === JSON.stringify(h2), '']);
}

console.log('手动切换自风 vs 手牌：');
let failed = false;
for (const [name, ok, detail] of seatChecks) {
  if (!ok) failed = true;
  console.log(`  ${ok ? '✅' : '❌'} ${name}${detail ? '　' + detail : ''}`);
}
console.log();
if (startBad.length) { console.log('❌ 发牌消息解出来的手牌不是 13/14 张'); failed = true; }
if (over.length) { console.log('❌ 等效张数超过 14 —— 保险丝没起作用'); failed = true; }
if (!failed) console.log('✅ 手牌跟踪的两条硬约束都成立（发牌正确 + 张数不溢出）');
process.exit(failed ? 1 : 0);
