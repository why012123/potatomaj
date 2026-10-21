#!/usr/bin/env node
// 发牌消息（ActionNewRound）密钥的进度、求解与验证 —— 一个入口。
//
//   node tools/protocol.js            看进度（默认）
//   node tools/protocol.js --probe    诊断字段起点（换长度/换版本时用）
//   node tools/protocol.js --solve    尝试求解，解出来就写进程序
//   node tools/protocol.js --verify   用「我打出的牌必在我手里」独立验证已解出的密钥
//
// 背景（完整推导见 docs/协议进展.md）：
//   手牌本来就在发牌消息的 `repeated string tiles = 4` 字段里，只是被 XOR 盖了一层。
//   密钥流**按消息长度分套**（消息长度随牌局变），所以每种长度要各自攒够样本。
//   每张牌编码成 `22 02 <数字> <花色>`：0x22 是字段 4 的标签，0x02 是长度。
//
// 已试过、**确认行不通**的思路（别再走一遍）：
//   * 把不同长度的样本合并起来解 —— 「密钥流只差一个固定偏移」是假规律，
//     实测合并后候选数为 0。原因：字节 0/2/4 都是恒定字节，它们的差值只反映
//     密文自身的结构，对密钥流没有约束力。
const fs = require('fs');
const path = require('path');
const pb = require('../electron/lib/pb');

const ROOT = path.resolve(__dirname, '..');
const DIR = process.env.POTATOMAJ_DUMP_DIR || path.join(ROOT, 'build', 'ws-frames');
const KEYFILE = path.join(ROOT, 'electron', 'lib', 'newround-keys.json');
const START = 6;              // 手牌字段起点（前面是 chang / ju / ben 三个 varint 字段）
const STEP = 4;               // [tag][len][数字][花色]
const N_TILES = 13;
//: 牌河密钥（已完全解开，线上验证 376 张零错）—— 验证步骤要用它读打牌
const K_DISCARD = [0x9d, 0x7c, 0x71, 0x6a, 0x66, 0xd4, 0x56, 0x9c, 0x55, 0xca, 0xb1, 0x94, 0xdf, 0x93];

const isDigit = (c) => c >= '0' && c <= '9';
const isSuit = (c) => 'mpsz'.includes(c);
const isTile = (t) => t && isDigit(t[0]) && isSuit(t[1]) && !(t[0] === '0' && t[1] === 'z');

/** 读帧，按「发牌消息切局」整理出每局的密文和这一局的打牌 */
function load() {
  const idxPath = path.join(DIR, 'index.jsonl');
  if (!fs.existsSync(idxPath)) return { rounds: [], path: DIR };
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
  for (const ev of evs) {
    if (ev.action === 'ActionNewRound') { cur = { payload: ev.payload, discards: [] }; rounds.push(cur); }
    else if (ev.action === 'ActionDiscardTile' && cur && ev.payload && ev.payload.length === 14) {
      const seat = ev.payload[1] ^ K_DISCARD[1];
      const tile = String.fromCharCode(ev.payload[4] ^ K_DISCARD[4]) + String.fromCharCode(ev.payload[5] ^ K_DISCARD[5]);
      if (seat <= 3 && isTile(tile)) cur.discards.push({ seat, tile });
    }
  }
  return { rounds, path: DIR };
}

/** 位置 i 上，所有样本都能解成合法值的密钥有哪些 */
function keysFor(samples, pos, ok) {
  const out = [];
  for (let k = 0; k < 256; k++) {
    if (samples.every((s) => pos < s.length && ok(String.fromCharCode(s[pos] ^ k)))) out.push(k);
  }
  return out;
}

/** 一组样本的手牌字段解到什么程度 */
function survey(S) {
  const digits = [], suits = [];
  for (let i = 0; i < N_TILES; i++) {
    const p = START + i * STEP;
    if (p + 3 >= Math.min(...S.map((s) => s.length))) break;
    digits.push(keysFor(S, p + 2, isDigit));
    suits.push(keysFor(S, p + 3, isSuit));
  }
  return { digits, suits, n: digits.length };
}

const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);

// ---------------------------------------------------------------------------

/**
 * 判断一段打牌属于三麻还是四麻。
 *
 * 依据很硬：**三麻没有 2m~8m**。而且这不是可选项 —— 实测抓到的录像里
 * 三麻四麻是**混着**的（8 场里只有 1 局是够样本的三麻），
 * 不先分清就会在混着的数据上做推导，得出「字段有 4 种取值而三麻只有 3 家」
 * 这种自相矛盾的结论。这个坑真踩过。
 */
function tableOf(round) {
  const mid = round.discards.filter((d) => d.tile[1] === 'm'
    && Number(d.tile[0]) >= 2 && Number(d.tile[0]) <= 8).length;
  if (round.discards.length < 10) return '样本太少';
  return mid > 0 ? '四麻' : '三麻';
}

function cmdStatus(rounds) {
  const byLen = {};
  for (const r of rounds) (byLen[r.payload.length] = byLen[r.payload.length] || []).push(r);

  // 先报牌桌构成 —— 这是所有推导的前提
  const tally = {};
  for (const r of rounds) {
    const t = tableOf(r);
    tally[t] = (tally[t] || 0) + 1;
  }
  const parts = Object.entries(tally).map(([k, v]) => `${k} ${v} 局`);
  console.log('牌桌构成：' + parts.join('，'));
  const sanma = tally['三麻'] || 0;
  if (!sanma) {
    console.log('⚠️  没有够样本的三麻局 —— 下面这些数字全部来自四麻样本。');
    console.log('    密钥本身不分几麻（编码方式一样），但「三麻没有 2m~8m」这类');
    console.log('    推导不能用在四麻样本上。要分析三麻，先打几局三麻。');
  } else if (tally['四麻']) {
    console.log('⚠️  三麻四麻混着 —— 基于「三麻只有 3 家」的推导只能用在三麻那些局上');
    console.log('    （跑 node tools/split-games.js 看逐场明细）。');
  }
  console.log();
  const lens = Object.keys(byLen).map(Number).sort((a, b) => byLen[b].length - byLen[a].length);

  console.log(`发牌消息共 ${rounds.length} 局，分 ${lens.length} 种长度（密钥按长度分套）\n`);
  if (!rounds.length) {
    console.log('还没抓到帧。先双击「土豆麻将.command」打一局。');
    return 0;
  }
  console.log('长度   局数  数字候选均值  花色候选均值  状态');
  let solvedAny = 0;
  for (const len of lens) {
    const S = byLen[len].map((r) => r.payload);
    const s = survey(S);
    if (!s.n) { console.log(`  ${String(len).padEnd(5)} ${String(S.length).padEnd(5)} —            —             样本太短`); continue; }
    const dA = avg(s.digits.map((d) => d.length));
    const sA = avg(s.suits.map((d) => d.length));
    const done = s.digits.every((d) => d.length === 1) && s.suits.every((d) => d.length === 1);
    if (done) solvedAny++;
    // 一个位置要唯一，样本取值得覆盖足够多的数字；经验上 10 局上下能收敛
    const need = Math.max(0, 10 - S.length);
    const state = done ? '✅ 已解开'
      : (S.length < 3 ? `样本太少，再打 ${need} 局左右` : `再打 ${need} 局左右`);
    console.log(`  ${String(len).padEnd(5)} ${String(S.length).padEnd(5)} ` +
      `${dA.toFixed(1).padEnd(13)} ${sA.toFixed(1).padEnd(13)} ${state}`);
  }
  console.log();
  if (solvedAny) console.log(`${solvedAny} 种长度已解开 —— 跑 --solve 写进程序。`);
  else console.log('还没解开。帧是自动记录的，继续打就行，不用做任何事。');
  console.log('（每种长度约需 10 局上下；手牌字段一旦唯一，程序会自动改用协议读手牌。）');
  return 0;
}

function cmdProbe(rounds) {
  const byLen = {};
  for (const r of rounds) (byLen[r.payload.length] = byLen[r.payload.length] || []).push(r.payload);
  console.log('诊断字段起点：判别依据是「数字」那一列的候选密钥数 ——');
  console.log('起点错了的话，数字位置会落在恒定字节上，任何密钥都能把它「解」成数字，');
  console.log('候选数就会是满值 10。正确起点上会明显低于 10。\n');
  console.log('长度  局数  起点0  起点2  起点4  起点6   结论');
  for (const len of Object.keys(byLen).map(Number).sort((a, b) => byLen[b].length - byLen[a].length)) {
    const S = byLen[len];
    if (S.length < 2) continue;
    const scores = [0, 2, 4, 6].map((st) => {
      let sum = 0, n = 0;
      for (let i = 0; i < N_TILES; i++) {
        const p = st + i * STEP;
        if (p + 3 >= Math.min(...S.map((s) => s.length))) break;
        sum += keysFor(S, p + 2, isDigit).length; n++;
      }
      return n ? sum / n : 99;
    });
    const best = scores.indexOf(Math.min(...scores));
    console.log(`  ${String(len).padEnd(5)} ${String(S.length).padEnd(5)} ` +
      scores.map((v) => v.toFixed(1).padEnd(6)).join(' ') +
      ` 起点 ${[0, 2, 4, 6][best]} 最像`);
  }
  return 0;
}

function cmdSolve(rounds, quiet) {
  const byLen = {};
  for (const r of rounds) (byLen[r.payload.length] = byLen[r.payload.length] || []).push(r);
  let wrote = 0;
  for (const len of Object.keys(byLen).map(Number)) {
    const group = byLen[len];
    if (group.length < 3) continue;
    const S = group.map((r) => r.payload);
    const s = survey(S);
    if (!s.n || !s.digits.every((d) => d.length === 1) || !s.suits.every((d) => d.length === 1)) continue;
    const keys = {
      length: len, tag: 0x22, tileCount: s.n,
      digit: s.digits.map((d) => d[0]),
      suit: s.suits.map((d) => d[0]),
      samples: S.length,
      note: `tools/protocol.js --solve 解出，${S.length} 局样本`,
      crackedAt: new Date().toISOString(),
    };
    fs.writeFileSync(KEYFILE, JSON.stringify(keys, null, 1));
    wrote++;
    if (!quiet) {
      console.log(`✅ 长度 ${len} 解开了（${S.length} 局样本），已写入 ${path.relative(ROOT, KEYFILE)}`);
      const hand = [];
      for (let i = 0; i < s.n; i++) {
        const p = START + i * STEP;
        hand.push(String.fromCharCode(S[0][p + 2] ^ keys.digit[i]) +
                  String.fromCharCode(S[0][p + 3] ^ keys.suit[i]));
      }
      console.log(`   第一局的手牌：${hand.join(' ')}`);
      console.log('   程序下次启动会自动加载，手牌就不需要识图了。');
      console.log('   建议再跑一次 --verify 独立确认。');
    }
  }
  if (!wrote && !quiet) console.log('还没有哪种长度的密钥唯一确定 —— 再打几局。');
  return wrote ? 0 : 1;
}

/**
 * 独立验证：用「我打出去的牌必须在我手里」这条**逻辑必然**去查已解出的密钥。
 *
 * 为什么值得单独做：破解是拿「解出来必须是合法牌名」当判据的，那个判据是**必要条件**、
 * 不是充分条件 —— 错的密钥也可能碰巧解出合法牌名。而这条不一样：
 * 我打掉的牌（除非是刚摸到就打）必然出自开局那 13 张，正确密钥下命中率约八成，
 * 错密钥下只有三成上下。两者差得很远，一看就知道。
 */
function cmdVerify(rounds) {
  if (!fs.existsSync(KEYFILE)) { console.log('还没有解出的密钥文件，先跑 --solve。'); return 1; }
  const K = JSON.parse(fs.readFileSync(KEYFILE, 'utf8'));
  const group = rounds.filter((r) => r.payload.length === K.length);
  if (!group.length) { console.log(`没有长度 ${K.length} 的样本可以验证。`); return 1; }

  const decode = (payload) => {
    const out = [];
    for (let i = 0; i < K.tileCount; i++) {
      const p = START + i * STEP;
      out.push(String.fromCharCode(payload[p + 2] ^ K.digit[i]) +
               String.fromCharCode(payload[p + 3] ^ K.suit[i]));
    }
    return out;
  };

  console.log(`用 ${group.length} 局样本验证（长度 ${K.length}）：\n`);
  let best = null;
  for (let seat = 0; seat < 4; seat++) {
    let hit = 0, tot = 0;
    for (const r of group) {
      if (r.payload.length !== K.length) continue;
      const pool = decode(r.payload);
      if (!pool.every(isTile)) { console.log('  ❌ 解出来的不全是合法牌名 —— 密钥有问题'); return 1; }
      for (const d of r.discards) {
        if (d.seat !== seat) continue;
        tot++;
        const k = pool.indexOf(d.tile);
        if (k >= 0) { pool.splice(k, 1); hit++; }
      }
    }
    if (tot < 10) continue;
    const rate = hit / tot;
    console.log(`  当作我是${['东', '南', '西', '北'][seat]}家：我的打牌落在手牌里 ${hit}/${tot} = ${(rate * 100).toFixed(1)}%`);
    if (!best || rate > best.rate) best = { rate, seat, hit, tot };
  }
  if (!best) { console.log('  打牌样本太少，验证不了。'); return 1; }
  console.log();
  if (best.rate >= 0.6) {
    console.log(`✅ 验证通过：座位 ${best.seat} 下命中 ${(best.rate * 100).toFixed(1)}%（错密钥只有三成上下）`);
    console.log('   如果这一局里我正好是那个座位，说明密钥是对的。');
    return 0;
  }
  console.log(`⚠️ 命中率只有 ${(best.rate * 100).toFixed(1)}%，接近随机 —— 密钥可能是错的。`);
  console.log('   （也可能只是这几局我恰好不在那个座位，多打几局再看。）');
  return 1;
}

// ---------------------------------------------------------------------------

const mode = process.argv.includes('--solve') ? 'solve'
  : process.argv.includes('--verify') ? 'verify'
  : process.argv.includes('--probe') ? 'probe' : 'status';
const { rounds, path: dumpPath } = load();
if (mode === 'status') console.log(`数据来源：${path.relative(ROOT, dumpPath)}\n`);

let code = 0;
if (mode === 'status') code = cmdStatus(rounds);
else if (mode === 'probe') code = cmdProbe(rounds);
else if (mode === 'solve') code = cmdSolve(rounds);
else code = cmdVerify(rounds);
process.exit(code);
