#!/usr/bin/env node
// 把录下来的帧解码成人能看的牌谱 —— 排查用。
//
//   node tools/decode.js              # 每局一行：手牌 + 宝牌
//   node tools/decode.js --full       # 连每一张摸打都打出来
//   node tools/decode.js --json       # 输出 JSON，方便喂给别的工具
//
// 这个脚本以前是「显示破解进度」的（密钥流按长度分套、还差几局样本……）。
// 那件事已经不用做了：解码算法在 shinkuan/Akagi 里公开着，见 electron/lib/liqi.js。
const fs = require('fs');
const path = require('path');
const pb = require('../electron/lib/pb');
const { decodeAction } = require('../electron/lib/actions');
const { sortHand } = require('../electron/lib/rowgeom');

const DIR = process.env.POTATOMAJ_DUMP_DIR || path.join(path.resolve(__dirname, '..'), 'build', 'ws-frames');
const FULL = process.argv.includes('--full');
const JSON_OUT = process.argv.includes('--json');
const SEAT = ['东', '南', '西', '北'];

const idxPath = path.join(DIR, 'index.jsonl');
if (!fs.existsSync(idxPath)) {
  console.log('还没有录到帧。双击 土豆麻将.command 打一局就有了。');
  process.exit(0);
}

const idx = fs.readFileSync(idxPath, 'utf8').trim().split('\n')
  .filter(Boolean).map((l) => { try { return JSON.parse(l); } catch (e) { return null; } })
  .filter(Boolean);
const bin = fs.readFileSync(path.join(DIR, 'frames.bin'));

const evs = [];
for (const e of idx) {
  if (e.off === undefined) continue;
  let m;
  try { m = pb.unwrap(bin.subarray(e.off, e.off + e.len)); } catch (err) { continue; }
  if (!m.action) continue;
  const d = decodeAction(m.action, m.actionPayload);
  if (d) evs.push({ action: m.action, d, seq: e.seq });
}
evs.sort((a, b) => a.seq - b.seq);

if (JSON_OUT) { console.log(JSON.stringify(evs.map((e) => ({ action: e.action, ...e.d })), null, 1)); process.exit(0); }

let round = 0;
let current = null;
const out = [];
for (const e of evs) {
  const { action, d } = e;
  if (action === 'ActionNewRound') {
    round++;
    current = { round, hand: sortHand(d.hand), dora: d.dora, events: [], state: '?' };
    out.push(current);
    if (FULL) {
      console.log(`\n第 ${round} 局　手牌 ${sortHand(d.hand).join(' ')}　宝牌指示牌 ${d.dora.join(' ')}`);
    }
    continue;
  }
  if (!current) continue;
  if (action === 'ActionDealTile' && d.tile) current.events.push(`${SEAT[d.seat]}摸 ${d.tile}`);
  if (action === 'ActionDiscardTile' && d.tile) {
    current.events.push(`${SEAT[d.seat]}打 ${d.tile}${d.isLiqi ? '（立直！）' : ''}${d.moqie ? '（摸切）' : ''}`);
  }
  if (action === 'ActionChiPengGang') current.events.push(`${SEAT[d.seat]}副露 ${d.tiles.join('')}`);
  if (action === 'ActionBaBei') current.events.push(`${SEAT[d.seat]}拔北`);
}

if (!JSON_OUT) {
  if (!FULL) {
    console.log(`共 ${out.length} 局（数据来源：${path.relative(path.resolve(__dirname, '..'), DIR)}）\n`);
    for (const r of out) {
      console.log(`第 ${String(r.round).padStart(2)} 局  ${r.hand.length} 张  ${r.hand.join(' ')}`);
      console.log(`          宝牌指示牌 ${r.dora.join(' ') || '（空）'}　投牌 ${r.events.length} 次`);
    }
  } else {
    console.log(`\n（共 ${out.length} 局）`);
  }
}
