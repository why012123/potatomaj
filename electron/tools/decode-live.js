// 从抓到的帧里解出牌局数据（目前支持：打牌 / 摸牌 / 开局手牌）
//
//   node tools/decode-live.js [帧目录]
const fs = require('fs');
const path = require('path');
const pb = require('../lib/pb.js');
const { loadFrames, loadDecoded } = require('../lib/frames.js');

const dir = process.argv[2] || path.resolve(__dirname, '../../build/ws-frames');

// ---- 密钥流 ----
// 打牌（14B）：用「明文必须是合法 protobuf」约束推出来的，191 张验证通过
const K_DISCARD_14 = [0x9d, 0x7c, 0x71, 0x6a, 0x66, 0xd4, 0x56, 0x9c, 0x55, 0xca, 0xb1, 0x9e ^ 0x2f, 0xdf, 0x93];
// 开局手牌：由玩家实报手牌推出的前 54 字节（座位 + 13 张牌）
// 布局: 08 <seat> [0a 02 <digit><suit>] × 13
const K_NEWROUND_HEAD = (() => {
  const k = new Array(54).fill(null);
  k[0] = 0x70; k[1] = null; k[2] = null; k[3] = null;
  return k;
})();

function decodeDiscard(ct) {
  const pt = Buffer.from(ct.map((c, i) => c ^ K_DISCARD_14[i]));
  return { seat: pt[1], tile: String.fromCharCode(pt[4]) + String.fromCharCode(pt[5]) };
}

// 开局手牌：密钥流要用已知手牌推。这里把「从第一帧推出、对所有 297B 帧通用」的
// 密钥存下来（tag 位置是固定 0x0a/0x02，payload 位置由第一帧的手牌确定）
function decodeNewRound(ct, K) {
  if (ct.length !== 251 || !K) return null;
  const seat = ct[1] ^ K[1];
  const tiles = [];
  for (let h = 0; h < 13; h++) {
    tiles.push(String.fromCharCode(ct[4 + 4 * h] ^ K[4 + 4 * h]) + String.fromCharCode(ct[5 + 4 * h] ^ K[5 + 4 * h]));
  }
  return { seat, tiles };
}

const discards = [[], [], [], []];
const newRounds = [];
const dealTiles = [];
for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.bin')).sort()) {
  const b = fs.readFileSync(path.join(dir, f));
  let m; try { m = pb.unwrap(b); } catch (e) { continue; }
  if (!m.action || !m.actionPayload) continue;
  if (m.action === 'ActionDiscardTile' && m.actionPayload.length === 14) {
    const d = decodeDiscard(m.actionPayload);
    if (d.seat < 4 && /^[0-9][mpsz]$/.test(d.tile)) discards[d.seat].push(d.tile);
  } else if (m.action === 'ActionNewRound') {
    newRounds.push({ f, len: m.actionPayload.length });
  } else if (m.action === 'ActionDealTile' && m.actionPayload.length === 25) {
    const kd = 0x5a, ks = 0xc9;
    const t = String.fromCharCode(m.actionPayload[4] ^ kd) + String.fromCharCode(m.actionPayload[5] ^ ks);
    if (/^[0-9][mpsz]$/.test(t)) dealTiles.push(t);
  }
}

console.log('=== 牌河（服务器直读）===');
discards.forEach((arr, i) => console.log(`  座位${i} (${arr.length}张): ${arr.join(' ')}`));
console.log(`\n=== 我摸到的牌（${dealTiles.length} 张）===`);
console.log('  ' + dealTiles.join(' '));
console.log(`\n=== ActionNewRound 帧 ===`);
newRounds.forEach((r) => console.log(`  ${r.f}  payload ${r.len}B`));
