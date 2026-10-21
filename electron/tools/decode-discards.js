// 解 ActionDiscardTile（14 字节那批）—— 密钥是在「明文必须是合法 protobuf」
// 的约束下推出来的，见对话记录。
const fs = require('fs');
const path = require('path');
const pb = require('../lib/pb.js');
const { loadFrames, loadDecoded } = require('../lib/frames.js');
const dir = process.argv[2] || path.resolve(__dirname, '../../build/ws-frames');

// 已知/推导出的密钥流（14 字节消息）
const K14 = [0x9d, 0x7c, 0x71, 0x6a, 0x66, 0xd4, 0x56, 0x9c, 0x55, 0xca, 0xb1, 0x9e ^ 0x2f, 0xdf, 0x93];

const rows = [];
for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.bin')).sort()) {
  const b = fs.readFileSync(path.join(dir, f));
  let m;
  try { m = pb.unwrap(b); } catch (e) { continue; }
  if (m.action === 'ActionDiscardTile' && m.actionPayload && m.actionPayload.length === 14)
    rows.push({ f, ct: m.actionPayload, step: m.step });
}

const seats = [[], [], [], []];
let bad = 0;
for (const r of rows) {
  const pt = Uint8Array.from(r.ct, (c, i) => c ^ K14[i]);
  const seat = pt[1];
  const tile = String.fromCharCode(pt[4]) + String.fromCharCode(pt[5]);
  if (!/^[0-9][mpsz]$/.test(tile) || seat > 3) { bad += 1; continue; }
  seats[seat].push(tile);
}
console.log(`解出 ${rows.length} 张打牌，其中格式异常 ${bad} 张\n`);
seats.forEach((arr, i) => {
  console.log(`座位 ${i} 的牌河（${arr.length} 张）:`);
  console.log('  ' + arr.join(' '));
});
