// 修复 index.jsonl 的偏移错位。
//
// 背景：frames.bin 用追加模式，如果进程启动时文件已有内容，而偏移从 0 计，
// 索引就会整体错位。messages.log 里记了每帧的大小和「存/略」标记，
// 用它把偏移重新算出来。
//
//   node tools/repair-index.js [帧目录]
const fs = require('fs');
const path = require('path');

const dir = process.argv[2] || path.resolve(__dirname, '../../build/ws-frames');
const logPath = path.join(dir, 'messages.log');
const binPath = path.join(dir, 'frames.bin');
const idxPath = path.join(dir, 'index.jsonl');

if (!fs.existsSync(logPath) || !fs.existsSync(binPath)) {
  console.error('缺少 messages.log 或 frames.bin');
  process.exit(1);
}
const binSize = fs.statSync(binPath).size;

// messages.log 每行: 000123 ←   138B  存  ActionPrototype/...
const entries = [];
for (const line of fs.readFileSync(logPath, 'utf8').split('\n')) {
  const m = line.match(/^\s*(\d+)\s+(←|→)\s+(\d+)B\s+(存|略)\s+(.+?)\s*$/);
  if (!m) continue;
  if (m[4] !== '存') continue;
  entries.push({ seq: Number(m[1]), direction: m[2] === '←' ? 'recv' : 'send', size: Number(m[3]), msg: m[5] });
}
const total = entries.reduce((a, e) => a + e.size, 0);
const base = binSize - total;
console.log(`frames.bin ${binSize}B，索引里应有 ${entries.length} 帧共 ${total}B，起始偏移 ${base}`);
if (base < 0) {
  console.error(`✗ 对不上（差 ${-base}B）—— messages.log 里可能记录了没落盘的帧，无法自动修复`);
  process.exit(2);
}

const bin = fs.readFileSync(binPath);
let off = base;
const lines = [];
let ok = 0;
for (const e of entries) {
  const buf = bin.subarray(off, off + e.size);
  // 校验：帧首字节应是 01/02/03（雀魂帧头）
  const good = buf.length === e.size && (buf[0] === 1 || buf[0] === 2 || buf[0] === 3);
  if (good) ok += 1;
  lines.push(JSON.stringify({ seq: e.seq, direction: e.direction, size: e.size, off, len: e.size, msg: e.msg }));
  off += e.size;
}
fs.writeFileSync(idxPath + '.repaired', lines.join('\n') + '\n');
console.log(`校验通过 ${ok}/${entries.length} 帧首字节合法`);
console.log(`已写出 ${idxPath}.repaired`);
