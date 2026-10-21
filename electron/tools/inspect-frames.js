// 看抓到的帧：消息类型统计 + ActionPrototype 的字段结构。
//
//   node tools/inspect-frames.js [帧目录] [--fields]
//
// 不带 --fields 只出统计；带上就把每种动作的字段号/类型列出来（用来推协议）。

const fs = require('fs');
const path = require('path');
const pb = require('../lib/pb');
const { loadFrames, loadDecoded } = require('../lib/frames.js');

const dir = process.argv[2] && !process.argv[2].startsWith('--')
  ? process.argv[2]
  : path.resolve(__dirname, '../../build/ws-frames');
const SHOW_FIELDS = process.argv.includes('--fields');

if (!fs.existsSync(dir)) {
  console.error(`帧目录不存在: ${dir}`);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// 读 messages.log
// ---------------------------------------------------------------------------
const logPath = path.join(dir, 'messages.log');
const counts = new Map();
const order = [];
let total = 0;
let kept = 0;

if (fs.existsSync(logPath)) {
  for (const line of fs.readFileSync(logPath, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    total += 1;
    const m = line.match(/^\s*(\d+)\s+(←|→)\s+(\d+)B\s+(存|略)\s+(.+)$/);
    if (!m) continue;
    const [, , arrow, size, keep, name] = m;
    if (keep === '存') kept += 1;
    const key = `${arrow} ${name}`;
    counts.set(key, (counts.get(key) || 0) + 1);
    if (!order.includes(key)) order.push(key);
  }
}

console.log(`帧目录: ${dir}`);
console.log(`消息总数: ${total}   其中落盘原始字节: ${kept}`);
console.log('');
if (!counts.size) {
  console.log('⚠ 一条消息都没有 —— 说明钩子没抓到。');
  console.log('  最可能的原因：Unity 把 WebSocket 建在 Worker 里，主线程钩子够不着。');
  process.exit(0);
}

console.log('=== 消息类型统计（按出现顺序）===');
for (const k of order) {
  console.log(`  ${String(counts.get(k)).padStart(5)}  ${k}`);
}

// ---------------------------------------------------------------------------
// 解 ActionPrototype 的字段结构
// ---------------------------------------------------------------------------
function fieldTypes(buf) {
  const types = new Map();
  for (const f of pb.parseFields(buf)) {
    const t = f.wire === 0 ? 'varint' : f.wire === 2 ? 'bytes' : `wire${f.wire}`;
    const cur = types.get(f.no) || new Set();
    cur.add(t);
    types.set(f.no, cur);
  }
  return types;
}

if (SHOW_FIELDS) {
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.bin')).sort();
  // 每种动作只取第一帧来看结构
  const seen = new Set();
  console.log('\n=== 各类动作的字段结构 ===');
  for (const f of files) {
    const buf = fs.readFileSync(path.join(dir, f));
    let msg;
    try {
      msg = pb.unwrap(buf);
    } catch (e) {
      continue;
    }
    if (!msg.action || seen.has(msg.action)) continue;
    seen.add(msg.action);

    // ActionPrototype.data 外面通常还包一层 GameAction{name=1,data=2}
    let inner = msg.actionPayload;
    let innerName = null;
    try {
      const g = pb.parseFields(inner);
      for (const fl of g) {
        if (fl.no === 1 && fl.wire === 2) {
          const s = Buffer.from(fl.bytes).toString('utf8');
          if (!s.includes('\ufffd')) innerName = s;
        }
        if (fl.no === 2 && fl.wire === 2) inner = fl.bytes;
      }
    } catch (e) {
      /* 保持原样 */
    }

    const t = fieldTypes(inner);
    const desc = [...t.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([no, s]) => `${no}:${[...s].join('/')}`)
      .join('  ');
    console.log(`\n  ${msg.action}${innerName ? `  ->  ${innerName}` : ''}`);
    console.log(`      字段: ${desc}`);
    console.log(`      原始: ${Buffer.from(inner).subarray(0, 32).toString('hex')}`);
  }
}
