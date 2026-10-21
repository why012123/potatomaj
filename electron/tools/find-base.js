// 修复 index.jsonl 的「整体偏移错位」。
//
// 背景：frames.bin 是追加模式，若进程启动时文件已有内容、而偏移从 0 计，
// 索引里所有 off 都会整体偏掉一个常数。这个常数在**同一次运行内不变**，
// 所以扫一次存下来即可，之后重复运行会直接沿用。
//
//   node tools/find-base.js [帧目录]
//
// 原始（错位的）索引会备份成 index.jsonl.raw，每次都从备份重算，
// 避免反复就地修改把偏移叠加两次。

const fs = require('fs');
const path = require('path');

const dir = process.argv[2] || path.resolve(__dirname, '../../build/ws-frames');
const binPath = path.join(dir, 'frames.bin');
const idxPath = path.join(dir, 'index.jsonl');
const rawPath = path.join(dir, 'index.jsonl.raw');
const basePath = path.join(dir, '.base');

const bin = fs.readFileSync(binPath);
if (!fs.existsSync(rawPath)) {
  if (!fs.existsSync(idxPath)) { console.error('没有 index.jsonl'); process.exit(1); }
  fs.copyFileSync(idxPath, rawPath);
  console.log('已备份原始索引 -> index.jsonl.raw');
}
// .raw 里是「原始错位」的条目；正在运行的应用还会继续往 index.jsonl
// 追加新的原始条目（它用的是旧逻辑），所以要把这些新的也并进来。
const rawRows = fs.readFileSync(rawPath, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const maxRawSeq = rawRows.reduce((m, r) => Math.max(m, r.seq), -1);
const fresh = fs.existsSync(idxPath)
  ? fs.readFileSync(idxPath, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l))
      .filter((r) => r.seq > maxRawSeq && !r.fixed)
  : [];
const idx = rawRows.concat(fresh);
if (fresh.length) {
  fs.writeFileSync(rawPath, idx.map((e) => JSON.stringify(e)).join('\n') + '\n');
  console.log(`并入 ${fresh.length} 条新抓到的原始条目`);
}

function score(base, rows) {
  let ok = 0;
  for (const e of rows) {
    const p = base + e.off;
    if (p + e.len > bin.length) return -1;
    if ([1, 2, 3].includes(bin[p])) ok += 1;
  }
  return ok;
}

// 抽样扫描（够用且快）
const sample = idx.filter((_, i) => i % 5 === 0).slice(0, 150);
let best = { base: 0, score: -1 };

if (fs.existsSync(basePath)) {
  const b = Number(fs.readFileSync(basePath, 'utf8').trim());
  const sc = score(b, sample);
  if (sc >= 0) {
    best = { base: b, score: sc };
    console.log(`沿用上次基准 ${b}（命中 ${sc}/${sample.length}）`);
  }
}
if (best.score < sample.length * 0.8) {
  for (let base = 0; base < bin.length; base++) {
    const sc = score(base, sample);
    if (sc > best.score) best = { base, score: sc };
    if (sc === sample.length) break;   // 全中，不用再扫
  }
  console.log(`扫描得到基准 ${best.base}（命中 ${best.score}/${sample.length}）`);
}

const out = [];
let valid = 0;
for (const e of idx) {
  const p = best.base + e.off;
  if (p + e.len > bin.length) continue;
  if ([1, 2, 3].includes(bin[p])) valid += 1;
  out.push({ ...e, off: p, fixed: true });
}
fs.writeFileSync(idxPath, out.map((e) => JSON.stringify(e)).join('\n') + '\n');
fs.writeFileSync(basePath, String(best.base));
console.log(`重写索引：${out.length} 条，帧首合法 ${valid} 条`);
