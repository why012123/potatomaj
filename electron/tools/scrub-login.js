// 把已经落盘的「登录/账号类」帧删掉。
//
// 用途：早期版本的帧头解析有 bug，导致消息名全是「未知」，
// 黑名单匹配到空字符串而失效，登录帧被误写到磁盘。
// 这个脚本用修好的解析器重新识别每帧，把敏感的删掉，其余保留。
//
//   node tools/scrub-login.js [帧目录]

const fs = require('fs');
const path = require('path');
const pb = require('../lib/pb');
const { loadFrames, loadDecoded } = require('../lib/frames.js');

const dir = process.argv[2] || path.resolve(__dirname, '../../build/ws-frames');
const NEVER =
  /Login|Auth|Oauth|Account|Register|Pay|Order|Bill|Mail|Friend|Chat|Lobby|Signup|Captcha|Password|Token|Session/i;

if (!fs.existsSync(dir)) {
  console.error(`目录不存在: ${dir}`);
  process.exit(1);
}

const files = fs.readdirSync(dir).filter((f) => f.endsWith('.bin')).sort();
let removed = 0;
let kept = 0;
const keptByName = {};

for (const f of files) {
  const full = path.join(dir, f);
  let name = '';
  try {
    name = pb.unwrap(fs.readFileSync(full)).name || '';
  } catch (e) {
    name = '';
  }
  // 解不出名字的（心跳等）体积都很小，留着无妨；有名字的按黑名单判
  const bad = name ? NEVER.test(name) : false;
  if (bad) {
    fs.unlinkSync(full);
    removed += 1;
    console.log(`  删除 ${f}  ←  ${name}`);
  } else {
    kept += 1;
    if (name) keptByName[name] = (keptByName[name] || 0) + 1;
  }
}

console.log(`\n删掉 ${removed} 个敏感帧，保留 ${kept} 个`);
console.log('保留的消息类型:');
for (const [k, v] of Object.entries(keptByName).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${String(v).padStart(4)}  ${k}`);
}
