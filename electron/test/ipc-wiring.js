// IPC 通道接线检查：页面侧要用到的通道，主进程必须真的注册了。
//
// 为什么要专门查这个：通道名写错**不会报错**，只会静默什么都不发生。
// 表现就是「功能莫名其妙不工作」，而且日志里干干净净 —— 这类问题最难查。
//
// 静态扫描即可，不需要跑 Electron。
const fs = require('fs');
const path = require('path');

const DIR = path.resolve(__dirname, '..');
const preloads = fs.readdirSync(DIR).filter((f) => /^preload.*\.js$/.test(f));
const main = fs.readFileSync(path.join(DIR, 'main.js'), 'utf8');

const grab = (s, re) => [...s.matchAll(re)].map((m) => m[1]);
const all = (arr) => [...new Set(arr)].sort();

// 主进程注册的处理器
const handled = all(grab(main, /ipcMain\.handle\(\s*'([^']+)'/g));
const onMain = all(grab(main, /ipcMain\.on\(\s*'([^']+)'/g));
// 主进程发给页面的通道，以及分别发给哪个窗口
const sentToPage = all(grab(main, /webContents\.send\(\s*'([^']+)'/g));

// 页面侧
const sends = [], invokes = [], listens = [], perPreload = {};
for (const f of preloads) {
  const s = fs.readFileSync(path.join(DIR, f), 'utf8');
  const sn = grab(s, /ipcRenderer\.send\(\s*'([^']+)'/g);
  const iv = grab(s, /ipcRenderer\.invoke\(\s*'([^']+)'/g);
  const ls = grab(s, /ipcRenderer\.on\(\s*'([^']+)'/g);
  perPreload[f] = { sn, iv, ls };
  sends.push(...sn); invokes.push(...iv); listens.push(...ls);
}

const checks = [];
const missOn = all(sends).filter((c) => !onMain.includes(c));
const missHandle = all(invokes).filter((c) => !handled.includes(c));
const missListen = sentToPage.filter((c) => !all(listens).includes(c));

checks.push([`页面 send 的通道都有主进程处理器（${all(sends).length} 个）`, missOn.length === 0, missOn.join(', ')]);
checks.push([`页面 invoke 的通道都有主进程 handle（${all(invokes).length} 个）`, missHandle.length === 0, missHandle.join(', ')]);
checks.push([`主进程发给页面的通道都有页面在监听（${sentToPage.length} 个）`, missListen.length === 0, missListen.join(', ')]);
checks.push(['每个 preload 都能被加载到（至少两个）', preloads.length >= 2, preloads.join(', ')]);
// 反向：注册了却没人用 —— 不算错，但要能看见（可能是改到一半的残留）
const unused = handled.filter((c) => !all(invokes).includes(c));
checks.push(['（参考）注册了但页面没用的 handle', true, unused.join(', ') || '（无）']);

let fail = 0;
for (const [name, pass, detail] of checks) {
  if (!pass) fail++;
  console.log(`${pass ? '✅' : '❌'} ${name}${detail ? '   ' + detail : ''}`);
}
console.log();
console.log('各 preload 的通道：');
for (const [f, c] of Object.entries(perPreload)) {
  console.log(`  ${f.padEnd(20)} send=${c.sn.length} invoke=${c.iv.length} on=${c.ls.length}`);
}
console.log();
console.log(fail === 0 ? '✅ IPC 接线全部对得上' : `❌ ${fail} 项不通过`);
process.exit(fail === 0 ? 0 : 1);
