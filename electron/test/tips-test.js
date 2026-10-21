// 验证「牌面提示气泡」：每张候选牌上方挂的收益牌子。
//
// 这是这次改动的核心显示，所以要验证四件事：
//   1. 牌子画出来了，位置/颜色/文字对
//   2. 最优的一手是金色、自己打出去的那张是红色「我打」
//   3. 同一份数据重复推**不重写 DOM**（之前一闪一闪就是这里没做好）
//   4. 窗口缩放后位置跟着变（比例坐标要重新乘）
const { app, BrowserWindow } = require('electron');
const path = require('path');

setTimeout(() => { console.log('⏰ 超时'); process.exit(3); }, 30000);

const BADGES = [
  // 手牌 4m 4m 6p 7p 2s 3s 0s 7s 8s 9s —— 假设第 2 张和第 5 张是候选
  { tile: '4m', box: [0.20, 0.80, 0.24, 0.95], score: 12, ukeire: 8, kinds: 4, loss: 0, danger: 0.2, warn: false, mode: 'model', best: false, mine: false },
  { tile: '7p', box: [0.30, 0.80, 0.34, 0.95], score: 84, ukeire: 24, kinds: 10, loss: 0, danger: 0.55, warn: true, mode: 'model', best: true, mine: false },
];

function state(extra) {
  return {
    hand: ['4m', '4m', '6p', '7p', '2s', '3s', '0s', '7s', '8s', '9s'],
    dora: ['6z'],
    discards: [['5z'], ['1p'], ['3m'], ['7z']],
    mySeat: 2,
    advice: { text: '2 向听　建议打 7p（模型 84%）', mode: 'model', mode_note: '神经网络 aimodel/policy.npz' },
    badges: BADGES,
    ...extra,
  };
}

const probe = () => `(() => {
  const layer = document.getElementById('__pm_tips');
  if (!layer) return { count: -1, innerW: innerWidth, innerH: innerHeight };
  const tips = [...layer.querySelectorAll('.pm-tip')];
  return {
    count: tips.length,
    innerW: innerWidth,
    innerH: innerHeight,
    html: layer.innerHTML.length,
    items: tips.map(t => ({
      text: t.textContent,
      cls: t.className,
      left: Math.round(parseFloat(t.style.left)),
      // 牌子中心 = style.left（CSS 用 translateX(-50%) 把它居中到那张牌上），
      // 所以这份数据里 left 就是「牌子中心对准的 x」。旧版本这里是牌的**左边缘**。
      centerX: Math.round(parseFloat(t.style.left)),
      top: Math.round(parseFloat(t.style.top)),
    })),
  };
})()`;

app.whenReady().then(async () => {
  const w = new BrowserWindow({
    width: 1200, height: 700, show: false,
    webPreferences: { preload: path.join(__dirname, '..', 'preload-game.js'), contextIsolation: false },
  });
  await w.loadURL('file://' + path.join(__dirname, 'fake-page.html'));
  await new Promise((r) => setTimeout(r, 800));

  // 1) 正常渲染
  w.webContents.send('gamestate', state());
  await new Promise((r) => setTimeout(r, 400));
  const a = await w.webContents.executeJavaScript(probe());
  console.log(`牌子数量 ${a.count}`);
  for (const it of a.items) console.log(`   ${it.cls.padEnd(16)} "${it.text}"  @(${it.left},${it.top})`);

  const gold = a.items.find((i) => i.cls.includes('best'));
  const blue = a.items.find((i) => !i.cls.includes('best') && !i.cls.includes('mine'));
  const checks = [
    ['两张牌子都画出来了', a.count === 2],
    ['最优是金色且写着「打 84%」', !!gold && gold.text === '打 84%'],
    ['其余是蓝色且写着「打 12%」', !!blue && blue.text === '打 12%'],
    ['危险度高的加了红圈', !!gold && gold.cls.includes('warn')],
    // 期望值按**页面自己的视口**算：窗口 frame 会占掉一点高度
    //
    // 位置对齐的是牌的**中心**（0.20~0.24 的中点 = 0.22），不是左边缘 ——
    // 以前只写了 Y 方向的 transform，牌子整体偏左，越靠右越明显。
    [`牌子中心对准牌中心（0.22×${a.innerW}=${Math.round(0.22 * a.innerW)}）`,
      Math.abs(a.items[0].centerX - Math.round(0.22 * a.innerW)) <= 2,
      `实际 ${a.items[0].centerX}`],
    [`y 贴着牌顶（0.80×${a.innerH}=${Math.round(0.8 * a.innerH)}）`,
      a.items[0].top === Math.round(0.8 * a.innerH)],
  ];

  // 2) 重复同数据 -> 不重写 DOM
  w.webContents.send('gamestate', state());
  await new Promise((r) => setTimeout(r, 300));
  const b = await w.webContents.executeJavaScript(probe());
  checks.push(['重复数据不重写 DOM（不会闪）', a.html === b.html]);

  // 3) 我打出去 -> 变红「我打」
  w.webContents.send('gamestate', state({
    badges: BADGES.map((x) => ({ ...x, mine: x.tile === '7p' })),
  }));
  await new Promise((r) => setTimeout(r, 300));
  const c = await w.webContents.executeJavaScript(probe());
  const mine = c.items.find((i) => i.cls.includes('mine'));
  checks.push(['打出去的那张标红「我打」', !!mine && mine.text === '我打']);

  // 4) 窗口缩放 -> 位置跟着变（**没有新数据推过来**也要重画）
  w.setContentSize(600, 350);
  await new Promise((r) => setTimeout(r, 700));
  const d = await w.webContents.executeJavaScript(probe());
  checks.push([`窗口缩小后中心跟着重算（0.22×${d.innerW}=${Math.round(0.22 * d.innerW)}）`,
    d.items.length === 2 && Math.abs(d.items[0].centerX - Math.round(0.22 * d.innerW)) <= 2,
    `实际 ${d.items[0].centerX}`]);

  // 5) 没有候选牌时不报错
  w.webContents.send('gamestate', state({ badges: [] }));
  await new Promise((r) => setTimeout(r, 300));
  const e = await w.webContents.executeJavaScript(probe());
  console.log(`   （清空后 count=${e.count}）`);
  checks.push(['没有候选牌时清空', e.count === 0]);

  let bad = 0;
  for (const [name, ok] of checks) {
    if (!ok) bad += 1;
    console.log(`${ok ? '✅' : '❌'} ${name}`);
  }
  console.log(bad === 0 ? '全部通过' : `${bad} 项不通过`);
  process.exit(bad === 0 ? 0 : 1);
});
