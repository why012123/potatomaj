const { app, BrowserWindow } = require('electron');
const path = require('path');
setTimeout(() => { console.log('⏰ 超时'); process.exit(3); }, 20000);
app.whenReady().then(async () => {
  const w = new BrowserWindow({ width: 1200, height: 700, show: false,
    webPreferences: { preload: path.join(__dirname, '..', 'preload-game.js'), contextIsolation: false } });
  await w.loadURL('file://' + path.join(__dirname, 'fake-page.html'));
  await new Promise((r) => setTimeout(r, 800));

  console.log('HUD 元素存在:', await w.webContents.executeJavaScript('!!document.getElementById("__pm_hud")'));

  w.webContents.send('gamestate', {
    hand: ['4m','4m','6p','7p','2s','3s','0s','7s','8s','9s'],
    dora: ['6z'],
    discards: [['5z','2z'],['1p','1s','2z'],['3m','4z'],['7z','9p','1m']],
    advice: { text: '1 向听，进张：5p 8p 1s 4s' },
    mySeat: 2,
  });
  await new Promise((r) => setTimeout(r, 500));
  const t = await w.webContents.executeJavaScript('document.getElementById("__pm_hud").innerText');
  console.log('HUD 内容:');
  console.log(t.split('\n').map((l) => '    ' + l).join('\n'));
  const ok = t.includes('4m 4m 6p') && t.includes('1 向听');
  console.log(ok ? '✅ 页内 HUD 正常' : '❌ HUD 没渲染出来');

  // 测重复发送同一份数据不会重写 DOM（防闪烁）
  const before = await w.webContents.executeJavaScript('document.getElementById("__pm_hud").innerHTML.length');
  w.webContents.send('gamestate', { hand: ['4m','4m','6p','7p','2s','3s','0s','7s','8s','9s'], dora: ['6z'],
    discards: [['5z','2z'],['1p','1s','2z'],['3m','4z'],['7z','9p','1m']],
    advice: { text: '1 向听，进张：5p 8p 1s 4s' }, mySeat: 2 });
  await new Promise((r) => setTimeout(r, 300));
  const after = await w.webContents.executeJavaScript('document.getElementById("__pm_hud").innerHTML.length');
  console.log(before === after ? '✅ 重复数据不重写 DOM（不会闪）' : `❌ DOM 被重写了 (${before} -> ${after})`);

  // 手牌空着的时候要能看出**卡在哪一步** —— 之前这里只有一个「—」，
  // 完全分不清是没收到数据、还是识图失败了
  w.webContents.send('gamestate', {
    hand: [], dora: [], discards: [[], [], [], []], mySeat: null,
    advice: null,
    diag: { actions: 12, lastAction: 'ActionDealTile', sockets: 1 },
  });
  await new Promise((r) => setTimeout(r, 400));
  const stuck = await w.webContents.executeJavaScript('document.getElementById("__pm_hud").innerText');
  console.log('空手牌时的 HUD:');
  console.log(stuck.split('\n').map((l) => '    ' + l).join('\n'));
  // 现在牌全走协议，「没手牌」只可能是没连上 / 还没进对局 —— 提示要能分清这两种
  // 有动作但没手牌 -> 应该说的是「已收到 12 条动作…」（说明协议通了、只是还没轮到发牌）
  const okDiag = stuck.includes('12 条动作') && stuck.includes('ActionDealTile');
  console.log(okDiag ? '✅ 没手牌时能看出卡在哪一步' : '❌ 诊断信息没显示出来');
  w.webContents.send('gamestate', {
    hand: [], dora: [], discards: [[], [], [], []], mySeat: null, advice: null,
    diag: { sockets: 0, actions: 0 },
  });
  await new Promise((r) => setTimeout(r, 300));
  const noWs = await w.webContents.executeJavaScript('document.getElementById("__pm_hud").innerText');
  console.log(noWs.includes('WebSocket') ? '✅ 没连上时会直说「还没连上 WebSocket」'
    : '❌ 没连上时的提示不清楚');

  // 认出手牌之后这行要消失（不能一直挂着吓唬人），同时要能看到本局截了几次图
  w.webContents.send('gamestate', {
    hand: ['1m', '2p', '3p'], dora: ['6z'], discards: [['5z'], [], [], []],
    advice: { text: '2 向听' }, diag: { actions: 20, vision: 4, caps: 3, capsFull: 1 },
  });
  await new Promise((r) => setTimeout(r, 400));
  const cleared = await w.webContents.executeJavaScript('document.getElementById("__pm_hud").innerText');
  console.log(!cleared.includes('还没') ? '✅ 有手牌之后提示消失' : '❌ 那行还挂着');
  // 副露：协议现在读得到**具体牌**，界面上要列出来
  w.webContents.send('gamestate', {
    hand: ['1m', '2p', '3p'], dora: ['6z'], discards: [['5z'], [], [], []],
    advice: { text: '2 向听' },
    melds: [{ type: 'pon', tiles: ['2s', '2s', '2s'], froms: [1, 1, 1] }],
    kita: [2, 0, 0],        // 拔北单独算，不混进副露（见 gamestate 的 meldCount）
    mySeat: 0,
  });
  await new Promise((r) => setTimeout(r, 300));
  const withMelds = await w.webContents.executeJavaScript('document.getElementById("__pm_hud").innerText');
  const meldOk = withMelds.includes('碰2s2s2s') && withMelds.includes('拔北×2');
  console.log(meldOk ? '✅ 副露显示了具体牌（碰2s2s2s + 拔北×2 分开列）' : '❌ 副露没显示');

  // 收益来源要写清楚用的是哪个模型 —— 三麻/四麻是两个不同的网络
  w.webContents.send('gamestate', {
    hand: ['1m', '2p', '3p'], dora: ['6z'], discards: [['5z'], [], [], []],
    advice: { text: '2 向听', mode_note: '神经网络 maj-flash4v1.1' },
  });
  await new Promise((r) => setTimeout(r, 120));
  const src = await w.webContents.executeJavaScript(
    "document.getElementById('__pm_hud').innerText");
  console.log(src.includes('maj-flash4v1.1') ? '✅ 显示用的是哪个模型'
    : '❌ 没写收益来源');
  if (!unsureOk) { console.log(unsure); process.exit(1); }

  // 确定时不该显示这行（免得天天挂着吓唬人）
  w.webContents.send('gamestate', {
    hand: ['1m', '2p', '3p'], dora: ['6z'], discards: [['5z'], [], [], []],
    advice: { text: '2 向听' }, drawnHint: null,
  });
  await new Promise((r) => setTimeout(r, 120));
  const clean = await w.webContents.executeJavaScript(
    "document.getElementById('__pm_hud').innerText");
  console.log(!clean.includes('拿不准') ? '✅ 确定时不显示这行' : '❌ 那行还挂着');

  // 立直必须显眼 —— 这是麻将里最该先看到的信息
  w.webContents.send('gamestate', {
    hand: ['1m', '2p', '3p'], dora: ['6z'], discards: [['5z'], [], [], []],
    advice: { text: '2 向听' }, mySeat: 0, riichi: [false, true, false, false],
  });
  await new Promise((r) => setTimeout(r, 120));
  const withR = await w.webContents.executeJavaScript(
    "document.getElementById('__pm_hud').innerText");
  const rOk = withR.includes('立直') && withR.includes('南');
  console.log(rOk ? '✅ 有人立直时会提醒（红底那行）' : '❌ 立直没提醒');
  if (!rOk) { console.log(withR); process.exit(1); }

  w.webContents.send('gamestate', {
    hand: ['1m', '2p', '3p'], dora: ['6z'], discards: [['5z'], [], [], []],
    advice: { text: '2 向听' }, mySeat: 0, riichi: [false, false, false, false],
  });
  await new Promise((r) => setTimeout(r, 120));
  const noR = await w.webContents.executeJavaScript(
    "document.getElementById('__pm_hud').innerText");
  console.log(!noR.includes('立直') ? '✅ 没人立直时不显示' : '❌ 那行还挂着');
  if (!capsOk) { console.log(cleared); process.exit(1); }

  // 信息框**绝对不能压住手牌**：它会被 capturePage 拍进截图，压住第一张牌就会让
  // 整行网格错位，于是一张牌都认不出来。这里检查它的底边确实落在牌顶上方。
  const H = await w.webContents.executeJavaScript('innerHeight');
  const probeBox = async () => w.webContents.executeJavaScript(`(() => {
    const el = document.getElementById('__pm_hud');
    const r = el.getBoundingClientRect();
    return { bottom: Math.round(r.bottom), top: Math.round(r.top), h: Math.round(r.height) };
  })()`);

  w.webContents.send('gamestate', {
    hand: ['1m', '2p', '3p'], dora: ['6z'], discards: [[], [], [], []],
    advice: { text: '2 向听' }, diag: {}, handTop: 0.86,
  });
  await new Promise((r) => setTimeout(r, 300));
  const box = await probeBox();
  const tileTop = Math.round(0.86 * H);
  const okPlace = box.bottom <= tileTop;
  console.log(okPlace
    ? `✅ 信息框在手牌上方（框底 ${box.bottom}px ≤ 牌顶 ${tileTop}px）`
    : `❌ 信息框压住手牌了（框底 ${box.bottom}px > 牌顶 ${tileTop}px）`);

  // 还没认出手牌时（没有 handTop）也得给一个保守位置，不能默认贴在底部
  w.webContents.send('gamestate', { hand: [], dora: [], discards: [[], [], [], []], advice: null, diag: {} });
  await new Promise((r) => setTimeout(r, 300));
  const box2 = await probeBox();
  const okFallback = box2.bottom <= Math.round(H * 0.78);
  console.log(okFallback
    ? `✅ 还没有手牌位置时用保守高度（框底 ${box2.bottom}px，屏高 ${H}px）`
    : `❌ 默认位置太靠下（框底 ${box2.bottom}px）`);

  process.exit(okDiag && okPlace && okFallback ? 0 : 1);
});
