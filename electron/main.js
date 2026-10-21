// 土豆麻将 · 主进程 —— **纯协议版**
//
// 数据全部来自牌局消息（见 lib/actions.js / lib/gamestate.js），
// 一句截图、一次识图都没有。
//
// 这条路是后来才走通的：那层「加密」其实就是 8 行位置相关的异或，
// 算法在 shinkuan/Akagi 的 src/bridge/majsoul/parser.rs 里公开着（见 lib/liqi.js）。
// 在此之前我花了很多轮去逆向它、还写了一大套识图兜底 —— 那些现在全删了。
const { app, BrowserWindow, Menu, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');

const { GameState, SEAT_NAMES } = require('./lib/gamestate');
const { sortHand } = require('./lib/rowgeom');
const { Bridge } = require('./lib/bridge');
const pb = require('./lib/pb');

const ROOT = path.resolve(__dirname, '..');
const DUMP_DIR = process.env.POTATOMAJ_DUMP_DIR || path.join(ROOT, 'build', 'ws-frames');
const GAME_URL = process.env.POTATOMAJ_URL || 'https://game.maj-soul.com/1/';
/**
 * 要不要把收到的帧落盘。
 *
 * **默认关**。以前默认开是为了继续解协议（研究密钥流用），现在协议全解开了，
 * 它只剩下「每帧一次磁盘写 + 一次 JSON 序列化」的开销 —— 打牌时纯属跟游戏抢时间。
 * 想存就 `--dump`。
 */
const DUMP = process.argv.includes('--dump') || process.env.POTATOMAJ_DUMP === '1';
const VERBOSE = process.env.POTATOMAJ_VERBOSE === '1' || process.argv.includes('--verbose');
const WIN_SIZE = (process.env.POTATOMAJ_WINDOW || '1600x950').split('x').map(Number);
const NO_BADGE = !process.argv.includes('--badge') || process.env.POTATOMAJ_NO_BADGE === '1';
/**
 * 牌面气泡（每张候选牌上方那个「打 N%」牌子）。
 *
 * **默认关**：实战里信息框那一行「建议打 X」就够了，牌子反而是干扰 ——
 * 而且它要跟着手牌行算位置、每次更新都重建一层 DOM。
 * 想要就说 `--tips`。
 */
const SHOW_TIPS = process.argv.includes('--tips') || process.env.POTATOMAJ_TIPS === '1';
const NEVER_DUMP = /Login|Auth|Oauth|Account|Register|Pay|Order|Bill|Mail|Friend|Chat|Lobby|Signup|Captcha|Password/i;
const MAX_DUMP_BYTES = 300 * 1024 * 1024;

// ---------------------------------------------------------------------------
// 手牌行的位置
//
// **不再靠识图量**。雀魂的手牌是排序显示、整行居中、每张等距，
// 所以位置只跟窗口尺寸有关 —— 用比例就够了。
//
// 这几个比例是之前在 1470x812 的真实窗口上量出来的（间距 71px、行顶 700px、
// 牌面高 108px）。换分辨率时游戏 UI 会等比缩放，所以按比例算仍然成立。
// 想微调就改 build/row.json（可选的覆盖文件）。
// ---------------------------------------------------------------------------
const ROW_DEFAULT = { pitch: 71 / 1470, cx: 0.5, y: 700 / 812, height: 108 / 812 };
//: 摸到的那张和前面隔开多少格（游戏里单独挂在右边）
const DRAWN_GAP = 0.3;
const MAX_TIPS = 5;

function loadRowConfig() {
  try {
    const j = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'row.json'), 'utf8'));
    return { ...ROW_DEFAULT, ...j };
  } catch (e) { return { ...ROW_DEFAULT }; }
}

// ---------------------------------------------------------------------------
// 状态
// ---------------------------------------------------------------------------
const game = new GameState();
const bridge = new Bridge();
let gameWindow = null;
let badgeWindow = null;
let advice = null;
let adviceBusy = false;
let adviceTimer = null;
/**
 * 鸣牌（吃/碰/杠/荣）建议。和 `advice` 分开存：
 *   * `advice`      = 轮到我打牌时的「打哪张」
 *   * `callAdvice`  = 别人打牌后、我能不能鸣的「要不要」
 * 两者不会同时有效（effective 13 vs 14），显示时按 `snap.call` 二选一。
 */
let callAdvice = null;
let callAdviceBusy = false;
let callAdviceTimer = null;
let callSig = '';                // 当前鸣牌窗口的签名，用来丢弃过期回应
let callWanted = null;
let mineDiscard = null;          // { tile, at } —— 刚打出去的那张，牌子上标「我打」
let mineFreezeTimer = null;
let mySeatOverride = null;       // 手动设的座位（点信息框那一行）
const rowCfg = loadRowConfig();

const stats = { sent: 0, received: 0, frames: 0, bytes: 0, kinds: {}, sockets: 0, actions: 0 };

function contentSize() {
  try {
    return gameWindow && !gameWindow.isDestroyed() ? gameWindow.getContentSize() : [WIN_SIZE[0], WIN_SIZE[1]];
  } catch (e) { return [WIN_SIZE[0], WIN_SIZE[1]]; }
}

/** 第 index 张牌（共 count 张）的框，比例坐标 */
function rowBox(index, count, drawnGap = DRAWN_GAP) {
  const [vw, vh] = contentSize();
  const pitch = rowCfg.pitch * vw;
  // 摸到的那张（index >= count）不在这套下标里，单独挂在右边
  const gap = index >= count ? drawnGap * pitch : 0;
  const left = rowCfg.cx * vw - (count * pitch) / 2;
  const x1 = left + index * pitch + gap;
  const y1 = rowCfg.y * vh;
  return [x1 / vw, y1 / vh, (x1 + pitch) / vw, (y1 + rowCfg.height * vh) / vh];
}

// ---------------------------------------------------------------------------
// 气泡
// ---------------------------------------------------------------------------

/** 危险警示取这一手牌里最危险的那几张（理由见 docs 里的说明：绝对值没有意义） */
function dangerWarnSet(options) {
  const scored = options.filter((o) => typeof o.danger === 'number')
    .sort((a, b) => (b.danger - a.danger) || String(a.tile).localeCompare(String(b.tile)));
  if (!scored.length) return new Set();
  const k = Math.max(1, Math.min(3, Math.round(scored.length * 0.25)));
  return new Set(scored.slice(0, k).map((o) => o.tile));
}

function buildBadges(snap) {
  if (!SHOW_TIPS) return [];          // 默认不画牌子，见 SHOW_TIPS 的注释
  if (snap.call) return [];           // 鸣牌窗口里不画「打哪张」的牌子（会误导）
  const opts = (advice && advice.options) || [];
  if (!opts.length || !snap.hand.length) return [];
  const usable = opts.filter((o) => (
    (advice && advice.mode) === 'model' ? (o.score || 0) > 0 : o.loss === 0
  ));
  const top = (usable.length ? usable : opts).slice(0, MAX_TIPS);
  const warnTiles = dangerWarnSet(top);

  // 手牌按雀魂顺序排 —— 位置就是「排序后的下标」
  const hand = sortHand(snap.hand);
  const myTurn = snap.effective >= 14;            // 刚摸完 -> 最后一张是摸到的
  const baseCount = myTurn ? Math.max(1, hand.length - 1) : hand.length;

  const firstAt = new Map();
  hand.forEach((t, i) => { if (t && !firstAt.has(t)) firstAt.set(t, i); });

  const out = [];
  for (const o of top) {
    const idx = firstAt.has(o.tile) ? firstAt.get(o.tile) : -1;
    if (idx < 0) continue;
    out.push({
      tile: o.tile,
      box: rowBox(idx, baseCount),
      score: o.score,
      ukeire: o.ukeire,
      kinds: o.kinds,
      loss: o.loss,
      danger: o.danger,
      warn: warnTiles.has(o.tile),
      mode: (advice && advice.mode) || 'ukeire',
      best: (advice && advice.best) === o.tile,
      mine: false,
    });
  }
  // 刚打出去的那张：标红「我打」，停在它原来的位置
  if (mineDiscard && mineDiscard.tile) {
    const idx = firstAt.has(mineDiscard.tile) ? firstAt.get(mineDiscard.tile) : -1;
    out.push({
      tile: mineDiscard.tile,
      box: rowBox(idx >= 0 ? idx : 0, baseCount),
      score: null, ukeire: 0, kinds: 0, loss: 0, danger: null,
      warn: false, mode: 'mine', best: false, mine: true,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// 推给界面
// ---------------------------------------------------------------------------
let stateWriteTimer = null;
function scheduleStateWrite(snap) {
  clearTimeout(stateWriteTimer);
  stateWriteTimer = setTimeout(() => {
    try {
      fs.mkdirSync(path.join(ROOT, 'build'), { recursive: true });
      fs.writeFileSync(path.join(ROOT, 'build', 'gamestate.json'), JSON.stringify(snap, null, 1));
    } catch (e) { /* 写不了就算了 */ }
  }, 500);
}

function pushGameState() {
  const snap = game.snapshot();
  const badges = buildBadges(snap);
  const payload = {
    ...snap,
    // 鸣牌窗口里显示「要不要吃/碰/杠」，其余时候显示「打哪张」。
    // call 是 gamestate 按协议里的 OptionalOperationList 算出来的，不是猜的。
    advice: (snap.call && callAdvice) ? callAdvice : advice,
    badges,
    diag: { ...stats, lastAction: game.lastAction, seatOverride: mySeatOverride },
    handTop: badges.length ? badges[0].box[1] : null,
    drawnHint: null,
  };
  scheduleStateWrite(snap);
  if (gameWindow && !gameWindow.isDestroyed()) {
    try { gameWindow.webContents.send('gamestate', payload); } catch (e) { /* 忽略 */ }
  }
  updateBadgeWindow();
}

/** 问 Python 要建议（牌效 / 神经网络 / 危险度） */
async function refreshAdvice(snap) {
  // 鸣牌窗口里不做「打哪张」分析：这时候根本还没轮到我打牌（13 张）。
  // 显示层会用 callAdvice，不能让它被旧的打牌建议盖掉。
  if (snap.call) return;
  if (adviceBusy || !snap.hand.length) return;
  adviceBusy = true;
  try {
    const r = await bridge.request('analyze', {
      hand: snap.hand,
      dora: snap.dora,
      discards: snap.discards,
      my_seat: mySeatOverride !== null ? mySeatOverride
        : (snap.mySeat === null || snap.mySeat === undefined ? 0 : snap.mySeat),
      sanma: snap.sanma !== false,
      three_player: snap.sanma !== false,
      // 副露的**具体牌**（协议给的）—— 有了它，带副露的手牌也能进网络
      meld_list: snap.melds || [],
      melds: (snap.melds || []).length,
      riichi: snap.riichi || [],
      scores: snap.scores || [],
      left_tile_count: snap.leftTileCount,
      round_wind: snap.chang,
      // 庄家（本局第一个打牌的人）—— 自风 = 离庄家差几家，见 gamestate 的 dealer
      dealer: snap.dealer,
      // 拔北数（三麻特有）：模型用它算「北还剩几张」
      kita: snap.kita || [0, 0, 0, 0],
      // 一发：协议里没有直接字段，由 gamestate 从立直/打牌推出来
      ippatsu: snap.ippatsu || [false, false, false, false],
      // 刚摸到的那张：模型靠它区分摸切和手切
      drawn: snap.lastDraw || null,
    });
    advice = r.ok
      ? {
          text: r.text, shanten: r.shanten,
          danger: r.danger && r.danger.recommend,
          options: r.options || [],
          mode: r.mode, mode_note: r.mode_note, best: r.best,
        }
      : { text: (r.error || '分析失败').slice(0, 80) };
    if (r.ok) {
      console.log(`[advice] ${r.text}   [${r.mode}] ${r.mode_note || ''}`);
    }
  } catch (e) {
    advice = { text: '分析异常：' + String(e && e.message).slice(0, 60) };
  } finally {
    adviceBusy = false;
    pushGameState();
  }
}

function scheduleAdvice(delay = 80) {
  clearTimeout(adviceTimer);
  adviceTimer = setTimeout(() => refreshAdvice(game.snapshot()), delay);
}

// ---------------------------------------------------------------------------
// 鸣牌建议（吃 / 碰 / 杠 / 荣）
//
// 数据来源是协议里的 `ActionDiscardTile.operation`（HUD 以前完全没用它）。
// 这里只**显示建议**，不自动点按钮 —— 自动点按钮是另一类风险（误点、抢焦点、
// 在对局里乱点），本次不引入。
// ---------------------------------------------------------------------------
function callSignature(c) {
  return [c.seat, c.from, c.tile,
    c.ops.map((o) => `${o.type}:${o.combination.join(',')}`).join(';')].join('|');
}

async function refreshCallAdvice(snap) {
  const c = snap.call;
  if (!c) {
    if (callAdvice !== null) { callAdvice = null; callSig = ''; callWanted = null; }
    return;
  }
  const sig = callSignature(c);
  if (sig === callSig) return;
  callSig = sig;
  callAdvice = { text: '🀄 正在判断要不要鸣牌…', mode: 'call' };
  pushGameState();
  if (callAdviceBusy) { callWanted = sig; return; }   // 上一手还没回来
  const mySig = sig;
  callAdviceBusy = true;
  try {
    const r = await bridge.request('analyze-call', {
      hand: snap.hand,
      dora: snap.dora,
      discards: snap.discards,
      my_seat: mySeatOverride !== null ? mySeatOverride
        : (snap.mySeat === null || snap.mySeat === undefined ? 0 : snap.mySeat),
      sanma: snap.sanma !== false,
      three_player: snap.sanma !== false,
      meld_list: snap.melds || [],
      melds: (snap.melds || []).length,
      // 别家的副露也要给：鸣牌判断最怕把对手当成门清
      melds_all: snap.meldsBySeat || null,
      riichi: snap.riichi || [],
      scores: snap.scores || [],
      left_tile_count: snap.leftTileCount,
      round_wind: snap.chang,
      dealer: snap.dealer,
      kita: snap.kita || [0, 0, 0, 0],
      ippatsu: snap.ippatsu || [false, false, false, false],
      drawn: null,                     // 鸣牌时没有「刚摸到的那张」
      ops: c.ops,
      call_tile: c.tile,
      call_from: c.from,
    });
    if (mySig !== callSig) return;      // 窗口已经换了，丢弃过期回应
    callAdvice = r.ok
      ? {
          text: r.text, mode: 'call', mode_note: r.mode_note,
          call: r.decision, options: r.options || [],
        }
      : { text: ('鸣牌分析失败：' + (r.error || '未知错误')).slice(0, 100), mode: 'call' };
    if (r.ok) console.log(`[call] ${r.text}   [${r.mode}] ${r.mode_note || ''}`);
  } catch (e) {
    if (mySig === callSig) {
      callAdvice = { text: '鸣牌分析异常：' + String(e && e.message).slice(0, 60), mode: 'call' };
    }
  } finally {
    callAdviceBusy = false;
    if (callWanted && callWanted !== mySig) {
      callWanted = null;
      scheduleCallAdvice(0);
    }
    pushGameState();
  }
}

function scheduleCallAdvice(delay = 30) {
  clearTimeout(callAdviceTimer);
  callAdviceTimer = setTimeout(() => {
    const snap = game.snapshot();
    if (snap.call) refreshCallAdvice(snap);
    else if (callAdvice !== null) {
      callAdvice = null; callSig = ''; callWanted = null;
      pushGameState();
    }
  }, delay);
}

// ---------------------------------------------------------------------------
// 抓帧
// ---------------------------------------------------------------------------
let frameStream = null;
let indexStream = null;
let dumpIndex = 0;
let dumpBytes = 0;

function ensureDumpDir() {
  if (!fs.existsSync(DUMP_DIR)) fs.mkdirSync(DUMP_DIR, { recursive: true });
}

function openStreams() {
  if (frameStream) return;
  ensureDumpDir();
  const binPath = path.join(DUMP_DIR, 'frames.bin');
  // frames.bin 是追加模式，偏移要从**文件实际大小**接着算，否则索引里的 off 全部错位
  let off = 0;
  try { off = fs.statSync(binPath).size; } catch (e) { off = 0; }
  frameStream = fs.createWriteStream(binPath, { flags: 'a' });
  indexStream = fs.createWriteStream(path.join(DUMP_DIR, 'index.jsonl'), { flags: 'a' });
  if (dumpBytes === 0) dumpBytes = off;
  frameOffset = off;
}
let frameOffset = 0;

function handleFrame(payload) {
  const { direction, bytes, info } = payload;
  if (!bytes || !bytes.length) return;
  stats.frames += 1;
  if (direction === 'recv') stats.received += 1; else stats.sent += 1;

  let msg = { kind: info && info.kind };
  try { msg = pb.unwrap(Buffer.from(bytes)); } catch (e) { /* 解不开就只记原始帧 */ }
  stats.kinds[msg.kind] = (stats.kinds[msg.kind] || 0) + 1;

  // ---- 协议：动作直接喂给状态机 ----
  if (msg.action && msg.actionPayload) {
    stats.actions += 1;
    const ev = game.onAction(msg.action, msg.actionPayload);
    if (ev && VERBOSE) console.log(`[game] ${msg.action}`, JSON.stringify(ev).slice(0, 120));
    if (ev) {
      if (ev.kind === 'newround') {
        console.log(`[game] 第 ${ev.handNo} 局发牌：手牌 ${ev.hand.length} 张　宝牌指示牌 ${ev.dora.join(' ') || '（空）'}`);
        mineDiscard = null;
      } else if (ev.kind === 'discard' && ev.mine) {
        // 我刚打出去：冻结 1.6 秒，把「我打」标在原来那张上
        mineDiscard = { tile: ev.tile, at: Date.now() };
        clearTimeout(mineFreezeTimer);
        mineFreezeTimer = setTimeout(() => { mineDiscard = null; pushGameState(); }, 1600);
      } else if (ev.kind === 'draw' && ev.mine) {
        console.log(`[game] 摸到 ${ev.tile}`);
      }
      // 鸣牌窗口：别人打牌后如果协议说「我可以吃/碰/杠/荣」，立刻去问 Python。
      // 窗口结束（有人摸牌/我过了/别人鸣了）就清掉，别把上一条建议留在界面上。
      const callNow = game.snapshot().call;
      if (callNow) {
        if (callSignature(callNow) !== callSig) {
          callAdvice = { text: '🀄 正在判断要不要鸣牌…', mode: 'call' };
        }
        scheduleCallAdvice(0);
      } else if (callAdvice !== null) {
        callAdvice = null; callSig = ''; callWanted = null;
      }
      pushGameState();
      // 轮到我打牌时才算收益
      if (game.effective >= 14 || ev.kind === 'newround') scheduleAdvice();
    }
  }

  // ---- 落盘（研究协议用）----
  const keepRaw = DUMP && msg.kind && !NEVER_DUMP.test(String(msg.kind))
    && dumpBytes < MAX_DUMP_BYTES;
  if (keepRaw) {
    openStreams();
    const buf = Buffer.from(bytes);
    const off = frameOffset;
    frameOffset += buf.length;
    dumpBytes += buf.length;
    frameStream.write(buf);
    indexStream.write(JSON.stringify({
      seq: dumpIndex, direction, size: buf.length, off, len: buf.length, msg: msg.kind,
    }) + '\n');
  }
  dumpIndex += 1;
  updateBadgeWindow();
}

ipcMain.on('ws-frame', (_e, payload) => handleFrame(payload));
ipcMain.on('ws-open', (_e, { url }) => {
  stats.sockets += 1;
  console.log('[ws] 连接:', String(url).slice(0, 80));
});

// ---------------------------------------------------------------------------
// 窗口
// ---------------------------------------------------------------------------
function createGameWindow() {
  gameWindow = new BrowserWindow({
    width: WIN_SIZE[0], height: WIN_SIZE[1],
    title: '土豆麻将 · 雀魂',
    backgroundColor: '#101418',
    webPreferences: {
      preload: path.join(__dirname, 'preload-game.js'),
      contextIsolation: false,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });
  gameWindow.loadURL(GAME_URL);
  gameWindow.webContents.on('did-finish-load', () => {
    gameWindow.webContents.send('gamestate', { ...game.snapshot(), badges: [], advice: null, diag: stats });
  });
  gameWindow.on('closed', () => { gameWindow = null; });
}

function createBadgeWindow() {
  badgeWindow = new BrowserWindow({
    width: 260, height: 320, show: true, frame: false, resizable: true,
    alwaysOnTop: true, skipTaskbar: true, title: '土豆麻将 · 状态',
    webPreferences: {
      preload: path.join(__dirname, 'preload-badge.js'),
      contextIsolation: false,
      nodeIntegration: false,
    },
  });
  badgeWindow.loadFile(path.join(__dirname, 'ui', 'badge.html'));
  badgeWindow.on('closed', () => { badgeWindow = null; });
}

function updateBadgeWindow() {
  if (!badgeWindow || badgeWindow.isDestroyed()) return;
  try { badgeWindow.webContents.send('stats', { ...stats, dump: DUMP }); } catch (e) { /* 忽略 */ }
}

//: 模型菜单的内容（启动后异步取一次，取不到就只留「打开配置」）
let modelMenu = { 3: null, 4: null, current: { 3: '?', 4: '?' }, config: '' };

async function loadModelMenu() {
  const r = await bridge.request('list-models', {}, 10000);
  if (r && r.ok) {
    modelMenu = {
      3: r.tables['3'].models, 4: r.tables['4'].models,
      current: { 3: r.tables['3'].current, 4: r.tables['4'].current },
      config: r.config,
    };
    buildMenu();
  }
}

async function pickModel(players, name) {
  const r = await bridge.request('set-model', { three_player: players === 3, model: name }, 10000);
  if (r && r.ok) {
    console.log(`[model] ${players} 麻改用 ${r.model}（${r.weights}）`);
    modelMenu.current[players] = r.model;
    advice = null;
    callAdvice = null; callSig = ''; callWanted = null;
    buildMenu();
    scheduleAdvice(0);          // 立刻用新模型重算一次
    scheduleCallAdvice(0);      // 如果正好在鸣牌窗口里，也按新模型重算
  } else {
    console.error('[model] 换不了：' + ((r && r.error) || '没有回应'));
  }
}

function modelSubmenu(players) {
  const list = modelMenu[players] || [];
  const items = list.map((name) => ({
    label: name + (name === modelMenu.current[players] ? '　✅' : ''),
    type: 'radio',
    checked: name === modelMenu.current[players],
    click: () => pickModel(players, name),
  }));
  if (!items.length) items.push({ label: '（没读到模型列表）', enabled: false });
  items.push({ type: 'separator' });
  items.push({
    label: '打开配置文件…',
    click: () => shell.openPath(modelMenu.config || path.join(ROOT, 'build', 'models.json')),
  });
  return items;
}

function buildMenu() {
  const tmpl = [
    { label: '土豆麻将', submenu: [
      { label: '打开抓包目录', click: () => shell.openPath(DUMP_DIR) },
      { type: 'separator' },
      { role: 'quit', label: '退出' },
    ] },
    { label: '模型', submenu: [
      { label: `三麻　（当前 ${modelMenu.current[3]}）`, submenu: modelSubmenu(3) },
      { label: `四麻　（当前 ${modelMenu.current[4]}）`, submenu: modelSubmenu(4) },
      { type: 'separator' },
      { label: '牌桌是自动认的：牌河出现 2m~8m 就是四麻',
        enabled: false },
    ] },
    { label: '视图', submenu: [
      { role: 'reload', label: '重新加载' },
      { role: 'toggleDevTools', label: '开发者工具' },
      { type: 'separator' },
      { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
    ] },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(tmpl));
}

// ---------------------------------------------------------------------------
// IPC：界面要用的
// ---------------------------------------------------------------------------
ipcMain.handle('get-gamestate', () => {
  const snap = game.snapshot();
  return {
    ...snap,
    advice: (snap.call && callAdvice) ? callAdvice : advice,
    badges: buildBadges(snap),
    diag: stats,
  };
});
ipcMain.handle('get-stats', () => ({ ...stats, dump: DUMP }));
ipcMain.handle('set-seat', (_e, seat) => {
  const n = Number(seat);
  if (n >= 0 && n <= 3) {
    mySeatOverride = n;
    // 只改「显示用」的座位 —— 手牌跟踪不受影响（见 gamestate.seatOverride 的注释）
    game.setMySeat(n);
    console.log(`[game] 手动设座位：${SEAT_NAMES[n]}家`);
    scheduleAdvice(0);
  }
  return mySeatOverride;
});
ipcMain.handle('open-dump-dir', () => shell.openPath(DUMP_DIR));

/** 换模型（菜单里调用；页面不再有这个入口） */
ipcMain.handle('cycle-model', async () => {
  const players = game.sanma === false ? 4 : 3;
  const list = modelMenu[players] || [];
  if (!list.length) { await loadModelMenu(); return { ok: false, error: '模型列表还没就绪' }; }
  const i = list.indexOf(modelMenu.current[players]);
  const next = list[(i + 1 + list.length) % list.length];
  await pickModel(players, next);
  return { ok: true, model: next };
});

/**
 * 长时间挂机的压测：把录好的帧重放很多遍，量内存涨了多少。
 *
 * 这个程序是设计成「挂着打一晚上」的，而内存泄漏只有跑久了才显形。
 * 真挂着等一晚不现实 —— 录下来的帧可以一两分钟重放几百局，
 * 走的是**完全相同**的代码路径（handleFrame 里那一整套）。
 */
async function runSoakForCli(roundsIn) {
  const rounds = Math.max(1, Math.min(500, roundsIn || 50));
  const dir = DUMP_DIR;
  let frames = [];
  try {
    const idx = fs.readFileSync(path.join(dir, 'index.jsonl'), 'utf8').trim().split('\n')
      .filter(Boolean).map((l) => { try { return JSON.parse(l); } catch (err) { return null; } })
      .filter(Boolean);
    const bin = fs.readFileSync(path.join(dir, 'frames.bin'));
    for (const e of idx) {
      if (e.off === undefined) continue;
      const b = bin.subarray(e.off, e.off + e.len);
      let m;
      try { m = pb.unwrap(b); } catch (err) { continue; }
      if (!m.action) continue;
      frames.push({ direction: e.direction || 'recv', info: { kind: m.kind, size: b.length },
                    bytes: new Uint8Array(b), seq: e.seq });
    }
  } catch (err) { return { ok: false, error: '读不到录制的帧：' + err.message }; }
  if (!frames.length) return { ok: false, error: 'build/ws-frames 里没有牌局帧' };

  const mem0 = process.memoryUsage();
  for (let i = 0; i < rounds; i++) {
    for (const f of frames) handleFrame(f);
    if (i % 10 === 9) await new Promise((r) => setTimeout(r, 0));
  }
  if (global.gc) global.gc();
  const mem1 = process.memoryUsage();
  const mb = (n) => Math.round(n / 1048576 * 10) / 10;
  return {
    ok: true, frames: frames.length, rounds,
    heapMB: [mb(mem0.heapUsed), mb(mem1.heapUsed)],
    rssMB: [mb(mem0.rss), mb(mem1.rss)],
    growthMB: mb(mem1.heapUsed - mem0.heapUsed),
    hands: game.snapshot().hand.length,
  };
}
ipcMain.handle('soak', (_e, opts) => runSoakForCli((opts && opts.rounds) || 50));

// 识图相关的调试接口已经用不上了，留着空壳免得旧界面报错
ipcMain.handle('debug-capture', () => ({ ok: false, error: '已经不用识图了（全部走协议）' }));
ipcMain.handle('probe-strip', () => null);

// ---------------------------------------------------------------------------
// 启动
// ---------------------------------------------------------------------------
async function selfCheck() {
  const r = await bridge.request('ping', {}, 8000);
  if (r && r.ok) { console.log('[selfcheck] Python 大脑正常'); return true; }
  const why = String((r && r.error) || '没有回应').replace(/^起不来：/, '').slice(0, 300);
  console.error('[selfcheck] Python 大脑起不来：' + why);
  console.error('[selfcheck] 多半是虚拟环境缺依赖。跑一下：');
  console.error('[selfcheck]   cd ' + ROOT);
  console.error('[selfcheck]   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt');
  if (process.env.POTATOMAJ_NO_DIALOG) return false;
  await dialog.showMessageBox({
    type: 'error', title: '土豆麻将：Python 环境有问题',
    message: '牌效和收益计算跑不起来，提示不会出现。',
    detail: '多半是 .venv 缺依赖（numpy / Pillow）。\\n\\n在项目目录里跑：\\n'
      + '  python3 -m venv .venv\\n  .venv/bin/pip install -r requirements.txt\\n\\n'
      + '具体错误：\\n' + why,
    buttons: ['知道了'], noLink: true,
  }).catch(() => null);
  return false;
}

app.whenReady().then(() => {
  buildMenu();
  createGameWindow();
  if (!NO_BADGE) createBadgeWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createGameWindow();
      createBadgeWindow();
    }
  });
  selfCheck().then((ok) => { if (ok) loadModelMenu(); });

  // 压测模式：在**真实应用**里重放录好的帧（见 build/soak-test.sh）
  const soakRounds = Number(process.env.POTATOMAJ_SOAK || 0);
  if (soakRounds > 0) {
    setTimeout(async () => {
      const r = await runSoakForCli(soakRounds);
      if (!r.ok) console.log('[soak] ❌ ' + r.error);
      else {
        const ratio = (r.heapMB[1] / Math.max(1, r.heapMB[0])).toFixed(2);
        console.log(`[soak] 重放 ${r.frames} 帧 × ${r.rounds} 局 = ${r.frames * r.rounds} 条消息`);
        console.log(`[soak] 堆内存 ${r.heapMB[0]}MB -> ${r.heapMB[1]}MB（涨 ${r.growthMB}MB，${ratio} 倍）`);
        console.log(`[soak] 常驻 ${r.rssMB[0]}MB -> ${r.rssMB[1]}MB　最后手牌 ${r.hands} 张`);
        console.log(r.growthMB < 40 && Number(ratio) < 1.6
          ? '[soak] ✅ 没有明显内存增长' : `[soak] ❌ 涨了 ${r.growthMB}MB —— 可能有泄漏`);
      }
      app.quit();
    }, 2500);
  }

  console.log(`[app] 纯协议版启动　窗口 ${WIN_SIZE[0]}x${WIN_SIZE[1]}　抓包 ${DUMP ? '开' : '关'}`);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
app.on('before-quit', () => { try { bridge.stop(); } catch (e) { /* 忽略 */ } });
