// 注入到雀魂页面的 preload。
//
// 关键点：把 window.WebSocket 换成我们的包装类，在**页面自己的上下文**里
// 记录收发帧。必须在页面脚本运行之前完成，所以放在 preload 里做。
//
// 抓到的帧通过 electronAPI 送到主进程（IPC），不走网络 —— 因此没有
// 「HTTPS 页面往 127.0.0.1 发请求」那种混合内容问题。

const { ipcRenderer } = require('electron');
let pb = null;
try { pb = require('./lib/pb'); } catch (e) { /* 没有也能跑，只是不过滤心跳 */ }

// 心跳占了将近一半的帧量，而且没有任何分析价值 —— 在页面里就丢掉，
// 省掉这部分 IPC 和主进程的解析开销
function isNoise(bytes) {
  if (!pb || !bytes || bytes.length > 200) return false;
  try {
    const msg = pb.unwrap(Buffer.from(bytes));
    const name = msg.name || '';
    return /heartbeat|loginBeat/i.test(name);
  } catch (e) {
    return false;
  }
}

// 只抓雀魂自己的 WebSocket，别把无关的也抓进来
const HOST_HINT = /maj-?soul|majsoul/i;

const state = {
  sockets: 0,
  skipped: 0,      // 被过滤掉、没进 IPC 的帧数（给角标看）
  sent: 0,
  received: 0,
};

function summarize(data) {
  try {
    if (typeof data === 'string') return { kind: 'text', size: data.length, head: data.slice(0, 120) };
    if (data instanceof ArrayBuffer) return { kind: 'binary', size: data.byteLength };
    if (ArrayBuffer.isView(data)) return { kind: 'binary', size: data.byteLength };
    if (data instanceof Blob) return { kind: 'blob', size: data.size };
  } catch (e) {
    /* 忽略 */
  }
  return { kind: 'unknown', size: 0 };
}

function toBytes(data) {
  // 统一成 Uint8Array，方便主进程落盘
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  if (typeof data === 'string') return new TextEncoder().encode(data);
  return null;
}

function isBlob(data) {
  return typeof Blob !== 'undefined' && data instanceof Blob;
}

/**
 * 这一帧里有没有对局动作（`.lq.ActionPrototype`）。
 *
 * 只看前 512 字节里有没有这段 ASCII —— 消息名在封装里是**明文**，
 * 不用先解 protobuf。够快，也够准（这个字符串不会出现在别处）。
 *
 * 为什么值得这么筛：雀魂的帧里绝大多数是房间/大厅/状态快照这类**大帧**，
 * 而我们真正要的 `ActionPrototype` 很小、很少。以前是每一帧都
 * 「Blob 转 ArrayBuffer + 整数扫描 + 整帧 IPC 序列化」，白花的量很大 ——
 * 打牌时的卡顿主要就来自这里。
 */
const ACTION_MARK = '.lq.ActionPrototype';
const ACTION_MARK_BYTES = new TextEncoder().encode(ACTION_MARK);
function hasAction(bytes) {
  if (!bytes || !bytes.length) return false;
  const n = ACTION_MARK_BYTES.length;
  const end = Math.min(bytes.length, 512) - n;
  for (let i = 0; i <= end; i++) {
    let ok = true;
    for (let k = 0; k < n; k++) {
      if (bytes[i + k] !== ACTION_MARK_BYTES[k]) { ok = false; break; }
    }
    if (ok) return true;
  }
  return false;
}

let frameSeq = 0;

function emit(direction, url, info, bytes, seq) {
  ipcRenderer.send('ws-frame', {
    direction,
    url,
    info,
    bytes: bytes && bytes.length <= 4 * 1024 * 1024 ? bytes : null, // 单帧超过 4MB 就不送了
    seq,
    counters: { ...state },
  });
}

function report(direction, url, data) {
  const info = summarize(data);
  if (direction === 'recv') state.received += 1;
  else state.sent += 1;
  const seq = frameSeq++;

  const direct = toBytes(data);
  if (direct && isNoise(direct)) {
    // 心跳：只更新计数给角标看，不发给主进程
    updateBadge();
    return;
  }
  // 不是对局动作帧 -> 不进 IPC（见 hasAction 的注释）
  if (direct && !hasAction(direct)) {
    state.skipped = (state.skipped || 0) + 1;
    updateBadge();
    return;
  }

  if (isBlob(data)) {
    // 浏览器 WebSocket 默认 binaryType='blob'，服务器发来的帧基本都是 Blob。
    //
    // **先只看前 512 字节**：克隆 Node 里 Blob.arrayBuffer() 是整帧拷贝，
    // 而雀魂的帧里 2/3 是我们根本不要的（房间、大厅、状态快照，而且那些正是
    // 最大的帧）。以前是先整帧拷贝、再判断要不要 —— 白拷的部分占了大头。
    // 现在只拷 512 字节做判断，命中了才整帧转。
    const needFull = () => {
      if (info && info.size !== undefined && info.size <= 512) return Promise.resolve(true);
      return data.slice(0, 512).arrayBuffer().then((head) => hasAction(new Uint8Array(head)));
    };
    needFull()
      .then((keep) => {
        if (!keep) {
          state.skipped = (state.skipped || 0) + 1;
          updateBadge();
          return null;
        }
        return data.arrayBuffer();
      })
      .then((buf) => {
        if (!buf) return;
        const bytes = new Uint8Array(buf);
        // 心跳：转成字节之后**必须再查一次**。上面那次 isNoise 走的是
        // toBytes(data)，而 Blob 在 toBytes 里返回 null —— 也就是说
        // Blob 帧从来没被过滤过，每一帧心跳都要转成 ArrayBuffer 再走一次 IPC。
        if (isNoise(bytes)) { updateBadge(); return; }
        emit(direction, url, { ...info, kind: 'binary' }, bytes, seq);
      })
      .catch(() => { /* 拿不到字节就算了，不值得把牌局搞崩 */ });
  } else {
    emit(direction, url, info, toBytes(data), seq);
  }
  updateBadge();
}

function installHook() {
  const NativeWebSocket = window.WebSocket;
  if (!NativeWebSocket || NativeWebSocket.__potatomajWrapped) return;

  class PatchedWebSocket extends NativeWebSocket {
    constructor(url, protocols) {
      super(url, protocols);
      this.__pm_url = String(url);
      this.__pm_track = true;
      state.sockets += 1;
      ipcRenderer.send('ws-open', { url: this.__pm_url });
      updateBadge();

      this.addEventListener('message', (ev) => {
        if (!this.__pm_track) return;
        report('recv', this.__pm_url, ev.data);
      });
    }

    send(data) {
      if (this.__pm_track) report('send', this.__pm_url, data);
      return super.send(data);
    }
  }

  PatchedWebSocket.__potatomajWrapped = true;
  // 保留原生常量，免得某些库读 WebSocket.OPEN 之类失败
  for (const k of ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED']) {
    Object.defineProperty(PatchedWebSocket, k, { value: NativeWebSocket[k] });
  }
  window.WebSocket = PatchedWebSocket;
  console.log('[potatomaj] WebSocket 钩子已装上');
}

// ---------------------------------------------------------------------------
// 游戏内 HUD：直接把提示画在游戏页面上
//
// 为什么不用独立的置顶窗口：那个窗口压在 WebGL 上，每次更新内容系统都要
// 重新合成一块区域 —— 表现就是卡 + 闪。画在页面里就没有这个问题。
// ---------------------------------------------------------------------------
const HUD_CSS = `
#__pm_hud {
  /* bottom 由脚本按手牌位置现算（见 placeHud）—— 压住手牌会让识别整行错位 */
  position: fixed; left: 10px; bottom: 210px; z-index: 2147483646;
  background: rgba(22, 26, 36, .92); color: #dfe7f1;
  font: 12px/1.5 -apple-system, "PingFang SC", sans-serif;
  padding: 8px 11px; border-radius: 9px;
  border: 1px solid rgba(120, 200, 255, .30);
  min-width: 210px; max-width: 330px; pointer-events: none;
  white-space: pre-wrap; word-break: break-all;
}
#__pm_hud .t { color: #8fd3ff; font-weight: 600; margin-bottom: 3px; }
#__pm_hud .k { color: #93a1b3; }
#__pm_hud .v { color: #eaf2fb; }
#__pm_hud .adv { color: #ffd479; font-weight: 600; margin-top: 4px; }
#__pm_hud .sub { color: #8794a6; font-size: 11px; margin-top: 2px; }
/* 座位那一行可点：整个信息框是 pointer-events:none，只给这一行开 */
#__pm_hud .seat { pointer-events: auto; cursor: pointer; }
#__pm_hud .seat:hover { text-decoration: underline; }
#__pm_hud .seat small { color: #6d7889; }
/* 需要注意的一行（比如「还没收到牌局消息」）：橙色，跟普通信息区分开 */
#__pm_hud .unsure { color: #ffbe55; }
#__pm_hud .unsure small { color: #b08a4a; }
/* 立直提醒：红底白字，扫一眼就能看到 —— 这是麻将里最该先看到的信息 */
#__pm_hud .riichi {
  margin-top: 3px; padding: 2px 6px; border-radius: 5px;
  background: rgba(200, 45, 35, .92); color: #fff;
  font-weight: 700; letter-spacing: .3px;
}
/* 「还没认出手牌」那一行要显眼 —— 卡住的时候这就是唯一有用的信息 */
#__pm_hud .sub.warn { color: #ffb26b; }

/* ---- 牌面提示：每张候选牌正上方的「打这张的收益」 ---- */
#__pm_tips {
  position: fixed; left: 0; top: 0; right: 0; bottom: 0;
  z-index: 2147483645; pointer-events: none; overflow: hidden;
}
.pm-model, .model { cursor: pointer; }
.pm-model:hover, .model:hover { background: rgba(255,255,255,.08); border-radius: 4px; }

.pm-tip {
  position: absolute;
  /* 离牌顶留 14px。为什么留这么多：识别时只截「牌面本身」，而上边界就在牌顶上方
     约 3~6px。气泡自己加上投影的向下溢出最多约 11px（best 那种大投影），
     所以只有留到 14px 才能保证**连阴影都在取景框外** ——
     贴边画的话会贴到窗口边缘，看着像是被裁掉了。 */
  /* 牌子要**居中在它那张牌上**：X 用 translateX(-50%)，Y 抬到牌顶上方 14px。
     以前只写了 Y，left 又给的是牌的**左边缘** —— 结果牌子整体偏左，
     而且越靠右的牌偏得越多（因为牌和牌之间还有间距在累加）。 */
  transform: translateY(-100%) translateY(-14px) translateX(-50%);
  display: flex; align-items: center; gap: 3px;
  padding: 2px 8px 3px; border-radius: 7px 7px 4px 4px;
  font: 700 14px/1.2 -apple-system, "PingFang SC", sans-serif;
  color: #fff; letter-spacing: .3px;
  background: linear-gradient(#5b9bdd, #3d76bd);
  box-shadow: 0 2px 6px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.35);
  border: 1px solid rgba(255,255,255,.28);
  white-space: nowrap;
}
/* 气泡层整体再上移一点，双保险 */
#__pm_tips { margin-top: -2px; }
.pm-tip .n { font-size: 11px; font-weight: 600; opacity: .92; }
/* 最优的一手：金色，跟牌效工具一样一眼看出来 */
.pm-tip.best {
  background: linear-gradient(#ffd75e, #f0a92c);
  color: #5a3200;
  border-color: rgba(255,255,255,.55);
  box-shadow: 0 2px 9px rgba(255,180,40,.55), inset 0 1px 0 rgba(255,255,255,.6);
}
/* 我真正打出去的那张 */
.pm-tip.mine {
  background: linear-gradient(#f2705a, #cf3f28);
  border-color: rgba(255,255,255,.5);
  box-shadow: 0 2px 9px rgba(220,60,40,.6);
}
/* 危险牌：红圈警示（读不到立直，只能给个相对提示） */
.pm-tip.warn { box-shadow: 0 0 0 2px rgba(255,86,64,.85), 0 2px 6px rgba(0,0,0,.45); }
#__pm_badge {
  position: fixed; right: 8px; bottom: 8px; z-index: 2147483647;
  background: rgba(20,24,32,.82); color: #8fd3ff;
  font: 11px/1.5 -apple-system, sans-serif;
  padding: 4px 8px; border-radius: 6px; pointer-events: none;
  border: 1px solid rgba(120,200,255,.35);
}
`;

function ensureHud() {
  if (document.getElementById('__pm_hud')) return;
  const style = document.createElement('style');
  style.textContent = HUD_CSS;
  document.head.appendChild(style);
  const el = document.createElement('div');
  el.id = '__pm_hud';
  el.innerHTML = '<div class="t">🀄 土豆麻将</div><div class="k">等待牌局…</div>';
  (document.body || document.documentElement).appendChild(el);
}

/**
 * 把信息框挪到手牌**上方**。
 *
 * 为什么必须待在手牌行**上面**：
 * 信息框一旦压在手牌上，你自己就看不清那几张牌了 —— 而这张框是你读牌的主要依据。
 * 手牌位置不用问识图：雀魂是排序显示、整行居中，位置由窗口尺寸算出来
 *（见 main.js 的 rowBox）。handTop 就是算出来的行顶比例 y。
 * 所以信息框必须待在手牌行的上面。
 * 拿到之前先用一个保守的默认值，宁可高一点。
 */
function placeHud(el, g) {
  const H = window.innerHeight || 1;
  const top = typeof g.handTop === 'number' ? g.handTop : null;
  // 牌高约 0.13 屏，所以留够「牌高 + 边距」就不会压到
  const bottom = top === null ? 210 : Math.max(90, Math.round((1 - top) * H + 10));
  const px = bottom + 'px';
  if (el.style.bottom !== px) el.style.bottom = px;
}

let hudCache = '';
function updateHud(g) {
  ensureHud();
  const el = document.getElementById('__pm_hud');
  if (!el) return;
  placeHud(el, g);
  const names = ['东', '南', '西', '北'];
  const seat = (typeof g.mySeat === 'number' && g.mySeat >= 0) ? g.mySeat : null;
  // **显示自风，不是座位号**。
  //
  // 庄家每小局轮转，自风跟着变 —— 而座位号是固定的。以前这里拿座位号当风显示，
  // 于是只有第一小局是对的（那时庄家恰好坐 0 位），之后每局都显示错的风。
  // 用户的原话：「场上各家对应的风还是只有第一小局能自动判断」。
  const wind = g.myWind || null;
  const seatLabel = seat === null
    ? '未知（点这里设）'
    : (wind ? wind + '家' : names[seat] + '家');
  const seatNote = (wind && g.dealer !== null)
    ? `<small>（坐 ${names[seat]} 位，庄家是 ${names[g.dealer]} 位）</small>` : '';
  // 谁立直了 —— 协议解出来的确定信息（见 electron/test/riichi.js）
  const rj = Array.isArray(g.riichi) ? g.riichi : [];
  const rjSeats = rj.map((v, i) => (v ? names[i] : null)).filter(Boolean);
  const mineRj = (seat !== null && rj[seat]) ? true : false;
  const riichiLine = rjSeats.length
    ? '⚠ ' + rjSeats.join('・') + '家立直'
      + (mineRj ? '（包括我）' : '　—— 打牌先看现物和筋')
    : '';
  const seatHint = seat === null ? '' : `　${seatNote}<small>（点一下换）</small>`;

  const hand = (g.hand || []).join(' ') || '—';
  const dora = (g.dora || []).join(' ') || '—';
  const ponds = (g.discards || []).map((a, i) => names[i] + (a || []).length).join(' ');
  const adv = (g.advice && g.advice.text) || '—';
  // 收益数字是哪来的要说清楚：「模型 %」和「进张枚数」不是一个单位。
  // 鸣牌建议的 mode_note 说的是「哪个模型/向听」，叫「判断依据」更贴切。
  const mode = (g.advice && g.advice.mode_note) || '';
  const modeLabel = (g.advice && g.advice.mode === 'call') ? '判断依据' : '收益来源';

  // 我的副露（协议给的**具体牌**）：碰/吃/杠都列出来。
  // 拔北**不算副露**（它和杠一样是拿掉一张再补摸一张，手牌张数不变），
  // 所以单独列在后面 —— 混在一起会让人以为少了三张牌。
  const MELD_LABEL = { chi: '吃', pon: '碰', minkan: '明杠', ankan: '暗杠',
                       kakan: '加杠', kan: '杠' };
  const meldParts = (g.melds || []).map((m) =>
    `${MELD_LABEL[m.type] || m.type}${(m.tiles || []).join('')}`);
  const myKita = (g.kita || [])[g.mySeat] || 0;
  if (myKita) meldParts.push(`拔北×${myKita}`);
  const meldLine = meldParts.join('　');

  // 手牌还空着的时候，把「卡在哪一步」直接写在脸上。
  // 现在牌全来自协议，所以「没手牌」只可能是**还没开局**或者**没钩上 WebSocket**。
  const d = g.diag || {};
  let stuck = '';
  if (!(g.hand || []).length) {
    const bits = [];
    if (!d.sockets) bits.push('还没连上雀魂的 WebSocket（登录了吗？）');
    else if (!d.actions) bits.push('连上了，但还没收到牌局消息（进对局就有了）');
    else bits.push(`已收到 ${d.actions} 条动作${d.lastAction ? `，最近 ${d.lastAction}` : ''}`);
    stuck = `<div class="sub warn">${bits.join('　')}</div>`;
  }

  const html =
    `<div class="t">🀄 土豆麻将</div>` +
    `<div><span class="k">手牌 </span><span class="v">${hand}</span></div>` +
    `<div><span class="k">宝牌 </span><span class="v">${dora}</span></div>` +
    // 副露：协议能读到**具体牌**，所以直接列出来。
    // 以前读不到，界面只能显示「有 N 个副露」，等于让你自己去牌桌上数。
    (meldLine ? `<div><span class="k">副露 </span><span class="v">${meldLine}</span></div>` : '') +
    `<div><span class="k">牌河 </span><span class="v">${ponds}</span></div>` +
    // 牌桌类型由程序自己认（三麻没有 2m~8m）。显示出来是为了**认错时你能看到** ——
    // 认成三麻而实际是四麻的话，危险度会少算一家，界面上本来完全看不出来
    (g.sanma === false ? `<div class="sub">四麻（用四麻模型 maj-flash4v1.1）</div>` : '') +
    // 立直：麻将里最该一眼看到的信息。红底，跟别的行区分开
    (riichiLine ? `<div class="riichi">${riichiLine}</div>` : '') +
    // 座位：靠「只有自己摸的牌才带牌名」推出来，绝大多数情况是对的；
    // 万一推错了可以点这一行手动改
    `<div class="seat"><span class="k">我坐 </span>` +
    `<span class="v">${seatLabel}</span>${seatHint}</div>` +
    // 手牌跟踪失准（漏了动作 / 中途重连 / 录像断点）：这时候的建议**不可信**，
    // 必须明说，否则你会拿一个基于错牌的建议去打
    (g.handSuspect
      ? `<div class="unsure">⚠ 手牌跟踪失准（漏了动作或中途重连）—— 下面这段仅供参考，下一局会自动恢复</div>`
      : '') +
    `<div class="adv">${adv}</div>` +
    (mode ? `<div class="sub">${modeLabel}：${mode}</div>` : '') +
    stuck;
  if (html === hudCache) return;      // 内容没变就不碰 DOM
  hudCache = html;
  el.innerHTML = html;
}

// ---------------------------------------------------------------------------
// 牌面提示气泡
//
// 参考牌效工具那种画法：每张「打了不亏」的候选牌上方挂一个小牌子，写这张
// 打出去的收益；最优的一手用金色，自己真打出去的那张标红「我打」。
//
// 位置是**比例坐标**（0~1），这里乘窗口尺寸 —— 所以窗口缩放、分辨率不同，
// 都不用改代码（算位置那步见 main.js 的 rowBox）。
// ---------------------------------------------------------------------------
let tipCache = '';

function tipSignature(badges) {
  // 只把**会影响显示**的字段拼进签名：内容没变就一个 DOM 都不碰，不然会闪
  return badges.map((b) => [
    b.tile, b.score, b.ukeire, b.best ? 1 : 0, b.mine ? 1 : 0,
    b.danger === null || b.danger === undefined ? -1 : Math.round(b.danger * 10),
    b.box ? Math.round(b.box[0] * 1000) + ',' + Math.round(b.box[1] * 1000) : '',
  ].join('|')).join(';');
}

function updateTips(g) {
  const badges = g.badges || [];
  const sig = tipSignature(badges);
  if (sig === tipCache) return;
  tipCache = sig;

  let layer = document.getElementById('__pm_tips');
  if (!badges.length) {
    if (layer) layer.textContent = '';
    return;
  }
  if (!layer) {
    layer = document.createElement('div');
    layer.id = '__pm_tips';
    (document.body || document.documentElement).appendChild(layer);
  }

  const W = window.innerWidth;
  const H = window.innerHeight;
  const frag = document.createDocumentFragment();
  for (const b of badges) {
    if (!b.box) continue;
    const tip = document.createElement('div');
    let cls = 'pm-tip';
    if (b.mine) cls += ' mine';
    else if (b.best) cls += ' best';
    // 危险警示由后端判好（相对这一手牌的位置，不是绝对阈值）——
    // 绝对阈值在这里会让 13 张牌里 12 张都亮红圈，等于没信息
    if (!b.mine && b.warn) cls += ' warn';
    tip.className = cls;
    // 给牌的**中心**（配合 CSS 的 translateX(-50%) 让牌子居中在牌上）
    tip.style.left = Math.round((b.box[0] + b.box[2]) / 2 * W) + 'px';
    tip.style.top = Math.round(b.box[1] * H) + 'px';
    // 牌子宽度跟着牌的宽度走，别让它盖住旁边那张
    const tileW = Math.max(28, Math.round((b.box[2] - b.box[0]) * W));
    tip.style.minWidth = Math.max(24, tileW - 14) + 'px';
    tip.style.justifyContent = 'center';
    // 有模型就是模型给的百分比，没有就退回进张枚数（单位在信息面板里写明）
    if (b.mine) {
      tip.textContent = '我打';
    } else {
      const n = (b.score === null || b.score === undefined) ? `${b.ukeire}枚` : `${b.score}%`;
      const pre = document.createElement('span');
      pre.className = 'n';
      pre.textContent = '打 ';
      const val = document.createElement('span');
      val.textContent = n;
      tip.append(pre, val);
    }
    tip.title = `${b.tile}　进张 ${b.ukeire} 枚（${b.kinds} 种）`
      + (b.danger === null || b.danger === undefined ? '' : `　危险度 ${b.danger}`);
    frag.appendChild(tip);
  }
  layer.textContent = '';
  layer.appendChild(frag);
}

let lastState = null;
// 点信息框里的座位行 -> 换下一个座位。
// 信息框本身是 pointer-events:none（不能挡住游戏操作），所以只给这一行开指针事件。
function bindHudClicks() {
  const el = document.getElementById('__pm_hud');
  if (!el || el.__pmHudBound) return;
  el.__pmHudBound = true;
  el.addEventListener('click', (ev) => {
    const t = ev.target;
    if (!t || !t.closest) return;
    if (t.closest('.seat')) {
      const cur = (lastState && typeof lastState.mySeat === 'number') ? lastState.mySeat : -1;
      ipcRenderer.invoke('set-seat', (cur + 1) % 4);
      return;
    }
  });
}

ipcRenderer.on('gamestate', (_e, g) => {
  lastState = g;
  bindHudClicks();
  try { updateHud(g); } catch (err) { /* 忽略 */ }
  try { updateTips(g); } catch (err) { /* 忽略 */ }
});

// 窗口大小变了：比例坐标要重新乘一遍，所以要拿**上一次的状态**主动重画
// （只清签名是不够的 —— 没有新数据推过来就永远不会重画）
let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (!lastState) return;
    tipCache = '';
    try { updateTips(lastState); } catch (err) { /* 忽略 */ }
  }, 120);
});

// 页面上显示一个很小的状态角标，确认钩子活着
let lastBadge = 0;
let badgeText = '';
function updateBadge() {
  // 这个角标是浮在 Unity 画布**上面**的：写它的 textContent 会让那块区域重新合成，
  // 所以两道保险 —— ① 最快 300ms 一次 ② 文字真的变了才写（数字没动就一个 DOM 都不碰）
  const now = Date.now();
  if (now - lastBadge < 300) return;
  lastBadge = now;
  const text = `🀄 收 ${state.received} 发 ${state.sent}` +
    (state.skipped ? `（滤掉 ${state.skipped}）` : '');
  if (text === badgeText) return;
  badgeText = text;
  let el = document.getElementById('__potatomaj_badge');
  if (!el) {
    el = document.createElement('div');
    el.id = '__potatomaj_badge';
    el.style.cssText = [
      'position:fixed', 'right:8px', 'bottom:8px', 'z-index:2147483647',
      'background:rgba(20,24,32,.82)', 'color:#8fd3ff', 'font:11px/1.5 -apple-system,sans-serif',
      'padding:4px 8px', 'border-radius:6px', 'pointer-events:none',
      'border:1px solid rgba(120,200,255,.35)',
    ].join(';');
    (document.body || document.documentElement).appendChild(el);
  }
  el.textContent = text;
}

// 给页面一个小接口。
//
// 注意：页面脚本里**没有 require**（nodeIntegration 关着），
// 所以想用 Electron 的能力只能由 preload 转发 —— 之前有个「角标永远显示 0/0」
// 就是页面直接 require('electron') 拿不到东西导致的。
window.__potatomaj = {
  state: () => ipcRenderer.invoke('get-gamestate'),
  stats: () => ipcRenderer.invoke('get-stats'),
  setSeat: (s) => ipcRenderer.invoke('set-seat', s),
  // 调试：手动跑一次识别（'full' 整屏 / 'hand' 只截手牌那一条）
  // 识图相关的调试接口已经废弃（牌全走协议了），留着空壳免得旧脚本报错
  capture: () => ipcRenderer.invoke('debug-capture'),
  tips: () => [...document.querySelectorAll('#__pm_tips .pm-tip')]
    .map((t) => ({ text: t.textContent, cls: t.className, left: t.style.left, top: t.style.top })),
  // 调试：截一次「手牌那一条」，数顶部几行的高饱和像素 —— 用来验证
  // 我们自己的气泡确实落在取景框外（气泡被拍进去会污染牌面上沿、让识别认错）
  probeStrip: () => ipcRenderer.invoke('probe-strip'),
  // 调试：长时间挂机的压测（重放录好的帧，量内存涨了多少）。
  // 页面里没有 require，所以只能从这里转发出去。
  soak: (rounds) => ipcRenderer.invoke('soak', { rounds }),
};

if (document.readyState === 'loading') {
  installHook();
  document.addEventListener('DOMContentLoaded', () => { updateBadge(); ensureHud(); });
} else {
  installHook();
  updateBadge();
  ensureHud();
}
