/*
 * 土豆麻将 · 页面外壳（直播页 live.html / 对战页 play.html 共用）
 *
 * 管三件事，牌桌本身仍然全在 table.js 里：
 *
 *   1. **连接**：`/ws` + 自动重连 + `hello` 之前排队（跟正式前端一个套路）；
 *   2. **中央信息区**：四条边的立直棒槽 + 点数条由这里生成（人数无关，
 *      三麻用不到的那两条由 renderState 负责清空藏起来）；
 *   3. **右侧面板**：对局信息 / 聊天 / 牌局日志 —— 数据就是 state 里的
 *      `room` / `round` / `seats` / `chat` / `log`，一条都不用自己造。
 *
 * 页面只负责「连上之后干什么」：直播页去看现成的纯 AI 房，对战页自己开一桌
 * 并坐上 0 号位。
 */
const $ = (id) => document.getElementById(id);

function majPage(opts) {
  const hud = $("hud");
  const side = $("side");
  let ws = null, greeted = false, queue = [], lastSeq = -1, joined = null, last = null;

  /* ---- 中央信息区：四条边的槽 + 点数条 ----
   * 只有牌桌页有 #center；大厅没有，这里必须容忍（否则建中心那一步就抛，
   * 大厅整页 JS 直接不跑了）。 */
  const center = $("center");
  if (center) {
    for (const d of ["bottom", "top", "left", "right"]) {
      const slot = document.createElement("div");
      slot.className = "stick-slot slot-" + d + " is-empty";
      slot.id = "slot-" + d;
      const strip = document.createElement("div");
      strip.className = "mid-side mid-" + d;
      strip.id = "side-" + d;
      strip.innerHTML = "<span></span><b></b>";
      center.append(slot, strip);
    }
  }

  /* ---- 手机上的面板折叠 ----
   * 只有对战页/直播页有 #side；大厅没有，这里要能容忍（不然 say() 之外的
   * 任何一步都会把整页 JS 卡死）。 */
  const fold = document.createElement("button");
  fold.id = "fold";
  const narrow = window.matchMedia("(max-width: 900px), (orientation: portrait)");
  const syncFold = () => {
    if (side && narrow.matches) side.classList.add("is-folded");
    fold.textContent = side && side.classList.contains("is-folded") ? "面板 ▸" : "面板 ▾";
  };
  fold.addEventListener("click", () => {
    if (!side) return;
    side.classList.toggle("is-folded");
    syncFold();
  });
  document.body.appendChild(fold);
  syncFold();
  narrow.addEventListener?.("change", syncFold);

  function say(html, bad) {
    if (!hud) { console.log("[maj]", html); return; }
    hud.innerHTML = html;
    hud.classList.toggle("off", !!bad);
  }
  function 换行(s) {
    return String(s == null ? "" : s).replace(/[&<>]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  function send(msg) {
    const text = JSON.stringify(msg);
    if (ws && ws.readyState === 1 && (greeted || msg.t === "hello")) ws.send(text);
    else queue.push(text);
  }
  function connect() {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(scheme + "//" + location.host + "/ws");
    ws.onopen = () => {
      say("已连接，握手…");
      send({ t: "hello", name: opts.name || "玩家" });
    };
    ws.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      onMsg(m);
    };
    ws.onclose = () => {
      say("连接断了，3 秒后重连…", true);
      greeted = false;
      queue = [];
      lastSeq = -1;
      window.setTimeout(connect, 3000);
    };
  }

  function onMsg(m) {
    if (m.t === "hello") {
      greeted = true;
      const name = opts.name || m.name;
      if (name) send({ t: "hello", name });
      for (const text of queue.splice(0)) ws.send(text);
      opts.onHello && opts.onHello(m.lobby || { rooms: [], models: [] });
      return;
    }
    if (m.t === "lobby") { opts.onLobby && opts.onLobby(m); return; }
    if (m.t === "joined") {
      joined = m;
      /* ★ 座位凭证**这里就存**：大厅建完房是整页跳到 /room/{id} 的，
         连接会断一次；不存 token，新页面就只剩「按 uid 找回座位」一条路，
         一旦找不回（房间已经开打）就只能退成观战。 */
      saveToken(m.room, m.token);
      opts.onJoined && opts.onJoined(m);
      return;
    }
    if (m.t === "state") {
      if (typeof m.seq === "number" && m.seq <= lastSeq) return;   // 旧帧丢掉
      lastSeq = m.seq;
      last = m;
      paintPanels(m);
      showResult(m);          // 局间盖住牌桌的和了确认页（直播页/对战页共用）
      opts.onState && opts.onState(m);
      return;
    }
    if (m.t === "error" || m.t === "reject") {
      say("服务端：" + 换行(m.message || "?"), true);
      opts.onRefuse && opts.onRefuse(m);
    }
  }

  /* ---- 右侧面板：对局信息 / 聊天 / 日志 ---- */
  function paintPanels(st) {
    if (!side) return;
    const r = st.round;
    const info = [
      ["房间", (st.room && st.room.id) + "　" + (st.room ? st.room.name : "")],
      ["人数", st.seats.length + " 人" + (st.room ? "（" + st.room.mode + "）" : "")],
      ["阶段", { lobby: "等待中", playing: "对局中", hand_end: "局间", finished: "已结束" }[st.phase] || st.phase],
      ["第几局", r ? r.round_wind + r.hand_no + "局（第 " + r.turn + " 巡）" : "—"],
      ["剩余牌", r ? r.wall + " 枚" : "—"],
      ["本场", r ? (r.honba || 0) + " 本场" : "—"],
      ["供托", r ? r.sticks + " 根立直棒" : "—"],
      ["宝牌指示", (r && r.dora && r.dora.length) ? r.dora.join(" ") : "—"],
      ["观战", String(st.spectator_count || 0)],
    ];
    if (st.seats) {
      info.push(["── 点数 ──", ""]);
      for (const s of st.seats) {
        /* 每家一行：风 + 名字 + 点数；后面挂徽章（立直/一发/拔北/亲/降级/离线） */
        const badge = [
          s.is_dealer ? "亲" : "",
          s.riichi ? (s.double_riichi ? "两立直" : "立直") : "",
          s.ippatsu ? "一发" : "",
          s.kita ? ("拔北 " + s.kita) : "",
          s.ai_degraded ? "降级" : "",
          s.connected === false ? "离线" : "",
          s.no_record ? "不记录" : "",
        ].filter(Boolean).join(" · ");
        info.push([
          s.seat_wind + " " + seatLabel(s),
          s.score + (s.score_delta ? "（" + (s.score_delta > 0 ? "+" : "") + s.score_delta + "）"
                                   : "") + (badge ? "　" + badge : ""),
        ]);
      }
    }
    /* 轮到谁 + 还剩几秒（服务端给的 deadline / ms_left） */
    const a = st.actions;
    if (a && st.phase === "playing") {
      const who = (st.seats[a.seat] || {});
      info.push(["── 轮到谁 ──", ""]);
      info.push([who.name || (a.seat + " 号位"), (a.kind_cn || a.kind) +
        (a.ms_left ? "　剩 " + Math.ceil(a.ms_left / 1000) + "s" : "")]);
    }
    $("info").innerHTML = info.map(([k, v]) =>
      '<div class="info__row"><span>' + 换行(k) + "</span><b>" + 换行(v) + "</b></div>").join("");
    /* 「宝牌指示」那一行画成**真的牌面小图**（值里塞 HTML，所以前面先 escape 了
       整行再单独替换这一格；id 只认 `^[0-9][mpsz]$`，防注入） */
    const infoEl = $("info");
    if (r && (r.dora || []).length) {
      const row = [...infoEl.querySelectorAll(".info__row")]
        .find((el) => el.textContent.indexOf("宝牌指示") === 0);
      if (row) {
        const mini = (id, gold) => (/^[0-9][mpsz]$/.test(id)
          ? '<img class="minitile' + (gold ? " is-dora" : "") + '" src="/tiles/'
            + id + '.svg" alt="' + id + '" title="' + id + '">' : "");
        /* ★ 宝牌指示**五个格子并排**（日麻的死墙就是这个数）：
             开局翻第 1 张，之后**每杠一次翻开一张**，最多 5 张。
             服务端给的 `round.dora` 是**已翻开**的那几张，剩下的画牌背。
             这里只放指示牌 —— 宝牌本身要玩家自己推（放上去反而不好认牌）。 */
        const shown = (r.dora || []).filter((id) => /^[0-9][mpsz]$/.test(id));
        const back = (typeof BACK_SRC !== "undefined") ? BACK_SRC : "";
        let html = shown.map((id) => mini(id, false)).join("");
        for (let i = shown.length; i < 5; i++) {
          html += '<img class="minitile is-back" src="' + back + '" alt="" title="未翻开">';
        }
        row.lastElementChild.innerHTML = html;
      }
    }

    $("log").innerHTML = (st.log || []).slice(-60).map((line) =>
      '<div class="log__line">' + 换行(line) + "</div>").join("");
    const logBox = $("log");
    logBox.scrollTop = logBox.scrollHeight;

    /* ★ 中间那块深蓝方块：**点一下切换**「剩余牌数 ↔ 宝牌指示牌」。
       手机上右侧面板默认折着，宝牌指示牌在那儿根本看不到（用户报的）；
       点中间最顺手，桌面端也能用。 */
    const core = document.querySelector("#center .mid-core");
    if (core) {
      let doraBox = core.querySelector(".mid-dora");
      if (!doraBox) {
        doraBox = document.createElement("div");
        doraBox.className = "mid-dora";
        core.appendChild(doraBox);
        core.title = "点一下切换：剩余牌数 / 宝牌指示牌";
        core.addEventListener("click", () => {
          const c = core.closest("#center");
          if (c) c.classList.toggle("show-dora");
        });
      }
      const shown = ((r && r.dora) || []).filter((id) => /^[0-9][mpsz]$/.test(id));
      /* 没翻开的格子画**牌背**（跟别家手牌一样那张橙黄 back），
         不要自己造深蓝方块 —— 牌背颜色全桌统一。 */
      const back = (typeof BACK_SRC !== "undefined") ? BACK_SRC : "";
      doraBox.innerHTML = shown.map((id) =>
        '<img class="minitile" src="/tiles/' + id + '.svg" alt="' + id + '">').join("")
        + Array.from({ length: Math.max(0, 5 - shown.length) }, () =>
          '<img class="minitile is-back" src="' + back + '" alt="" title="未翻开">').join("");
    }

    $("chat").innerHTML = (st.chat || []).slice(-60).map((c) =>
      '<div class="chat__line' + (c.system ? " is-sys" : "") + '">' +
      (c.system ? "" : "<i>" + 换行(c.name) + "</i> ") + 换行(c.text) + "</div>").join("");
    const chatBox = $("chat");
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  /* ---- 聊天发送 ---- */
  const chatSend = $("chatsend");
  const chatInput = $("chatinput");
  if (chatSend && chatInput) {
    const push = () => {
      const text = chatInput.value.trim();
      if (text) send({ t: "chat", text });
      chatInput.value = "";
    };
    chatSend.addEventListener("click", push);
    chatInput.addEventListener("keydown", (e) => { if (e.key === "Enter") push(); });
  }

  /* ---- 和了确认页 ----
   *
   * 一局打完盖住牌桌，亮出赢家的牌面 / 役 / 番符 / 点数，点「确认」才开下一局。
   * 服务端 `_between()` 会等在线的真人**或者**（纯 AI 房）在线的观战者 ——
   * 所以直播页和对战页都要有这一页，索性放在共用外壳里，两边自动都有。
   */
  let over = $("over");
  if (!over) {
    over = document.createElement("div");
    over.id = "over";
    over.hidden = true;
    ($("stage") || document.body).appendChild(over);
  }

  function tileImg(id, cls) {
    return /^[0-9][mpsz]$/.test(id)
      ? '<img class="rtile' + (cls ? " " + cls : "") + '" src="/tiles/' + id
        + '.svg" alt="' + id + '">'
      : "";
  }

  function showResult(st) {
    /* ★ 外面套一层 try/catch（用户报的 bug：**只有第一局的结算是对的，
       后面每一局都跟第一局一模一样**）。最可能的成因是渲染中途抛异常 ——
       我的写法是「全部拼好再一次写进 innerHTML」，一抛就停在旧内容上，
       表现就是「不重绘、永远是第一把」。所以这里：
         1) 出错也**一定要写一次**（写一条显眼的错误），绝不留旧内容；
         2) 顺手把这一局的局号写进 data-hand，肉眼/脚本都能看出有没有刷新。 */
    try {
      showResultInner(st);
    } catch (err) {
      console.error("[maj] 结算页渲染失败", err);
      over.innerHTML = '<div class="over__panel"><div class="over__hd"><h2>结算页渲染失败</h2>'
        + "<em>" + 换行(String(err && err.message || err)) + "</em></div>"
        + '<div class="rrow">把这条消息发我，我照它修。</div></div>';
      over.hidden = false;
    }
  }

  function showResultInner(st) {
    const r = st.result;
    if (!r || st.phase !== "hand_end") { over.hidden = true; return; }
    /* ★ 「和了 / 流局」**不能只看 agari 数组**：服务端只要有一条明细没填上，
       明明是荣和也会被显示成流局（用户抓到的 bug）。这里以 `kind` 为准。 */
    const won = (r.agari || []).length > 0
      || r.kind === "tsumo" || r.kind === "ron";
    const meSeat = (st.viewer && st.viewer.seat >= 0) ? st.viewer.seat : -1;
    const rows = [];
    for (const ag of r.agari || []) {
      const who = typeof seatLabel === "function" ? seatLabel(st.seats[ag.seat]) : "";
      rows.push(
        '<div class="rrow"><span class="who"><b>' + 换行(who) + "</b> "
        + (ag.tsumo ? "自摸" : "荣和") + "</span>"
        + (ag.yakuman ? '<span class="rnum">役满</span>' : "")
        + '<span class="rnum">' + ag.total_han + " 番 " + ag.fu + " 符</span>"
        + '<span class="rnum">+' + ag.gain + "</span></div>");
      let used = false;
      const win = ag.win_tile || ag.tile || null;
      /* 和了牌**单独画一张**（用户要的）：手里要是有同一张（自摸）就先拿走，
         再在末尾单独画一张带红边的 —— 主行 13 张 + 单独 1 张 = 14 张，
         自摸/荣和两种情形都一样。不写字、不加分隔，靠红边认。 */
      const tiles = (ag.hand || []).map((id) => {
        if (!used && win && id === win) { used = true; return ""; }
        return tileImg(id);
      }).join("");
      const extra = win ? tileImg(win, "rtile--win") : "";
      const melds = (ag.melds || []).map((m) =>
        '<span class="rmeld">' + (m.tiles || []).map((id) => tileImg(id)).join("")
        + "</span>").join("");
      rows.push('<div class="rrow rrow--tiles"><span class="rhand">'
        + tiles + extra + melds + "</span></div>");
      rows.push('<div class="rrow"><span class="who"></span><span class="ryaku">'
        + ((ag.yaku || []).map((y) => 换行(y.name)
            + (y.han > 1 ? "(" + y.han + ")" : "")).join(" · ") || "—")
        + ((ag.dora || []).length
          ? "　<span class='hint'>宝牌 "
            + (ag.dora || []).map((y) => 换行(y.name) + "×" + y.han).join(" ") + "</span>"
          : "")
        + "</span></div>");
    }
    if (!won) {
      rows.push('<div class="rrow"><span class="who"><b>流局</b></span>'
        + '<span class="ryaku">'
        + ((r.tenpai || []).length
          ? "听牌：" + (r.tenpai || [])
            .map((s) => 换行(typeof seatLabel === "function" ? seatLabel(st.seats[s]) : s))
            .join("、")
          : "全员不聴")
        + "</span></div>");
    } else if (!(r.agari || []).length) {
      // 明细缺了但确实有人和了：至少把赢家写出来（别显示成流局）
      rows.push('<div class="rrow"><span class="who"><b>'
        + (r.winners || [r.winner]).filter((x) => x !== null && x !== undefined)
          .map((s) => 换行(seatLabel ? seatLabel(st.seats[s]) : s)).join("、")
        + "</b> " + 换行(r.label || "") + "</span>"
        + '<span class="ryaku">（这一手的役种明细没送到，看日志）</span></div>');
    }
    const deltas = st.seats.map((s, i) => {
      const d = (r.deltas || [])[i] || 0;
      return '<div style="color:' + (d < 0 ? "#ff9b86" : d > 0 ? "#ffd479" : "#8fb0a3") + '">'
        + s.seat_wind + " " + 换行(typeof seatLabel === "function" ? seatLabel(s) : s.name)
        + "　" + (d > 0 ? "+" : "") + d
        + "　→ " + ((r.scores || [])[i] ?? s.score) + "</div>";
    }).join("");
    /* 还等谁确认：服务端给了 `pending_acks`（座位或观战者都可能是它） */
    const pending = (st.room && st.room.pending_acks) || [];
    const spec = (st.room && st.room.spectator_next) || null;
    const mineAcked = meSeat >= 0
      ? !!(st.room && (st.room.next_acks || [])[meSeat])
      : !!(st.viewer && st.viewer.next_ack);
    over.innerHTML = '<div class="over__panel">'
      + '<div class="over__hd"><h2>' + (won ? "和了" : "流局") + "</h2>"
      + "<em>" + 换行(r.label || r.note || "") + "</em>"
      + "<em>第 " + ((st.round && st.round.hand_no) || 1) + " 局 · "
      + ((st.round && st.round.honba) || 0) + " 本场</em>"
      + (spec && spec.total > 0 && !(st.room.next_acks || []).some(Boolean)
        ? "<em>观战 " + spec.acked + "/" + spec.total + " 已确认</em>" : "")
      + "</div>"
      + rows.join("")
      + '<div class="over__deltas">' + deltas + "</div>"
      + '<div class="over__ft">'
      + (mineAcked
        ? '<span class="over__wait">已确认' + (pending.length
          ? "，等 " + pending.map(换行).join("、") + " 确认…"
          : "，等服务端开下一局…") + "</span>"
        : '<button class="btn" id="ovok">确认，下一局</button>'
          + '<span class="over__wait">'
          + (pending.length
            ? "还要 " + pending.map(换行).join("、") + " 确认"
            : "所有在场的人确认后才开下一局")
          + "</span>")
      + "</div></div>";
    const ok = $("ovok");
    if (ok) ok.addEventListener("click", () => send({ t: "next" }));
    /* 局号挂在元素上：刷新没刷新，肉眼和脚本都能立刻看出来（这次 bug 的判据） */
    over.dataset.hand = String((st.round && st.round.hand_no) || "?");
    over.dataset.players = String(st.seats.length);
    over.hidden = false;
  }
  const tokenKey = (rid) => "maj:token:" + String(rid || "").toUpperCase();
  const readToken = (rid) => {
    try { return window.localStorage.getItem(tokenKey(rid)) || ""; }
    catch { return ""; }
  };
  const saveToken = (rid, token) => {
    if (!rid || !token) return;
    try { window.localStorage.setItem(tokenKey(rid), token); } catch { /* 无痕模式 */ }
  };

  return {
    say, send, connect,
    get joined() { return joined; },
    get lastState() { return last; },
    readToken, saveToken,
    /* 常用几条 */
    join: (id, o) => send({
      t: "room.join", room: id,
      seat: o && o.seat !== undefined ? o.seat : null,
      token: (o && o.token) || "",
      spectator: !!(o && o.spectator),
      reveal: !!(o && o.reveal),
      autostart: !!(o && o.autostart),
    }),
    spectate: (id) => send({ t: "room.join", room: id, seat: null, token: "",
                             spectator: true, reveal: true, autostart: false }),
    seatAt: (id, seat) => send({ t: "room.join", room: id, seat, token: "",
                                 spectator: false, reveal: false, autostart: false }),
    create: (o) => send(Object.assign({ t: "room.create" }, o)),
    start: () => send({ t: "room.start" }),
    reveal: (on) => send({ t: "reveal", on }),
    act: (kind, index) => send(index === undefined
      ? { t: "action", kind } : { t: "action", kind, index }),
    next: () => send({ t: "next" }),
    leave: () => send({ t: "room.leave" }),
    paint: paintPanels,
    showResult,
  };
}
