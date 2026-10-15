/**
 * 应用装配 + 路由。
 *
 * 服务端对 `/` 和 `/room/{id}` 发的是**同一个** `index.html`，所以路由只能
 * 靠 `location.pathname` 判断（这就是为什么 index.html 里不能有服务端模板
 * 变量）。`/room/XXXX` 进来会等 WebSocket 一连上就自动 `room.join` ——
 * 邀请链接点开就是入座/观战，不用再点一次。
 */

import "./styles.css";

import { Net, type ConnStatus } from "./net";
import type {
  ActionsView,
  HelloMsg,
  JoinedMsg,
  LobbyMsg,
  RejectMsg,
  RevealMsg,
  ErrorMsg,
  StateMsg,
} from "./protocol";
import { LobbyView, type BuildRequest } from "./ui/lobby";
import { ResultPanel } from "./ui/result";
import { TableView } from "./ui/table";

function toastHost(): HTMLDivElement {
  const n = document.createElement("div");
  n.className = "toast";
  document.body.appendChild(n);
  return n;
}

class App {
  private readonly conn: Net;
  private readonly lobby: LobbyView;
  private readonly table: TableView;
  private readonly result: ResultPanel;
  private readonly toastEl: HTMLDivElement;
  private readonly connEl: HTMLDivElement;
  private gameEl: HTMLDivElement | null = null;
  private screen: "lobby" | "game" | null = null;
  private state: StateMsg | null = null;
  private actionBar: HTMLDivElement | null = null;
  private timerEl: HTMLDivElement | null = null;
  private timerRaf = 0;
  private resizeRaf = 0;
  private timerBaseline = 0;
  private timerStartedAt = 0;
  private logEl: HTMLDivElement | null = null;
  private chatEl: HTMLDivElement | null = null;
  private chatInput: HTMLInputElement | null = null;
  private revealWrap: HTMLElement | null = null;
  private revealBox: HTMLInputElement | null = null;
  private toastTimer = 0;
  /** 结算面板：同一局只自动弹一次，关掉之后不再打扰 */
  private dismissedKey = "";
  private readonly ridFromPath: string | null;

  constructor(private readonly root: HTMLElement) {
    this.toastEl = toastHost();
    this.connEl = document.createElement("div");
    this.connEl.className = "connbar";
    this.connEl.hidden = true;
    document.body.appendChild(this.connEl);

    this.conn = new Net({
      onStatus: (s: ConnStatus, detail: string) => this.onStatus(s, detail),
      onHello: (m: HelloMsg) => {
        this.lobby.applyHello(m);
        this.showToast(
          `已连接 · 服务端 v${m.server.version} · 模型 ${m.server.aimodel}`,
          "ok",
        );
      },
      onLobby: (m: LobbyMsg) => {
        this.lobby.applyLobby(m);
        // 只有「还没进房间」的浏览态才需要大厅；牌桌上收到 lobby 不动画面
        if (!this.state) this.setScreen("lobby");
      },
      onState: (m: StateMsg) => this.onState(m),
      onJoined: (m: JoinedMsg) => {
        this.showToast(
          m.spectator ? `进入观战：房间 ${m.room}` : `入座成功：房间 ${m.room} · ${m.seat + 1} 号位`,
          "ok",
        );
        // ★ **入座之后要把画面切到牌桌**。
        //
        // 只弹个 toast 不算完：`setScreen("game")` 是由 `onState` 触发的，
        // 而服务端在 `_enter_room()` 里是先推 `joined` 再推第一帧 `state`。
        // 正常情况下这两条会到，画面也就切过去了；但只要第一帧因为任何原因
        // 没到（或晚到），玩家就会**卡在大厅里，只看到一句「入座成功」**——
        // 明明已经坐在牌桌上了，屏幕上却还是房间列表（实测踩过）。
        //
        // 所以这里显式切一下，并且把地址栏也改成 `/room/{id}`：
        // 刷新之后能直接回到这张桌子（服务端对 `/room/{id}` 发的是同一份
        // index.html，前端靠 `location.pathname` 判断该显示哪个画面）。
        this.setScreen("game");
        const want = `/room/${m.room}`;
        if (location.pathname !== want) {
          history.pushState({ room: m.room }, "", want);
        }
      },
      onReveal: (m: RevealMsg) => {
        // 服务端可能拒绝（桌上有真人），**要把原因显示出来**
        this.showToast(m.ok ? (m.on ? "已开启全亮牌" : "已关闭全亮牌") : m.message || "全亮牌被拒绝", m.ok ? "ok" : "warn");
      },
      onReject: (m: RejectMsg) => this.showToast(`动作被拒绝：${m.message}`, "bad"),
      onError: (m: ErrorMsg) => this.showToast(m.message, "bad"),
    });

    this.lobby = new LobbyView({
      onName: (n) => this.conn.setName(n),
      onJoin: (rid) => this.conn.joinRoom(rid),
      onWatch: (rid) => this.conn.watchRoom(rid),
      onNoRecord: (v) => this.handleNoRecord(v),
      onRefresh: () => this.conn.refreshLobby(),
      onCreate: (req: BuildRequest) =>
        this.conn.createRoom({
          name: req.name,
          players: req.players,
          rounds: req.rounds,
          aiDelay: req.aiDelay,
          seats: req.seats,
          autostart: true,
        }),
      onToast: (text, level) => this.showToast(text, level ?? "ok"),
    });

    this.table = new TableView({
      onDiscard: (tile) => this.conn.act("discard", undefined, tile),
      onAction: (kind, index) => this.conn.act(kind, index),
      onNext: () => this.conn.next(),
    });

    this.result = new ResultPanel({
      onNext: () => this.conn.next(),
      onClose: () => {
        this.result.hide();
        this.dismissedKey = this.state ? this.result.fingerprint(this.state) : "";
        if (this.state?.finished) this.leaveToLobby();
      },
    });

    const m = /^\/room\/([A-Za-z0-9]+)/.exec(location.pathname);
    this.ridFromPath = m ? m[1].toUpperCase() : null;

    this.bindKeys();
    // 视口变化要重排：牌的尺寸是 vw/vh 算出来的，窗口一改，左右两家的
    // translate 和牌河偏移全部失效。用 rAF 去抖，避免拖动窗口时每像素算一次。
    window.addEventListener("resize", () => {
      if (this.resizeRaf) cancelAnimationFrame(this.resizeRaf);
      this.resizeRaf = requestAnimationFrame(() => {
        this.resizeRaf = 0;
        this.table.fitLayout();
      });
    });
    this.setScreen("lobby");
    this.conn.connect();
    // 等 WebSocket 建好（onopen 里会带 name 一起 hello）；join 由
    // Net 在 onopen 时按 wantedRoom 发出，这里只负责「告诉它想去哪」。
    if (this.ridFromPath) {
      window.setTimeout(() => this.conn.joinRoom(this.ridFromPath as string), 400);
    }
  }

  // ---------------------------------------------------------------- 通用 --

  private showToast(text: string, level: "ok" | "warn" | "bad" = "ok"): void {
    this.toastEl.textContent = text;
    this.toastEl.className = `toast is-on${level === "warn" ? " is-warn" : level === "bad" ? " is-bad" : ""}`;
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => {
      this.toastEl.className = "toast";
    }, 3600);
  }

  private onStatus(status: ConnStatus, detail: string): void {
    if (status === "open") {
      this.connEl.hidden = true;
      return;
    }
    this.connEl.hidden = false;
    this.connEl.textContent = status === "closed" ? `连接断开，正在重连…（${detail}）` : "正在连接服务器…";
  }

  private setScreen(which: "lobby" | "game"): void {
    if (this.screen === which) return;
    this.screen = which;
    this.root.replaceChildren();
    if (which === "lobby") {
      this.gameEl = null;
      this.actionBar = null;
      this.timerEl = null;
      this.logEl = null;
      this.chatEl = null;
      this.chatInput = null;
      this.revealWrap = null;
      this.revealBox = null;
      cancelAnimationFrame(this.timerRaf);
      this.table.reset();
      this.result.hide();
      this.root.appendChild(this.lobby.root);
      this.conn.refreshLobby();
    } else {
      this.gameEl = this.buildGame();
      this.root.appendChild(this.gameEl);
    }
  }

  private leaveToLobby(): void {
    this.conn.leaveRoom();
    this.state = null;
    this.setScreen("lobby");
  }

  private handleNoRecord(v: boolean): void {
    // 观战者没有记录开关（服务端也会回一句「观战没有记录开关」）
    if (this.state?.viewer.spectator) {
      this.showToast("观战没有记录开关（只有入座的玩家才落盘）", "warn");
      return;
    }
    this.conn.setNoRecord(v);
    this.showToast(v ? "本局不记录你的数据" : "已恢复记录", "ok");
  }

  // ------------------------------------------------------------ 牌桌外壳 --

  private buildGame(): HTMLDivElement {
    const game = document.createElement("div");
    game.className = "game";

    const topbar = document.createElement("div");
    topbar.className = "topbar";
    topbar.innerHTML = `
      <span class="topbar__id" id="rid">----</span>
      <span class="topbar__name" id="rname"></span>
      <span class="pill" id="rphase"></span>
      <span class="topbar__spacer"></span>
      <label class="spectbar" id="revealwrap" hidden>
        <input type="checkbox" id="revealbox" /> 全亮牌（仅纯 AI 房）
      </label>
      <button class="btn btn--sm" id="btn-panel" title="显示 / 隐藏右侧信息栏（快捷键 Tab）">侧栏</button>
      <button class="btn btn--sm" id="btn-leave">离开房间</button>
    `;

    const playfield = document.createElement("div");
    playfield.className = "playfield";
    playfield.appendChild(this.table.root);

    const side = document.createElement("div");
    side.className = "side";
    side.innerHTML = `
      <div class="card">
        <h3>对局信息</h3>
        <div id="roommeta" class="small muted"></div>
      </div>
      <div class="card">
        <h3>牌局日志</h3>
        <div class="logbox" id="logbox"></div>
      </div>
      <div class="card chatbox">
        <h3>聊天</h3>
        <div class="chatlog" id="chatlog"></div>
        <div class="chatinput">
          <input id="chatinput" maxlength="200" placeholder="说点什么…" />
          <button class="btn btn--sm" id="chatsend">发送</button>
        </div>
      </div>
    `;
    playfield.appendChild(side);
    this.logEl = side.querySelector("#logbox");
    this.chatEl = side.querySelector("#chatlog");
    this.chatInput = side.querySelector("#chatinput");

    const bottombar = document.createElement("div");
    bottombar.className = "bottombar";
    const bar = document.createElement("div");
    bar.className = "actionbar";
    bottombar.appendChild(bar);
    this.actionBar = bar;

    game.append(topbar, playfield, bottombar);

    topbar.querySelector("#btn-leave")?.addEventListener("click", () => this.leaveToLobby());
    // 右侧信息栏收起 / 展开。收起时把 `--side-bar` 归零 —— 牌桌立刻把
    // 右边那一条收回来，下家的手牌和牌河就完整可见了。
    const toggleSide = (): void => {
      const hidden = side.classList.toggle("is-hidden");
      game.classList.toggle("is-side-hidden", hidden);
      const btn = topbar.querySelector("#btn-panel");
      if (btn) btn.textContent = hidden ? "侧栏 ▸" : "侧栏 ◂";
      this.table.scheduleRefit();
    };
    topbar.querySelector("#btn-panel")?.addEventListener("click", toggleSide);
    // Tab 也能切（键盘流用起来顺手；Tab 默认会把焦点挪走，所以拦掉）
    window.addEventListener("keydown", (ev) => {
      if (ev.key === "Tab" && this.screen === "game") {
        ev.preventDefault();
        toggleSide();
      }
    });
    this.revealWrap = topbar.querySelector("#revealwrap");
    this.revealBox = topbar.querySelector("#revealbox");
    this.revealBox?.addEventListener("change", () => {
      this.conn.setReveal(this.revealBox?.checked ?? false);
    });

    const sendChat = (): void => {
      const t = this.chatInput?.value.trim();
      if (!t) return;
      this.conn.chat(t);
      if (this.chatInput) this.chatInput.value = "";
    };
    topbar.querySelector("#chatsend")?.addEventListener("click", sendChat);
    this.chatInput?.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") sendChat();
    });

    // 结算面板挂在 body 上：它要盖住整页（包括侧栏），挂在 game 里会被
    // grid 布局裁掉。
    this.result.root.remove();
    document.body.appendChild(this.result.root);
    return game;
  }

  // ---------------------------------------------------------------- 状态 --

  private onState(state: StateMsg): void {
    this.state = state;
    // 倒计时基线：服务端给的 ms_left + 本机此刻。见 tickTimer() 的注释。
    this.timerBaseline = Math.max(0, state.actions?.ms_left ?? 0);
    this.timerStartedAt = performance.now();
    if (this.screen !== "game") this.setScreen("game");
    if (!this.gameEl) return;

    (this.gameEl.querySelector("#rid") as HTMLElement).textContent = state.room.id;
    (this.gameEl.querySelector("#rname") as HTMLElement).textContent = state.room.name;
    const phaseEl = this.gameEl.querySelector("#rphase") as HTMLElement;
    phaseEl.className = `pill pill--${state.phase}`;
    phaseEl.textContent =
      state.phase === "playing"
        ? `对局中 · 第 ${state.round?.hand_no ?? 1} 局`
        : state.phase === "hand_end"
          ? "局间休息"
          : state.phase === "finished"
            ? "对局结束"
            : "等待中";

    if (this.revealWrap) this.revealWrap.hidden = !state.viewer.spectator;
    if (this.revealBox) {
      this.revealBox.disabled = !state.spectator_ok;
      this.revealBox.checked = state.viewer.reveal;
      this.revealWrap?.setAttribute(
        "title",
        state.spectator_ok
          ? "纯 AI 房：可以看所有人的手牌"
          : "桌上有真人玩家，服务端不允许全亮牌",
      );
    }

    this.table.render(state);

    const meta = this.gameEl.querySelector("#roommeta") as HTMLElement;
    const mode = state.room.players === 4 ? "四麻" : "三麻";
    meta.textContent =
      `${mode} · ${state.room.rounds} 局 · AI 节拍 ${state.room.ai_delay.toFixed(1)}s · ` +
      `${state.viewer.seat >= 0 ? `${state.viewer.seat + 1} 号位` : state.viewer.reveal ? "观战（全亮）" : "观战中"}` +
      (state.spectator_count ? ` · 观众 ${state.spectator_count}` : "") +
      (state.room.record ? "" : " · 本房不记录");

    this.renderLog(state);
    this.renderChat(state);
    this.renderActions(state);
    this.renderResult(state);

    // 服务端在快照里带了 error（比如房间被回收）：显示出来，别静默
    if (state.error) this.showToast(state.error, "bad");
  }

  private renderLog(state: StateMsg): void {
    const box = this.logEl;
    if (!box) return;
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    box.replaceChildren();
    for (const line of state.log) {
      const d = document.createElement("div");
      d.textContent = line;
      box.appendChild(d);
    }
    if (atBottom) box.scrollTop = box.scrollHeight;
  }

  private renderChat(state: StateMsg): void {
    const box = this.chatEl;
    if (!box) return;
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
    box.replaceChildren();
    for (const line of state.chat) {
      const d = document.createElement("div");
      if (line.system) {
        d.className = "sys";
        d.textContent = `· ${line.text}`;
      } else {
        const who = document.createElement("span");
        who.className = "who";
        who.textContent = `${line.name}: `;
        d.append(who, document.createTextNode(line.text));
      }
      box.appendChild(d);
    }
    if (atBottom) box.scrollTop = box.scrollHeight;
  }

  /**
   * 底部操作区。
   *
   * **可选项只能来自服务端**：`actions.options` 里有什么就画什么，
   * 这里绝不自己造「碰 / 吃 / 立直」按钮。
   */
  private renderActions(state: StateMsg): void {
    const bar = this.actionBar;
    if (!bar) return;
    const a = state.actions;
    const mine =
      state.viewer.seat >= 0 && a != null && !a.waiting && a.seat === state.viewer.seat;
    const sig = JSON.stringify({
      k: a?.kind ?? "",
      w: a?.waiting ?? true,
      s: a?.seat ?? -1,
      o: a?.options.map((o) => `${o.label ?? ""}/${o.tile ?? ""}/${o.riichi_ok ? 1 : 0}`) ?? [],
      p: state.phase,
      v: state.viewer.seat,
      next: state.room.next_acks?.[state.viewer.seat] ?? false,
    });

    if (bar.dataset["sig"] !== sig) {
      bar.dataset["sig"] = sig;
      bar.replaceChildren();

      if (state.phase === "hand_end") {
        const acked = state.room.next_acks?.[state.viewer.seat] ?? false;
        const btn = document.createElement("button");
        btn.className = "btn btn--action btn--primary";
        btn.textContent = acked ? "已准备" : state.viewer.seat < 0 ? "等下一局…" : "下一局";
        btn.disabled = acked || state.viewer.seat < 0;
        btn.dataset["action"] = "next";
        btn.addEventListener("click", () => this.conn.next());
        bar.appendChild(btn);
        bar.appendChild(hint("点「下一局」继续；超时会自动开下一局"));
      } else if (state.phase === "finished") {
        const show = document.createElement("button");
        show.className = "btn btn--action btn--primary";
        show.textContent = "查看结算";
        show.dataset["action"] = "result";
        show.addEventListener("click", () => this.result.show(state, { final: true }));
        bar.appendChild(show);
        const back = document.createElement("button");
        back.className = "btn btn--action";
        back.textContent = "回到大厅";
        back.dataset["action"] = "leave";
        back.addEventListener("click", () => this.leaveToLobby());
        bar.appendChild(back);
      } else if (mine && a) {
        bar.appendChild(this.buildButtons(a));
      } else if (a && a.waiting && a.seat !== state.viewer.seat) {
        const who = state.seats.find((s) => s.seat === a.seat);
        bar.appendChild(hint(`等待 ${who?.name ?? ""} ${a.kind_cn}…`));
      } else if (state.viewer.seat < 0) {
        bar.appendChild(
          hint(state.spectator_ok ? "观战中（全亮牌）" : "观战中（有真人在座，看不到手牌）"),
        );
      } else {
        bar.appendChild(hint("等待其他玩家…"));
      }
    }

    if (!this.timerEl || !this.timerEl.isConnected) {
      this.timerEl = document.createElement("div");
      this.timerEl.className = "actionbar__timer";
      this.timerEl.innerHTML = "<i></i>";
      bar.appendChild(this.timerEl);
    }
    this.tickTimer();
  }

  private buildButtons(a: ActionsView): DocumentFragment {
    const frag = document.createDocumentFragment();
    const positive = new Set(["ron", "tsumo", "riichi", "kita", "kyuushu"]);

    for (const o of a.options) {
      const btn = document.createElement("button");
      btn.className = "btn btn--action";
      const isPass = o.label === "pass" || o.label === "no";
      if (positive.has(a.kind) ? o.label === "yes" : !isPass) btn.classList.add("btn--primary");
      btn.dataset["action"] = a.kind;
      btn.dataset["index"] = String(o.index);
      if (o.tile) btn.dataset["tile"] = o.tile;
      btn.textContent =
        a.kind === "discard"
          ? `打 ${o.tile ?? ""}`
          : isPass
            ? "过"
            : (o.cn ?? o.label ?? "?");
      btn.addEventListener("click", () => {
        if (a.kind === "discard" && o.tile) this.conn.act("discard", undefined, o.tile);
        else this.conn.act(a.kind, o.index);
      });
      frag.appendChild(btn);
    }

    if (a.kind === "discard") {
      frag.appendChild(hint("点手牌 / 数字键 1-9 选牌；蓝边的是可立直的牌"));
    } else if (a.tile_cn) {
      frag.appendChild(hint(`对象：${a.tile_cn}`));
    }
    if (a.timeout) {
      frag.appendChild(hint(`超时 ${a.timeout.toFixed(0)}s 后由教师托管`));
    }
    return frag;
  }

  /**
   * 倒计时条。
   *
   * `ms_left` 是**服务端算的**，本机不重新算 deadline —— 两边的钟不可能
   * 一样准。这里记住「快照到手的时刻」和「那一刻的 ms_left」，之后按本机
   * 流逝的时间往下减，只在本地显示上走动，真正的超时判定始终在服务端。
   */
  private tickTimer(): void {
    cancelAnimationFrame(this.timerRaf);
    const step = (): void => {
      const st = this.state;
      const box = this.timerEl;
      if (!st || !box || !box.isConnected) return;
      const a = st.actions;
      const total = (a?.timeout ?? st.room.human_timeout) * 1000;
      const left = a ? Math.max(0, this.timerBaseline - (performance.now() - this.timerStartedAt)) : 0;
      const fill = box.querySelector("i");
      if (a && total > 0 && left > 0) {
        if (fill instanceof HTMLElement) fill.style.width = `${Math.min(100, (left / total) * 100).toFixed(1)}%`;
        box.classList.toggle("is-urgent", left < 6000);
      } else {
        if (fill instanceof HTMLElement) fill.style.width = "0%";
        box.classList.remove("is-urgent");
      }
      this.timerRaf = requestAnimationFrame(step);
    };
    this.timerRaf = requestAnimationFrame(step);
  }

  private renderResult(state: StateMsg): void {
    if (!ResultPanel.hasContent(state)) {
      if (state.phase === "playing") {
        this.result.hide();
        this.dismissedKey = "";
      }
      return;
    }
    if (state.phase !== "hand_end" && state.phase !== "finished") return;
    const key = this.result.fingerprint(state);
    if (this.dismissedKey === key) return;
    if (this.result.visible && this.result.alreadyShown(state)) return;
    this.result.markShown(state);
    this.result.show(state, { final: state.phase === "finished" });
  }

  // ---------------------------------------------------------------- 键盘 --

  private bindKeys(): void {
    window.addEventListener("keydown", (ev) => {
      const t = ev.target;
      if (t instanceof HTMLElement && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) {
        return;
      }
      const st = this.state;
      if (!st || this.screen !== "game") return;

      if (ev.key === "Enter" && st.phase === "hand_end") {
        this.conn.next();
        return;
      }

      if (/^[1-9]$/.test(ev.key)) {
        // 数字键选牌：按 `.seat--bottom .edge__hand` 里第 n 张来算
        const hand = this.table.root.querySelector(".seat--bottom .edge__hand");
        if (!hand) return;
        const node = Array.from(hand.querySelectorAll<HTMLElement>(".tile"))[Number(ev.key) - 1];
        if (node && node.classList.contains("is-clickable")) {
          const tile = node.dataset["tile"];
          if (tile) this.conn.act("discard", undefined, tile);
        }
        return;
      }

      const a = st.actions;
      if (!a || a.waiting || st.viewer.seat < 0 || a.seat !== st.viewer.seat) return;
      const key = ev.key.toUpperCase();
      const allowed: Record<string, string[]> = {
        R: ["ron", "riichi"],
        T: ["tsumo"],
        P: ["call", "pon"],
        K: ["kan", "kita"],
        C: ["chi"],
        " ": ["call", "chi", "kan", "kita", "tsumo", "ron", "riichi", "kyuushu"],
      };
      const kinds = allowed[key];
      if (!kinds || !kinds.includes(a.kind)) return;
      const pick = a.options.find((o) => o.label !== "pass" && o.label !== "no" && o.label !== "yes") ??
        a.options.find((o) => o.label !== "pass" && o.label !== "no") ??
        a.options[0];
      if (!pick) return;
      ev.preventDefault();
      if (a.kind === "discard" && pick.tile) this.conn.act("discard", undefined, pick.tile);
      else this.conn.act(a.kind, pick.index);
    });
  }

  /** CDP 自检 / 排障用：`window.maj.state()` 和 `window.maj.debug()`。 */
  debug(): Record<string, unknown> {
    return {
      state: this.state,
      screen: this.screen,
      diag: this.table.lastDiag,
      status: this.conn.status,
      seq: this.conn.lastState?.seq ?? -1,
    };
  }

  /** 排障入口：自检脚本要能主动触发一次布局修正。 */
  tableApi(): TableView {
    return this.table;
  }
}

function hint(text: string): HTMLSpanElement {
  const n = document.createElement("span");
  n.className = "actionbar__hint";
  n.textContent = text;
  return n;
}

// ---------------------------------------------------------------------------

const host = document.getElementById("app");
if (host) {
  const app = new App(host);
  // 排障 / 截图自检用的钩子（只读，不写任何状态）。无头浏览器里没有别的
  // 办法把内部状态挖出来 —— `_dbg_layout*.py` 那套脚本就靠它。
  (window as unknown as { maj: unknown }).maj = {
    state: () => app.debug().state as StateMsg | null,
    debug: () => app.debug(),
    el: () => document.querySelector(".table"),
    table: () => app.tableApi(),
  };
}
