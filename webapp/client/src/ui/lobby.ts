/**
 * 大厅：昵称、房间列表、建房面板、模型下拉、隐私开关。
 *
 * 「模型下拉」的选项**全部来自服务端**（`lobby.models`，即
 * `agents.available_models()`），前端不硬编码任何模型名 —— 权重换了、
 * 加了新版本，页面不用改。
 *
 * 隐私那段是硬要求：必须写出「本局对局数据会被记录，用于模型训练」，
 * 并且给一个「本局不记录我的数据」的勾选框（勾上发 `privacy`）。
 */

import type { LobbyMsg, LobbyStats, ModelInfo, RoomSummary } from "../protocol";

/** 内置的两个非权重「模型」，服务端的 `room.create` 认这两个名字。 */
const BUILTIN = [
  { value: "teacher", label: "内置教师（无权重）" },
  { value: "random", label: "随机（基线）" },
];

export interface BuildRequest {
  name: string;
  players: 3 | 4;
  rounds: number;
  aiDelay: number | null;
  seats: Array<{ kind: "human" | "ai"; ai?: string }>;
}

export interface LobbyCallbacks {
  onName: (name: string) => void;
  onJoin: (room: string) => void;
  onWatch: (room: string) => void;
  onNoRecord: (noRecord: boolean) => void;
  onRefresh: () => void;
  onCreate: (req: BuildRequest) => void;
  onToast: (text: string, level?: "ok" | "warn" | "bad") => void;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function esc(s: string): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const PHASE_CN: Record<string, string> = {
  lobby: "等待中",
  playing: "对局中",
  hand_end: "局间休息",
  finished: "已结束",
};

export class LobbyView {
  readonly root: HTMLDivElement;
  private readonly listEl: HTMLDivElement;
  private readonly statsEl: HTMLSpanElement;
  private readonly nameInput: HTMLInputElement;
  private readonly buildEl: HTMLElement;
  private readonly privacyEl: HTMLElement;
  private models: ModelInfo[] = [];
  private recordOn = true;
  private noRecord = false;
  /** 建房面板当前的选择，重绘时保留 */
  private draft: BuildRequest = {
    name: "",
    players: 4,
    rounds: 0,
    aiDelay: null,
    seats: ["human", "ai", "ai", "ai"].map((k) => ({ kind: k as "human" | "ai" })),
  };
  private lastRooms: RoomSummary[] = [];
  private lastStats: LobbyStats | null = null;

  constructor(private readonly cb: LobbyCallbacks) {
    this.root = el("div", "lobby");
    this.root.innerHTML = `
      <header class="lobby__top">
        <div class="brand">
          <span class="brand__mark">東</span>
          <div>
            <h1>土豆麻将</h1>
            <p>网页版日本麻将对战平台 · 三麻 / 四麻 · AI 对战与观战</p>
          </div>
        </div>
        <div class="lobby__id">
          <label>昵称
            <input id="nick" maxlength="16" placeholder="取个名字" autocomplete="nickname" />
          </label>
          <span class="stats" id="stats"></span>
        </div>
      </header>

      <section class="lobby__actions">
        <button class="btn btn--primary" id="mk">建立房间</button>
        <button class="btn" id="join">用房间号加入</button>
        <button class="btn btn--ghost" id="reload">刷新列表</button>
      </section>

      <section class="lobby__build" id="build" hidden></section>

      <section class="lobby__rooms">
        <div class="lobby__roomshead">
          <h2>房间列表</h2>
          <span class="hint">点「入座」坐下，点「观战」看牌（桌上有真人时不会全亮）</span>
        </div>
        <div class="roomlist" id="rooms"></div>
      </section>

      <section class="lobby__rules" id="rules"></section>
      <section class="lobby__privacy" id="privacy"></section>
      <footer class="lobby__foot">
        土豆麻将 · 服务端与前端同源部署；本页不加载任何第三方脚本。
      </footer>
    `;
    this.listEl = this.root.querySelector("#rooms") as HTMLDivElement;
    this.statsEl = this.root.querySelector("#stats") as HTMLSpanElement;
    this.nameInput = this.root.querySelector("#nick") as HTMLInputElement;
    this.buildEl = this.root.querySelector("#build") as HTMLElement;
    this.privacyEl = this.root.querySelector("#privacy") as HTMLElement;

    this.nameInput.addEventListener("change", () => this.commitName());
    this.nameInput.addEventListener("blur", () => this.commitName());

    (this.root.querySelector("#mk") as HTMLButtonElement).addEventListener("click", () => {
      this.toggleBuild(this.buildEl.hidden);
    });
    (this.root.querySelector("#join") as HTMLButtonElement).addEventListener("click", () => {
      const rid = window.prompt("输入房间号（例如 7K3QP）");
      if (rid && rid.trim()) this.cb.onJoin(rid.trim().toUpperCase());
    });
    (this.root.querySelector("#reload") as HTMLButtonElement).addEventListener("click", () => {
      this.cb.onRefresh();
    });

    this.renderRules();
    this.renderPrivacy();
  }

  // ---------------------------------------------------------------- 昵称 --

  private commitName(): void {
    const n = this.nameInput.value.trim().slice(0, 16);
    this.nameInput.value = n;
    if (n) this.cb.onName(n);
  }

  set name(v: string) {
    this.nameInput.value = v;
  }

  get name(): string {
    return this.nameInput.value.trim();
  }

  // ------------------------------------------------------------ 服务端数据 --

  applyHello(msg: { name: string; config: { record?: boolean }; lobby: LobbyMsg }): void {
    if (!this.nameInput.value && msg.name) this.nameInput.value = msg.name;
    this.models = msg.lobby.models ?? [];
    this.recordOn = msg.config.record !== false;
    this.noRecord = false;
    this.renderPrivacy();
    this.applyLobby(msg.lobby);
  }

  applyLobby(msg: LobbyMsg): void {
    if (msg.models) this.models = msg.models;
    if (msg.record !== undefined) {
      const changed = this.recordOn !== !!msg.record;
      this.recordOn = !!msg.record;
      if (changed) this.renderPrivacy();
    }
    this.lastRooms = msg.rooms ?? [];
    this.lastStats = msg.stats ?? null;
    this.renderRooms(this.lastRooms, this.lastStats);
    if (!this.buildEl.hidden) this.renderBuild();
  }

  // ------------------------------------------------------------ 房间列表 --

  private renderRooms(rooms: RoomSummary[], stats: LobbyStats | null): void {
    this.statsEl.textContent = stats
      ? `${stats.rooms} 桌 · ${stats.humans} 人 · ${stats.ais} AI · ${stats.conns} 连接`
      : "";
    if (!rooms.length) {
      this.listEl.innerHTML =
        '<div class="empty">还没有房间。点「建立房间」开一桌，或者直接建一桌纯 AI 观战。</div>';
      return;
    }
    this.listEl.replaceChildren();
    for (const r of rooms) this.listEl.appendChild(this.roomCard(r));
  }

  private roomCard(r: RoomSummary): HTMLElement {
    const card = el("article", "roomcard");
    card.dataset["phase"] = r.phase;
    const mode = r.humans === 0 ? "纯 AI" : r.ais ? "人机混合" : "纯真人";
    const seats = [...r.seats].sort((a, b) => a.seat - b.seat);
    const seatType = r.players === 4 ? "四麻" : "三麻";
    card.innerHTML = `
      <div class="roomcard__head">
        <span class="roomcard__id">${esc(r.id)}</span>
        <span class="roomcard__name">${esc(r.name)}</span>
        <span class="pill pill--${esc(r.phase)}">${esc(PHASE_CN[r.phase] ?? r.phase)}${
          r.phase === "playing" && r.hand_no ? ` · 第 ${r.hand_no} 局` : ""
        }</span>
      </div>
      <div class="roomcard__meta">
        <span>${seatType}</span>
        <span>${mode}</span>
        <span>${r.humans} 真人 / ${r.ais} AI</span>
        <span>AI 节拍 ${r.ai_delay.toFixed(1)}s</span>
        <span>观众 ${r.spectators}</span>
        ${r.spectator_ok ? '<span class="pill pill--ok">可全亮观战</span>' : ""}
      </div>
      <div class="roomcard__seats">
        ${seats
          .map(
            (s) => `
          <div class="seatpill ${s.kind === "ai" ? "is-ai" : ""} ${s.online ? "" : "is-off"}">
            <b>${s.seat + 1}</b>
            <span>${esc(s.name)}</span>
            ${s.host ? '<i class="host">房主</i>' : ""}
            ${s.ai_degraded ? '<i class="warn">降级</i>' : ""}
          </div>`,
          )
          .join("")}
      </div>
      <div class="roomcard__foot">
        <button class="btn btn--sm btn--primary" data-act="seat">入座</button>
        <button class="btn btn--sm" data-act="watch">观战</button>
        <button class="btn btn--sm btn--ghost" data-act="link">复制邀请链接</button>
      </div>
    `;
    card.querySelectorAll<HTMLButtonElement>("[data-act]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const act = btn.dataset["act"] ?? "";
        if (act === "seat") this.cb.onJoin(r.id);
        else if (act === "watch") this.cb.onWatch(r.id);
        else {
          const link = `${location.origin}/room/${r.id}`;
          void navigator.clipboard?.writeText(link).catch(() => undefined);
          this.cb.onToast(`邀请链接：${link}`, "ok");
        }
      });
    });
    return card;
  }

  // ------------------------------------------------------------ 建房面板 --

  private toggleBuild(show: boolean): void {
    this.buildEl.hidden = !show;
    if (show) this.renderBuild();
  }

  private renderBuild(): void {
    const d = this.draft;
    const seatRows = d.seats
      .slice(0, d.players)
      .map((spec, i) => {
        const models = this.models.filter((m) => m.players === d.players);
        const cur = spec.ai ?? "";
        return `
          <div class="build__seat" data-seat="${i}">
            <span class="build__seatno">${i + 1} 号位</span>
            <select data-role="seatkind">
              <option value="human"${spec.kind === "human" ? " selected" : ""}>真人</option>
              <option value="ai"${spec.kind === "ai" ? " selected" : ""}>AI</option>
            </select>
            <select data-role="model"${spec.kind === "human" ? " disabled" : ""}>
              ${BUILTIN.map(
                (b) => `<option value="${b.value}"${cur === b.value ? " selected" : ""}>${b.label}</option>`,
              ).join("")}
              ${models
                .map(
                  (m) =>
                    `<option value="${esc(m.name)}"${cur === m.name ? " selected" : ""}>${esc(m.cn)} · ${esc(m.name)}${m.usable ? "" : "（缺权重）"}</option>`,
                )
                .join("")}
            </select>
          </div>`;
      })
      .join("");

    const preset = (label: string, kinds: Array<"human" | "ai">, players: 3 | 4): string =>
      `<button class="btn btn--sm" data-preset="${kinds.join(",")}|${players}">${label}</button>`;

    this.buildEl.innerHTML = `
      <div class="build__grid">
        <label>房间名<input id="bname" maxlength="24" placeholder="随便写" value="${esc(d.name)}" /></label>
        <label>人数
          <select id="bplayers">
            <option value="4"${d.players === 4 ? " selected" : ""}>四麻（136 张 · 有吃）</option>
            <option value="3"${d.players === 3 ? " selected" : ""}>三麻（108 张 · 有拔北）</option>
          </select>
        </label>
        <label>局数
          <select id="brounds">
            <option value="0"${d.rounds === 0 ? " selected" : ""}>东风战（按人数：三麻 3 / 四麻 4 局）</option>
            <option value="1"${d.rounds === 1 ? " selected" : ""}>只打 1 局</option>
            <option value="2"${d.rounds === 2 ? " selected" : ""}>打 2 局</option>
            <option value="4"${d.rounds === 4 ? " selected" : ""}>打 4 局</option>
          </select>
        </label>
        <label>AI 节拍
          <select id="bdelay">
            <option value=""${d.aiDelay === null ? " selected" : ""}>跟随服务端</option>
            <option value="0">全速（0s）</option>
            <option value="0.1">很快（0.1s）</option>
            <option value="0.5">较快（0.5s）</option>
            <option value="1.2">正常（1.2s）</option>
            <option value="2">慢（2s）</option>
          </select>
        </label>
      </div>
      <div class="build__seats">${seatRows}</div>
      <div class="build__presets">
        <span>快速预设：</span>
        ${preset("我 + 3 AI", ["human", "ai", "ai", "ai"], 4)}
        ${preset("纯 AI 观战（可全亮）", ["ai", "ai", "ai", "ai"], 4)}
        ${preset("三麻：我 + 2 AI", ["human", "ai", "ai"], 3)}
        ${preset("纯 AI 三麻观战", ["ai", "ai", "ai"], 3)}
      </div>
      <div class="build__foot">
        <button class="btn btn--primary" id="bgo">开桌并开始</button>
        <button class="btn btn--ghost" id="bcancel">取消</button>
        <span class="hint">建好后把房间号或邀请链接发给别人，他们可以入座或观战。</span>
      </div>
    `;

    const firstModel = (): string => {
      const usable = this.models.find((m) => m.players === d.players && m.usable);
      return usable ? usable.name : "teacher";
    };

    this.buildEl.querySelectorAll<HTMLElement>(".build__seat").forEach((row) => {
      const i = Number(row.dataset["seat"] ?? "0");
      const kindSel = row.querySelector<HTMLSelectElement>('[data-role="seatkind"]');
      const modelSel = row.querySelector<HTMLSelectElement>('[data-role="model"]');
      kindSel?.addEventListener("change", () => {
        const kind = kindSel.value === "ai" ? "ai" : "human";
        // 座位表就是**完整的一张桌子**，第 0 格是房主自己。
        // 把某一格标成 AI 就真的没有真人坐在那儿。
        this.draft.seats[i] = kind === "ai" ? { kind: "ai", ai: firstModel() } : { kind: "human" };
        this.renderBuild();
      });
      modelSel?.addEventListener("change", () => {
        this.draft.seats[i] = { kind: "ai", ai: modelSel.value };
      });
    });

    this.buildEl.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const [kinds, players] = (btn.dataset["preset"] ?? "").split("|");
        const n = (Number(players) === 3 ? 3 : 4) as 3 | 4;
        const arr = kinds.split(",").slice(0, n);
        this.draft.players = n;
        this.draft.seats = arr.map((k) =>
          k === "ai" ? { kind: "ai", ai: firstModel() } : { kind: "human" },
        );
        this.renderBuild();
      });
    });

    (this.buildEl.querySelector("#bname") as HTMLInputElement).addEventListener("input", (ev) => {
      this.draft.name = (ev.target as HTMLInputElement).value.slice(0, 24);
    });
    (this.buildEl.querySelector("#bplayers") as HTMLSelectElement).addEventListener("change", (ev) => {
      const n = (ev.target as HTMLSelectElement).value === "3" ? 3 : 4;
      this.draft.players = n;
      const seats = this.draft.seats.slice(0, n);
      while (seats.length < n) seats.push({ kind: "ai", ai: firstModel() });
      this.draft.seats = seats;
      this.renderBuild();
    });
    (this.buildEl.querySelector("#brounds") as HTMLSelectElement).addEventListener("change", (ev) => {
      this.draft.rounds = Number((ev.target as HTMLSelectElement).value) || 0;
    });
    (this.buildEl.querySelector("#bdelay") as HTMLSelectElement).addEventListener("change", (ev) => {
      const v = (ev.target as HTMLSelectElement).value;
      this.draft.aiDelay = v === "" ? null : Number(v);
    });
    (this.buildEl.querySelector("#bgo") as HTMLButtonElement).addEventListener("click", () => {
      if (!this.name) {
        this.cb.onToast("先在右上角取个昵称", "warn");
        this.nameInput.focus();
        return;
      }
      const seats = this.draft.seats.slice(0, this.draft.players);
      while (seats.length < this.draft.players) seats.push({ kind: "ai", ai: firstModel() });
      this.cb.onCreate({ ...this.draft, seats });
    });
    (this.buildEl.querySelector("#bcancel") as HTMLButtonElement).addEventListener("click", () => {
      this.buildEl.hidden = true;
    });
  }

  // -------------------------------------------------------------- 规则 / 隐私 --

  private renderRules(): void {
    const rulesEl = this.root.querySelector("#rules") as HTMLElement;
    rulesEl.innerHTML = `
      <h3>规则</h3>
      <ul>
        <li>默认东风战：三麻 3 局、四麻 4 局；一位未满返し点会南入。</li>
        <li>三麻 108 张（万子只有 1m / 9m）、有拔北；四麻 136 张、有吃。</li>
        <li>赤 5：三麻赤 5p / 5s，四麻再加赤 5m；双响有效，不聴罰符三麻 2000 / 四麻 3000。</li>
        <li>真人思考超时会由内置教师托管，不会把整桌卡死。</li>
        <li>观战全亮牌只在纯 AI 房生效；桌上有真人时服务端直接拒绝。</li>
      </ul>
    `;
  }

  private renderPrivacy(): void {
    if (!this.recordOn) {
      this.privacyEl.innerHTML = `
        <h3>数据记录</h3>
        <p class="muted small">本服务当前<strong>未开启</strong>对局数据记录，牌局不会被落盘。</p>`;
      return;
    }
    this.privacyEl.innerHTML = `
      <h3>数据记录与隐私</h3>
      <p class="small">
        <strong>本局对局数据会被记录，用于模型训练。</strong>
        记录的是决策级数据：局面（自己的手牌 / 四家牌河 / 副露 / 宝牌 / 立直状态 /
        巡目 / 点数）、引擎给出的全部合法动作、你的选择、思考用时。
        一局结束时四家手牌也会一并存下来（服务端本来就知道）。
      </p>
      <p class="small muted">
        玩家 ID 是随机匿名串，不是昵称 / 邮箱；带 ID 的原始日志不对外展示。
        数据按天分文件存放，有容量上限和轮转。
      </p>
      <label class="small">
        <input type="checkbox" id="norec"${this.noRecord ? " checked" : ""} />
        本局不记录我的数据
      </label>
    `;
    const box = this.privacyEl.querySelector<HTMLInputElement>("#norec");
    box?.addEventListener("change", () => {
      this.noRecord = box.checked;
      this.cb.onNoRecord(box.checked);
    });
  }

  /** 新的一局开始后把「不记录」的勾去掉（服务端那边也是每局重置语义）。 */
  resetPrivacy(): void {
    if (!this.noRecord) return;
    this.noRecord = false;
    this.renderPrivacy();
  }
}
