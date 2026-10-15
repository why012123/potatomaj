/**
 * 牌桌视图。
 *
 * ## 方位（这条是整块代码里最容易搞错的地方）
 *
 * 自己永远在**下方**，其余**逆时针**排：下家 → 右、对家 → 上、上家 → 左。
 * 三麻只有下家和上家。方位由服务端给的 `seat.rel` 直接决定，前端**不做**
 * `(seat - viewer) % n` 这种二次推导 —— 服务端 `rel_label()` 已经是权威，
 * 前端再算一遍只会在三麻上算岔。
 *
 * ## 布局：为什么不能只靠 CSS
 *
 * 左右两家是整块 `.edge` 转 ±90° 之后摆到桌边的。旋转之后：
 * - `left/top` 百分比是相对**未旋转的盒子**解析的，跟屏幕上看到的完全不是
 *   一回事；
 * - 盒子的宽高也在旋转后互换了。
 *
 * 所以位置由 JS **实测反解**：先给一个粗略的 translate，装上之后
 * `getBoundingClientRect()` 量真实包围盒，把「量到的中心」和「想要的中心」
 * 的残差补进 `--fix-x` / `--fix-y`，再来一次。因为 translate 在 transform
 * 链里是最后应用的（矩阵乘法意义上等价于对中心做平移），这个迭代
 * 2~3 次就收敛到亚像素；实测一般第 2 次残差已经 <0.5px。
 *
 * 另外牌河 / 副露的偏移是**量出手牌真实宽度**之后写死的像素值（`--pond-dx`
 * 等）。手牌张数从 14 变到 1，百分比偏移就会把牌河甩到隔壁去。
 */

import type { MeldInfo, RoundInfo, SeatState, StateMsg } from "../protocol";
import { buildHiddenHand, buildMeld, tileEl } from "./tiles";

export type Dir = "bottom" | "right" | "top" | "left";

export interface TableCallbacks {
  /** 点手牌出牌（`tile` 是牌 id） */
  onDiscard: (tile: string) => void;
  /** 操作按钮（`kind` + 选项下标） */
  onAction: (kind: string, index: number) => void;
  onNext: () => void;
}

interface SeatView {
  root: HTMLDivElement;
  dir: Dir;
  plate: HTMLDivElement;
  name: HTMLElement;
  score: HTMLElement;
  wind: HTMLElement;
  badges: HTMLElement;
  edge: HTMLDivElement;
  hand: HTMLDivElement;
  melds: HTMLDivElement;
  pond: HTMLDivElement;
  riichiStick: HTMLDivElement;
  /** 上一次渲染的签名，用来判断「这块要不要重建」 */
  sig: { hand: string; melds: string; meldCount: number; pond: string; first: boolean };
  /** fitHandSpan 算出的盒子尺寸（未旋转），反解 translate 时用 */
  boxW: number;
  boxH: number;
}

const SIDE_CLASS = "edge--side";

/** `rel` 字符串 → 屏幕方位。三麻没有「对家」。 */
export function dirOf(rel: string): Dir {
  switch (rel) {
    case "下家":
      return "right";
    case "对家":
      return "top";
    case "上家":
      return "left";
    default:
      return "bottom";
  }
}

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

/** 名牌 + 空的边块骨架。 */
function seatSkeleton(dir: Dir): string {
  const side = dir === "left" || dir === "right" ? ` ${SIDE_CLASS}` : "";
  return `
    <div class="seat__plate">
      <span class="plate__wind"></span>
      <span class="plate__name"></span>
      <span class="plate__score"></span>
      <span class="plate__badges"></span>
    </div>
    <div class="edge edge--${dir}${side}">
      <div class="edge__hand"></div>
      <div class="edge__melds"></div>
      <div class="edge__pond"></div>
    </div>
    <div class="seat__riichi"></div>
  `;
}

/** 期望的包围盒中心（视口坐标 + 尺寸）。 */
export class TableView {
  readonly root: HTMLDivElement;
  private readonly cb: TableCallbacks;
  private readonly seats = new Map<number, SeatView>();
  private readonly roundEl: HTMLDivElement;
  private readonly wallEl: HTMLDivElement;
  private readonly doraEl: HTMLDivElement;
  private readonly sticksEl: HTMLDivElement;
  private readonly turnEl: HTMLDivElement;
  private readonly curtain: HTMLDivElement;
  /** 上一帧牌河长度：用来判断哪张是新打出的（做动画） */
  private readonly prevPond = new Map<number, number>();
  private prevSeq = -1;
  private lastFit = "";
  private fitCount = 0;
  private refitPending = false;
  /** 最近一次诊断信息（CDP 自检用） */
  lastDiag: Record<string, unknown> = {};

  constructor(cb: TableCallbacks) {
    this.cb = cb;
    this.root = h("div", "table");
    this.root.innerHTML = `
      <div class="table__felt"></div>
      <div class="table__rim"></div>
      <div class="table__center">
        <div class="centerbox">
          <div class="centerbox__round">等待开始</div>
          <div class="centerbox__wall"><b>--</b><span>枚</span></div>
          <div class="centerbox__dora"><span class="lbl">宝牌</span><div class="dora-row"></div></div>
          <div class="centerbox__sticks"><span class="lbl">供托</span><span class="stick"></span><b>0</b></div>
          <div class="centerbox__turn"></div>
        </div>
      </div>
      <div class="table__curtain"></div>
    `;
    this.roundEl = this.root.querySelector(".centerbox__round") as HTMLDivElement;
    this.wallEl = this.root.querySelector(".centerbox__wall") as HTMLDivElement;
    this.doraEl = this.root.querySelector(".dora-row") as HTMLDivElement;
    this.sticksEl = this.root.querySelector(".centerbox__sticks") as HTMLDivElement;
    this.turnEl = this.root.querySelector(".centerbox__turn") as HTMLDivElement;
    this.curtain = this.root.querySelector(".table__curtain") as HTMLDivElement;
  }

  /** 换房间 / 回大厅时清空。 */
  reset(): void {
    for (const s of this.seats.values()) s.root.remove();
    this.seats.clear();
    this.prevPond.clear();
    this.prevSeq = -1;
  }

  // ---------------------------------------------------------------- 座位 --

  private ensureSeat(seat: number, dir: Dir): SeatView {
    const found = this.seats.get(seat);
    if (found) {
      if (found.dir !== dir) {
        found.dir = dir;
        found.root.className = `seat seat--${dir}`;
        found.root.dataset["dir"] = dir;
        found.edge.className = `edge edge--${dir}${dir === "left" || dir === "right" ? ` ${SIDE_CLASS}` : ""}`;
        found.sig.hand = "";
        found.sig.melds = "\u0000";
        found.sig.meldCount = 0;
        found.sig.pond = "";
        found.sig.first = true;
      }
      return found;
    }
    const root = h("div", `seat seat--${dir}`);
    root.dataset["dir"] = dir;
    root.dataset["seat"] = String(seat);
    root.innerHTML = seatSkeleton(dir);
    const view: SeatView = {
      root,
      dir,
      plate: root.querySelector(".seat__plate") as HTMLDivElement,
      name: root.querySelector(".plate__name") as HTMLElement,
      score: root.querySelector(".plate__score") as HTMLElement,
      wind: root.querySelector(".plate__wind") as HTMLElement,
      badges: root.querySelector(".plate__badges") as HTMLElement,
      edge: root.querySelector(".edge") as HTMLDivElement,
      hand: root.querySelector(".edge__hand") as HTMLDivElement,
      melds: root.querySelector(".edge__melds") as HTMLDivElement,
      pond: root.querySelector(".edge__pond") as HTMLDivElement,
      riichiStick: root.querySelector(".seat__riichi") as HTMLDivElement,
      sig: { hand: "", melds: "\u0000", meldCount: 0, pond: "", first: true },
      boxW: 0,
      boxH: 0,
    };
    this.seats.set(seat, view);
    this.root.appendChild(root);
    return view;
  }

  // ---------------------------------------------------------------- 渲染 --

  render(state: StateMsg): void {
    const n = state.room.players;
    const viewer = state.viewer.seat;
    const freshFrame = state.seq !== this.prevSeq;
    this.prevSeq = state.seq;

    this.root.dataset["phase"] = state.phase;
    this.root.dataset["players"] = String(n);
    this.root.dataset["viewer"] = String(viewer);

    // 该出现几个座位：服务端 `seats` 里有哪些就画哪些
    const wanted = new Set<number>();
    for (const s of state.seats) wanted.add(s.seat);
    for (const [seat, view] of this.seats) {
      if (!wanted.has(seat)) {
        view.root.remove();
        this.seats.delete(seat);
        this.prevPond.delete(seat);
      }
    }

    for (const s of state.seats) {
      // 观战者的「自己」不存在：服务端给观战的 rel 是相对 0 号位算的，
      // 直接拿来用会让 0 号位跑到下面去。观战时按「0 号位在下方」摆，
      // 这样左右上下仍然是逆时针的，只是「下方」是 0 号位而已。
      const rel = viewer < 0 ? relFromViewer(s.seat, 0, n) : s.rel;
      this.ensureSeat(s.seat, dirOf(rel));
    }
    for (const s of state.seats) this.renderSeat(s, state, freshFrame);

    this.renderCenter(state);
    this.renderCurtain(state);

    // 每帧都推进一轮布局修正。
    //
    // **不能只在「尺寸变了」的时候做**：这一帧手牌 / 牌河刚被重建，上一帧存下来的
    // `--fix-*` 就是按旧内容算的，不重跑就会一直偏着；而一帧里改完自定义属性
    // 立刻量，读到的还是旧布局，所以必须**逐帧**迭代到收敛（见 fitEdge）。
    // 代价很小：一帧几次 getBoundingClientRect，只在有帧推送时发生（服务端
    // 30fps 合并过）。换视口尺寸时顺手重置收敛状态。
    const fitKey = `${window.innerWidth}x${window.innerHeight}:${n}:${state.seats.length}`;
    if (fitKey !== this.lastFit) {
      this.lastFit = fitKey;
      this.fitCount = 0;
    }
    void freshFrame;
    this.fitLayout();
  }

  /**
   * 「下一帧再量一次」。
   *
   * 手牌 / 牌河刚被重建的那一帧，浏览器还没把布局算完；而且我们在同一帧里写了
   * `--edge-w/h`、`--fix-x/y` 这些自定义属性，**立刻**去读 `getBoundingClientRect`
   * 拿到的还是改动之前的值。所以修正必须跨帧迭代：一帧一轮，量到稳定为止。
   * 用一个标记去重，不重复排队。
   */
  private scheduleRefit(): void {
    if (this.refitPending) return;
    this.refitPending = true;
    requestAnimationFrame(() => {
      this.refitPending = false;
      this.fitLayout();
    });
  }

  private renderSeat(s: SeatState, state: StateMsg, freshFrame: boolean): void {
    const v = this.seats.get(s.seat);
    if (!v) return;
    // 这一家的内容变了（手牌 / 副露 / 牌河重建）→ 现在存着的 `--fix-*` 是按旧
    // 内容算的。**必须重新收敛**，否则它会一直偏着，直到下一次恰好有帧推送。
    let dirty = false;
    const mine = s.seat === state.viewer.seat;
    const active =
      state.round != null &&
      state.phase === "playing" &&
      state.actions != null &&
      state.actions.seat === s.seat;

    v.root.classList.toggle("is-active", active);
    v.root.classList.toggle("is-dealer", s.is_dealer);
    v.root.classList.toggle("is-offline", !s.connected);
    if (s.seat === state.viewer.seat) v.root.classList.add("is-me");
    else v.root.classList.remove("is-me");

    // -- 名牌 --
    v.wind.textContent = s.seat_wind || "?";
    v.name.textContent = s.name;
    v.score.textContent = String(s.score);
    const badges: string[] = [];
    if (s.is_dealer) badges.push('<i class="bg bg--dealer">亲</i>');
    if (s.host) badges.push('<i class="bg">房主</i>');
    if (s.riichi) badges.push(`<i class="bg bg--riichi">${s.double_riichi ? "双立直" : "立直"}</i>`);
    if (s.ippatsu) badges.push('<i class="bg bg--ippatsu">一发</i>');
    if (s.kita > 0) badges.push(`<i class="bg bg--kita">拔北 ${s.kita}</i>`);
    if (s.kind === "ai") badges.push('<i class="bg bg--ai">AI</i>');
    if (s.ai_degraded) {
      badges.push(`<i class="bg bg--warn" title="${escapeAttr(s.ai_degrade_reason)}">已降级</i>`);
    }
    if (s.no_record) badges.push('<i class="bg bg--off">不记录</i>');
    if (!s.connected) badges.push('<i class="bg bg--off">离线</i>');
    if (state.room.next_acks?.[s.seat]) badges.push('<i class="bg bg--ok">已准备</i>');
    v.badges.innerHTML = badges.join("");
    v.riichiStick.classList.toggle("is-on", s.riichi);

    // -- 手牌 --
    const size = state.room.players === 3 ? "lg" : "md";
    const handSig = `${s.hand_count}|${s.hand ? s.hand.join(",") : "-"}|${s.drawn ?? "-"}`;
    if (v.sig.hand !== handSig) {
      v.sig.hand = handSig;
      v.hand.replaceChildren();
      v.hand.classList.toggle("is-mine", mine);
      if (s.hand) this.buildMyHand(v.hand, s, state, size);
      else v.hand.appendChild(buildHiddenHand(s.hand_count, size));
      dirty = true;
    } else if (mine) {
      this.updateHandSelection(v.hand, state);
    }

    // -- 副露 --
    const meldSig = JSON.stringify(s.melds.map((m: MeldInfo) => m.kind + m.tiles.join("")));
    if (v.sig.melds !== meldSig) {
      const nowCount = countMeldTiles(s.melds);
      // 首帧不弹动画：刷新页面时整桌副露都在，全弹一遍会像放烟花
      const grew = nowCount > v.sig.meldCount && !v.sig.first;
      v.sig.melds = meldSig;
      v.sig.meldCount = nowCount;
      v.sig.first = false;
      v.melds.replaceChildren();
      for (const m of s.melds) {
        const box = buildMeld(m, "sm");
        if (grew) box.classList.add("is-meld-new");
        v.melds.appendChild(box);
      }
      dirty = true;
    }

    // -- 牌河 --
    const pondSig = `${s.pond.join(",")}|${s.pond_red.join("")}|${s.riichi_tile ?? ""}`;
    if (v.sig.pond !== pondSig) {
      v.sig.pond = pondSig;
      this.rebuildPond(v, s, size, freshFrame);
      dirty = true;
    }

    if (dirty) this.scheduleRefit();
  }

  /**
   * 自己的手牌：理好的 13 张 + 一道缝 + 刚摸的那张。
   *
   * 只有服务端在 `actions.options` 里列出来的牌才可点 —— **可选项只能来自
   * 服务端**。立直可宣言的牌（`riichi_ok`）会多一圈蓝边，点它服务端会在
   * 出牌的同时把立直宣言一起处理掉。
   */
  private buildMyHand(
    box: HTMLDivElement,
    s: SeatState,
    state: StateMsg,
    size: "md" | "lg",
  ): void {
    const actions = state.actions;
    const isDiscard = !!actions && actions.kind === "discard" && actions.waiting === false;
    const options = isDiscard && actions ? actions.options : [];
    const byTile = new Map<string, number>();
    for (const o of options) {
      if (o.tile && !byTile.has(o.tile)) byTile.set(o.tile, o.index);
    }
    const riichiOk = new Set(options.filter((o) => o.riichi_ok).map((o) => o.tile ?? ""));

    const hand = s.hand ?? [];
    const drawn = s.drawn;

    const make = (id: string, isDrawn: boolean): HTMLElement => {
      const idx = byTile.get(id);
      const clickable = isDiscard && idx !== undefined;
      const el = tileEl({
        id,
        size,
        cls: [
          isDrawn ? "is-drawn" : "",
          clickable ? "is-clickable" : "",
          clickable && riichiOk.has(id) ? "is-riichi-ok" : "",
        ]
          .filter(Boolean)
          .join(" "),
        title: clickable ? `${id}（点击打出）` : id,
        onClick: clickable ? () => this.cb.onDiscard(id) : undefined,
      });
      if (clickable && idx !== undefined) el.dataset["index"] = String(idx);
      return el;
    };

    for (const t of hand) box.appendChild(make(t, false));
    if (drawn) {
      // 摸到的那张与理好的牌之间留一道缝
      box.appendChild(h("span", "hand__gap"));
      box.appendChild(make(drawn, true));
    }
  }

  /** 快照没变时只更新「哪些牌可点」，不重建 DOM（保住动画与焦点）。 */
  private updateHandSelection(box: HTMLDivElement, state: StateMsg): void {
    const actions = state.actions;
    const isDiscard = !!actions && actions.kind === "discard" && actions.waiting === false;
    const map = new Map<string, number>();
    const riichiOk = new Set<string>();
    if (isDiscard && actions) {
      for (const o of actions.options) {
        if (o.tile && !map.has(o.tile)) map.set(o.tile, o.index);
        if (o.riichi_ok && o.tile) riichiOk.add(o.tile);
      }
    }
    for (const node of Array.from(box.children)) {
      if (!(node instanceof HTMLElement)) continue;
      const t = node.dataset["tile"];
      if (!t) continue;
      const idx = map.get(t);
      const clickable = isDiscard && idx !== undefined;
      node.classList.toggle("is-clickable", clickable);
      node.classList.toggle("is-riichi-ok", clickable && riichiOk.has(t));
      if (clickable && idx !== undefined) node.dataset["index"] = String(idx);
      else delete node.dataset["index"];
    }
  }

  /**
   * 牌河：每行 6 张、最多 3 行（第 19 张起接着往下排，牌桌真实规则本来
   * 也是往右边挤；这里用 overflow 类做视觉降级）。
   *
   * **立直宣言牌要横放 90°**：服务端在 `seats[i].riichi_tile` 给了宣言牌，
   * 但牌河里可能有好几张同牌型的牌，所以取**最后一张**等于它的 ——
   * 那才是他立直时打出的那一张。`lastIndexOf` 正好是这个语义。
   */
  private rebuildPond(v: SeatView, s: SeatState, size: "md" | "lg", animate: boolean): void {
    const before = this.prevPond.get(s.seat) ?? 0;
    const pond = s.pond;
    const riichiAt = riichiIndex(s);
    // 只有「多了恰好一张」才做动画：一帧里塞进两张（断线重连）就直接画。
    const animFrom = animate && pond.length === before + 1 ? before : -1;

    v.pond.replaceChildren();
    v.pond.dataset["count"] = String(pond.length);
    pond.forEach((t, i) => {
      const red = s.pond_red[i] === 1;
      v.pond.appendChild(
        tileEl({
          id: t,
          size,
          rotate: i === riichiAt,
          cls: [
            i >= 18 ? "is-overflow" : "",
            red ? "is-red" : "",
            i === animFrom ? "is-new" : "",
          ]
            .filter(Boolean)
            .join(" "),
          title: i === riichiAt ? `立直宣言牌 ${t}` : t,
        }),
      );
    });
    this.prevPond.set(s.seat, pond.length);
  }

  // ------------------------------------------------------------ 中央信息 --

  private renderCenter(state: StateMsg): void {
    const r: RoundInfo | null = state.round;
    if (!r) {
      this.roundEl.textContent = state.phase === "lobby" ? "等待开始" : "准备中";
      this.wallEl.innerHTML = "<b>--</b><span>枚</span>";
      this.doraEl.replaceChildren();
      this.sticksEl.querySelector("b")!.textContent = "0";
      this.turnEl.textContent = "";
      return;
    }
    const handNo = r.hand_no;
    const dealer = state.seats.find((s) => s.seat === r.dealer);
    this.roundEl.innerHTML =
      `${escapeHtml(r.round_wind)}${handNo} 局` +
      `<span class="sub">第 ${r.turn} 巡 · 亲 ${escapeHtml(dealer?.name ?? `${r.dealer + 1} 号位`)}</span>`;
    this.wallEl.innerHTML = `<b>${r.wall}</b><span>枚</span>`;

    this.doraEl.replaceChildren();
    const addDora = (ids: string[], cls: string): void => {
      ids.forEach((id, i) => {
        if (i > 0) this.doraEl.appendChild(h("span", "dora-sep"));
        this.doraEl.appendChild(tileEl({ id, size: "xs", cls: `is-dora ${cls}`.trim(), title: `宝牌指示 ${id}` }));
      });
    };
    addDora(r.dora, "");
    if (r.ura.length) addDora(r.ura, "is-ura");

    this.sticksEl.querySelector("b")!.textContent = String(r.sticks);

    // 轮到谁 + 倒计时
    const a = state.actions;
    if (a && state.phase === "playing") {
      const who = state.seats.find((s) => s.seat === a.seat);
      const mine = a.seat === state.viewer.seat;
      const left = Math.max(0, a.ms_left) / 1000;
      const urgent = a.ms_left > 0 && a.ms_left < 6000;
      this.turnEl.innerHTML =
        `<span class="turn__who${mine ? " is-mine" : ""}">${escapeHtml(who?.name ?? `${a.seat + 1} 号位`)}${mine ? "（你）" : ""}</span>` +
        `<span class="turn__timer${urgent ? " is-urgent" : ""}">${left.toFixed(1)}s</span>` +
        `<span class="muted small"> · ${escapeHtml(a.kind_cn)}</span>`;
    } else {
      this.turnEl.textContent = "";
    }
  }

  private renderCurtain(state: StateMsg): void {
    let text = "";
    if (state.phase === "hand_end") text = "本局结束";
    this.curtain.textContent = text;
    this.curtain.classList.toggle("is-on", !!text);
  }

  // ---------------------------------------------------------------- 布局 --

  /**
   * 布局修正：一帧只做一轮「量 → 改」，靠 `scheduleRefit()` 逐帧迭代到收敛。
   *
   * 顺序：先把手牌尺寸 / 盒内偏移算好（纯算术，见 `fitHandSpan`），再让整家
   * 的自校正往前走一步（`fitEdge`，量现状补差值）。
   *
   * 为什么不在同一帧里跑两遍：改完自定义属性立刻量，读到的还是旧布局 ——
   * 两遍里第二遍量到的仍是旧值，纯属白跑（踩过：修正写了但位置纹丝不动）。
   */
  fitLayout(): void {
    this.fitCount++;
    let moved = 0;
    for (const v of this.seats.values()) this.fitHandSpan(v);
    for (const v of this.seats.values()) moved += this.fitEdge(v);
    this.lastDiag = {
      fit: this.fitCount,
      moved: Math.round(moved * 10) / 10,
      size: `${window.innerWidth}x${window.innerHeight}`,
      seats: [...this.seats.entries()].map(([seat, v]) => {
        const r = v.edge.getBoundingClientRect();
        return {
          seat,
          dir: v.dir,
          fix: [v.edge.style.getPropertyValue("--fix-x"), v.edge.style.getPropertyValue("--fix-y")],
          edgeBox: [
            v.edge.style.getPropertyValue("--edge-w"),
            v.edge.style.getPropertyValue("--edge-h"),
          ],
          handSpan: v.edge.style.getPropertyValue("--hand-span"),
          rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
          pond: v.pond.dataset["count"] ?? "0",
          pondRot: [...v.pond.children].findIndex((c) => c.classList.contains("is-rot")),
          melds: v.melds.children.length,
        };
      }),
      overflow: this.overflowReport(),
    };
    // 还没收敛就下一帧接着来（最多几帧，不会一直跑）
    if (moved > 0.5) this.scheduleRefit();
  }

  /**
   * 手牌真实跨度 → 盒子尺寸与所有内部偏移（全部是算好的像素值）。
   *
   * 跨度**逐张加**，不读 `hand.offsetWidth`：手牌在带旋转的祖先里会被压扁
   * （实测 19px vs 378px），读它就会把整家摆到屏幕外。逐张读子元素的
   * `offsetWidth` 是准的 —— 它们自己没旋转，而且 `flex: 0 0 auto` 不会被压缩。
   *
   * 牌高也**量元素**，不读 CSS 变量：`--tile-h` 是 `calc(min(...))`，
   * `getPropertyValue()` 返回的是**没解析的表达式字符串**，parseFloat 是 NaN。
   */
  private fitHandSpan(v: SeatView): void {
    const tile = v.hand.querySelector<HTMLElement>(".tile");
    const tileW = tile && tile.offsetWidth ? tile.offsetWidth : 30;
    const tileH = tile && tile.offsetHeight ? tile.offsetHeight : tileW / 0.96;
    const n = v.hand.children.length;
    if (!n) return;
    // 跨度按「张数 × 牌宽 + 牌距」算，不依赖任何一次 DOM 收缩计算 ——
    // 手牌在带旋转的祖先里会被压扁，父元素的尺寸一律不可信。
    const gap = parseFloat(getComputedStyle(v.hand).columnGap) || 0;
    const span = tileW * n + gap * (n - 1);
    const side = v.dir === "left" || v.dir === "right";
    const slack = Math.max(2, tileH * 0.1);
    const thick = tileH + slack * 2;

    // 盒子尺寸：一边是「手牌跨度」，另一边是「牌厚 + 留白」。**不能比手牌窄** ——
    // 手牌要在盒子里居中，盒子比手牌窄就会把它顶出去（下家只剩几张牌的时候
    // 最容易碰上）。短边用 max，多出来的空间由居中偏移吃掉，这样牌河 / 副露
    // 相对手牌的位置不随手牌张数抖动。
    const across = Math.max(span, thick);
    v.boxW = side ? thick : across;
    v.boxH = side ? across : thick;
    v.edge.style.setProperty("--edge-w", `${v.boxW.toFixed(2)}px`);
    v.edge.style.setProperty("--edge-h", `${v.boxH.toFixed(2)}px`);
    v.edge.style.setProperty("--side-slack", `${slack.toFixed(2)}px`);
    v.edge.style.setProperty("--hand-span", `${span.toFixed(2)}px`);
    v.edge.style.setProperty("--hand-thick", `${thick.toFixed(2)}px`);
    // 手牌居中偏移 =「盒子长边 - 手牌跨度」的一半
    v.edge.style.setProperty("--hand-center", `${((across - span) / 2).toFixed(2)}px`);
    if (side) {
      // ---- 侧家 ----
      // 手牌在局部 x = slack .. slack + 牌厚（贴着墙那一边）。
      // 副露紧挨手牌里侧；牌河再往里让开副露那一条带子（最多 4 张副露）。
      v.edge.style.setProperty("--melds-lead", `${thick.toFixed(2)}px`);
      v.edge.style.setProperty("--pond-lead", `${(thick + slack).toFixed(2)}px`);
      // 牌河的上边缘 = 手牌起点 + 让开的距离。手牌在局部 y ∈ [0, span]，
      // 用 `top`（自盒子顶边往下量）最直观，不跟 `bottom` 的百分比打架。
      v.edge.style.setProperty("--pond-gap", `${(tileH * 2.6 + slack).toFixed(2)}px`);
    } else {
      // ---- 下 / 上家 ----
      // 副露紧挨手牌右端；牌河右端与手牌右端对齐，整体在手牌内侧。
      // 顶家整体转了 180°，局部 x/y 都翻了，所以它的 CSS 用的是
      // `right` / `bottom` —— 同一套「往盒子里量」的偏移在两边都成立。
      v.edge.style.setProperty("--melds-lead", `${((across - span) / 2 - slack).toFixed(2)}px`);
      v.edge.style.setProperty(
        "--pond-lead",
        `${((across - span) / 2 - slack).toFixed(2)}px`,
      );
      v.edge.style.setProperty("--pond-gap", `${(thick + tileH * 0.5).toFixed(2)}px`);
    }
  }

  /**
   * 副露 / 牌河的位置**全部靠算**（见 `fitHandSpan` 里那几个 `--melds-lead`
   * / `--pond-lead` / `--pond-gap`），没有单独的修正函数。
   *
   * 为什么不留「量残差再补」的兜底（踩过的坑）：这类元素的落点在带旋转的祖先
   * 里量不准，一旦某个 `--x` 被补反了方向，`requestAnimationFrame` 每帧都会
   * 再补一次同样的量，几十帧之后牌河就跑到离桌子几千像素外去了。
   * **能算就别量**，这条比「量了再补」稳得多。
   */

  /**
   * 把整家摆到桌边 —— 一次自校正迭代。
   *
   *     translate ← translate + (目标位置 - 现在的实际位置)
   *
   * 因为 `translate(t) rotate(θ)` 里 rotate 绕自身中心、**旋转不改包围盒中心**，
   * 所以「包围盒中心 = 静止中心 + t」恒成立，一次迭代就把差值补平，不会振荡。
   *
   * ## 为什么必须跨帧迭代，不能在同一帧里跑两遍
   *
   * 我们在同一帧里改了 `--edge-w/h`、`--fix-x/y` 这些自定义属性，然后**立刻**
   * 用 `getBoundingClientRect()` 去量 —— 拿到的还是**改动之前**的布局。同一帧里
   * 跑两遍「改 → 量」是白跑：第二遍量到的仍然是旧值。表现就是「明明算了修正，
   * 位置却纹丝不动」，或者更糟 —— 用旧尺寸解方程把整家推到屏幕外。
   *
   * 正确做法：一帧只做一轮「量 → 改」，靠 `requestAnimationFrame` 迭代到收敛
   * （见 `scheduleRefit`）。实测 2~3 帧就稳定到亚像素。
   *
   * 返回这一轮改了多少（收敛判据用）。
   */
  private fitEdge(v: SeatView): number {
    const parent = this.root.getBoundingClientRect();
    const side = v.dir === "left" || v.dir === "right";
    // 量**手牌里每一张牌的并集**：容器在带旋转的祖先下会塌掉，而牌一定画出来了
    const r = spanOf(v.hand, ".tile, .hand__gap");
    if (!r) return 0;

    // 目标：让手牌贴到桌子对应的沿上、另一个方向居中
    const wantCx = side
      ? v.dir === "right"
        ? parent.right - r.width / 2
        : parent.left + r.width / 2
      : parent.left + parent.width / 2;
    const wantCy = side
      ? parent.top + parent.height / 2
      : v.dir === "bottom"
        ? parent.bottom - r.height / 2
        : parent.top + r.height / 2;

    const haveX = num(v.edge.style.getPropertyValue("--fix-x"));
    const haveY = num(v.edge.style.getPropertyValue("--fix-y"));
    // 限幅：先按小步长逼近，量到脏值也不会把整家甩出屏幕
    const clamp = (d: number): number => Math.max(-200, Math.min(200, d));
    const dx = clamp(wantCx - (r.left + r.width / 2));
    const dy = clamp(wantCy - (r.top + r.height / 2));
    v.edge.style.setProperty("--fix-x", `${(haveX + dx).toFixed(2)}px`);
    v.edge.style.setProperty("--fix-y", `${(haveY + dy).toFixed(2)}px`);
    return Math.abs(dx) + Math.abs(dy);
  }

  /** 自检用：哪些家具跑出了牌桌。 */
  overflowReport(): Array<Record<string, unknown>> {
    const parent = this.root.getBoundingClientRect();
    const out: Array<Record<string, unknown>> = [];
    for (const [seat, v] of this.seats) {
      for (const [what, node] of [
        ["hand", v.hand],
        ["melds", v.melds],
        ["pond", v.pond],
        ["plate", v.plate],
      ] as Array<[string, HTMLElement]>) {
        if (!node.children.length) continue;
        const r = node.getBoundingClientRect();
        const over = {
          left: r.left < parent.left - 1,
          right: r.right > parent.right + 1,
          top: r.top < parent.top - 1,
          bottom: r.bottom > parent.bottom + 1,
        };
        if (over.left || over.right || over.top || over.bottom) {
          out.push({
            seat,
            what,
            over,
            rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
          });
        }
      }
    }
    return out;
  }
}

// ---------------------------------------------------------------- 工具 --

/** CSS 自定义属性里的像素值（写进去的都是 `12.34px`，解析失败当 0）。 */
function num(raw: string): number {
  const n = parseFloat(raw);
  return Number.isFinite(n) ? n : 0;
}

/**
 * 一组子元素的包围盒并集（取不到就返回 null）。
 *
 * **不要直接量容器**：左右两家的祖先带 `rotate(±90deg)`，容器自己的包围盒会
 * 被压成「一张牌的宽度」（实测 19px vs 378px）。子元素（牌、副露组）自己没有
 * 额外的旋转，逐个子元素量出来的并集才是真的。
 */
function spanOf(box: HTMLElement, sel: string): DOMRect | null {
  let l = Infinity;
  let t = Infinity;
  let r = -Infinity;
  let b = -Infinity;
  for (const child of Array.from(box.querySelectorAll(sel))) {
    if (!(child instanceof HTMLElement)) continue;
    const cr = child.getBoundingClientRect();
    if (!cr.width && !cr.height) continue;
    l = Math.min(l, cr.left);
    t = Math.min(t, cr.top);
    r = Math.max(r, cr.right);
    b = Math.max(b, cr.bottom);
  }
  if (!Number.isFinite(l)) return null;
  return new DOMRect(l, t, r - l, b - t);
}

function countMeldTiles(melds: MeldInfo[]): number {
  return melds.reduce((a, m) => a + m.tiles.length, 0);
}

/** 立直宣言牌在牌河里的下标（最后一张同牌型的）。 */
function riichiIndex(s: SeatState): number {
  if (!s.riichi_tile) return -1;
  return s.pond.lastIndexOf(s.riichi_tile);
}

/** 观战时没有「自己」：按 0 号位在下方来推方位，保持逆时针不变。 */
function relFromViewer(seat: number, viewer: number, n: number): string {
  const d = (seat - viewer + n) % n;
  if (n === 3) return ["自己", "下家", "上家"][d] ?? "自己";
  return ["自己", "下家", "对家", "上家"][d] ?? "自己";
}

function escapeHtml(s: string): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeAttr(s: string): string {
  return escapeHtml(s).replace(/"/g, "&quot;");
}
