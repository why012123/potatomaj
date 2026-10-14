/**
 * 结算面板（`state.result`，`phase` 为 `hand_end` 或 `finished`）。
 *
 * 必须表达清楚的四件事：
 * 1. **完整手牌**：`result.agari[].hand` + `melds`，其中和牌张（`win_tile`）
 *    单独高亮 —— 不然一堆牌里根本看不出和的是哪张。
 * 2. **役种分两组**：真役（`yaku`）和宝牌（`dora`）。混在一起会让人以为
 *    「宝牌」是自己做出来的役。
 * 3. **番 / 符**：`total_han` 与 `fu`。
 * 4. **收支表**：`result.deltas`，输的人红、赢的人金。
 *
 * `state.match` 存在时（对局结束）再加一张终局顺位表。
 */

import type { AgariInfo, HandResult, MatchResult, StateMsg } from "../protocol";
import { buildMeld, tileEl } from "./tiles";

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

function signed(n: number): string {
  return n > 0 ? `+${n}` : `${n}`;
}

export interface ResultCallbacks {
  onNext: () => void;
  onClose: () => void;
}

export class ResultPanel {
  /** 覆盖层根节点。`.overlay` 是 `position: fixed`，会盖住整页（含侧栏）。 */
  readonly root: HTMLElement;
  private readonly body: HTMLDivElement;
  private readonly foot: HTMLElement;
  private readonly title: HTMLHeadingElement;
  /** 已经展示过的结果指纹，避免同一局被反复弹出来 */
  private shownFor = "";

  constructor(private readonly cb: ResultCallbacks) {
    this.root = el("div", "overlay");
    this.root.hidden = true;

    const panel = el("div", "panel panel--result");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");

    const head = el("header", "panel__head");
    this.title = el("h2", "panel__title", "本局结算");
    const x = el("button", "panel__x", "×");
    x.setAttribute("aria-label", "关闭");
    x.addEventListener("click", () => this.cb.onClose());
    head.append(this.title, x);

    this.body = el("div", "panel__body");
    this.foot = el("footer", "panel__foot");
    panel.append(head, this.body, this.foot);
    this.root.appendChild(panel);
  }

  get visible(): boolean {
    return !this.root.hidden;
  }

  hide(): void {
    this.root.hidden = true;
  }

  /** `state` 里有没有值得弹的东西（没有就别弹空面板）。 */
  static hasContent(state: StateMsg): boolean {
    return !!(state.result || (state.finished && state.match));
  }

  show(state: StateMsg, opts: { final?: boolean } = {}): void {
    const res = state.result;
    const match = state.match ?? null;

    this.root.hidden = false;
    this.title.textContent = opts.final
      ? "对局结束"
      : `第 ${res?.hand_no ?? "?"} 局结算`;
    this.body.replaceChildren();
    this.foot.replaceChildren();

    if (!res) {
      this.body.appendChild(el("p", "muted", "还没有结算数据。"));
      if (match) this.body.appendChild(this.matchBlock(state, match));
      this.buildFoot(state, opts.final === true);
      return;
    }

    this.body.appendChild(this.headline(state, res));

    if (res.agari.length) {
      for (const a of res.agari) this.body.appendChild(this.agariBlock(state, a));
    } else {
      this.body.appendChild(this.drawBlock(state, res));
    }

    this.body.appendChild(this.payments(state, res));
    if (match) this.body.appendChild(this.matchBlock(state, match));

    this.buildFoot(state, opts.final === true);
  }

  /** 指纹：房间 + 第几局 + 阶段。同一个指纹不重复弹。 */
  fingerprint(state: StateMsg): string {
    return `${state.room.id}:${state.result?.hand_no ?? 0}:${state.phase}`;
  }

  markShown(state: StateMsg): void {
    this.shownFor = this.fingerprint(state);
  }

  alreadyShown(state: StateMsg): boolean {
    return this.shownFor === this.fingerprint(state);
  }

  private seatName(state: StateMsg, seat: number): string {
    const s = state.seats.find((x) => x.seat === seat) ?? state.room.seats.find((x) => x.seat === seat);
    return s?.name ?? `${seat + 1} 号位`;
  }

  private buildFoot(state: StateMsg, final: boolean): void {
    if (final) {
      const lobby = el("button", "btn btn--primary", "回到大厅");
      lobby.addEventListener("click", () => this.cb.onClose());
      this.foot.appendChild(lobby);
      const again = el("button", "btn", "看最后一眼牌桌");
      again.addEventListener("click", () => this.hide());
      this.foot.appendChild(again);
      return;
    }
    const next = el("button", "btn btn--primary", "下一局");
    next.disabled = state.viewer.seat < 0;
    next.addEventListener("click", () => this.cb.onNext());
    this.foot.appendChild(next);
    this.foot.appendChild(
      el(
        "span",
        "hint",
        state.viewer.seat < 0 ? "观战中：等真人点「下一局」" : "点「下一局」继续；超时会自动开",
      ),
    );
  }

  private headline(state: StateMsg, res: HandResult): HTMLElement {
    const box = el("div", "resline");
    box.appendChild(el("span", "pill pill--gold", res.label));
    if (res.winners.length > 1) box.appendChild(el("span", "pill", `双响 × ${res.winners.length}`));
    if (res.winner !== null && res.winner !== undefined) {
      box.appendChild(el("span", undefined, `${this.seatName(state, res.winner)} 和了`));
    }
    if (res.loser !== null && res.loser !== undefined) {
      box.appendChild(el("span", "muted", `放铳：${this.seatName(state, res.loser)}`));
    }
    box.appendChild(el("span", "muted small", `第 ${res.hand_no} 局 · 共 ${res.turns} 巡`));
    if (res.sticks) box.appendChild(el("span", "pill", `供托 ${res.sticks} 本场`));
    if (res.note) box.appendChild(el("span", "hint", res.note));
    return box;
  }

  private agariBlock(state: StateMsg, a: AgariInfo): HTMLElement {
    const box = el("div", "agari");

    const who = el("div", "agari__who");
    who.appendChild(el("span", "pill pill--gold", this.seatName(state, a.seat)));
    who.appendChild(el("span", undefined, a.tsumo ? "自摸" : "荣和"));
    who.appendChild(el("span", "pill", `${a.total_han} 番 ${a.fu} 符`));
    if (a.yakuman) who.appendChild(el("span", "pill pill--gold", `役满 ×${a.yakuman}`));
    const gain = el("span", `gain ${a.gain >= 0 ? "gold" : "red"}`, signed(a.gain));
    who.appendChild(gain);
    if (a.describe) who.appendChild(el("span", "muted small", a.describe));
    box.appendChild(who);

    // 手牌：和牌张单独标出来（金色描边 + 弹一下）
    const hand = el("div", "agari__hand");
    const win = a.win_tile;
    let winUsed = false;
    for (const t of a.hand) {
      const isWin = !winUsed && win !== null && t === win;
      if (isWin) winUsed = true;
      hand.appendChild(
        tileEl({
          id: t,
          size: "md",
          cls: [isWin ? "is-win-tile" : "", t.startsWith("0") ? "is-red" : ""].filter(Boolean).join(" "),
          title: isWin ? `${t}（和牌张）` : t,
        }),
      );
    }
    // 和牌张不在手里的情况（荣和那张来自别人）：补画一张
    if (win && !winUsed) {
      hand.appendChild(tileEl({ id: win, size: "md", cls: "is-win-tile", title: `${win}（和牌张）` }));
    }
    for (const m of a.melds) {
      const wrap = el("div", "meld-inline");
      wrap.appendChild(buildMeld(m, "sm"));
      hand.appendChild(wrap);
    }
    box.appendChild(hand);

    // 役种两组：真役 / 宝牌
    const grid = el("div", "agari__yaku");

    const yakuCol = el("div", "yaku-col");
    yakuCol.appendChild(el("h4", undefined, "役种"));
    const yl = el("ul", "yaku-list");
    if (a.yaku.length) {
      for (const y of a.yaku) {
        const li = el("li");
        li.appendChild(el("span", undefined, y.name || y.key));
        li.appendChild(el("b", undefined, `${y.han} 番`));
        yl.appendChild(li);
      }
    } else {
      yl.appendChild(el("li", "muted", "（无役，仅宝牌）"));
    }
    const sum = el("li", "sum");
    sum.appendChild(el("span", undefined, "合计"));
    sum.appendChild(el("b", undefined, `${a.total_han} 番 ${a.fu} 符`));
    yl.appendChild(sum);
    yakuCol.appendChild(yl);
    grid.appendChild(yakuCol);

    const doraCol = el("div", "yaku-col");
    doraCol.appendChild(el("h4", undefined, "宝牌"));
    const dl = el("ul", "yaku-list yaku-list--dora");
    if (a.dora.length) {
      for (const d of a.dora) {
        const li = el("li");
        li.appendChild(el("span", undefined, d.name || d.key));
        li.appendChild(el("b", undefined, `${d.han} 番`));
        dl.appendChild(li);
      }
    } else {
      dl.appendChild(el("li", "muted", "（无宝牌）"));
    }
    const dsum = el("li", "sum");
    dsum.appendChild(el("span", undefined, "宝牌番数"));
    dsum.appendChild(el("b", undefined, `${a.dora_han} 番`));
    dl.appendChild(dsum);
    doraCol.appendChild(dl);
    grid.appendChild(doraCol);

    box.appendChild(grid);
    return box;
  }

  private drawBlock(state: StateMsg, res: HandResult): HTMLElement {
    const box = el("div", "agari");
    const who = el("div", "agari__who");
    who.appendChild(el("span", "pill pill--hand_end", res.label || "流局"));
    who.appendChild(el("span", "muted small", `第 ${res.hand_no} 局 · 共 ${res.turns} 巡`));
    if (res.note) who.appendChild(el("span", "hint", res.note));
    box.appendChild(who);

    const tenpai = new Set(res.tenpai ?? []);
    const grid = el("div", "drawgrid");
    for (const s of state.seats) {
      const card = el("div", `draw-seat ${tenpai.has(s.seat) ? "is-tenpai" : "is-noten"}`);
      card.appendChild(el("span", undefined, s.name));
      card.appendChild(el("i", undefined, tenpai.has(s.seat) ? "听牌" : "未听"));
      card.appendChild(el("i", "muted", `${s.score} 点`));
      grid.appendChild(card);
    }
    box.appendChild(grid);
    return box;
  }

  /** 收支表：**输的人红、赢的人金**。 */
  private payments(state: StateMsg, res: HandResult): HTMLElement {
    const box = el("div");
    box.appendChild(el("h4", undefined, "收支"));
    const table = el("table", "paytable");
    const thead = el("thead");
    const hr = el("tr");
    for (const h of ["座位", "名字", "收支", "点数"]) hr.appendChild(el("th", undefined, h));
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = el("tbody");
    const winners = new Set(res.winners.length ? res.winners : res.winner !== null ? [res.winner] : []);
    const losers = new Set(res.loser !== null && res.loser !== undefined ? [res.loser] : []);
    for (const s of state.seats) {
      const d = res.deltas[s.seat] ?? s.score_delta ?? 0;
      const tr = el("tr");
      if (d < 0) tr.classList.add("is-loser");
      else if (d > 0) tr.classList.add("is-winner");
      const nameCell = el("td");
      nameCell.appendChild(el("span", "pill", `${s.seat + 1} 号位`));
      if (s.is_dealer) nameCell.appendChild(el("span", "bg bg--dealer", "亲"));
      tr.appendChild(nameCell);
      tr.appendChild(el("td", undefined, s.name));
      const dcell = el(
        "td",
        `num ${d < 0 ? "gain-loser" : d > 0 ? "gain-winner" : "gain-zero"}`,
        signed(d),
      );
      tr.appendChild(dcell);
      tr.appendChild(el("td", "num", `${res.scores[s.seat] ?? s.score}`));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    box.appendChild(table);
    if (losers.size && !winners.size) {
      box.appendChild(el("p", "hint", "流局：听牌方收未听方的罚符。"));
    }
    return box;
  }

  /** 终局顺位表（只在 `finished` 且服务端给了 `match` 时存在）。 */
  private matchBlock(state: StateMsg, m: MatchResult): HTMLElement {
    const box = el("div");
    box.appendChild(el("h4", undefined, "终局顺位"));
    const table = el("table", "paytable rank-table");
    const thead = el("thead");
    const hr = el("tr");
    for (const h of ["顺位", "名字", "终局点数", "顺位点", "硬币"]) {
      hr.appendChild(el("th", undefined, h));
    }
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = el("tbody");
    state.seats.forEach((s, i) => {
      const rank = m.ranks[i] ?? i + 1;
      const tr = el("tr");
      if (rank === 1) tr.classList.add("is-winner");
      tr.appendChild(el("td", rank === 1 ? "rank1" : undefined, `${rank} 位`));
      tr.appendChild(el("td", undefined, s.name));
      tr.appendChild(el("td", "num", `${m.scores[i] ?? s.score}`));
      tr.appendChild(el("td", "num", `${m.rank_points[i] ?? 0}`));
      tr.appendChild(el("td", "num", `${m.coins[i] ?? 0}`));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    box.appendChild(table);

    if (m.hands.length) {
      const det = el("details", "recent");
      det.appendChild(el("summary", undefined, `每局结果（${m.hands.length} 局）`));
      const ul = el("ul");
      m.hands.forEach((h, i) => {
        const li = el("li");
        li.textContent = `第 ${i + 1} 局：${h.text || h.kind} · ${h.deltas.map(signed).join(" / ")}`;
        ul.appendChild(li);
      });
      det.appendChild(ul);
      box.appendChild(det);
    }
    return box;
  }
}
