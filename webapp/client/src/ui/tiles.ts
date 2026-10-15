/**
 * 牌的渲染 —— 全前端**唯一**一处拼 `<img src="/tiles/…">` 的地方。
 *
 * 牌面素材是服务端静态目录里的 37 张 SVG（`1m`..`9m` / `0m`.. / `1z`..`7z`
 * 加 `back.svg` / `blank.svg`），走 `/tiles/{name}` 路由，带一周的强缓存。
 * 前端不做任何「素材映射表」—— 牌 id 就是文件名，服务端给什么画什么。
 */

import type { MeldInfo, TileId } from "../protocol";

export type TileSize = "xs" | "sm" | "md" | "lg";

export interface TileOpts {
  /** null / 省略 = 牌背 */
  id?: TileId | null;
  size?: TileSize;
  /** 横放 90°（立直宣言牌、碰/杠里被叫的那张） */
  rotate?: boolean;
  dim?: boolean;
  /** 附加的高亮类：`new` / `drawn` / `win-tile` / `dora` / `riichi` … 会变成 `is-<glow>` */
  glow?: string;
  /** 额外的 class */
  cls?: string;
  badge?: string;
  title?: string;
  onClick?: (el: HTMLButtonElement) => void;
}
/**
 * 牌面图的**版本戳**。
 *
 * `/tiles/*.svg` 是带 `Cache-Control: immutable, max-age=7d` 发的（牌面要求
 * 清晰、不变，缓存是对的），但这也意味着**素材换了以后老浏览器不会重新拉** ——
 * 用户就会一直看到旧版（实测踩过：牌面白底 / 新牌背早就上线了，用户那边
 * 还是老图，怎么看都是「看不到白底」）。
 *
 * `main()` 启动时会把服务端 index.html 里注入的 `?v=<构建时间戳>` 抓出来，
 * 这里拼到每张牌上。抓不到就不加（本地直连 / 老服务端也不会坏）。
 */
let tileVersion = "";

/** 从当前页面的 `app.js?v=…` 里取出版本戳（只做一次）。 */
export function initTileVersion(): void {
  if (tileVersion) return;
  const s = document.querySelector<HTMLScriptElement>('script[src*="/app/"]');
  const m = s ? /[?&]v=(\d+)/.exec(s.getAttribute("src") ?? "") : null;
  if (m) tileVersion = m[1];
}

/** 牌面图路径。赤 5 的 id 是 `0m` / `0p` / `0s`，文件名一样。 */
export function tileSrc(id: TileId | null | undefined): string {
  const name = id ? `/tiles/${id}.svg` : "/tiles/back.svg";
  return tileVersion ? `${name}?v=${tileVersion}` : name;
}

/**
 * 所有牌面图的 `naturalWidth` 都 > 0 吗？用来做**整页自检**：
 * 有一张挂了就说明素材或路由出了问题，牌桌的布局跟着就会歪。
 */
export function tilesHealthy(): { total: number; broken: number } {
  const all = Array.from(document.querySelectorAll<HTMLImageElement>(".tile img"));
  return {
    total: all.length,
    broken: all.filter((i) => i.complete && i.naturalWidth === 0).length,
  };
}

/** 这张牌是不是赤宝牌（id 以 0 开头）。 */
export function isRed(id: TileId | null | undefined): boolean {
  return !!id && id.startsWith("0");
}

/**
 * 造一张牌。
 *
 * 有 `onClick` 的时候用 `<button>`（可聚焦、回车能按），否则用 `<div>` ——
 * 别家手牌和牌河里那几十张牌都做成 button 的话，Tab 键会变成灾难，
 * 而且浏览器会给它们加上一堆默认样式。
 */
/**
 * 一张牌里的 `<img>` 加载失败时的兜底。
 *
 * 为什么需要：牌的尺寸**全靠图片本身**（`.tile` 用 `--tile-w/h` 定盒子，
 * 但布局那边要量每张牌的 `offsetWidth` 去算手牌跨度）。图一挂，
 * `naturalWidth` 变 0、`offsetWidth` 也可能塌掉，于是手牌跨度算成 0，
 * 副露 / 牌河 / 整家的 `translate` 全跟着错 —— 屏幕上就是**牌乱飞**。
 *
 * 所以这里挂 `error` 事件，失败就给这张牌加 `is-broken`：
 * CSS 里用纯色 + 文字把它画出来（见 styles.css 的 `.tile.is-broken`），
 * 尺寸由 CSS 保证，不依赖图片。
 *
 * 顺带记一条 `console.warn`，排障时一眼能看到是哪张牌没取到。
 */
function attachFallback(el: HTMLElement, img: HTMLImageElement, id: TileId | null): void {
  const mark = (): void => {
    if (el.classList.contains("is-broken")) return;
    el.classList.add("is-broken");
    el.dataset["broken"] = id ?? "back";
    console.warn(`[tiles] 牌面图加载失败，已用纯色兜底：${img.src}`);
  };
  img.addEventListener("error", mark, { once: true });
  // 有些浏览器在缓存里拿到坏响应时不会触发 error，只在 complete 后
  // `naturalWidth === 0`。补一次检查（用 rAF 等这一轮加载结算）。
  requestAnimationFrame(() => {
    if (img.complete && img.naturalWidth === 0) mark();
  });
}

export function tileEl(o: TileOpts): HTMLElement {
  const tag = o.onClick ? "button" : "div";
  const el = document.createElement(tag);
  el.className = `tile tile--${o.size ?? "md"}`;
  if (o.rotate) el.classList.add("is-rot");
  if (o.dim) el.classList.add("is-dim");
  if (o.glow && o.glow !== "none") el.classList.add(`is-${o.glow}`);
  if (o.cls) {
    for (const c of o.cls.split(/\s+/)) if (c) el.classList.add(c);
  }
  if (o.id) el.dataset["tile"] = o.id;
  if (isRed(o.id)) el.classList.add("is-red");

  const img = document.createElement("img");
  img.draggable = false;
  img.alt = o.id ?? "牌背";
  img.src = tileSrc(o.id);
  img.loading = "eager";
  // 兜底用的文字（图片正常时被 CSS 藏起来）
  const label = document.createElement("span");
  label.className = "tile__fallback";
  label.textContent = o.id ? tileText(o.id as TileId) : "";
  el.append(img, label);
  attachFallback(el, img, o.id ?? null);

  if (o.badge) {
    const b = document.createElement("span");
    b.className = "tile__badge";
    b.textContent = o.badge;
    el.appendChild(b);
  }
  if (o.title) el.title = o.title;
  if (o.onClick && el instanceof HTMLButtonElement) {
    el.type = "button";
    el.addEventListener("click", () => o.onClick?.(el));
  }
  return el;
}

/**
 * 一组副露。
 *
 * 摆放规则（天凤式）：
 * - `chi` / `pon` / `kan`：`called` 指出的那一张**横放**（吃是顺子，横过来
 *   也还是顺子，所以吃的横放只是习惯；碰、明杠的横放是为了标出「这张是
 *   别人打的」）。
 * - `ankan`（暗杠）：四张全是自己的 —— 中间两张扣过来画牌背，
 *   这是牌桌上唯一能看出「暗杠」的方式。
 */
export function buildMeld(meld: MeldInfo, size: TileSize = "sm"): HTMLElement {
  const box = document.createElement("div");
  box.className = `meld meld--${meld.kind}`;
  box.title = meld.label;

  if (meld.kind === "ankan") {
    const first = meld.tiles[0] ?? null;
    const last = meld.tiles[meld.tiles.length - 1] ?? first;
    box.appendChild(tileEl({ id: first, size, cls: "is-ankan-face" }));
    box.appendChild(tileEl({ id: null, size, cls: "is-back is-ankan" }));
    box.appendChild(tileEl({ id: null, size, cls: "is-back is-ankan" }));
    box.appendChild(tileEl({ id: last, size, cls: "is-ankan-face" }));
    return box;
  }

  meld.tiles.forEach((t, i) => {
    const called = i === meld.called;
    box.appendChild(
      tileEl({
        id: t,
        size,
        // 吃是顺子：横放反而看不清，所以只有碰 / 杠才把被叫的那张横过来
        rotate: called && meld.kind !== "chi",
        cls: [called ? "is-called" : "", meld.kind === "kakan" && i === 0 ? "is-kakan" : ""]
          .filter(Boolean)
          .join(" "),
      }),
    );
  });
  return box;
}

/** 别家手牌：一叠牌背。张数由服务端的 `hand_count` 说了算。 */
export function buildHiddenHand(count: number, size: TileSize): DocumentFragment {
  const frag = document.createDocumentFragment();
  const n = Math.max(0, count);
  for (let i = 0; i < n; i++) frag.appendChild(tileEl({ id: null, size, cls: "is-back" }));
  return frag;
}

export function tileText(id: TileId): string {
  const num = id.charAt(0);
  const suit = id.charAt(1);
  const n = num === "0" ? "赤5" : num;
  const s = suit === "m" ? "万" : suit === "p" ? "筒" : suit === "s" ? "索" : "字";
  return `${n}${s}`;
}
