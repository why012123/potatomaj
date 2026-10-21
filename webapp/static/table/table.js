const el = document.getElementById("table");
const W = 1080, H = 1080;

/*
 * 等比缩放 + 居中：牌桌固定 1080×1080，缩到「装得下它的那个盒子」里。
 *
 * 盒子优先取 `#stage`（对战页/直播页的外壳：牌桌只是其中一栏，左边是按钮、
 * 右边是信息栏），没有 `#stage` 就退回整个窗口（沙盘页 index.html 就是这种）。
 */
function fit() {
  const stage = document.getElementById("stage");
  const vw = Math.floor(stage ? stage.clientWidth : window.innerWidth);
  const vh = Math.floor(stage ? stage.clientHeight : window.innerHeight);
  const s = Math.min(vw / W, vh / H, 1);
  const ox = (vw - W * s) / 2;
  const oy = (vh - H * s) / 2;
  el.style.transform = `translate(${ox}px, ${oy}px) scale(${s})`;
}
window.addEventListener("resize", fit);
window.visualViewport?.addEventListener("resize", fit);
fit();

/* ================================================================
 *                        摆 牌 河
 * ================================================================
 *
 * 全部用**屏幕坐标**算，一次算清：第几张牌在哪个像素位置。
 *
 * 规则：
 *   - 六张一排、居中
 *   - 第一排紧贴蓝色核心区的边，后面一排除往外推
 *   - 每边排数：上 1、右 2、下 3、左 3（＋加放区）
 *   - 每边各摆**一张立直宣言牌**（横置），同一排排在它后面的牌整体顶开 5px。
 *     每家**最多立直一次**，所以一条牌河里最多只有一张横置牌。
 *   - 装不下的牌（超过「排数 × 6」）进**加放区**：在牌河**右边**，
 *     隔 10px 的空隙，从**第一排**起接着摆，加放区**一排 4 张**。
 *     立直牌那 5px 的顶开**挤不到加放区**（空隙 10px，比 5px 大一倍）。
 *
 * 之所以不再用「旋转的容器 + CSS 偏移」：那种写法在带 rotate 的祖先下，
 * 局部的 x/y 会在屏幕上互换，「沿牌河」和「朝外」两个方向没法分别表达，
 * 怎么调都有一边是歪的。直接给屏幕坐标就没有这个问题。
 */
const TW = 40;                      // 牌宽（与 CSS 的 --tw 一致）
const TH = 45;                      // 牌高 = 牌宽 × 1.125（标准比例）
const GAP = 2;
const STEP = TW + GAP;              // 相邻两张的中心距
const MID = 6 * TW + 2 * 9;         // 核心区边长 = 258（与 CSS 的 --mid 一致）
const HALF = MID / 2;               // 129
const CX = W / 2, CY = H / 2;
const ROW = 6;                      // 一排六张

const POND = [
  "1m", "9p", "3s", "1z", "9m", "5p", "2s", "6z", "4m", "8p", "1s", "7z",
  "3m", "2p", "9s", "5z", "6m", "4s", "1p", "7m", "2m", "8s", "3p", "9m",
];

/**
 * 摆一条牌河：一排 6 张**居中**，第 N 排往外推 row 个「牌高 + 牌距」。
 *
 * @param host  容器元素
 * @param n     张数
 * @param rows  磊几排（正常牌河最多装 rows × 6 张，多出来的进加放区）
 * @param deg   牌面基准旋转角（侧家 ±90° 才是「朝自己」的）
 * @param at    (out, along, i) => {x,y}：把「往外距离、沿边距离」换算成板内坐标
 * @param riichi 立直宣言牌的下标（-1 = 这家没立直）。它在 CSS 里横置
 *               （牌头朝上一张），自己往后挪半个多出来的宽度，并把
 *               **同一条线**上排在它后面的牌各顶开 TH − TW = 5px。
 */
function pond(host, tiles, rows, deg, at, riichi = -1) {
  host.innerHTML = "";
  /* ★ `tiles` 是**这一家的牌河内容**（牌 id 数组）。传数字 = 沙盘页的老用法
     （按 POND 那份假数据取前 n 张）。以前这里无条件用 `POND[i]`，
     接上真数据之后四家牌河就画得一模一样了 —— 用户一眼看出来的那个 bug。 */
  const ids = typeof tiles === "number"
    ? Array.from({ length: tiles }, (_, k) => POND[k % POND.length])
    : (tiles || []).slice();
  const n = ids.length;

  /*
   * 一张牌落在第几条「线」的哪一格。
   *
   *   牌河：rows 排 × 6 格。
   *   加放区：张数超过 rows × 6 之后，多出来的牌摆在牌河**右边**，
   *          隔着 OVER_GAP 的空隙，**从第一排起**照样六张一排。
   *
   * 两块地方的坐标都是「这条线的起点 + 格号 × 牌距」，所以用一个式子算。
   */
  const CAP = rows * ROW;                        // 牌河装得下的张数
  const OV_ROW = 4;                              // 加放区一排 **4 张**
  const EDGE = (ROW - 1) / 2 * STEP + TW / 2;    // 牌河沿排方向的半宽（= 125）
  /*
   * 牌河与加放区之间的空隙。
   *
   * 这 10px 是**夹在两个硬约束中间**抠出来的：
   *   下限 —— 立直牌顶开后，同一条线上后面的牌各往右挪 5px。
   *           空隙必须比这 5px 大，那 5px 才挤不进加放区（留 5px 余量）。
   *   上限 —— 加放区是往「自己右边」接着摆的，空隙给多大，它就整体
   *           往外多长多少。42px（一个牌距）那版直接顶到旁边那家的
   *           手牌区去了（用户看到的），所以这里压到 10px。
   */
  const OVER_GAP = 10;
  const zoneOf = (k) => (k < CAP ? 0 : 1);       // 0 = 牌河，1 = 加放区
  const lineOf = (k) => (k < CAP
    ? Math.floor(k / ROW)                         // 牌河第 k/6 排
    : Math.floor((k - CAP) / OV_ROW));            // 加放区从第一排重新数，一排 4 张
  const slotOf = (k) => (k < CAP ? k % ROW : (k - CAP) % OV_ROW);
  const baseOf = (k) => (k < CAP
    ? -(ROW - 1) / 2 * STEP                       // 牌河第一格的中心
    : EDGE + OVER_GAP + TW / 2);                  // 加放区第一格的中心

  /* 立直宣言牌在哪条线上：它只顶开**自己这条线**后面的牌。
     加放区是另一块地方（中间隔着 OVER_GAP），一根都不动 ——
     空隙 42px，顶开的 5px 挤不过来。 */
  const rZone = riichi >= 0 ? zoneOf(riichi) : -1;
  const rLine = riichi >= 0 ? lineOf(riichi) : -1;

  for (let i = 0; i < n; i++) {
    const line = lineOf(i);
    const slot = slotOf(i);
    const turned = deg === 90 || deg === -90;
    /* 屏幕上占的矩形：横画就换宽高。这条铁律不能破 ——
       角度转了、矩形没换，牌就会挤到旁边那一格去。 */
    const bw = turned ? TH : TW;
    const bh = turned ? TW : TH;

    /*
     * 立直宣言牌是**横置**的（横过来 90°，见 CSS 的 `.tile.is-side`），
     * 它在「沿这一排的方向」上比普通牌多占 TH − TW = 5px。
     *
     * 两段偏移，缺一不可：
     *
     * 1) `self` —— 它**自己往后挪 2.5px**（半个 5px）。不挪的话，
     *    它比普通牌宽出来的 5px 会左右各分 2.5px，**左边那 2.5px 直接
     *    怼进上一张牌里**（上一张的右边 539，它的左边 538.5 —— 重叠了
     *    0.5px，实测出来的）。挪完它的**顶边正好落在普通牌的起手位**上，
     *    离上一张只有标准牌距 2px，跟这一排其它缝一样宽，绝不重合。
     *
     * 2) `pushed` —— **同一条线**上排在它后面的牌各顶开 5px（它多占的
     *    5px 得从后面让出来）。它前面的牌、别的线、加放区，一根都不动。
     */
    const self = i === riichi ? (TH - TW) / 2 : 0;
    const pushed = riichi >= 0 && i > riichi
      && zoneOf(i) === rZone && line === rLine ? TH - TW : 0;

    const out = line * (TH + GAP);
    const along = baseOf(i) + slot * STEP + self + pushed;
    const { x, y } = at(out, along, i);

    const el = document.createElement("div");
    /* 横置的牌只加一个类：盒子的宽高仍然是上面那对值（JS 给的 left/top
       是按**没转**的盒子算的），转的是整块，牌面跟着一起转。 */
    el.className = i === riichi ? "tile is-side" : "tile";
    /* 落在哪块地方的第几条线第几格 —— 自检脚本按它核对，不靠反推 */
    el.dataset.zone = zoneOf(i) ? "over" : "pond";
    el.dataset.line = String(line);
    el.dataset.slot = String(slot);
    el.style.width = bw + "px";
    el.style.height = bh + "px";
    el.style.left = x - bw / 2 + "px";
    el.style.top = y - bh / 2 + "px";
    if (deg) el.style.setProperty("--face", `${deg}deg`);
    const img = document.createElement("img");
    img.src = `/tiles/${ids[i]}.svg`;
    img.alt = ids[i];
    el.appendChild(img);
    host.appendChild(el);
  }
}

/** 一排六张、居中：第 col 张相对中线的偏移。 */
const along = (col) => (col - (ROW - 1) / 2) * STEP;

/*
 * ================================================================
 *                    四个起始牌位（硬算）
 * ================================================================
 *
 * 我先把要落的位置**实测**出来，再让算式去凑这四个数 ——
 * 不反过来推。
 *
 * 目标（板内坐标 = 离板子左上角多少像素）：
 *     下家（自己）  (540, 690)   牌面  0°
 *     上家（对家）  (540, 390)   牌面 180°
 *     右家（下家）  (876, 540)   牌面 -90°
 *     左家（上家）  (204, 540)   牌面  90°
 *
 * ★ 容器 `#table` 有 26px 的边框，所以 `.pond` 的原点在板内是 **(26,26)**，
 *   不是 (0,0)。一开始忘了这一步，四个起始位全体偏 (26,26)
 *   （实测变成 (461,737) 这种），看着像「又歪了」。
 *   所以给容器的位移要**减掉这个边框**。
 */
const BORDER = 26;                  // 与 CSS 里 #table 的 border 一致
const PITCH = TW + GAP;             // 42：同一排里相邻两张的中心距

/*
 * 四个起始点（板内目标坐标 → 容器内坐标）。
 *
 * 算式两行，写清楚免得算错：
 *     核心区边 = 540 ± 129        →  411 / 669
 *     容器内   = 板内 − 26        →  再各减 26（板子有 26px 边框）
 *
 * ★ 牌河**贴着核心区的外边缘**，不再往外让 —— 之前按 `边 ± 42` 给的，
 *   那是给核心区自己的「边框 + 立直棒槽 + 点数条」留的位，结果牌河
 *   离核心区有 42px 的空隙，看着没贴边（用户反馈）。
 *   现在第一张牌的**近边**正好压在核心区的外边缘上：
 *       牌心 = 边 ± 半个牌尺寸（横排 ± 牌高/2，竖排 ± 牌宽/2）
 */
const EDGE_LO = 540 - 129;          // 411
const EDGE_HI = 540 + 129;          // 669
const TW2 = TW / 2;                 // 20
const TH2 = TH / 2;                 // 21

const START = {
  // 下：第一排的上边缘贴核心区下边缘 → 牌心 = 边 + 牌高/2
  bottom: { x: 540 - BORDER, y: EDGE_HI + TH2 - BORDER },
  top: { x: 540 - BORDER, y: EDGE_LO - TH2 - BORDER },
  // 右：第一列的左边缘贴核心区右边缘 → 牌心 = 边 + 牌宽/2（竖排用半个牌宽）
  right: { x: EDGE_HI + TW2 - BORDER, y: 540 - BORDER },
  left: { x: EDGE_LO - TW2 - BORDER, y: 540 - BORDER },
};
// 排与排之间往外的推进量 = 牌高 + 牌距 = 47。
// （`PITCH` / `ROWPITCH` 这两个常量目前没被引用 —— 实际牌距写在 `pond()`
//   里，这里只留个名字和正确的数，免得下次照着旧的「44」去改。）
const ROWPITCH = TH + GAP;          // 47

/*
 * ==================== 四个方向的坐标换算 ====================
 *
 * `at(out, along)` 收到「往外距离」和「沿边距离」（along 已经居中），
 * 换算成板内坐标。out = 0 就是紧贴核心区外边缘的那一排。
 *
 * ★★ 铁律：**四家都只认自己的视角** —— 牌河从自己看过去的左上角起，
 *    先往右磊满 6 张，再换到下一排（从上到下）。四家朝向不同，
 *    所以「自己的左→右」「自己的上→下」在**屏幕上指向四个不同的方向**：
 *
 *       座位      面朝      自己的左→右 = 屏幕     自己的上→下 = 屏幕
 *       下（自己）  上（北）      +x（往右）            +y（往下，朝自己）
 *       上（对家）  下（南）      −x（往左）            −y（往上，朝自己）
 *       左（上家）  右（东）      +y（往下）            −x（往左，朝自己）
 *       右（下家）  左（西）      −y（往上）            +x（往右，朝自己）
 *
 *    「自己的左是哪一边」按「面向北时左手朝西」推：
 *       面朝北 → 左 = 屏幕左      面朝南 → 左 = 屏幕右
 *       面朝东 → 左 = 屏幕上      面朝西 → 左 = 屏幕下
 *
 *    ★ 左右两家**朝向相反，沿着牌河走的方向必然也相反**，一家 +y
 *      另一家就得 −y。写成同一个符号，等于把其中一家的第一张牌
 *      放到了自己视角的右上角 —— 整条牌河变成从右往左磊（踩过：
 *      下家原本就是 `y + a`，跟左家一模一样，四家里只有它反了）。
 *
 *    「自己的上→下」= 一排排往自己身前推：第一排贴核心区，
 *      后面的排一排排往外（朝自己）挪 —— out 的符号就是上面那一列。
 */
const atBottom = (out, a) => ({ x: START.bottom.x + a, y: START.bottom.y + out });
// 对家整体转 180°：自己的右是屏幕左，自己的下是屏幕上
const atTop = (out, a) => ({ x: START.top.x - a, y: START.top.y - out });
// 下家面朝西：自己的右是屏幕上（−y），自己的下是屏幕右（+x）
const atRight = (out, a) => ({ x: START.right.x + out, y: START.right.y - a });
// 上家面朝东：自己的右是屏幕下（+y），自己的下是屏幕左（−x）
const atLeft = (out, a) => ({ x: START.left.x - out, y: START.left.y + a });

/* ================================================================
 *                      四家的手牌 + 副露
 * ================================================================
 *
 * 一条手牌 = 13 张**紧挨着** + 第 13/14 张之间 **0.5 牌宽**的空隙 +
 * 摸到的那张。副露（吃/碰/杠）摆在**自己家的最右边**，
 * 再来一组就摆**上一组的左侧**。摆法见 CSS 里 `.hand` 那一段。
 *
 * ★★ 两条铁律：
 *   1. **手牌第一张的位置永远不变** —— 吃碰杠之后手牌从 13 张掉到 10 张
 *      （每副露少 3 张），第一张还钉在原地，手牌**不重新居中**。
 *   2. **副露永远贴自己那面墙的最右边**，第二组在第一组左侧；
 *      摸到的那张始终紧贴**非副露部分的右端**。
 *
 * 尺寸：**手牌整体比牌河大**（1.2 倍），其中**主视角（下侧）再大 12%**
 * （跟正式前端的 lg 一档一致）；副露的牌再小一档（手牌的 0.6）。
 * 其他三家是**牌背**（橙黄色）。
 *
 * 侧家的牌身要**横过来**（宽高互换）：牌面那张图还是竖的，
 * 转 90° 之后正好填满横过来的牌身，不会溢出、也不用负 margin。
 */
const HAND_SCALE = 1.2;                  // 手牌比牌河大这一档（四家都用）
const VIEW_SCALE = 1.12;                 // 主视角**再**大一点
const OTHER_W = TW * HAND_SCALE;         // 48：其他三家的牌宽
const MY_W = OTHER_W * VIEW_SCALE;       // 53.76：自己的牌宽
const MELD_RATIO = 0.9;                  // 副露的牌：比手牌**略小**
const MELD_GAP = 0.35;                   // 组与组之间空这么多个「副露牌宽」

/* ---------------- 牌背：橙黄色 ----------------
 * 公共素材 `tiles/back.svg` 是绿的；这里按同一套画法（白边 + 内框 + 圈）
 * 换成橙黄渐变，直接内联成 data URI —— 预览的配色不往公共素材目录里塞。
 */
const BACK_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 400">'
  + '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
  + '<stop offset="0" stop-color="#f7cf63"/>'
  + '<stop offset="0.5" stop-color="#e3a62c"/>'
  + '<stop offset="1" stop-color="#b9761a"/></linearGradient></defs>'
  + '<rect x="0" y="0" width="300" height="400" rx="26" fill="#fbf8f1"/>'
  + '<rect x="4" y="4" width="292" height="392" rx="24" fill="none"'
  + ' stroke="#ddd6c8" stroke-width="3"/>'
  + '<rect x="22" y="22" width="256" height="356" rx="16" fill="url(#g)"/>'
  + '<rect x="22" y="22" width="256" height="356" rx="16" fill="none"'
  + ' stroke="rgba(0,0,0,.30)" stroke-width="3"/>'
  + '<rect x="36" y="36" width="228" height="328" rx="10" fill="none"'
  + ' stroke="rgba(255,255,255,.32)" stroke-width="2"/>'
  + '<circle cx="150" cy="200" r="34" fill="none"'
  + ' stroke="rgba(255,255,255,.26)" stroke-width="3"/></svg>';
const BACK_SRC = "data:image/svg+xml;charset=utf-8,"
  + encodeURIComponent(BACK_SVG);

/* 四家：牌多大 / 牌面朝哪 / 牌身在屏幕上是否横过来 */
const SEAT = {
  bottom: { base: MY_W, face: 0, turned: false },
  top: { base: OTHER_W, face: 180, turned: false },
  left: { base: OTHER_W, face: 90, turned: true },
  right: { base: OTHER_W, face: -90, turned: true },
};

/**
 * 造一张牌。
 *
 * @param base   这张牌**自己**的宽（高 = 宽 × 1.125）
 * @param face   牌面朝向（对家 180、上家 90、下家 −90、自己 0）
 * @param turned 牌身**在屏幕上**是否横过来（侧家、副露里被叫的那张）
 * @param id     牌面 id；null = 牌背
 */
function tile(base, face, turned, id) {
  const baseH = base * 1.125;
  const el = document.createElement("div");
  el.className = "tile";
  /* 牌身宽高互换 = 这张牌横过来；牌面图**始终是竖的** base × baseH，
     转 face 度之后正好嵌进牌身（见 CSS 里 `.hand .tile img` 的说明）。 */
  el.style.width = (turned ? baseH : base) + "px";
  el.style.height = (turned ? base : baseH) + "px";
  const img = document.createElement("img");
  img.src = id ? `/tiles/${id}.svg` : BACK_SRC;
  img.alt = id ?? "牌背";
  img.style.width = base + "px";
  img.style.height = baseH + "px";
  img.style.transform = `translate(-50%, -50%) rotate(${face}deg)`;
  el.appendChild(img);
  return el;
}

/**
 * 摆一个座位。
 *
 *   手牌盒（定宽 14.5 张牌、居中、贴墙）：手牌（13 − 3×副露数 张）
 *   → 0.5 牌宽的空隙 → 摸到的那张。**盒子位置永远不动。**
 *   副露另起一条，挂在**自己那面墙的最外圈**：第一组在最右，
 *   后面的往左排（顺序见 CSS 的 `.melds--*`）。
 *
 * @param name    座位名（bottom / top / left / right）
 * @param handIds 手牌（null = 牌背）
 * @param drawn   摸到的那张（null = 牌背）
 * @param melds   副露列表，按时间顺序；每项 {kind, tiles, from}
 */
function seat(name, handIds, drawn, melds) {
  const S = SEAT[name];
  const host = document.getElementById("hand-" + name);
  host.style.setProperty("--hw", S.base + "px");   // 14.5 张牌条的定宽靠它算
  host.replaceChildren();
  handIds.forEach((id, i) => {
    const el = tile(S.base, S.face, S.turned, id);
    el.dataset.role = "hand";
    el.dataset.idx = String(i);
    host.appendChild(el);
  });
  /* 空隙 = 0.5 牌宽。侧家整条是竖着的，所以量的是「高」。
     `drawn === false` = 这家手上没摸牌（已经打出去了）—— 空隙和第 14 格都不画。 */
  if (drawn !== false) {
    const gap = document.createElement("div");
    gap.className = "hand__gap";
    gap.style[S.turned ? "height" : "width"] = S.base * 0.5 + "px";
    host.appendChild(gap);
    const last = tile(S.base, S.face, S.turned, drawn);   // 摸到的：非副露部分的最右
    last.dataset.role = "drawn";
    host.appendChild(last);
  }

  const box = document.getElementById("melds-" + name);
  box.style.gap = (S.base * MELD_RATIO * MELD_GAP) + "px";
  box.replaceChildren();
  melds.forEach((m, k) => box.appendChild(meldEl(S, m, k)));
}

/**
 * 拔北（三麻）：把拔出来的北牌亮在自己那面墙的第 14 格上方。
 *
 * @param name 座位（bottom / top / left / right）
 * @param ids  拔出来的牌（三麻里恒为 `4z` = 北），**按拔的先后顺序**
 *
 * 位置在 CSS 的 `.kita--*`：锚在**固定的第 14 格**上（手牌盒子定宽，这一格
 * 永不移动），略斜 −14°，**从右往左**排 —— DOM 里第一个落在自己右手边。
 */
function kita(name, ids) {
  const S = SEAT[name];
  const host = document.getElementById("kita-" + name);
  if (!host) return;
  host.style.setProperty("--hw", S.base + "px");
  host.replaceChildren();
  ids.forEach((id) => {
    const el = tile(S.base, S.face, S.turned, id);
    el.dataset.role = "kita";
    host.appendChild(el);
  });
}

/* ======================================================================
 *                  服务端快照 → 牌桌（直播页 / 对战页共用）
 * ====================================================================
 *
 * 输入就是 `webapp/server/protocol.py` 那一份 `state` 消息。**两家共用**
 * 这一段，区别只在页面上那些按钮和面板：
 *
 *   直播页（live.html）：纯 AI 观战，只画；`state.actions` 不点。
 *   对战页（play.html）：自己坐一家，把 `actions.options` 画成按钮。
 */
const DIRS = { 3: ["bottom", "right", "left"], 4: ["bottom", "right", "top", "left"] };
const DEG = { bottom: 0, top: 180, right: -90, left: 90 };
const SIDES = ["bottom", "top", "left", "right"];

/**
 * 一个座位在界面上叫什么。
 *
 * ★ AI 座位**写具体模型名**（`maj-flashv1.1` 这种），不写「AI·速攻」这种
 *   风格标签 —— 服务端给的 `name` 是给人看的风格名，牌桌上要认的是模型。
 *   模型名太长时掐掉版本号尾巴：`maj-normal4v1.2` → `maj-normal4`。
 *   （用户点名要的：maj-N / maj-M / maj-F / maj-H 这类。）
 */
function seatLabel(s) {
  if (!s) return "";
  if (s.kind === "ai") {
    const m = String(s.ai || "");
    return m ? m.replace(/v\d+(\.\d+)*$/i, "") : "AI";
  }
  return s.name || ("P" + s.seat);
}

/** 服务端那套副露 → 摆牌函数要的形状 {kind, tiles, from}。
 *  `from` 是**谁打的**（自己视角：left 上家 / across 对家 / right 下家），
 *  横放那张摆哪一格由它决定 —— 日麻规矩：上家 → 最左、对家 → 中间、下家 → 最右。 */
function meldModel(st, s, m) {
  const n = st.seats.length;
  const kind = (m.kind === "kan" || m.kind === "kakan") ? "minkan" : m.kind;
  let from = null;
  if (kind !== "ankan" && typeof m.from === "number" && m.from >= 0) {
    const rel = (m.from - s.seat + n) % n;   // 1 = 下家、2 = 对家（三麻是上家）、3 = 上家
    from = rel === 1 ? "right" : rel === 2 ? (n === 3 ? "left" : "across") : "left";
  }
  return { kind, tiles: m.tiles, from };
}

/** 某一家那一套容器清空（三麻用不到对家那套）。 */
function clearSide(d) {
  for (const p of ["pond-", "hand-", "melds-", "kita-"]) {
    const el = document.getElementById(p + d);
    if (el) el.replaceChildren();
  }
  setSeatName(d, "");
  const side = document.getElementById("side-" + d);
  const slot = document.getElementById("slot-" + d);
  if (side) side.style.visibility = "hidden";
  if (slot) slot.style.visibility = "hidden";
}

/**
 * 这一家的名牌：写在自己那条边的**右端**桌布上（浅色）。
 *
 * 没有就**现建一个** —— 摆牌代码四个页面共用，不想让每个页面的 HTML 里
 * 都手抄四行空 div（漏一个就是「这家没名字」）。
 */
function seatNameEl(d) {
  let el = document.getElementById("name-" + d);
  if (!el) {
    const host = document.getElementById("table");
    if (!host) return null;
    el = document.createElement("div");
    el.id = "name-" + d;
    el.className = "seatname seatname--" + d;
    host.appendChild(el);
  }
  return el;
}

function setSeatName(d, text) {
  const el = seatNameEl(d);
  if (el) el.textContent = text == null ? "" : String(text);
}

/*
 * 宝牌指示牌 → 宝牌本身 = 「顺位下一张」：
 *   数牌 1→2→…→9→1（**三麻的万子也走这个循环**，9m 的下一位是 1m），
 *   风牌 东→南→西→北→东，三元牌 白→發→中→白。
 */
const DORA_NEXT = {
  "1m": "2m", "2m": "3m", "3m": "4m", "4m": "5m", "5m": "6m", "6m": "7m",
  "7m": "8m", "8m": "9m", "9m": "1m",
  "1p": "2p", "2p": "3p", "3p": "4p", "4p": "5p", "5p": "6p", "6p": "7p",
  "7p": "8p", "8p": "9p", "9p": "1p",
  "1s": "2s", "2s": "3s", "3s": "4s", "4s": "5s", "5s": "6s", "6s": "7s",
  "7s": "8s", "8s": "9s", "9s": "1s",
  "1z": "2z", "2z": "3z", "3z": "4z", "4z": "1z",
  "5z": "6z", "6z": "7z", "7z": "5z",
};

/** 这一局的宝牌集合（赤 5 在协议里是 `5x` + 单独的 red 标记，所以不用另算）。 */
function doraSet(st) {
  const out = new Set();
  const ind = (st.round && st.round.dora) || [];
  for (const id of ind) {
    const next = DORA_NEXT[id];
    if (next) out.add(next);
    if (id === "5m" || id === "5p" || id === "5s") out.add("0" + id[1]);  // 赤 5 也算
  }
  return out;
}

/* 上一帧各家牌河张数：用来判断「四家合起来最新打出的那张」是谁打的 */
let prevPondLen = [];
/* 现在牌河上最新那张是谁的、当时他那条牌河多长（红框跟着它） */
let lastDiscard = { seat: -1, len: 0 };

/** 别家打牌时，他那条立直棒栏也闪一下（AI 出牌在协议里没有 actions）。 */
function flashSlot(d) {
  const slot = document.getElementById("slot-" + d);
  if (!slot) return;
  slot.classList.add("is-turn");
  window.clearTimeout(flashSlot._t);
  flashSlot._t = window.setTimeout(() => slot.classList.remove("is-turn"), 1200);
}

/**
 * 把一份 state 画到牌桌上。返回本次真正画出来的座位名（三麻就是 3 个）。
 *
 * @param st   服务端 state 消息
 * @param me   自己坐的座位号（−1 = 观战）—— 决定「谁在下面」
 */
function renderState(st, me) {
  const n = st.seats.length;
  const dirs = DIRS[n] || DIRS[3];
  const viewer = typeof me === "number" ? me : (st.viewer ? st.viewer.seat : -1);
  const base = viewer < 0 ? 0 : viewer;      // 观战者按 0 号位摆
  const at = { bottom: atBottom, top: atTop, right: atRight, left: atLeft };
  const live = new Set();

  for (const s of st.seats) {
    const d = dirs[(s.seat - base + n) % n];
    if (!d) continue;
    live.add(d);

    // 牌河：一排 6 张、最多 3 排，装不下的进加放区（都在 pond() 里）
    const riichiAt = s.riichi_tile ? s.pond.lastIndexOf(s.riichi_tile) : -1;
    const rows = Math.max(1, Math.min(3, Math.ceil(s.pond.length / 6)));
    pond(document.getElementById("pond-" + d), s.pond, rows, DEG[d],
         at[d], riichiAt);

    /* 手牌：自己 / 全亮观战给的是牌面数组；看不到就只画张数。
       ★ `hand_count` 是**含刚摸那张**的：余 2（14/11/8…）= 手上正摸着牌，
         那张单独画在第 14 格；否则传 `false` —— 传 null 会多画一张牌背。 */
    let ids, drawn;
    if (Array.isArray(s.hand)) {
      const hand = s.hand.slice();
      if (s.drawn) {
        const i = hand.lastIndexOf(s.drawn);
        if (i >= 0) hand.splice(i, 1);
      }
      ids = hand;
      drawn = s.drawn || false;
    } else {
      const hasDraw = s.hand_count % 3 === 2;
      ids = Array(Math.max(0, s.hand_count - (hasDraw ? 1 : 0))).fill(null);
      drawn = hasDraw ? null : false;
    }
    seat(d, ids, drawn, (s.melds || []).map((m) => meldModel(st, s, m)));
    kita(d, Array(s.kita || 0).fill("4z"));      // 拔北：三麻才有，四麻恒 0
    setSeatName(d, seatLabel(s));                // 桌布上的名牌（AI 写模型名）

    // 点数条 + 立直棒槽
    const side = document.getElementById("side-" + d);
    const slot = document.getElementById("slot-" + d);
    side.style.visibility = "visible";
    slot.style.visibility = "visible";
    side.querySelector("span").textContent = s.seat_wind || "?";
    side.querySelector("b").textContent = s.score;
    slot.classList.toggle("is-empty", !s.riichi);
    slot.classList.toggle("is-turn", !!st.actions && st.actions.seat === s.seat
                          && st.phase === "playing");
    if (s.riichi && !slot.firstChild) slot.appendChild(document.createElement("i"));
    if (!s.riichi) slot.replaceChildren();
  }

  // 用不到的那两家清空藏起来（不然会留着上一局或上一桌的牌）
  for (const d of SIDES) if (!live.has(d)) clearSide(d);

  /* ---- 牌桌上的小装饰（每帧重算，全是幂等的 class 操作）----
   *   1) 宝牌牌面镀一层金；
   *   2) 四家合起来**最新打出的那张**加红框（就是现在能被鸣的那张）；
   *   3) 打牌那家的立直棒栏闪一下（AI 出牌走这条，协议里没有 actions）。
   */
  const dora = doraSet(st);
  document.querySelectorAll(".tile img").forEach((img) => {
    const t = img.closest(".tile");
    if (t) t.classList.toggle("is-dora", dora.has(img.getAttribute("alt")));
  });
  let grew = -1;
  st.seats.forEach((s, i) => {
    if ((s.pond || []).length > (prevPondLen[i] === undefined ? 0 : prevPondLen[i])) {
      grew = i;
    }
  });
  prevPondLen = st.seats.map((s) => (s.pond || []).length);
  if (grew >= 0) {
    lastDiscard = { seat: st.seats[grew].seat, len: (st.seats[grew].pond || []).length };
    if (st.seats[grew].seat !== base) flashSlot(dirs[(st.seats[grew].seat - base + n) % n]);
  }
  /* 红框要**留住**：只有那家牌河还是当时的长度（说明之后没人再打）才继续标。
     变短 = 那张被鸣走了（牌都不在牌河里了），标记自然作废。 */
  const markAt = st.seats.findIndex((s) => s.seat === lastDiscard.seat);
  if (markAt >= 0) {
    const seatState = st.seats[markAt];
    const len = (seatState.pond || []).length;
    if (len === lastDiscard.len) {
      const d = dirs[(seatState.seat - base + n) % n];
      const host = document.getElementById("pond-" + d);
      const tis = host ? host.querySelectorAll(".tile") : [];
      if (tis.length) tis[tis.length - 1].classList.add("is-last");
    } else if (len < lastDiscard.len) {
      lastDiscard = { seat: -1, len: 0 };
    }
  }

  // 正中：第几局 / 剩余牌数 / 几本场
  const r = st.round;
  if (r) {
    const c = document.getElementById("center");
    c.querySelector(".mid-round").textContent = r.round_wind + r.hand_no + "局";
    c.querySelector(".mid-wall").textContent = String(r.wall);
    /* 本场 = **连庄次数**（服务端 `round.honba`）。以前拿 `sticks`（立直棒）
       顶替，既不是本场、又会在有人立直时乱跳。 */
    c.querySelector(".mid-honba").textContent = (r.honba || 0) + " 本场";
  }
  return [...live];
}

/**
 * 一组副露：3 张（吃/碰）或 4 张（杠）。
 *
 * ★ 横放的那张**放在哪一格**，是拿哪一家的牌决定的（日麻规矩）：
 *     上家（自己左手边）→ 最左那张横放
 *     对家（正对面）  → 中间那张横放
 *     下家（自己右手边）→ 最右那张横放
 *   吃只能吃上家，所以吃固定是**最左**那张横放。
 *   暗杠没有「被叫的那张」，按牌桌规矩画成「亮—暗—暗—亮」。
 */
function meldEl(S, m, k) {
  const box = document.createElement("div");
  box.className = "meld";
  box.dataset.kind = m.kind;
  box.dataset.meld = String(k);
  const w = S.base * MELD_RATIO;
  /* 横放那张的下标：3 张 = 左/中/右，4 张（杠）= 最左 / 第 2 张 / 最右。
     暗杠没有「被叫的那张」，`from` 给 null → 一张都不横放。 */
  const n = m.tiles.length;
  const at = m.from === "left" ? 0
    : m.from === "across" ? 1
      : m.from === "right" ? (n === 3 ? 2 : 3)
        : -1;
  m.tiles.forEach((id, i) => {
    const rot = i === at;                             // 被叫的那张横放
    const back = m.kind === "ankan" && (i === 1 || i === 2);
    /* 横放的牌：牌身在屏幕上跟同组其它牌**反着来**（宽高互换）。
       侧家本来就横着，再横一次就竖回来了 —— 所以这里是「异或」。 */
    const el = tile(w, S.face + (rot ? 90 : 0), S.turned !== rot,
                    back ? null : id);
    el.dataset.role = "meld";
    el.dataset.meld = String(k);
    el.dataset.called = rot ? "1" : "0";
    el.dataset.back = back ? "1" : "0";
    box.appendChild(el);
  });
  return box;
}
