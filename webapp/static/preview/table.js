/*
 * 纯静态牌桌预览 —— **不连后端**，数据全是写死的。
 *
 * ============================================================
 *  这一版的核心思路（前几版乱动的根因和解法）
 * ============================================================
 *
 * 天凤 / 雀魂的牌桌，每一家看到的都是「自己的手牌在下方、从左往右排」。
 * 也就是说**每一家都有自己的局部坐标系**：
 *
 *      local x → 沿手牌方向（自己视角里从左往右）
 *      local y → 从手牌指向桌心
 *
 * 侧家之所以要转，只是因为这个局部坐标系在屏幕上转了 ±90°。
 * 前几版就是在「旋转后的坐标系」里直接写 CSS 偏移，于是：
 *   - 局部 x/y 在屏幕上互换，`left` / `bottom` 表达不清；
 *   - 同一条边映射到哪个屏幕边还分左右，两家落点完全不同；
 *   - `getBoundingClientRect()` 返回旋转后的包围盒，宽高互换，
 *     任何「量了再补」的修正都会越修越偏。
 *
 * 现在的做法把这件事一次说清楚：
 *   1. **先在自己的局部坐标系里排版** —— 手牌是从 (0,0) 往 +x 排的一行，
 *      副露在手牌右边一组组往 -y 磊，牌河是 6 列 × 3 行的固定格子。
 *      这一层完全不关心屏幕，四家共用同一段代码。
 *   2. **再把整个平面转过去** —— 一家的所有牌包在一个 `div.plane` 里，
 *      对 plane 施加 `translate(锚点) rotate(角度)`。局部坐标只出现一次，
 *      之后就交给矩阵，不再有任何「按边距凑」的写法。
 *
 * 这样：
 *   - 牌河第一张 = 局部的 (0,0)，**钉死在锚点上**，后面的按格子往后排；
 *   - 四家的排版代码完全一样，左右对称不需要打补丁；
 *   - 「只有手牌会动」也成立：副露和牌河的局部坐标是常量。
 */

// 舞台尺寸（定尺，按窗口等比缩放）
const W = 1920;
const H = 1080;

// 牌尺寸 —— 全场唯一的尺寸源头。真牌 26x28mm，所以高略大于宽。
const TW = 62;
const TH = Math.round(TW / 0.96);
const GAP = Math.max(1, Math.round(TW * 0.045));

/** 牌距（相邻两张牌的中心距）。 */
const STEP = TW + GAP;

const board = document.getElementById("board");
const layer = document.getElementById("layer");

function tileEl(id, cls, deg = 0, w = TW, h = TH) {
  const el = document.createElement("div");
  el.className = "tile" + (cls ? " " + cls : "");
  el.style.width = w + "px";
  el.style.height = h + "px";
  // 牌面朝向：每家看到的牌都应该是「正的」，所以每家的牌都要跟着转。
  // 这一步跟平面旋转是两件事：平面旋转决定**摆在哪**，这里决定**图案朝哪**。
  if (deg) el.style.transform = `rotate(${deg}deg)`;
  const img = document.createElement("img");
  img.src = id ? `/tiles/${id}.svg` : "/tiles/back.svg";
  img.alt = id || "牌背";
  img.draggable = false;
  el.appendChild(img);
  if (id && id.startsWith("0")) el.classList.add("is-red");
  return el;
}

/**
 * 摆一家。
 *
 * @param cfg.origin 这一家的**局部原点**在屏幕上的位置，也就是手牌第一张的
 *                   中心（局部坐标 (0,0) 落在这里）
 * @param cfg.deg    这一家的局部坐标系相对屏幕转了多少度
 *                   （自己 0、对家 180、下家 -90、上家 90）
 *
 * 局部坐标系约定（四家完全一致）：
 *   +x = 沿手牌方向（自己视角里从左往右）
 *   +y = 从手牌指向桌心
 * 所有偏移都在这个坐标系里写，**不出现任何屏幕方向**。
 */
function seat(cfg) {
  const { origin, deg } = cfg;
  // 牌面要转多少度才朝着这一家的玩家（和 `deg` 是同一套数，但用途不同：
  // `deg` 决定平面摆在屏幕哪，`face` 决定图案朝哪）
  const face = cfg.face ?? deg;

  // 一个 plane 装下这一家的全部家具。`transform-origin: 0 0` 让
  // 「把局部原点搬到 origin」这件事变成一次干净的平移。
  const plane = document.createElement("div");
  plane.className = "plane";
  plane.style.transformOrigin = "0 0";
  plane.style.transform =
    `translate(${origin.x}px, ${origin.y}px) rotate(${deg}deg)`;

  /**
   * 局部位移 (u, v) → 往 plane 里放一张牌。
   *
   * `u` / `v` 是这张牌**中心**相对局部原点的偏移；`w` / `h` 是它在局部坐标里
   * 占的宽高（横放的牌宽高互换）。**「居中」在这里做掉**（减掉半个宽高），
   * CSS 那边就只剩「宽高 + 旋转」，不掺 `translate` 属性 ——
   * 两套位移机制混在一起，在带旋转的元素上实测会整体偏移。
   */
  const put = (u, v, id, cls, extraDeg = 0, w = TW, h = TH) => {
    const el = tileEl(id ?? null, cls, face + extraDeg, w, h);
    el.style.left = u - w / 2 + "px";
    el.style.top = v - h / 2 + "px";
    plane.appendChild(el);
    return el;
  };

  // ---- 手牌：从局部原点沿 +x 排一行。最后摸的那张单独隔开一道缝。
  const hand = cfg.hand ?? [];
  const drawn = cfg.drawn ?? null;
  const drawnGap = TW * 0.42; // 摸牌的缝，比普通牌距宽一倍多，一眼能看出来
  hand.forEach((id, i) => {
    // 有摸牌时，理好的 13 张先排，缝＋摸牌接在后面
    put(i * STEP, 0, id);
  });
  if (drawn) {
    const u = (hand.length - 1) * STEP + drawnGap + TW;
    put(u, 0, drawn, "is-drawn");
  }
  const handRight = (hand.length - 1) * STEP + (drawn ? drawnGap + TW : 0);

  // ---- 副露：钉在**手牌右边**，一组贴一组地往 -y（远离桌心）磊。
  //      「第二个磊在第一个上方」在局部就是 y 更小 —— 转过去之后
  //      在屏幕上自然就是「上面那一组」，四家一致。
  const melds = cfg.melds ?? [];
  const meldX = handRight + STEP * 0.7; // 手牌右边让开一点
  melds.forEach((meld, mi) => {
    const v = -mi * (TH + GAP * 3); // 第 mi 组：一组一组往外磊
    let u = meldX;
    meld.forEach((id, i) => {
      // 碰 / 明杠里被叫的那张横放（吃是顺子，横过来反而看不清）
      const called = meld.called === i && meld.kind !== "chi";
      const w = called ? TH : TW;
      const h = called ? TW : TH;
      put(u + w / 2, v, id, called ? "is-called" : "", called ? 90 : 0, w, h);
      u += w + 1;
    });
  });

  // ---- 牌河：**6 列 × 3 行的固定格子**，第一张钉在 (pondU, pondV)。
  //      第 i 张永远落在同一格 —— 不会因为总数变化而重排。
  const pond = cfg.pond ?? [];
  const pondU = cfg.pondU ?? 0; // 固定值，跟手牌张数无关
  const pondV = cfg.pondV ?? TH * 1.6;
  pond.forEach((id, i) => {
    const col = i % 6;
    const row = Math.floor(i / 6);
    const riichi = cfg.riichiAt === i;
    const w = riichi ? TH : TW;
    const h = riichi ? TW : TH;
    put(pondU + col * STEP, pondV + row * (TH + GAP), id,
        riichi ? "is-riichi" : "", riichi ? 90 : 0, w, h);
  });

  // ---- 名牌：挂到手牌上方的固定位置
  const plate = document.createElement("div");
  plate.className = "plate";
  plate.innerHTML =
    `<span class="wind">${cfg.wind}</span>` +
    `<span class="name">${cfg.name}</span>` +
    `<span class="score">${cfg.score}</span>` +
    (cfg.dealer ? `<i class="bg bg--dealer">亲</i>` : "") +
    (cfg.riichi ? `<i class="bg bg--riichi">立直</i>` : "") +
    (cfg.ai ? `<i class="bg bg--ai">AI</i>` : "");
  plate.style.left = (cfg.plateU ?? STEP * 6.5) + "px";
  plate.style.top = (cfg.plateV ?? -TH * 1.5) + "px";
  plane.appendChild(plate);

  layer.appendChild(plane);
}

// ------------------------------------------------------------------ 假数据

const HAND4 = [
  "1m", "3m", "4m", "7m", "4p", "7p", "9p", "2s", "7s", "4z", "5z", "5z", "6z",
];
const POND_LONG = [
  "1m", "9p", "3s", "1z", "9m", "5p", "2s", "6z", "4m", "8p", "1s", "7z",
  "3m", "2p", "9s", "5z", "6m", "4s", "1p", "7m", "2m", "8s", "3p", "9m",
];

/**
 * 四个座位的锚点。
 *
 * 自己（下）和对家（上）：手牌在屏幕上是横的，所以 origin 就是「手牌第一张
 * 的中心」，四家都用同一个规矩 —— **origin 永远是手牌第一张的中心，
 * 各家的局部 +x 从那里开始**。这样手牌长度变了，牌河也不会动
 * （牌河的位置只跟 origin 和固定偏移有关）。
 */
function draw() {
  layer.innerHTML = "";
  const cx = W / 2;
  const cy = H / 2;
  const HAND_PAD = 92; // 手牌离桌沿的距离

  const seats = [
    {
      // ---- 自己（下方）：局部 x 向右，局部 +y 朝上（桌心在屏幕上方）
      name: "我", wind: "东", score: 25000, dealer: true, ai: false,
      origin: { x: cx - 7 * STEP, y: H - HAND_PAD }, deg: 0, face: 0,
      hand: HAND4, drawn: "2m", pond: POND_LONG.slice(0, 14),
      pondU: 0, pondV: -TH * 2.6,
      plateU: 7 * STEP, plateV: -TH * 1.3,
    },
    {
      /*
       * ---- 对家（上方）。
       *
       * ★ `deg` 和 `face` 在这里**故意分开**，这是这一版想清楚的一件事：
       *   - `deg` 决定这一家的平面在屏幕上怎么摆（手牌横着、桌心在下），
       *     它和对家是**镜像**关系，不是旋转 180° —— 用 180° 的话
       *     「+y 朝桌心」会翻过去，牌河和副露全跑到屏幕外面；
       *   - `face` 决定**牌面的图案朝哪**，对家的牌在自己屏幕上看是倒的，
       *     所以这里转 180°。
       * 之前把两者混成一个数，就出现「位置对但牌面倒」或者反过来。
       */
      name: "小美", wind: "南", score: 31000, ai: true,
      origin: { x: cx - 7 * STEP, y: HAND_PAD }, deg: 0, face: 180,
      hand: ["2m", "3m", "5m", "8m", "1p", "3p", "6p", "1s", "4s", "8s",
             "2z", "3z", "7z"],
      drawn: "5s", pond: POND_LONG.slice(0, 9),
      pondU: 0, pondV: TH * 2.6,
      plateU: 7 * STEP, plateV: TH * 1.3,
    },
    {
      // ---- 下家（右侧）：转 -90°，手牌在屏幕上竖着、从下往上
      name: "AI·标准", wind: "西", score: 22000, ai: true,
      origin: { x: W - HAND_PAD, y: cy + 6.5 * STEP }, deg: -90, face: -90,
      hand: ["1m", "2m", "6m", "9m", "2p", "5p", "8p", "3s", "6s", "9s",
             "1z", "4z", "6z"],
      drawn: "7p",
      melds: [
        { kind: "pon", called: 0, tiles: ["5z", "5z", "5z"] },
        { kind: "chi", called: 1, tiles: ["3m", "4m", "5m"] },
      ],
      pond: POND_LONG.slice(0, 16),
      pondU: 0, pondV: -TH * 2.6,
      plateU: 6.5 * STEP, plateV: -TH * 1.3,
    },
    {
      // ---- 上家（左侧）：转 +90°，手牌竖着、从上往下
      name: "AI·防守", wind: "北", score: 22000, ai: true,
      origin: { x: HAND_PAD, y: cy - 6.5 * STEP }, deg: 90, face: 90,
      hand: ["4m", "5m", "7m", "1p", "2p", "4p", "9p", "1s", "5s", "7s",
             "2z", "5z", "7z"],
      drawn: "8m",
      melds: [{ kind: "kan", called: 0, tiles: ["1z", "1z", "1z", "1z"] }],
      pond: POND_LONG.slice(0, 21),
      pondU: 0, pondV: -TH * 2.6,
      plateU: 6.5 * STEP, plateV: -TH * 1.3,
    },
  ];
  seats.forEach(seat);
}

// ------------------------------------------------------------- 中央信息区

function center() {
  const dora = ["6s", "1m", "9p"];
  document.getElementById("center").innerHTML = `
    <div class="c-round">东 1 局<span class="sub">第 8 巡 · 亲 我</span></div>
    <div class="c-wall"><b>42</b><span>枚</span></div>
    <div class="c-dora"><span class="lbl">宝牌</span>
      <div class="dora-row">${dora.map((d) => `<img src="/tiles/${d}.svg">`).join("")}</div>
    </div>
    <div class="c-sticks"><span class="lbl">供托</span><span class="stick"></span><b>1</b></div>
    <div class="scorestrip">
      <div class="scoreline is-me is-dealer is-active">
        <span class="wind">东</span><span class="name">我（你）</span><span class="sc">25000</span>
      </div>
      <div class="scoreline">
        <span class="wind">南</span><span class="name">小美</span><span class="sc">31000</span>
      </div>
      <div class="scoreline is-riichi">
        <span class="wind">西</span><span class="name">AI·标准</span><span class="sc">22000</span>
      </div>
      <div class="scoreline">
        <span class="wind">北</span><span class="name">AI·防守</span><span class="sc">22000</span>
      </div>
    </div>
    <div class="c-turn"><span class="who is-mine">我（你）</span><span class="timer">28.4s</span>
      <span class="muted">· 出牌</span></div>
  `;
}

window.addEventListener("resize", () => {
  const s = Math.min(window.innerWidth / W, window.innerHeight / H, 1);
  board.style.transform = `scale(${s})`;
  board.style.left = (window.innerWidth - W * s) / 2 + "px";
  board.style.top = (window.innerHeight - H * s) / 2 + "px";
});

center();
draw();
window.dispatchEvent(new Event("resize"));
