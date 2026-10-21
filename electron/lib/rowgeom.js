// 手牌行的几何：**从第一张牌 + 排序下标推算每张牌的位置**。
//
// 为什么改成这样：原来每巡截一次手牌条、靠识图给出每张牌的坐标。
// 那条路有三个问题：
//   1. 慢 —— 每次截图都要把画面从 GPU 读回来
//   2. 会错 —— 局间压暗、游戏自己的面板压住牌顶，都会让识图认不出或认偏
//   3. 坐标会漂 —— 摸牌/打牌时还要靠推测去挪坐标，和识图结果一叠加就错位
//
// 而**位置本来是可以算出来的**：雀魂的手牌是**排序显示**的
//（同一花色内从小到大，花色顺序 万→筒→索→字，实测确认），
// 整行居中，每张牌等距。所以只要知道「第一张在哪」和「牌与牌的间距」，
// 后面每一张的位置就确定了 —— 跟识图没有关系。
//
// 这两个量只需要量**一次**（开局那次整屏识图），之后整局都不用再认。

/** 牌张之间的间距、整行中心、行顶部 —— 量一次，一直用 */
class RowGeometry {
  constructor() {
    this.pitch = 0;      // 相邻两张牌的距离（像素）
    this.cx = 0;         // 整行的中心 x（像素）—— 手牌是居中的
    this.y = 0;          // 行顶部 y（像素）
    this.height = 0;     // 牌面高度（像素）
    this.valid = false;
  }

  /** 从一次识图的结果里量出来（取所有手牌框的中位，抗单张误差） */
  measure(handItems, viewportWidth, viewportHeight) {
    const boxes = (handItems || []).map((it) => it.box).filter(Boolean);
    if (boxes.length < 4) return false;
    const widths = boxes.map((b) => b[2] - b[0]).filter((w) => w > 0).sort((a, b) => a - b);
    const pitch = widths[Math.floor(widths.length / 2)] * viewportWidth;
    if (!(pitch > 4)) return false;
    const xs = boxes.map((b) => (b[0] + b[2]) / 2 * viewportWidth);
    const ys = boxes.map((b) => b[1] * viewportHeight).sort((a, b) => a - b);
    const hs = boxes.map((b) => (b[3] - b[1]) * viewportHeight).sort((a, b) => a - b);
    this.pitch = pitch;
    // 整行中心 = 所有牌中心的中点（不是平均，避免多认/少认把中心拉偏）
    this.cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    this.y = ys[0];
    this.height = hs[Math.floor(hs.length / 2)];
    this.vw = viewportWidth;
    this.vh = viewportHeight;
    this.valid = true;
    return true;
  }

  /**
   * 算出第 index 张牌（共 count 张）的框，单位是**比例坐标**（跟原来的接口一致）。
   *
   * 整行居中：n 张牌占 n*pitch 宽，左边缘 = 中心 - n*pitch/2。
   *
   * @param {number} index  第几张（0 开始，按雀魂排序）
   * @param {number} count  这一行几张（不含摸到的那张）
   * @param {object} opts   { drawnGap } 摸到的那张和前面隔开多少格
   */
  boxAt(index, count, opts = {}) {
    if (!this.valid) return null;
    const gapTiles = opts.drawnGap === undefined ? 0.3 : opts.drawnGap;
    const total = count;
    const left = this.cx - (total * this.pitch) / 2;
    const x1 = left + index * this.pitch + (index >= total ? gapTiles * this.pitch : 0);
    return [x1, this.y, x1 + this.pitch, this.y + this.height];
  }

  /** 比例坐标版本（界面直接乘窗口尺寸用） */
  boxRatio(index, count, viewportWidth, viewportHeight, opts = {}) {
    const b = this.boxAt(index, count, opts);
    if (!b) return null;
    return [b[0] / viewportWidth, b[1] / viewportHeight,
            b[2] / viewportWidth, b[3] / viewportHeight];
  }
}

module.exports = { RowGeometry };

// ---------------------------------------------------------------------------
// 雀魂的手牌顺序
//
// 实测确认（拿真实截图逐张放大核对）：**同一花色内从小到大，花色顺序 万→筒→索→字**。
//
// 这件事是整套「按位置算坐标」的前提 —— 顺序错了，所有气泡都会错位到别的牌上，
// 而且看上去还挺整齐（所以不容易发现）。
// ---------------------------------------------------------------------------

const SUIT_ORDER = { m: 0, p: 1, s: 2, z: 3 };

/** 一张牌的排序键。赤五（0p/0s）按「5」参与排序 —— 它在显示上就在 5 的位置。 */
function tileKey(tile) {
  if (typeof tile !== 'string' || tile.length < 2) return [99, 99];
  const suit = SUIT_ORDER[tile[1]];
  if (suit === undefined) return [99, 99];
  let num = Number(tile[0]);
  if (Number.isNaN(num)) return [99, 99];
  if (num === 0) num = 5;              // 赤五当 5 排
  return [suit, num];
}

/** 按雀魂的显示顺序排序（不改变原数组） */
function sortHand(tiles) {
  return (tiles || []).slice().sort((a, b) => {
    const ka = tileKey(a), kb = tileKey(b);
    if (ka[0] !== kb[0]) return ka[0] - kb[0];
    if (ka[1] !== kb[1]) return ka[1] - kb[1];
    // 同一张牌里赤五排普通五后面（显示上它们挨着，顺序无所谓，定死便于比较）
    return String(a).localeCompare(String(b));
  });
}

module.exports.sortHand = sortHand;
module.exports.tileKey = tileKey;


// ---------------------------------------------------------------------------
// 存盘 / 读盘
//
// 几何只跟**窗口尺寸**有关，量一次就固定了。存下来的意义是：
// 下次启动直接读，**连那一次识图都不需要** —— 也就是真正「不用识图」。
// 换窗口大小或改缩放时尺寸对不上会自动作废、重新量。
// ---------------------------------------------------------------------------

/** 转成可存的对象（带上窗口尺寸，尺寸变了就作废） */
RowGeometry.prototype.toJSON = function toJSON() {
  return { pitch: this.pitch, cx: this.cx, y: this.y, height: this.height,
           vw: this.vw, vh: this.vh };
};

/** 从存下来的对象恢复。尺寸对不上就返回 false（当没量过）。 */
RowGeometry.prototype.restore = function restore(data, viewportWidth, viewportHeight) {
  if (!data || !(data.pitch > 4)) return false;
  if (data.vw && Math.abs(data.vw - viewportWidth) > 2) return false;
  if (data.vh && Math.abs(data.vh - viewportHeight) > 2) return false;
  this.pitch = data.pitch;
  this.cx = data.cx;
  this.y = data.y;
  this.height = data.height || 0;
  this.vw = viewportWidth;
  this.vh = viewportHeight;
  this.valid = true;
  this.measuredOnce = true;
  return true;
};
