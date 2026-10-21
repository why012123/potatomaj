// 把 `.lq.ActionPrototype` 解成结构化数据。
//
// 两层：先 wtf_decode 去掉那层位置相关的异或（见 lib/liqi.js），
// 再按 `.refs/mahjongsoul.proto` 里的字段号读 protobuf。
//
// 字段号都是**照着 proto 抄的**，不是我试出来的。每条都写在下面的 SCHEMA 里，
// 改协议时对着 proto 更新这一张表就行。
const { wtfDecode } = require('./liqi');

// ---------------------------------------------------------------------------
// 极简 protobuf 读取（够用就行，不引依赖）
// ---------------------------------------------------------------------------

/** 把一段 protobuf 拆成 [{f: 字段号, w: 线格式, v: 数字 | Buffer}] */
function readFields(buf) {
  const out = [];
  let p = 0;
  while (p < buf.length) {
    let tag = 0, sh = 0;
    while (p < buf.length) {
      const b = buf[p++];
      tag += (b & 0x7f) * Math.pow(2, sh);
      if ((b & 0x80) === 0) break;
      sh += 7;
      if (sh > 63) return out;
    }
    const f = tag >>> 3, w = tag & 7;
    if (f === 0) return out;
    if (w === 0) {
      let v = 0; sh = 0;
      while (p < buf.length) {
        const b = buf[p++];
        v += (b & 0x7f) * Math.pow(2, sh);
        if ((b & 0x80) === 0) break;
        sh += 7;
      }
      out.push({ f, w, v });
    } else if (w === 2) {
      let len = 0; sh = 0;
      while (p < buf.length) {
        const b = buf[p++];
        len += (b & 0x7f) * Math.pow(2, sh);
        if ((b & 0x80) === 0) break;
        sh += 7;
      }
      if (p + len > buf.length) return out;
      out.push({ f, w, b: buf.subarray(p, p + len) });
      p += len;
    } else if (w === 5) { out.push({ f, w, b: buf.subarray(p, p + 4) }); p += 4; }
    else if (w === 1) { out.push({ f, w, b: buf.subarray(p, p + 8) }); p += 8; }
    else return out;                       // 不认识的线格式：停下，别硬解
  }
  return out;
}

const strsAt = (F, n) => F.filter((x) => x.f === n && x.w === 2).map((x) => x.b.toString('utf8'));
const strAt = (F, n) => (strsAt(F, n)[0] !== undefined ? strsAt(F, n)[0] : '');

/**
 * 解一段 packed repeated 整数（proto3 里 `repeated int32/uint32` 默认 packed）：
 * 字段是线格式 2，里面是一串 varint。
 *
 * **为什么必须有**：`ActionChiPengGang.froms` 是 `repeated uint32`，实测都是
 * packed 发过来的（例如 `02 02 01` = [2,2,1]）。以前 `numsAt` 只看线格式 0，
 * 于是 froms 永远解成空数组 —— 吃牌时只能猜「哪张是别人打的」，猜错就扣错手牌，
 * 整局失准（见 gamestate `_meld` 的说明）。`ActionNewRound.scores` 也是 packed，
 * 同样一直读成空。
 */
const packedVarints = (buf) => {
  const out = [];
  let p = 0;
  while (p < buf.length) {
    let v = 0;
    let sh = 0;
    let b;
    do {
      if (p >= buf.length) return out;    // 截断的包：能读多少算多少
      b = buf[p++];
      v += (b & 0x7f) * Math.pow(2, sh);
      sh += 7;
      if (sh > 63) return out;
    } while (b & 0x80);
    out.push(v);
  }
  return out;
};

const numsAt = (F, n) => {
  const out = [];
  for (const x of F) {
    if (x.f !== n) continue;
    if (x.w === 0) out.push(x.v);
    else if (x.w === 2) out.push(...packedVarints(x.b));
  }
  return out;
};
const numAt = (F, n, d = 0) => {
  const xs = numsAt(F, n);
  return xs.length ? xs[0] : d;
};

const TILE_RE = /^[0-9][mpsz]$/;
const isTile = (t) => typeof t === 'string' && TILE_RE.test(t) && t !== '0z';
const tilesAt = (F, n) => strsAt(F, n).filter(isTile);

/**
 * 解 `OptionalOperationList`（谁可以操作 + 每个操作的种类/组合）。
 *
 * 字段（照抄 proto 2219 / 2212 行）：
 *   OptionalOperationList { seat = 1, repeated OptionalOperation operation_list = 2 }
 *   OptionalOperation    { type = 1, repeated string combination = 2 }
 *
 * `type` 的取值是**从真实帧里对出来的**（不是猜的）：
 *   2 = 吃（combination 形如 "7s|8s"，是手里的两张）
 *   3 = 碰（"1z|1z"）
 *   5 = 大明槓（"4p|4p|4p"）
 *   9 = 荣和（combination 为空）
 * 另外 4/6/7/8 等是摸牌后自己的暗杠/加杠/立直/九种九牌，不在「打牌后鸣牌」
 * 这条路上，这里只保留 2/3/5/9，其余原样带出（供排查）。
 *
 * **为什么现在才加**：以前这个字段被整个丢掉，HUD 只能对「打哪张」给建议，
 * 鸣牌窗口里什么都不说 —— 用户看到的就是「没法判断是否应该鸣牌」。
 */
const OP_KIND = { 2: 'chi', 3: 'pon', 5: 'kan', 9: 'ron' };
const operationListAt = (F, field) => {
  const x = F.find((y) => y.f === field && y.w === 2);
  if (!x) return null;
  let OF;
  try { OF = readFields(x.b); } catch (e) { return null; }
  const seat = numAt(OF, 1);
  const ops = [];
  for (const y of OF) {
    if (y.f !== 2 || y.w !== 2) continue;
    let OP;
    try { OP = readFields(y.b); } catch (e) { continue; }
    const type = numAt(OP, 1);
    // combination 是**原样字符串**（可能带 `|`），不要按单张牌过滤
    const combination = strsAt(OP, 2);
    if (!combination.length && type !== 9) continue;   // 2/3/5 必须有组合
    ops.push({ type, kind: OP_KIND[type] || `op${type}`, combination });
  }
  return ops.length ? { seat, ops } : null;
};

// ---------------------------------------------------------------------------
// 每条动作怎么读 —— 字段号照抄 .refs/mahjongsoul.proto
// ---------------------------------------------------------------------------

/** 里宝牌/宝牌指示牌：字段 5（单个）和 14（重复），不同版本服务器填的不一样 */
const doraOf = (F) => {
  const many = tilesAt(F, 14);
  if (many.length) return many;
  const one = strAt(F, 5);
  return isTile(one) ? [one] : [];
};

const PARSERS = {
  /** 发牌：手牌就在这里（字段 4） */
  ActionNewRound(F) {
    return {
      chang: numAt(F, 1), ju: numAt(F, 2), ben: numAt(F, 3),
      hand: tilesAt(F, 4),
      dora: doraOf(F),
      scores: numsAt(F, 6),
      leftTileCount: numAt(F, 13),
    };
  },

  /** 摸牌：只有**自己**摸的那张带牌名，别人摸的是空串 */
  ActionDealTile(F) {
    const tile = strAt(F, 2);
    return {
      seat: numAt(F, 1),
      tile: isTile(tile) ? tile : null,
      leftTileCount: numAt(F, 3),
      dora: tilesAt(F, 6),
      zhenting: numAt(F, 7) === 1,
    };
  },

  /** 打牌：座位、牌、立直、摸切、振听 */
  ActionDiscardTile(F) {
    const tile = strAt(F, 2);
    return {
      seat: numAt(F, 1),
      tile: isTile(tile) ? tile : null,
      isLiqi: numAt(F, 3) === 1,
      moqie: numAt(F, 5) === 1,
      zhenting: numAt(F, 6) === 1,
      dora: tilesAt(F, 8),
      isWliqi: numAt(F, 9) === 1,
      revealed: numAt(F, 12) === 1,
      // 打牌之后「我可以吃/碰/杠/荣」的提示（字段 4）。没有就是 null。
      // 这是 HUD 终于能对「要不要鸣牌」说话的数据来源。
      operations: operationListAt(F, 4),
    };
  },

  /** 副露：type 0=吃 1=碰 2=大明槓（三麻没有吃） */
  ActionChiPengGang(F) {
    return {
      seat: numAt(F, 1),
      type: numAt(F, 2),
      tiles: tilesAt(F, 3),
      froms: numsAt(F, 4),
    };
  },

  /** 拔北（三麻特有）：摸到北可以拔出来当宝牌 */
  ActionBaBei(F) {
    return { seat: numAt(F, 1), moqie: numAt(F, 9) === 1, dora: tilesAt(F, 6) };
  },

  /** 暗杠 / 加杠：tiles 是**一张**牌（字符串，不是数组） */
  ActionAnGangAddGang(F) {
    const t = strAt(F, 3);
    return { seat: numAt(F, 1), type: numAt(F, 2), tile: isTile(t) ? t : null };
  },

  /** 和了：分数在这里 */
  ActionHule(F) {
    return { deltaScores: numsAt(F, 3), scores: numsAt(F, 5) };
  },

  ActionNoTile() { return {}; },
  ActionLiuJu() { return {}; },
  ActionMJStart() { return {}; },
  ActionGameEnd() { return {}; },
};

const MELD_TYPE = { 0: 'chi', 1: 'pon', 2: 'kan' };

/**
 * 解一条 ActionPrototype 的 data。
 * @param {string} action 动作名，如 `ActionNewRound`
 * @param {Buffer|Uint8Array} payload 被打码的 data
 * @returns {object|null} 结构化数据；解不出来返回 null
 */
function decodeAction(action, payload) {
  const fn = PARSERS[action];
  if (!fn) return null;
  // 有些动作**本来就没有参数**（ActionMJStart / ActionGameEnd …），
  // 它们的 payload 是空的。以前这里直接 return null，于是「新的一局开始」
  // 这条消息被整个丢掉 —— 该重置的手牌没重置，下一局从上一局的残局接着算。
  if (!payload || !payload.length) {
    try { return fn([]); } catch (e) { return null; }
  }
  let plain;
  try { plain = wtfDecode(payload); } catch (e) { return null; }
  try {
    const out = fn(readFields(plain));
    if (out && out.seat === undefined) return out;
    return out;
  } catch (e) { return null; }
}

module.exports = { decodeAction, readFields, isTile, MELD_TYPE, PARSERS };
