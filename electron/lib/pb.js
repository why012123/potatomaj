// 最小 protobuf 读取器 + 雀魂 liqi 消息拆包。
//
// 为什么先写这个：雀魂的 lq.Wrapper 结构很简单 ——
//   字段 1 = 方法名(string)，字段 2 = 载荷(bytes)
// 光靠这两条就能把「收到了哪些消息」列出来，不用完整 .proto 定义。
// ActionPrototype 再往里剥一层：字段 1=step(int)，2=动作名(string)，3=数据(bytes)。

function readVarint(buf, pos) {
  let result = 0;
  let shift = 0;
  while (pos < buf.length) {
    const b = buf[pos++];
    result += (b & 0x7f) * Math.pow(2, shift);
    if ((b & 0x80) === 0) return [result, pos];
    shift += 7;
    if (shift > 63) break;
  }
  return [null, pos];
}

/** 把一段 protobuf 拆成字段列表（不做 schema 校验） */
function parseFields(buf) {
  const out = [];
  let pos = 0;
  while (pos < buf.length) {
    const [tag, p1] = readVarint(buf, pos);
    if (tag === null) break;
    pos = p1;
    const no = tag >>> 3;
    const wire = tag & 7;
    if (wire === 0) {
      const [v, p2] = readVarint(buf, pos);
      if (v === null) break;
      pos = p2;
      out.push({ no, wire, int: v });
    } else if (wire === 2) {
      const [len, p2] = readVarint(buf, pos);
      if (len === null || p2 + len > buf.length) break;
      pos = p2;
      out.push({ no, wire, bytes: buf.subarray(pos, pos + len) });
      pos += len;
    } else if (wire === 5) {
      pos += 4;
    } else if (wire === 1) {
      pos += 8;
    } else {
      break; // 不认识的 wire type，放弃
    }
  }
  return out;
}

const asText = (b) => {
  try {
    const s = Buffer.from(b).toString('utf8');
    // protobuf 的 string 字段一定合法 UTF-8，含替换符就说明不是字符串
    return s.includes('\ufffd') ? null : s;
  } catch (e) {
    return null;
  }
};

/**
 * 找到 protobuf 的起点。
 *
 * 雀魂的真实帧外面还有一层自己的帧头（这一层我一开始漏了，导致 98 帧全解不出来）：
 *   01 <protobuf>                —— notify，服务器主动推送
 *   02 <id:2字节小端> <protobuf>  —— request，客户端请求
 *   03 <id:2字节小端> <protobuf>  —— response，服务器应答（常是空 wrapper = 心跳）
 */
function findPayloadStart(buf) {
  const t = buf.length ? buf[0] : -1;
  const guess = t === 1 ? 1 : t === 2 || t === 3 ? 3 : 0;
  const order = [guess, 1, 3, 0, 2, 4];
  for (const off of order) {
    if (off >= buf.length || buf[off] !== 0x0a) continue;
    const [len, p] = readVarint(buf, off + 1);
    if (len === null || len === 0 || p + len > buf.length || len > 200) continue;
    const s = buf.subarray(p, p + len).toString('utf8');
    if (/^\.?lq(\.[A-Za-z0-9_]+)+$/.test(s)) return off;
  }
  return guess;
}

/**
 * 拆一层 liqi 消息。
 * @returns {{name: string|null, payload: Buffer|null, kind: string, action?: string, step?: number}}
 */
function unwrap(buf) {
  const off = findPayloadStart(buf);
  const fields = parseFields(buf.subarray(off));
  let name = null;
  let payload = null;
  for (const f of fields) {
    if (f.no === 1 && f.wire === 2 && name === null) name = asText(f.bytes);
    if (f.no === 2 && f.wire === 2 && payload === null) payload = f.bytes;
  }
  const out = { name, payload, kind: name || '未知' };

  if (name === '.lq.ActionPrototype' && payload) {
    let step = null;
    let action = null;
    let data = null;
    for (const f of parseFields(payload)) {
      if (f.no === 1 && f.wire === 0) step = f.int;
      if (f.no === 2 && f.wire === 2 && action === null) action = asText(f.bytes);
      if (f.no === 3 && f.wire === 2 && data === null) data = f.bytes;
    }
    out.action = action;
    out.step = step;
    out.actionPayload = data;
    out.kind = action ? `ActionPrototype/${action.replace(/^\.lq\./, '')}` : 'ActionPrototype';
  }
  return out;
}

module.exports = { readVarint, parseFields, unwrap, findPayloadStart };
