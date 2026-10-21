// 本地测试台的服务端：手写一个最小 WebSocket（不依赖任何第三方包）。
//
// 目的：**不依赖雀魂、不依赖外网**，就能验证 preload 里的 WebSocket 钩子
// 真的能抓到收发帧。钩子验证通过后，换成真游戏只要改 URL。
//
//   node test/fake-server.js [port]

const http = require('http');
const crypto = require('crypto');

const PORT = Number(process.argv[2] || 8799);
const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

function encodeFrame(payload, opcode = 0x2) {
  const len = payload.length;
  let header;
  if (len < 126) {
    header = Buffer.from([0x80 | opcode, len]);
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x80 | opcode;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x80 | opcode;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payload]);
}

function decodeFrames(buf) {
  const out = [];
  let off = 0;
  while (off + 2 <= buf.length) {
    const opcode = buf[off] & 0x0f;
    const masked = (buf[off + 1] & 0x80) !== 0;
    let len = buf[off + 1] & 0x7f;
    let p = off + 2;
    if (len === 126) {
      if (p + 2 > buf.length) break;
      len = buf.readUInt16BE(p); p += 2;
    } else if (len === 127) {
      if (p + 8 > buf.length) break;
      len = Number(buf.readBigUInt64BE(p)); p += 8;
    }
    let mask = null;
    if (masked) {
      if (p + 4 > buf.length) break;
      mask = buf.subarray(p, p + 4); p += 4;
    }
    if (p + len > buf.length) break;
    const data = Buffer.from(buf.subarray(p, p + len));
    if (mask) for (let i = 0; i < data.length; i++) data[i] ^= mask[i % 4];
    out.push({ opcode, data });
    off = p + len;
  }
  return { frames: out, rest: buf.subarray(off) };
}

function varint(n) {
  const out = [];
  while (n > 0x7f) { out.push((n & 0x7f) | 0x80); n >>>= 7; }
  out.push(n);
  return Buffer.from(out);
}
const pbString = (f, s) => { const b = Buffer.from(s, 'utf8'); return Buffer.concat([varint((f << 3) | 2), varint(b.length), b]); };
const pbBytes = (f, b) => Buffer.concat([varint((f << 3) | 2), varint(b.length), b]);
const pbInt = (f, v) => Buffer.concat([varint(f << 3), varint(v)]);
const wrapper = (name, payload) => Buffer.concat([pbString(1, name), pbBytes(2, payload)]);
const actionPrototype = (step, name, data) => Buffer.concat([pbInt(1, step), pbString(2, name), pbBytes(3, data)]);

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('potatomaj fake server\n');
});

server.on('upgrade', (req, socket) => {
  const key = req.headers['sec-websocket-key'];
  const accept = crypto.createHash('sha1').update(key + GUID).digest('base64');
  socket.write('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n' +
    `Sec-WebSocket-Accept: ${accept}\r\n\r\n`);
  console.log(`[fake-server] 有连接进来了: ${req.url}`);

  let step = 0;
  let pending = Buffer.alloc(0);
  const timer = setInterval(() => {
    step += 1;
    let payload;
    if (step % 3 === 0) {
      const body = actionPrototype(step, '.lq.ActionDiscardTile', pbBytes(2, Buffer.alloc(12, step)));
      payload = wrapper('.lq.ActionPrototype', body);
    } else {
      payload = wrapper('.lq.NotifyRoomGameStart', pbString(1, 'game-' + step));
    }
    try { socket.write(encodeFrame(payload)); } catch (e) { clearInterval(timer); }
  }, 400);

  socket.on('data', (chunk) => {
    pending = Buffer.concat([pending, chunk]);
    const { frames, rest } = decodeFrames(pending);
    pending = rest;
    for (const f of frames) {
      if (f.opcode === 0x8) { clearInterval(timer); socket.end(); return; }
      console.log(`[fake-server] 收到客户端 ${f.data.length} 字节`);
      socket.write(encodeFrame(wrapper('.lq.NotifyClientEcho', pbInt(1, f.data.length))));
    }
  });
  socket.on('close', () => { clearInterval(timer); console.log('[fake-server] 连接关闭'); });
  socket.on('error', () => clearInterval(timer));
});

server.listen(PORT, '127.0.0.1', () => console.log(`[fake-server] 监听 ws://127.0.0.1:${PORT}`));
