// 跟 Python 大脑（potatomaj.bridge）通话：一行一个 JSON，按 id 配对。
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '../..');

function findPython() {
  // 允许覆盖：方便用自定义环境，也方便测试「环境坏掉」这条路
  if (process.env.POTATOMAJ_PYTHON) return process.env.POTATOMAJ_PYTHON;
  const venv = path.join(ROOT, '.venv', 'bin', 'python');
  if (fs.existsSync(venv)) return venv;
  return 'python3';
}

class Bridge {
  constructor() {
    this.proc = null;
    this.pending = new Map();
    this.nextId = 1;
    this.buf = '';
  }

  start() {
    if (this.proc) return;
    this.proc = spawn(findPython(), ['-m', 'potatomaj.bridge'], { cwd: ROOT });
    this.proc.stdout.on('data', (chunk) => {
      this.buf += chunk.toString('utf8');
      let i;
      while ((i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i).trim();
        this.buf = this.buf.slice(i + 1);
        if (!line) continue;
        let resp;
        try { resp = JSON.parse(line); } catch (e) { continue; }
        const p = this.pending.get(resp.id);
        if (p) { this.pending.delete(resp.id); p(resp); }
      }
    });
    this.proc.stderr.on('data', (d) => console.log('[bridge:stderr]', d.toString().trim().slice(0, 200)));
    // spawn 失败（比如解释器路径不存在）**不会同步抛**，它走 error 事件 ——
    // 不监听的话请求会一直挂到超时，最后报一句「超时」，
    // 而真正的原因是「找不到 python」。排查时这点信息差别很大。
    this.proc.on('error', (e) => {
      const why = `起不来：${e && e.message ? e.message : e}`;
      console.log('[bridge]', why);
      this.proc = null;
      for (const [, p] of this.pending) p({ ok: false, error: why });
      this.pending.clear();
    });
    this.proc.on('exit', (code) => {
      console.log('[bridge] 退出，code =', code);
      this.proc = null;
      for (const [, p] of this.pending) p({ ok: false, error: 'bridge 已退出' });
      this.pending.clear();
    });
  }

  /** 发一条请求，返回 Promise<响应> */
  request(cmd, params = {}, timeoutMs = 20000) {
    this.start();
    const id = this.nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        resolve({ ok: false, error: '超时' });
      }, timeoutMs);
      this.pending.set(id, (resp) => { clearTimeout(timer); resolve(resp); });
      try {
        this.proc.stdin.write(JSON.stringify({ id, cmd, ...params }) + '\n');
      } catch (e) {
        clearTimeout(timer);
        this.pending.delete(id);
        resolve({ ok: false, error: String(e) });
      }
    });
  }

  stop() {
    if (this.proc) { try { this.proc.kill(); } catch (e) {} this.proc = null; }
  }
}

module.exports = { Bridge };
