/**
 * WebSocket 连接层。
 *
 * 设计要点（每一条都是踩过的坑）：
 *
 * 1. **地址按页面协议推**：页面是 https 就必须用 wss，否则浏览器直接拦掉
 *    混合内容，用户只会看到「一直连不上」。
 * 2. **指数退避重连**：服务端重启 / 笔记本合盖 / 切代理都会断。退避从 0.5s
 *    起、×1.7、封顶 8s。没有退避的话，服务端一挂就是几百个客户端每秒重连。
 * 3. **重连之后必须重新 `room.join`**：服务端在连接断开时保留座位
 *    `seat_grace`（默认 300s），凭证是 `joined` 消息里的 `token`。没有这一步，
 *    刷新页面就会掉到观战席，牌局还在等你出牌 —— 也就是「卡死」。
 * 4. **`state.seq` 单调**：服务端每推一帧 `seq += 1`。乱序（或重连后收到旧
 *    快照）的帧必须丢掉，否则牌河会「倒退」，看上去像是丢牌。
 * 5. **`hello` 之前发的消息要排队**：`room.create` / `room.join` 依赖连接上的
 *    `client.name`，早发一步服务端会回「先取个昵称再开房」。
 */

import type {
  CreateRoomOpts,
  HelloMsg,
  JoinedMsg,
  LobbyMsg,
  RejectMsg,
  RevealMsg,
  ErrorMsg,
  ServerMsg,
  StateMsg,
} from "./protocol";

/** 昵称。 */
export const LS_NAME = "maj:name";
/** 匿名玩家 ID（服务端也可能用签名 cookie 覆盖它）。 */
export const LS_UID = "maj:uid";
/** 座位凭证前缀：`maj:token:<房间号>`。 */
export const LS_TOKEN_PREFIX = "maj:token:";

export type ConnStatus = "connecting" | "open" | "closed";

export interface NetHandlers {
  onStatus?: (status: ConnStatus, detail: string) => void;
  onHello?: (msg: HelloMsg) => void;
  onLobby?: (msg: LobbyMsg) => void;
  onJoined?: (msg: JoinedMsg) => void;
  onState?: (msg: StateMsg) => void;
  onReveal?: (msg: RevealMsg) => void;
  onReject?: (msg: RejectMsg) => void;
  onError?: (msg: ErrorMsg) => void;
}

/** `localStorage` 在某些隐私模式下会直接抛异常，全部包一层。 */
function lsGet(key: string): string {
  try {
    return window.localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function lsSet(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* 无痕模式：忽略，功能降级但不崩 */
  }
}

export function tokenKey(room: string): string {
  return LS_TOKEN_PREFIX + room.toUpperCase();
}

export class Net {
  private ws: WebSocket | null = null;
  /** 退避基数（毫秒）。连上就重置。 */
  private backoff = 500;
  private reconnectTimer = 0;
  private heartbeat = 0;
  /** `hello` 之前排队的消息 */
  private queue: string[] = [];
  private greeted = false;
  private closedByUs = false;

  /** 想坐的房间（重连时用它重新 join） */
  private wantedRoom: string | null = null;
  private wantSeat: number | null = null;
  private wantSpectator = false;
  /** 已经成功 join 过：重连时用来决定要不要自动 join */
  private joinedRoom: string | null = null;

  name = "";
  uid = "";
  status: ConnStatus = "connecting";
  serverVersion = "";
  aimodel = "";
  lastState: StateMsg | null = null;
  /** 收到过的最大 seq，用来丢旧帧 */
  private lastSeq = -1;

  constructor(private readonly h: NetHandlers) {
    this.name = lsGet(LS_NAME);
    this.uid = lsGet(LS_UID);
  }

  // -- 凭证 ---------------------------------------------------------------

  token(room: string): string {
    return lsGet(tokenKey(room));
  }

  saveToken(room: string, token: string): void {
    if (room && token) lsSet(tokenKey(room), token);
  }

  setName(name: string): void {
    const n = name.trim().slice(0, 16);
    if (!n) return;
    this.name = n;
    lsSet(LS_NAME, n);
    // 服务端的名片（签名 cookie）走 HTTP，一条 fetch 足够；失败也不影响
    // WebSocket 那条路（`hello` 里也会带名字）。
    void fetch("/api/name", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: n, uid: this.uid }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((j: { uid?: string } | null) => {
        if (j?.uid) {
          this.uid = j.uid;
          lsSet(LS_UID, j.uid);
        }
      })
      .catch(() => undefined);
    this.send({ t: "hello", name: n });
  }

  // -- 连接 ---------------------------------------------------------------

  connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this.closedByUs = false;
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${scheme}//${location.host}/ws`;
    this.status = "connecting";
    this.h.onStatus?.("connecting", url);
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      this.scheduleReconnect(String(err));
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.backoff = 500;
      this.greeted = false;
      this.status = "open";
      this.h.onStatus?.("open", "");
      this.sendNow({ t: "hello", name: this.name });
      // 重连后第一件事是回自己的座位 —— 座位凭证在 localStorage 里，
      // 服务端在 seat_grace 内认它。
      if (this.wantedRoom && this.joinedRoom === this.wantedRoom) {
        this.sendNow(this.joinPayload(this.wantedRoom));
      }
      for (const text of this.queue.splice(0)) this.raw(text);
      this.startHeartbeat();
    };

    ws.onmessage = (ev: MessageEvent<string>) => this.dispatch(ev.data);

    ws.onclose = () => {
      this.ws = null;
      this.stopHeartbeat();
      if (this.closedByUs) return;
      this.status = "closed";
      this.h.onStatus?.("closed", "连接已断开");
      this.scheduleReconnect("连接已断开");
    };

    ws.onerror = () => {
      /* onclose 一定会跟着来，这里不重复处理 */
    };
  }

  private scheduleReconnect(_reason: string): void {
    if (this.reconnectTimer) return;
    const wait = this.backoff;
    this.backoff = Math.min(Math.round(this.backoff * 1.7), 8000);
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = 0;
      this.connect();
    }, wait);
  }

  /**
   * 心跳：`ping` → `pong`。
   *
   * 服务端自己也有 WebSocket 心跳（25s），但有些中间代理会「假死」
   * （TCP 还在、其实已经不通），应用层心跳能让前端更快发现要重连。
   * 25s 是刻意跟服务端的 heartbeat 错开的。
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeat = window.setInterval(() => {
      this.sendNow({ t: "ping", at: Math.round(performance.now()) });
    }, 25000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeat) {
      window.clearInterval(this.heartbeat);
      this.heartbeat = 0;
    }
  }

  // -- 发送 ---------------------------------------------------------------

  send(msg: Record<string, unknown>): void {
    // `hello` 已经握过手就直接发；否则排队（见文件头第 5 条）。
    if (this.ws && this.ws.readyState === WebSocket.OPEN && this.greeted) {
      this.sendNow(msg);
      return;
    }
    if (this.ws && this.ws.readyState === WebSocket.OPEN && !this.greeted && msg["t"] === "hello") {
      this.sendNow(msg);
      return;
    }
    this.queue.push(JSON.stringify(msg));
  }

  private sendNow(msg: Record<string, unknown>): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      this.queue.push(JSON.stringify(msg));
      return;
    }
    this.raw(JSON.stringify(msg));
  }

  private raw(text: string): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      this.queue.push(text);
      return;
    }
    try {
      ws.send(text);
    } catch {
      this.queue.push(text);
    }
  }

  // -- 协议动作 -----------------------------------------------------------

  refreshLobby(): void {
    this.send({ t: "lobby" });
  }

  private joinPayload(room: string): Record<string, unknown> {
    return {
      t: "room.join",
      room,
      seat: this.wantSeat,
      token: this.token(room),
      spectator: this.wantSpectator,
      // 观战者的全亮请求只在纯 AI 房生效，服务端说了算
      reveal: true,
      autostart: false,
    };
  }

  createRoom(opts: CreateRoomOpts): void {
    this.send({
      t: "room.create",
      name: opts.name.slice(0, 24),
      players: opts.players,
      rounds: opts.rounds,
      ai_delay: opts.aiDelay,
      seats: opts.seats,
      spectator_reveal: true,
      autostart: opts.autostart,
    });
  }

  joinRoom(room: string, opts: { seat?: number | null; spectator?: boolean } = {}): void {
    const rid = room.trim().toUpperCase();
    this.wantedRoom = rid;
    this.wantSeat = opts.seat ?? null;
    this.wantSpectator = opts.spectator ?? false;
    this.send(this.joinPayload(rid));
  }

  watchRoom(room: string): void {
    this.joinRoom(room, { spectator: true });
  }

  /** 牌桌里主动离开：解绑，回到大厅。 */
  leaveRoom(): void {
    const rid = this.wantedRoom;
    this.send({ t: "room.leave" });
    this.wantedRoom = null;
    this.wantSeat = null;
    this.wantSpectator = false;
    this.joinedRoom = null;
    this.lastSeq = -1;
    this.lastState = null;
    // 主动离开就把凭证丢掉：下次进来应该是「重新入座」，不是「回到那个
    // 可能已经不属于你的座位」。
    if (rid) lsSet(tokenKey(rid), "");
  }

  startRoom(): void {
    this.send({ t: "room.start" });
  }

  /**
   * 出牌。
   *
   * `tile` 优先：服务端 `submit_action()` 支持用牌 id 反查 index，
   * 而牌 id 在「同一种牌有两张」时比 index 稳（index 是服务端那份
   * `pending.options` 里的位置，前端一旦把列表错位就会打错牌）。
   */
  act(kind: string, index?: number, tile?: string): void {
    const msg: Record<string, unknown> = { t: "action", kind };
    if (tile) msg["tile"] = tile;
    else if (index !== undefined) msg["index"] = index;
    this.send(msg);
  }

  next(): void {
    this.send({ t: "next" });
  }

  chat(text: string): void {
    const t = text.trim();
    if (t) this.send({ t: "chat", text: t });
  }

  setReveal(on: boolean): void {
    this.send({ t: "reveal", on });
  }

  setNoRecord(noRecord: boolean): void {
    this.send({ t: "privacy", no_record: noRecord });
  }

  setProfile(rank: string, mode: string): void {
    this.send({ t: "profile", rank, mode });
  }

  // -- 接收 ---------------------------------------------------------------

  private dispatch(rawData: string): void {
    let msg: ServerMsg;
    try {
      msg = JSON.parse(rawData) as ServerMsg;
    } catch {
      return;
    }
    switch (msg.t) {
      case "hello": {
        const m = msg as HelloMsg;
        this.greeted = true;
        this.uid = m.uid;
        lsSet(LS_UID, m.uid);
        if (!this.name && m.name) this.name = m.name;
        this.serverVersion = m.server?.version ?? "";
        this.aimodel = m.server?.aimodel ?? "";
        this.h.onHello?.(m);
        break;
      }
      case "lobby":
        this.h.onLobby?.(msg as LobbyMsg);
        break;
      case "state": {
        const m = msg as StateMsg;
        // 旧帧丢掉：牌河只能往前长，seq 倒退一定是乱序/陈旧推送
        if (typeof m.seq === "number" && m.seq <= this.lastSeq) return;
        this.lastSeq = m.seq;
        this.lastState = m;
        this.wantedRoom = m.room.id;
        this.joinedRoom = m.room.id;
        this.wantSeat = m.viewer.seat >= 0 ? m.viewer.seat : null;
        this.wantSpectator = m.viewer.spectator;
        this.h.onState?.(m);
        break;
      }
      case "joined": {
        const m = msg as JoinedMsg;
        this.joinedRoom = m.room;
        this.wantedRoom = m.room;
        this.wantSeat = m.seat >= 0 ? m.seat : null;
        this.wantSpectator = m.spectator;
        if (m.token) this.saveToken(m.room, m.token);
        this.h.onJoined?.(m);
        break;
      }
      case "reveal":
        this.h.onReveal?.(msg as RevealMsg);
        break;
      case "reject":
        this.h.onReject?.(msg as RejectMsg);
        break;
      case "error":
        this.h.onError?.(msg as ErrorMsg);
        break;
      case "identity": {
        const m = msg as { uid?: string; name?: string };
        if (m.uid) {
          this.uid = m.uid;
          lsSet(LS_UID, m.uid);
        }
        break;
      }
      default:
        break;
    }
  }

  close(): void {
    this.closedByUs = true;
    this.stopHeartbeat();
    if (this.reconnectTimer) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = 0;
    }
    this.ws?.close();
    this.ws = null;
  }
}
