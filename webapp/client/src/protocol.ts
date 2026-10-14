/**
 * 协议类型 —— 与 `webapp/server/protocol.py` 的 `build_state()` 一一对应。
 *
 * 这里**只描述服务端真的会下发的字段**（宁可多写几个 `| null`，也不要靠
 * `any` 蒙过去）。协议版本是 1，`hello.v` / `state.v` 都会带。
 *
 * 有一条规矩必须写死在注释里：**别家的 `hand` 永远是 null**。服务端在
 * `protocol.seat_view()` 里就不往字典里写，所以前端「想显示也显示不出来」。
 * 别为了「观战好看」在这里造数据。
 */

export const PROTOCOL_VERSION = 1;

export type Phase = "lobby" | "playing" | "hand_end" | "finished";

/** 牌 id：`1m`..`9m` / `0m`(赤5) / `1p`.. / `1s`.. / `1z`..`7z`。 */
export type TileId = string;

export type SeatKind = "human" | "ai" | "waiting";

export interface RoomSeatInfo {
  seat: number;
  name: string;
  kind: SeatKind;
  ai: string | null;
  host: boolean;
  online: boolean;
  ai_degraded: boolean;
  ai_degrade_reason: string;
  no_record: boolean;
  declared_rank: string;
  main_mode: string;
  ready?: boolean;
  next_ack?: boolean;
}

export interface RoomRules {
  aka: number;
  kita: boolean;
  chi: boolean;
  noten_penalty: number;
  double_ron: boolean;
  start_score: number;
  return_score: number;
}

export interface RoomInfo {
  id: string;
  name: string;
  players: 3 | 4;
  mode: "ai" | "mixed" | "human";
  rounds: number;
  ai_delay: number;
  human_timeout: number;
  seats: RoomSeatInfo[];
  max_conns: number;
  spectator_reveal: boolean;
  start_score: number;
  return_score: number;
  created_at: number;
  record: boolean;
  rules: RoomRules;
  /** 由 `push_state()` 补上 */
  next_acks?: boolean[];
  dealt?: boolean;
}

/** `tiles.meld_json()` 的结构。 */
export interface MeldInfo {
  kind: "chi" | "pon" | "kan" | "kakan" | "ankan";
  tile: TileId;
  tiles: TileId[];
  /** `tiles` 里哪几个位置是赤 5 */
  red: number[];
  from: number;
  aka: number;
  /** 哪一张是从别人那里来的（横放）；暗杠是 -1 */
  called: number;
  label: string;
}

export interface SeatState {
  seat: number;
  name: string;
  kind: SeatKind;
  ai: string | null;
  online: boolean;
  host: boolean;
  score: number;
  /** 相对本局开始时（或本场开始时）的增减 */
  score_delta: number;
  hand_count: number;
  /** 只有自己（或纯 AI 房的观战者）拿得到，别家恒为 null */
  hand: TileId[] | null;
  drawn: TileId | null;
  melds: MeldInfo[];
  pond: TileId[];
  pond_red: number[];
  riichi: boolean;
  double_riichi: boolean;
  /** 立直宣言牌：牌河里等于它的那一张要横放 */
  riichi_tile: TileId | null;
  riichi_ippatsu: boolean;
  ippatsu: boolean;
  kita: number;
  seat_wind: string;
  is_dealer: boolean;
  rel: string;
  menzen: boolean;
  connected: boolean;
  ai_degraded: boolean;
  ai_degrade_reason: string;
  no_record: boolean;
  declared_rank: string;
  main_mode: string;
}

export interface ActionOption {
  index: number;
  /** 只有出牌选项有 */
  tile?: TileId;
  /** 只有出牌选项有（牌的中文名） */
  cn?: string;
  label?: string;
  riichi_ok?: boolean;
  tsumogiri?: boolean;
}

export interface ActionsView {
  kind: string;
  seat: number;
  kind_cn: string;
  options: ActionOption[];
  tile?: TileId | null;
  tile_cn?: string | null;
  waiting: boolean;
  deadline: number;
  ms_left: number;
  timeout?: number;
  fallback?: number;
}

export interface YakuInfo {
  key: string;
  name: string;
  han: number;
}

export interface AgariInfo {
  seat: number;
  tsumo: boolean;
  hand: TileId[];
  melds: MeldInfo[];
  win_tile: TileId | null;
  /** 真役 */
  yaku: YakuInfo[];
  /** 宝牌 / 里宝牌 / 赤宝牌，单独一组 */
  dora: YakuInfo[];
  han: number;
  dora_han: number;
  total_han: number;
  fu: number;
  yakuman: number;
  describe: string;
  gain: number;
}

export interface HandResult {
  hand_no: number;
  kind: "tsumo" | "ron" | "draw" | "kyuushu" | "abort";
  turns: number;
  note: string;
  deltas: number[];
  scores: number[];
  sticks: number;
  winner: number | null;
  loser: number | null;
  winners: number[];
  label: string;
  agari: AgariInfo[];
  /** 流局时：听牌的家 */
  tenpai?: number[];
}

export interface MatchHandInfo {
  kind: string;
  text: string;
  deltas: number[];
  turns: number;
  note: string;
}

export interface MatchResult {
  scores: number[];
  ranks: number[];
  deltas: number[];
  start_scores: number[];
  rank_points: number[];
  coins: number[];
  seed: number;
  stats: unknown[];
  hands: MatchHandInfo[];
}

export interface RoundInfo {
  hand_no: number;
  hands_played: number;
  dealer: number;
  round_wind: string;
  round_wind_idx: number;
  turn: number;
  wall: number;
  dora: TileId[];
  ura: TileId[];
  sticks: number;
  rounds: number;
  max_hands: number;
  scores_before: number[];
  sticks_before: number;
}

export interface ViewerInfo {
  seat: number;
  spectator: boolean;
  reveal: boolean;
  name: string;
}

export interface ChatLine {
  at: number;
  seat: number;
  name: string;
  text: string;
  system: boolean;
}

/** 引擎结构化动作流（`table._on_action` 攒的 `last_event`）。 */
export interface LastEvent {
  kind: string;
  player?: number;
  tile?: number;
  [k: string]: unknown;
}

export interface StateMsg {
  t: "state";
  v: number;
  seq: number;
  now: number;
  phase: Phase;
  room: RoomInfo;
  viewer: ViewerInfo;
  round: RoundInfo | null;
  seats: SeatState[];
  actions: ActionsView | null;
  last: LastEvent | null;
  result: HandResult | null;
  log: string[];
  chat: ChatLine[];
  spectator_ok: boolean;
  spectator_count: number;
  finished: boolean;
  server_time: number;
  /** 只有对局结束（`finished`）时才带 */
  match?: MatchResult;
  error?: string;
}

export interface RoomSummary {
  id: string;
  name: string;
  players: 3 | 4;
  phase: Phase;
  humans: number;
  ais: number;
  spectators: number;
  conns: number;
  ai_delay: number;
  spectator_ok: boolean;
  hand_no: number;
  rounds: number;
  created_at: number;
  seats: Array<{
    seat: number;
    name: string;
    kind: SeatKind;
    ai: string | null;
    online: boolean;
    host: boolean;
    ai_degraded: boolean;
  }>;
}

export interface LobbyStats {
  rooms: number;
  conns: number;
  humans: number;
  ais: number;
  playing: number;
  max_rooms: number;
}

/** AI 模型下拉用的一项（`agents.available_models()`）。 */
export interface ModelInfo {
  name: string;
  players: number;
  style: string;
  style_cn: string;
  cn: string;
  version: string;
  usable: boolean;
  missing_heads: string[];
  note: string;
}

export interface ServerConfig {
  ai_delay: number;
  human_timeout: number;
  allow_chat?: boolean;
  record?: boolean;
  need_password?: boolean;
  max_rooms?: number;
}

export interface HelloMsg {
  t: "hello";
  v: number;
  client: string;
  uid: string;
  name: string;
  server: { version: string; aimodel: string; host: string };
  config: ServerConfig;
  lobby: LobbyMsg;
}

export interface LobbyMsg {
  t: "lobby";
  rooms: RoomSummary[];
  stats: LobbyStats;
  models: ModelInfo[];
  record: boolean;
}

export interface JoinedMsg {
  t: "joined";
  room: string;
  seat: number;
  spectator: boolean;
  token: string;
  name: string;
  reveal_ok: boolean;
  shared: string;
}

export interface RevealMsg {
  t: "reveal";
  ok: boolean;
  on: boolean;
  message: string;
}

export interface ErrorMsg {
  t: "error";
  message: string;
}

export interface RejectMsg {
  t: "reject";
  message: string;
  kind?: string;
}

export interface IdentityMsg {
  t: "identity";
  uid: string;
  name: string;
}

/** 座位定义（建房 `seats` 数组里的一格）。 */
export interface SeatSpec {
  kind: "human" | "ai";
  /** `kind === "ai"` 时的模型名：`teacher` / `random` / `maj-normal4v1.2` … */
  ai?: string;
}

export interface CreateRoomOpts {
  name: string;
  players: 3 | 4;
  rounds: number;
  aiDelay: number | null;
  seats: SeatSpec[];
  autostart: boolean;
}

/** 服务端可能推过来的所有消息。 */
export type ServerMsg =
  | HelloMsg
  | LobbyMsg
  | JoinedMsg
  | StateMsg
  | RevealMsg
  | ErrorMsg
  | RejectMsg
  | IdentityMsg
  | { t: "pong"; at?: number }
  | { t: string; [k: string]: unknown };
