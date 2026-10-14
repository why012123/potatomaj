package maj;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code GET /api/state} 的一份快照。
 *
 * <p>后端现在同时给三个字段（server.py 注释里写明了）：
 * <ul>
 *   <li>{@code player_count}：人数，int —— 新代码用这个；</li>
 *   <li>{@code players_detail}：座位明细数组 —— 推荐用这个；</li>
 *   <li>{@code players}：同一份明细（历史别名，老客户端在读）。</li>
 * </ul>
 * 三个都解析、互相兜底，任何一个缺失都不会让牌桌空掉。
 *
 * <p>其它用到的：{@code pending.tile / pending.tile_cn}（被鸣 / 被荣的那张牌）、
 * 每家的 {@code pond / melds / kita / riichi / ippatsu / seat_wind / hand_count}。
 */
public final class State {

    /** 人数（player_count）。 */
    public int playerCount;
    public double delay;
    public boolean allowChi;
    public boolean allowKita;
    public long seed;
    public boolean finished;
    /** 服务器给的视角座位。 */
    public int you;
    public int wallLeft;
    public int turn;
    public int dealer;
    public int handNo;
    public String roundWind = "东";
    public String waitingFor = "";
    /** 回放 / 观战全亮：所有座位的手牌都画正面。 */
    public boolean revealAll;
    /** 局间等确认（POST /api/next 继续）。 */
    public boolean awaitingNext;
    public String lastHandNote = "";
    public boolean confirmBetween;
    /** 桌上没有人类 → 可以用 reveal=1 全亮（后端说了算）。 */
    public boolean spectatorOk;
    /** 供托（立直棒）数量。 */
    public int sticks;
    /** 回放帧的那句话（牌桌上方横幅）。 */
    public String bannerText = "";
    /** 铳牌 / 被鸣的牌（牌河里标红）。 */
    public String dangerTile;
    /** 谁放铳（-1 = 不知道，全桌找）。 */
    public int dangerSeat = -1;
    /** 回放模式。 */
    public boolean replay;
    public int replayIndex = -1;
    public int replayFrames;
    public final List<Seat> seats = new ArrayList<>();
    public final List<String> log = new ArrayList<>();
    public final List<String> dora = new ArrayList<>();
    public final List<Player> details = new ArrayList<>();
    public Pending pending;
    public Hint hint;
    public Result result;
    /** 本局（一手）结算，awaiting_next / finished 时后端会给。 */
    public RoundResult handResult;

    /** 原始 JSON（出问题时打日志用）。 */
    public String raw = "";

    private State() {
    }

    /** 有效人数：三处取最大可信值，缺字段也不会变 0。 */
    public int players() {
        int n = Math.max(playerCount, Math.max(seats.size(), details.size()));
        return n <= 0 ? 3 : n;
    }

    public Seat seat(int i) {
        return i >= 0 && i < seats.size() ? seats.get(i) : null;
    }

    public Player player(int i) {
        return i >= 0 && i < details.size() ? details.get(i) : null;
    }

    /** 这个座位是不是人 / 外部接口（能收 /api/act）。 */
    public boolean isHumanSeat(int seat) {
        Seat s = seat(seat);
        return s != null && ("human".equals(s.type) || "external".equals(s.type));
    }

    /** 这个座位此刻是不是轮到我决策。 */
    public boolean myTurn() {
        return pending != null && pending.yours;
    }

    // ------------------------------------------------------------ DTO

    /** 一次决策请求。 */
    public static final class Pending {
        public int seat;
        public String kind = "";
        public String kindCn = "";
        public double waited;
        public boolean yours;
        /** 刚摸到的那张（打牌决策时有）。 */
        public String drawn;
        /** 被鸣 / 被荣的那张牌（新字段，UI 要高亮）。 */
        public String tile;
        public String tileCn = "";
        public final List<Option> options = new ArrayList<>();
        public final List<String> yourHand = new ArrayList<>();
        public final List<String> yourMelds = new ArrayList<>();

        public boolean isDiscard() {
            return "discard".equals(kind);
        }

        /** 找一个 label（牌 id 或 pass/pon/…）对应的下标，找不到返回 -1。 */
        public int indexOfLabel(String label) {
            String want = Tile.normalize(label);
            for (Option o : options) {
                if (o.label.equals(label) || Tile.normalize(o.label).equals(want)) {
                    return o.index;
                }
            }
            return -1;
        }

        /** 按钮 / 手牌重建用的签名（变了才重建，免得每 500ms 闪一下）。 */
        public String signature() {
            StringBuilder sb = new StringBuilder();
            sb.append(seat).append('/').append(kind).append('/').append(yours).append('/')
                    .append(tile).append('/');
            for (Option o : options) {
                sb.append(o.index).append(':').append(o.label).append(',');
            }
            return sb.toString();
        }
    }

    /** 一个候选项；{@code label} 就是提交给 /api/act 的值。 */
    public static final class Option {
        public int index;
        public String label = "";
        public String cn = "";
        public String tile = "";

        /** 打牌选项对应的牌图 id（非打牌选项返回空串）。 */
        public String tileId() {
            if (!tile.isEmpty() && Tile.isValid(tile)) {
                return tile;
            }
            return Tile.isValid(label) ? label : "";
        }
    }

    /** 一个座位（配置层面的信息）。 */
    public static final class Seat {
        public int index;
        public String type = "";
        public String name = "";
    }

    /** 一个座位的牌桌状态。 */
    public static final class Player {
        public int score;
        public int kita;
        public boolean riichi;
        public boolean ippatsu;
        public boolean isDealer;
        public int handCount;
        public String seatWind = "";
        public final List<String> pond = new ArrayList<>();
        public final List<Meld> melds = new ArrayList<>();
        /** 只有自己的座位不是 null。 */
        public List<String> hand;
        /** 自己手里的赤 5（牌型 id 列表）。 */
        public List<String> red;

        /** 门前清：没有副露，或者只有暗杠。 */
        public boolean menzen() {
            for (Meld m : melds) {
                if (!m.isClosedKan()) {
                    return false;
                }
            }
            return true;
        }
    }

    /** 一组副露。kind: pon / chi / minkan / ankan / kakan。 */
    public static final class Meld {
        public String kind = "";
        public String tile = "";
        public String label = "";

        public boolean isKan() {
            return kind.equals("minkan") || kind.equals("ankan") || kind.equals("kakan");
        }

        /** 暗杠要扣着画。 */
        public boolean isClosedKan() {
            return kind.equals("ankan");
        }

        public String kindCn() {
            return switch (kind) {
                case "pon" -> "碰";
                case "chi" -> "吃";
                case "minkan" -> "明杠";
                case "ankan" -> "暗杠";
                case "kakan" -> "加杠";
                default -> kind;
            };
        }

        /** 摊开的牌（吃是连续三张）。 */
        public List<String> tiles() {
            List<String> out = new ArrayList<>();
            int n = kind.equals("pon") || kind.equals("chi") ? 3 : 4;
            int base = Tile.index(tile);
            for (int i = 0; i < n; i++) {
                out.add(kind.equals("chi") ? Tile.id(base + i) : tile);
            }
            return out;
        }
    }

    /** 牌效提示。 */
    public static final class Hint {
        public int shanten;
        public int ukeire;
        public boolean menzen;
    }

    /** 终局结算。 */
    public static final class Result {
        public final List<Integer> scores = new ArrayList<>();
        public final List<Integer> ranks = new ArrayList<>();
        public final List<Double> rankPoints = new ArrayList<>();
        public final List<Integer> deltas = new ArrayList<>();
        public final List<HandResult> hands = new ArrayList<>();
        public final List<Map<String, Double>> stats = new ArrayList<>();
    }

    /** 一手（本局）的结算 —— /api/state 的 hand_result。 */
    public static final class RoundResult {
        public int handNo;
        public String kind = "";
        public String label = "";
        public String notes = "";
        public String describe = "";
        public boolean tsumo;
        public int winner = -1;
        public int loser = -1;
        public int players;
        public int turns;
        public int sticks;
        public int han;
        public int doraHan;
        public int totalHan;
        public int fu;
        public int yakuman;
        public int points;
        public String winTile = "";
        public final List<Integer> winners = new ArrayList<>();
        public final List<String> hand = new ArrayList<>();
        public final List<Meld> melds = new ArrayList<>();
        public final List<Yaku> yaku = new ArrayList<>();
        public final List<Integer> deltas = new ArrayList<>();
        public final List<Integer> scores = new ArrayList<>();
        public final List<Boolean> tenpai = new ArrayList<>();

        /** 流局 / 九种九牌 / 途中流局。 */
        public boolean isDraw() {
            return kind.equals("draw") || kind.equals("kyuushu") || kind.equals("abort");
        }

        /** 宝牌类（宝牌 / 里宝 / 赤 / 拔北）—— 结算页要跟真役分开显示。 */
        public static boolean isBonus(String key) {
            return key.equals("dora") || key.equals("ura") || key.equals("aka")
                    || key.equals("kita") || key.equals("dora_ura");
        }

        public List<Yaku> realYaku() {
            List<Yaku> out = new ArrayList<>();
            for (Yaku y : yaku) {
                if (!isBonus(y.key)) {
                    out.add(y);
                }
            }
            return out;
        }

        public List<Yaku> bonusYaku() {
            List<Yaku> out = new ArrayList<>();
            for (Yaku y : yaku) {
                if (isBonus(y.key)) {
                    out.add(y);
                }
            }
            return out;
        }
    }

    /** 一个役。 */
    public static final class Yaku {
        public String name = "";
        public String key = "";
        public int han;
    }

    /** 一局的和牌记录。 */
    public static final class HandResult {
        public String kind = "";
        public String text = "";
        public int turns;
        public final List<Integer> deltas = new ArrayList<>();
    }

    // ------------------------------------------------------------ 解析

    public static State fromJson(String text) throws Json.JsonException {
        return fromMap(Json.obj(Json.parse(text)), text);
    }

    private static State fromMap(Map<String, Object> m, String rawText) {
        State s = new State();
        s.raw = rawText;
        s.playerCount = Json.i(m.get("player_count"), 0);
        s.delay = Json.d(m.get("delay"), 0);
        s.allowChi = Json.b(m.get("allow_chi"), false);
        s.allowKita = Json.b(m.get("allow_kita"), false);
        s.seed = (long) Json.d(m.get("seed"), 0);
        s.finished = Json.b(m.get("finished"), false);
        s.you = Json.i(m.get("you"), 0);
        s.wallLeft = Json.i(m.get("wall_left"), 0);
        s.turn = Json.i(m.get("turn"), 0);
        s.dealer = Json.i(m.get("dealer"), 0);
        s.handNo = Json.i(m.get("hand_no"), 1);
        s.roundWind = Json.str(m.get("round_wind"), "东");
        s.waitingFor = Json.str(m.get("waiting_for"), "");
        s.awaitingNext = Json.b(m.get("awaiting_next"), false);
        s.lastHandNote = Json.str(m.get("last_hand_note"), "");
        s.confirmBetween = Json.b(m.get("confirm_between"), false);
        s.spectatorOk = Json.b(m.get("spectator_ok"), false);
        s.sticks = Json.i(m.get("sticks"), 0);

        for (Object o : Json.arr(m.get("seats"))) {
            Map<String, Object> sm = Json.obj(o);
            Seat seat = new Seat();
            seat.index = Json.i(sm.get("index"), s.seats.size());
            seat.type = Json.str(sm.get("type"), "");
            seat.name = Json.str(sm.get("name"), "P" + seat.index);
            s.seats.add(seat);
        }
        s.log.addAll(Json.strList(m.get("log")));
        s.dora.addAll(Json.strList(m.get("dora")));

        // 座位明细：players_detail 优先，退回 players（历史别名）
        List<Object> rows = Json.arr(m.get("players_detail"));
        if (rows.isEmpty()) {
            rows = Json.arr(m.get("players"));
        }
        for (Object o : rows) {
            s.details.add(parsePlayer(Json.obj(o)));
        }

        if (m.get("pending") instanceof Map) {
            Map<String, Object> dm = Json.obj(m.get("pending"));
            Pending pd = new Pending();
            pd.seat = Json.i(dm.get("seat"), -1);
            pd.kind = Json.str(dm.get("kind"), "");
            pd.kindCn = Json.str(dm.get("kind_cn"), pd.kind);
            pd.waited = Json.d(dm.get("waited"), 0);
            pd.yours = Json.b(dm.get("yours"), false);
            pd.drawn = dm.get("drawn") == null ? null : Json.str(dm.get("drawn"), null);
            pd.tile = dm.get("tile") == null ? null : Json.str(dm.get("tile"), null);
            pd.tileCn = Json.str(dm.get("tile_cn"),
                    pd.tile == null ? "" : Tile.cn(pd.tile));
            for (Object o : Json.arr(dm.get("options"))) {
                Map<String, Object> om = Json.obj(o);
                Option op = new Option();
                op.index = Json.i(om.get("index"), pd.options.size());
                op.label = Json.str(om.get("label"), "");
                op.cn = Json.str(om.get("cn"), op.label);
                op.tile = Json.str(om.get("tile"), "");
                pd.options.add(op);
            }
            pd.yourHand.addAll(Json.strList(dm.get("your_hand")));
            pd.yourMelds.addAll(Json.strList(dm.get("your_melds")));
            s.pending = pd;
        }

        if (m.get("hint") instanceof Map) {
            Map<String, Object> hm = Json.obj(m.get("hint"));
            Hint h = new Hint();
            h.shanten = Json.i(hm.get("shanten"), -1);
            h.ukeire = Json.i(hm.get("ukeire"), 0);
            h.menzen = Json.b(hm.get("menzen"), true);
            s.hint = h;
        }

        if (m.get("result") instanceof Map) {
            s.result = parseResult(Json.obj(m.get("result")));
        }
        if (m.get("hand_result") instanceof Map) {
            s.handResult = parseRoundResult(Json.obj(m.get("hand_result")));
        }
        return s;
    }

    /**
     * 把 /api/replay 的一帧变成一份 {@link State}，这样牌桌那套渲染可以直接复用。
     * 帧里每家的 hand 都是全的（没有视角过滤），所以这里一定 revealAll。
     *
     * @param seatNames 座位名（来自 /api/replay 的 seats）
     */
    public static State fromReplayFrame(Map<String, Object> f, List<String> seatNames,
            int viewSeat) {
        State s = new State();
        s.replay = true;
        s.revealAll = true;
        s.replayIndex = Json.i(f.get("i"), 0);
        s.bannerText = Json.str(f.get("text"), "");
        s.handNo = Json.i(f.get("hand_no"), 1);
        s.turn = Json.i(f.get("turn"), 0);
        s.wallLeft = Json.i(f.get("wall"), 0);
        s.roundWind = Json.str(f.get("round_wind"), "东");
        s.dealer = Json.i(f.get("dealer"), 0);
        s.sticks = Json.i(f.get("sticks"), 0);
        s.dora.addAll(Json.strList(f.get("dora")));
        List<Object> rows = Json.arr(f.get("players"));
        s.playerCount = rows.size();
        s.you = viewSeat < 0 || viewSeat >= rows.size() ? 0 : viewSeat;
        for (int i = 0; i < rows.size(); i++) {
            s.details.add(parsePlayer(Json.obj(rows.get(i))));
            Seat seat = new Seat();
            seat.index = i;
            seat.type = "replay";
            seat.name = i < seatNames.size() ? seatNames.get(i) : "P" + i;
            s.seats.add(seat);
        }
        // 分数从帧里单独给（players 里没有 score）
        List<Object> scores = Json.arr(f.get("scores"));
        for (int i = 0; i < s.details.size() && i < scores.size(); i++) {
            s.details.get(i).score = Json.i(scores.get(i), 0);
        }
        // 和牌帧：荣和的话把放铳那张标红
        if (f.get("agari") instanceof Map) {
            Map<String, Object> ag = Json.obj(f.get("agari"));
            boolean tsumo = Json.b(ag.get("tsumo"), false);
            String tile = Json.str(ag.get("tile"), "");
            if (!tsumo && !tile.isEmpty()) {
                s.dangerTile = tile;
                s.dangerSeat = Json.i(ag.get("loser"), -1);
            }
        }
        return s;
    }

    private static Player parsePlayer(Map<String, Object> pm) {
        Player p = new Player();
        p.score = Json.i(pm.get("score"), 0);
        p.kita = Json.i(pm.get("kita"), 0);
        p.riichi = Json.b(pm.get("riichi"), false);
        p.ippatsu = Json.b(pm.get("ippatsu"), false);
        p.isDealer = Json.b(pm.get("is_dealer"), false);
        p.handCount = Json.i(pm.get("hand_count"), 0);
        p.seatWind = Json.str(pm.get("seat_wind"), "");
        p.pond.addAll(Json.strList(pm.get("pond")));
        for (Object mo : Json.arr(pm.get("melds"))) {
            Map<String, Object> mm = Json.obj(mo);
            Meld meld = new Meld();
            meld.kind = Json.str(mm.get("kind"), "");
            meld.tile = Json.str(mm.get("tile"), "");
            meld.label = Json.str(mm.get("label"), "");
            p.melds.add(meld);
        }
        if (pm.get("hand") instanceof List) {
            p.hand = Json.strList(pm.get("hand"));
        }
        if (pm.get("red") instanceof List) {
            p.red = Json.strList(pm.get("red"));
        }
        return p;
    }

    private static RoundResult parseRoundResult(Map<String, Object> m) {
        RoundResult r = new RoundResult();
        r.handNo = Json.i(m.get("hand_no"), 0);
        r.kind = Json.str(m.get("kind"), "");
        r.label = Json.str(m.get("label"), "");
        r.notes = Json.str(m.get("notes"), "");
        r.describe = Json.str(m.get("describe"), "");
        r.tsumo = Json.b(m.get("tsumo"), false);
        r.winner = Json.i(m.get("winner"), -1);
        r.loser = Json.i(m.get("loser"), -1);
        r.players = Json.i(m.get("players"), 0);
        r.turns = Json.i(m.get("turns"), 0);
        r.sticks = Json.i(m.get("sticks"), 0);
        r.han = Json.i(m.get("han"), 0);
        r.doraHan = Json.i(m.get("dora_han"), 0);
        r.totalHan = Json.i(m.get("total_han"), 0);
        r.fu = Json.i(m.get("fu"), 0);
        r.yakuman = Json.i(m.get("yakuman"), 0);
        r.points = Json.i(m.get("points"), 0);
        r.winTile = Json.str(m.get("win_tile"), "");
        for (Object o : Json.arr(m.get("winners"))) {
            r.winners.add(Json.i(o, -1));
        }
        r.hand.addAll(Json.strList(m.get("hand")));
        for (Object o : Json.arr(m.get("melds"))) {
            Map<String, Object> mm = Json.obj(o);
            Meld meld = new Meld();
            meld.kind = Json.str(mm.get("kind"), "");
            meld.tile = Json.str(mm.get("tile"), "");
            meld.label = Json.str(mm.get("label"), "");
            r.melds.add(meld);
        }
        for (Object o : Json.arr(m.get("yaku"))) {
            Map<String, Object> ym = Json.obj(o);
            Yaku y = new Yaku();
            y.name = Json.str(ym.get("name"), "");
            y.key = Json.str(ym.get("key"), "");
            y.han = Json.i(ym.get("han"), 0);
            r.yaku.add(y);
        }
        for (Object o : Json.arr(m.get("deltas"))) {
            r.deltas.add(Json.i(o, 0));
        }
        for (Object o : Json.arr(m.get("scores"))) {
            r.scores.add(Json.i(o, 0));
        }
        for (Object o : Json.arr(m.get("tenpai"))) {
            r.tenpai.add(Json.b(o, false));
        }
        return r;
    }

    private static Result parseResult(Map<String, Object> rm) {
        Result r = new Result();
        for (Object o : Json.arr(rm.get("scores"))) {
            r.scores.add(Json.i(o, 0));
        }
        for (Object o : Json.arr(rm.get("ranks"))) {
            r.ranks.add(Json.i(o, 0));
        }
        for (Object o : Json.arr(rm.get("rank_points"))) {
            r.rankPoints.add(Json.d(o, 0));
        }
        for (Object o : Json.arr(rm.get("deltas"))) {
            r.deltas.add(Json.i(o, 0));
        }
        for (Object o : Json.arr(rm.get("hands"))) {
            Map<String, Object> hm = Json.obj(o);
            HandResult hr = new HandResult();
            hr.kind = Json.str(hm.get("kind"), "");
            hr.text = Json.str(hm.get("text"), "");
            hr.turns = Json.i(hm.get("turns"), 0);
            for (Object d : Json.arr(hm.get("deltas"))) {
                hr.deltas.add(Json.i(d, 0));
            }
            r.hands.add(hr);
        }
        for (Object o : Json.arr(rm.get("stats"))) {
            Map<String, Double> one = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : Json.obj(o).entrySet()) {
                one.put(e.getKey(), Json.d(e.getValue(), 0));
            }
            r.stats.add(one);
        }
        return r;
    }

    /** 我自己的手牌：pending 里的优先，否则座位明细里的。 */
    public List<String> myHand() {
        if (pending != null && !pending.yourHand.isEmpty()) {
            return new ArrayList<>(pending.yourHand);
        }
        Player me = player(you);
        if (me != null && me.hand != null) {
            return new ArrayList<>(me.hand);
        }
        return new ArrayList<>();
    }

    /** 我手里的赤 5 牌型 id 列表。 */
    public List<String> myRed() {
        Player me = player(you);
        return me != null && me.red != null ? me.red : new ArrayList<>();
    }
}
