package maj;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 后端 HTTP 接口的 Java 客户端（对应 aimodel/ui/server.py 的路由）。
 *
 * <p>不碰 Swing，后台线程随便调。业务错误（比如 /api/act 下标越界时
 * 后端回 400 + {@code {"ok":false,"error":"下标越界：999…"}}）不抛异常，
 * 而是放进 {@link ActResult}，界面照样能把原因显示出来。
 */
public final class MajClient {

    private static final Duration STATE_TIMEOUT = Duration.ofSeconds(6);
    private static final Duration ACT_TIMEOUT = Duration.ofSeconds(6);
    private static final Duration CONFIG_TIMEOUT = Duration.ofSeconds(20);
    private static final Duration TILE_TIMEOUT = Duration.ofSeconds(8);

    private final String base;

    public MajClient(String base) {
        String b = base == null || base.isBlank() ? "http://127.0.0.1:8765" : base.trim();
        while (b.endsWith("/")) {
            b = b.substring(0, b.length() - 1);
        }
        this.base = b;
    }

    public String base() {
        return base;
    }

    // ------------------------------------------------------------ 局面

    /** 拉一份以 seat 视角看的局面。 */
    public State state(int seat) throws MajException {
        return state(seat, false);
    }

    /**
     * 拉一份局面。
     *
     * @param reveal 观战全亮（{@code reveal=1}）—— 只有后端 {@code spectator_ok=true}
     *               （桌上没人）时才真的给别家手牌，有人类在座后端会拒绝，这是防作弊，
     *               我们不去绕它。
     */
    public State state(int seat, boolean reveal) throws MajException {
        String url = base + "/api/state?seat=" + Math.max(0, seat) + (reveal ? "&reveal=1" : "");
        String text = Http.getText(url, STATE_TIMEOUT);
        try {
            return State.fromJson(text);
        } catch (Json.JsonException e) {
            throw new MajException("state JSON 解析失败：" + e.getMessage(), e);
        }
    }

    // ------------------------------------------------------------ 回放

    /** {@code /api/replay} 的元信息。 */
    public record ReplayMeta(int frames, int players, boolean spectatorOk, boolean finished,
            List<String> seats) {
        static ReplayMeta from(Map<String, Object> m) {
            return new ReplayMeta(Json.i(m.get("frames"), 0),
                    Json.i(m.get("players"), Json.i(m.get("player_count"), 3)),
                    Json.b(m.get("spectator_ok"), false),
                    Json.b(m.get("finished"), false),
                    Json.strList(m.get("seats")));
        }
    }

    /** 回放总帧数 / 人数 / 座位名。 */
    public ReplayMeta replayMeta() throws MajException {
        return ReplayMeta.from(parse(Http.getText(base + "/api/replay", CONFIG_TIMEOUT)));
    }

    /** 取一批帧（含头含尾）。一次 100~200 帧，别一帧一个请求。 */
    public List<Map<String, Object>> replayFrames(int from, int to) throws MajException {
        String url = base + "/api/replay?from=" + Math.max(0, from) + "&to=" + Math.max(0, to);
        Map<String, Object> m = parse(Http.getText(url, CONFIG_TIMEOUT));
        List<Map<String, Object>> out = new ArrayList<>();
        for (Object o : Json.arr(m.get("frames"))) {
            out.add(Json.obj(o));
        }
        return out;
    }

    /** 单帧。 */
    public Map<String, Object> replayFrame(int i) throws MajException {
        return parse(Http.getText(base + "/api/replay?i=" + Math.max(0, i), STATE_TIMEOUT));
    }

    /** 局间确认：继续下一局。 */
    public Map<String, Object> nextHand() throws MajException {
        return post("/api/next", null, CONFIG_TIMEOUT).raw();
    }

    /** 服务器活着吗（拿 /api/models 探活）。 */
    public boolean ping() {
        try {
            models();
            return true;
        } catch (MajException e) {
            return false;
        }
    }

    // ------------------------------------------------------------ 动作

    /** 一次 /api/act 的结果。 */
    public record ActResult(boolean ok, String error, int status, Map<String, Object> raw) {
        public String describe() {
            return ok ? "ok" : "失败(" + status + "): " + error;
        }
    }

    /** 提交一个选项下标。 */
    public ActResult act(int seat, int index) throws MajException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("seat", seat);
        body.put("index", index);
        return post("/api/act", body, ACT_TIMEOUT);
    }

    /** 用 label（牌 id / pass / pon …）提交。 */
    public ActResult actLabel(int seat, String label) throws MajException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("seat", seat);
        body.put("label", label);
        return post("/api/act", body, ACT_TIMEOUT);
    }

    // ------------------------------------------------------------ 配置

    /** 可用模型 / 内置座位名。 */
    public Models models() throws MajException {
        return Models.from(parse(Http.getText(base + "/api/models", STATE_TIMEOUT)));
    }

    /** 当前配置。 */
    public Config config() throws MajException {
        return Config.from(parse(Http.getText(base + "/api/config", STATE_TIMEOUT)));
    }

    /** 换模式重开一局。 */
    public Map<String, Object> setConfig(Config cfg) throws MajException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("players", cfg.players);
        body.put("seats", cfg.seats);
        body.put("delay", cfg.delay);
        body.put("seed", cfg.seed);
        if (cfg.humanWait > 0) {
            body.put("human_wait", cfg.humanWait);
        }
        return post("/api/config", body, CONFIG_TIMEOUT).raw();
    }

    /** 同配置再来一局。 */
    public Map<String, Object> newGame() throws MajException {
        return post("/api/new", null, CONFIG_TIMEOUT).raw();
    }

    // ------------------------------------------------------------ 牌图

    /** 取一张牌图的原始字节。 */
    public byte[] tilePng(String tileId) throws MajException {
        if (!Tile.isValid(tileId)) {
            throw new MajException("不是合法的牌 id: " + tileId);
        }
        return Http.getBytes(base + "/tiles/" + tileId + ".png", TILE_TIMEOUT);
    }

    // ------------------------------------------------------------ 内部

    private ActResult post(String path, Map<String, Object> body, Duration timeout)
            throws MajException {
        Http.Response r = Http.postJson(base + path,
                Json.write(body == null ? Map.of() : body), timeout);
        Map<String, Object> parsed;
        try {
            parsed = Json.obj(Json.parse(r.body()));
        } catch (Json.JsonException e) {
            if (!r.ok2xx()) {
                throw new MajException("HTTP " + r.status() + " " + path + "：" + r.body());
            }
            throw new MajException("JSON 解析失败：" + e.getMessage(), e);
        }
        String error = parsed.get("error") == null ? "" : Json.str(parsed.get("error"));
        boolean ok = Json.b(parsed.get("ok"), false) && r.ok2xx();
        return new ActResult(ok, error, r.status(), parsed);
    }

    private Map<String, Object> parse(String text) throws MajException {
        try {
            return Json.obj(Json.parse(text));
        } catch (Json.JsonException e) {
            throw new MajException("JSON 解析失败：" + e.getMessage(), e);
        }
    }

    /** {@code /api/models} 的返回（含中文名 / 风格 / 适用人数 / 推荐组合）。 */
    public static final class Models {
        public final List<String> models = new ArrayList<>();
        public final List<String> builtin = new ArrayList<>();
        public final List<String> seats = new ArrayList<>();
        /** 模型名 -> 说明。 */
        public final Map<String, ModelInfo> info = new LinkedHashMap<>();
        /** 人数（"3" / "4"）-> 推荐模型顺序。 */
        public final Map<String, List<String>> suggest = new LinkedHashMap<>();

        /** 一个模型的说明。 */
        public static final class ModelInfo {
            public String cn = "";
            public String style = "";
            public String reward = "";
            /** 0 = 没写（通用）。 */
            public int players;

            /** 下拉里显示的名字：{@code maj-flash · 速攻}。 */
            public String label(String name) {
                if (!style.isEmpty() && !cn.isEmpty()) {
                    return name + " · " + style;
                }
                if (!cn.isEmpty()) {
                    return name + " · " + cn;
                }
                if (!style.isEmpty()) {
                    return name + " · " + style;
                }
                return name;
            }
        }

        static Models from(Map<String, Object> m) {
            Models out = new Models();
            out.models.addAll(Json.strList(m.get("models")));
            out.builtin.addAll(Json.strList(m.get("builtin")));
            out.seats.addAll(Json.strList(m.get("seats")));
            for (Map.Entry<String, Object> e : Json.obj(m.get("info")).entrySet()) {
                Map<String, Object> im = Json.obj(e.getValue());
                ModelInfo mi = new ModelInfo();
                mi.cn = Json.str(im.get("cn"), "");
                mi.style = Json.str(im.get("style"), "");
                mi.reward = Json.str(im.get("reward"), "");
                mi.players = Json.i(im.get("players"), 0);
                out.info.put(e.getKey(), mi);
            }
            for (Map.Entry<String, Object> e : Json.obj(m.get("suggest")).entrySet()) {
                out.suggest.put(e.getKey(), Json.strList(e.getValue()));
            }
            return out;
        }

        /** 这个人数下「正经能用」的模型：推荐组合优先，然后同人数，最后没标人数的通用模型。 */
        public List<String> modelsFor(int players) {
            List<String> out = new ArrayList<>();
            for (String s : suggest.getOrDefault(String.valueOf(players), List.of())) {
                if (models.contains(s) && !out.contains(s)) {
                    out.add(s);
                }
            }
            for (String s : models) {
                if (playersOf(s) == players && !out.contains(s)) {
                    out.add(s);
                }
            }
            for (String s : models) {
                if (playersOf(s) == 0 && !out.contains(s)) {
                    out.add(s);
                }
            }
            return out;
        }

        /** 模型适用人数；没这个模型返回 -1，没标人数返回 0。 */
        public int playersOf(String name) {
            ModelInfo mi = info.get(name);
            if (mi != null) {
                return mi.players;
            }
            return models.contains(name) ? 0 : -1;
        }

        /** 这个人数推荐的座位表（P0=人类，其余尽量填模型）。 */
        public List<String> defaultSeats(int players) {
            List<String> pool = modelsFor(players);
            List<String> out = new ArrayList<>();
            for (int i = 0; i < players; i++) {
                if (i == 0) {
                    out.add("human");
                } else if (!pool.isEmpty()) {
                    out.add(pool.get((i - 1) % pool.size()));
                } else {
                    out.add("teacher");
                }
            }
            return out;
        }

        /** 全是模型（观战）时的座位表。 */
        public List<String> spectateSeats(int players) {
            List<String> pool = modelsFor(players);
            List<String> out = new ArrayList<>();
            for (int i = 0; i < players; i++) {
                out.add(pool.isEmpty() ? "teacher" : pool.get(i % pool.size()));
            }
            return out;
        }

        /** 下拉里的文字：模型带风格，内置项给人话。 */
        public String label(String value) {
            ModelInfo mi = info.get(value);
            if (mi != null) {
                return mi.label(value);
            }
            return switch (value) {
                case "human" -> "人类（我自己打）";
                case "external" -> "外部接口（external）";
                case "teacher" -> "教师（内置 AI）";
                case "random" -> "随机（random）";
                default -> models.contains(value) ? value : value + "（未知）";
            };
        }

        /** 这个座位值是不是模型（而不是 human/teacher/…）。 */
        public boolean isModel(String value) {
            return models.contains(value);
        }

        /** 座位值认不认。 */
        public boolean known(String value) {
            return models.contains(value) || builtin.contains(value)
                    || seats.contains(value) || value.equals("teacher") || value.equals("random")
                    || value.equals("human") || value.equals("external");
        }

        /** 一个座位能选的全部取值：先模型（按人数过滤），再内置项。 */
        public List<String> seatChoices(int players) {
            List<String> out = new ArrayList<>(modelsFor(players));
            for (String s : List.of("human", "teacher", "random", "external")) {
                if (!out.contains(s)) {
                    out.add(s);
                }
            }
            for (String s : builtin) {
                if (!out.contains(s)) {
                    out.add(s);
                }
            }
            for (String s : seats) {
                if (!out.contains(s)) {
                    out.add(s);
                }
            }
            return out;
        }

        /** 兼容老调用：默认三麻的候选。 */
        public List<String> seatChoices() {
            return seatChoices(3);
        }

        /** 观战 / 默认用的模型名（没有模型就退回教师）。 */
        public String spectateSeat() {
            List<String> pool = modelsFor(3);
            return pool.isEmpty() ? (builtin.isEmpty() ? "teacher" : builtin.get(0)) : pool.get(0);
        }
    }

    /** {@code /api/config} 的返回 / 请求体。 */
    public static final class Config {
        public int players = 3;
        public final List<String> seats = new ArrayList<>();
        public double delay = 1.0;
        public long seed = 1;
        /** 人 / 外部座位等多久算超时（秒，0 = 用后端默认）。 */
        public double humanWait;

        public Config() {
        }

        public Config(int players, List<String> seats, double delay, long seed) {
            this.players = players;
            this.seats.addAll(seats);
            this.delay = delay;
            this.seed = seed;
        }

        /** 座位表里第一个「人」的座位号，没有就 -1。 */
        public int firstHumanSeat() {
            for (int i = 0; i < seats.size(); i++) {
                if ("human".equals(seats.get(i)) || "external".equals(seats.get(i))) {
                    return i;
                }
            }
            return -1;
        }

        static Config from(Map<String, Object> m) {
            Config c = new Config();
            c.players = Json.i(m.get("players"), Json.i(m.get("player_count"), 3));
            c.seats.addAll(Json.strList(m.get("seats")));
            if (c.players <= 0) {
                c.players = Math.max(3, c.seats.size());
            }
            c.delay = Json.d(m.get("delay"), 1.0);
            c.seed = (long) Json.d(m.get("seed"), 1);
            c.humanWait = Json.d(m.get("human_wait"), 0);
            return c;
        }

        @Override
        public String toString() {
            return players + "人 " + seats + " 节拍 " + delay + "s 种子 " + seed;
        }
    }
}
