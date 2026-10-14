package maj;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 牌桌布局的「铁律」—— 一处定义，两处用：
 * <ul>
 *   <li>{@link TableLayout#build} 用它在大 / 中 / 小几档牌里挑<b>第一档真的摆得下的</b>
 *       （所以牌会尽量大：同样是 1000×700，能塞下大牌就用大牌）；</li>
 *   <li>{@link LayoutCheck} 用它跑全尺寸矩阵自检。</li>
 * </ul>
 *
 * <p>规矩：
 * <ol>
 *   <li>每个座位的四块（名牌 / 手牌 / 副露 / 牌河）两两不重叠，且都在桌面里；</li>
 *   <li>每一张牌都在自己那块矩形里，不越界；</li>
 *   <li>跨块（换座位 / 换区）一张都不许压；同块挤不下时允许叠一点，
 *       但同一行/列相邻两张至少露出 {@value #MIN_VISIBLE_RATIO}；</li>
 *   <li>数据里有几张牌就画几张，不许悄悄吞掉；</li>
 *   <li>中央信息区和顶部横幅不压任何座位、也不压任何牌。</li>
 * </ol>
 */
public final class LayoutRules {

    /** 同块叠放时，同一行/列相邻两张最少露出多少（跟 TableLayout 的常量对齐）。 */
    static final double MIN_VISIBLE_RATIO = TableLayout.MIN_VISIBLE - 0.02;
    /** 牌河的底线更松一点（整片网格，压一半也看得清）。 */
    static final double POND_VISIBLE_RATIO = TableLayout.POND_MIN_VISIBLE - 0.02;
    /** 判定用的容差：像素取整别把刚好及格的判成违规。 */
    private static final int EPS = 1;

    /** 一条违规。 */
    public record Violation(String where, String what) {
        @Override
        public String toString() {
            return "    ✗ " + where + "：" + what;
        }
    }

    private LayoutRules() {
    }

    // ------------------------------------------------------------ 规则

    /**
     * 查一份已经算好的布局。
     *
     * @param tag    报错前缀（一般写窗口尺寸）
     * @param stats  顺手记统计（全矩阵里最小牌宽 / 最窄露出比例），只在跑自检时为 true
     */
    public static List<Violation> check(TableLayout L, State s, String tag, boolean stats) {
        List<Violation> out = new ArrayList<>();
        if (L == null || L.felt.w <= 0) {
            out.add(new Violation("整体", tag + " 太小，布局没算出来"));
            return out;
        }

        // 座位 → slot 反查
        Map<Integer, TableLayout.SeatView> bySeat = new LinkedHashMap<>();
        for (int slot : L.usedSlots()) {
            TableLayout.SeatView v = L.view(slot);
            if (v != null) {
                bySeat.put(v.seat, v);
            }
        }

        // 规则 1 + 2：四块矩形两两不重叠、都在桌面里
        for (TableLayout.SeatView v : bySeat.values()) {
            String[] names = {"名牌", "手牌", "副露", "牌河"};
            Rect[] rects = {v.plate, v.hand, v.meld, v.pond};
            for (int i = 0; i < rects.length; i++) {
                if (rects[i].w <= 0 || rects[i].h <= 0) {
                    continue;                       // 空块（没副露 / 没牌河）跳过
                }
                if (!inside(rects[i], L.felt)) {
                    out.add(new Violation(tag + " P" + v.seat + " " + names[i],
                            names[i] + " " + rects[i] + " 超出桌面 " + L.felt));
                }
                for (int k = i + 1; k < rects.length; k++) {
                    int ov = overlap(rects[i], rects[k]);
                    if (ov > 0) {
                        out.add(new Violation(tag + " P" + v.seat,
                                names[i] + " ∩ " + names[k] + " = " + ov + " px²（"
                                        + rects[i] + " vs " + rects[k] + "）"));
                    }
                }
            }
            // 名牌跟手牌也各占一块（上面已经两两查过）
        }

        // 规则 3 + 4：每张牌在自己的块里，且任意两张不重叠
        for (int i = 0; i < L.spots.size(); i++) {
            TableLayout.Spot sp = L.spots.get(i);
            if (!inside(sp.rect, L.felt)) {
                out.add(new Violation(tag + " " + zoneName(sp.zone) + " P" + sp.seat,
                        "牌 " + sp.tileId + " " + sp.rect + " 超出桌面"));
            }
            TableLayout.SeatView v = bySeat.get(sp.seat);
            Rect zoneRect = switch (sp.zone) {
                case HAND -> v == null ? null : v.hand;
                case MELD -> v == null ? null : v.meld;
                case POND -> v == null ? null : v.pond;
                case DORA -> L.centerDora;
            };
            if (zoneRect != null && zoneRect.w > 0 && !inside(sp.rect, zoneRect)) {
                out.add(new Violation(tag + " " + zoneName(sp.zone) + " P" + sp.seat,
                        "牌 " + sp.tileId + " " + sp.rect + " 越出自己的块 " + zoneRect));
            }
            for (int k = i + 1; k < L.spots.size(); k++) {
                TableLayout.Spot other = L.spots.get(k);
                // 同一块里允许相邻牌叠一点（窗口挤不下时的常规画法，牌大比不叠重要），
                // 但**跨块**（换座位 / 换区）一张都不许压 —— 那才是「压到别人牌上」。
                // 同块叠放的底线下面单独查：每张至少露 MIN_VISIBLE。
                // 同一块（同一家同一类）允许叠一点，底线由下面的「露出比例」把关；
                // 手牌 / 副露 / 牌河都可能挤，宝牌在中央不会挤。
                if (sp.zone == other.zone && sp.seat == other.seat
                        && sp.zone != TableLayout.Zone.DORA) {
                    continue;
                }
                int ov = overlap(sp.rect, other.rect);
                if (ov > 0) {
                    out.add(new Violation(tag + " 牌与牌",
                            zoneName(sp.zone) + " P" + sp.seat + " " + sp.tileId + " " + sp.rect
                                    + " ∩ " + zoneName(other.zone) + " P" + other.seat + " "
                                    + other.tileId + " " + other.rect + " = " + ov + " px²"));
                }
            }
        }

        // 规则 4b：同块内叠放可以，但每张牌至少得露出一条 ≥MIN_VISIBLE 的边；
        // 顺便记下全矩阵里最惨的一档（「牌是不是太小/叠得太狠」看这个数）
        for (TableLayout.Zone zone : new TableLayout.Zone[] {
                TableLayout.Zone.HAND, TableLayout.Zone.POND, TableLayout.Zone.MELD}) {
            for (int seat = 0; seat < s.players(); seat++) {
                for (double[] step : pitches(L, zone, seat)) {
                    String key = zoneName(zone);
                    if (stats) {
                        LEAST_VISIBLE.merge(key, step[0], Math::min);
                    }
                    double floor = zone == TableLayout.Zone.POND
                            ? POND_VISIBLE_RATIO : MIN_VISIBLE_RATIO;
                    if (step[0] < floor) {
                        out.add(new Violation(tag + " " + key + " P" + seat,
                                "同一行/列相邻两张只露出 " + Math.round(step[0] * 100)
                                        + "%（叠放底线 " + Math.round(floor * 100) + "%）"));
                    }
                }
            }
        }

        // 顺手记一下各类牌最小画到多大（支持范围内才记）
        if (stats) {
            for (TableLayout.Spot sp : L.spots) {
                String key = zoneName(sp.zone);
                int[] cur = SMALLEST.get(key);
                if (cur == null || sp.rect.w < cur[0]) {
                    SMALLEST.put(key, new int[] {sp.rect.w, sp.rect.h});
                    SMALLEST_AT.put(key, tag);
                }
            }
        }

        // 规则 7：数据里有几张牌就得画几张 —— 摆不下的时候不许悄悄丢掉
        for (int seat = 0; seat < s.players(); seat++) {
            checkCount(out, tag, L, seat, TableLayout.Zone.POND, pondExpected(s, seat));
            checkCount(out, tag, L, seat, TableLayout.Zone.MELD, meldExpected(s, seat));
            checkCount(out, tag, L, seat, TableLayout.Zone.HAND, handExpected(s, seat));
        }
        checkCount(out, tag, L, -1, TableLayout.Zone.DORA, s.dora.size());

        // 规则 5：中央信息区不压座位、不压牌
        if (L.center.w > 0) {
            if (!inside(L.center, L.felt)) {
                out.add(new Violation(tag + " 中央", "信息区 " + L.center + " 超出桌面"));
            }
            for (TableLayout.SeatView v : bySeat.values()) {
                Rect[] rects = {v.plate, v.hand, v.meld, v.pond};
                for (Rect r : rects) {
                    if (r.w > 0 && overlap(L.center, r) > 0) {
                        out.add(new Violation(tag + " 中央 ∩ P" + v.seat,
                                "信息区 " + L.center + " 压住 " + r));
                    }
                }
            }
            for (TableLayout.Spot sp : L.spots) {
                if (sp.zone != TableLayout.Zone.DORA && overlap(L.center, sp.rect) > 0) {
                    out.add(new Violation(tag + " 中央 ∩ 牌",
                            "信息区 " + L.center + " 压住 " + sp.rect + " (" + sp.tileId + ")"));
                }
            }
        }

        // 规则 6：顶部横幅不压座位 / 不压牌（只有这次真给横幅留了位置才查）
        if (L.bannerReserved && L.banner.w > 0) {
            if (!inside(L.banner, L.felt)) {
                out.add(new Violation(tag + " 横幅", "横幅 " + L.banner + " 超出桌面"));
            }
            for (TableLayout.SeatView v : bySeat.values()) {
                for (Rect r : new Rect[] {v.plate, v.hand, v.meld, v.pond}) {
                    if (r.w > 0 && overlap(L.banner, r) > 0) {
                        out.add(new Violation(tag + " 横幅 ∩ P" + v.seat,
                                "横幅 " + L.banner + " 压住 " + r));
                    }
                }
            }
            for (TableLayout.Spot sp : L.spots) {
                if (overlap(L.banner, sp.rect) > 0) {
                    out.add(new Violation(tag + " 横幅 ∩ 牌",
                            "横幅 " + L.banner + " 压住 " + sp.rect));
                }
            }
        }
        return out;
    }

    /** 全矩阵里每一类牌的最小边长 / 最窄露出比例（自检用）。 */
    private static final Map<String, int[]> SMALLEST = new LinkedHashMap<>();
    private static final Map<String, String> SMALLEST_AT = new LinkedHashMap<>();
    private static final Map<String, Double> LEAST_VISIBLE = new LinkedHashMap<>();

    public static Map<String, int[]> smallest() {
        return SMALLEST;
    }

    public static Map<String, String> smallestAt() {
        return SMALLEST_AT;
    }

    public static Map<String, Double> leastVisible() {
        return LEAST_VISIBLE;
    }

    /** 关键几条：给 {@link TableLayout#build} 挑档位用（不看统计、不做格式化）。 */
    public static boolean ok(TableLayout L, State s) {
        return check(L, s, "", false).isEmpty();
    }

    private static String zoneName(TableLayout.Zone z) {
        return switch (z) {
            case HAND -> "手牌";
            case MELD -> "副露";
            case POND -> "牌河";
            case DORA -> "宝牌";
        };
    }

    // ------------------------------------------------------------ 张数

    /** 这一家的手牌应该画几张（后端只给主视角 hand，别家只给 hand_count）。 */
    private static int handExpected(State s, int seat) {
        State.Player p = s.player(seat);
        if (seat == s.you) {
            return s.myHand().size();          // 刚摸的那张是从手牌里挪到末尾，张数不变
        }
        if (p == null) {
            return 0;
        }
        return p.hand != null && !p.hand.isEmpty() ? p.hand.size() : p.handCount;
    }

    private static int meldExpected(State s, int seat) {
        State.Player p = s.player(seat);
        if (p == null) {
            return 0;
        }
        int n = 0;
        for (State.Meld m : p.melds) {
            n += m.tiles().size();
        }
        return n;
    }

    private static int pondExpected(State s, int seat) {
        State.Player p = s.player(seat);
        return p == null ? 0 : p.pond.size();
    }

    private static void checkCount(List<Violation> out, String tag, TableLayout L, int seat,
            TableLayout.Zone zone, int expected) {
        int got = 0;
        for (TableLayout.Spot sp : L.spots) {
            if (sp.zone == zone && sp.seat == seat) {
                got++;
            }
        }
        if (got != expected) {
            out.add(new Violation(tag + " " + zoneName(zone) + " P" + seat,
                    "数据里 " + expected + " 张，画出来 " + got + " 张（不许悄悄丢牌）"));
        }
    }

    // ------------------------------------------------------------ 几何

    /**
     * 一块里同一行（列）相邻两张的「露出比例」= 步进 ÷ 牌长。
     *
     * <p>手牌 / 牌河只有真挤不下时才叠一点，这个比例就是每张实际露出来的那一条边；
     * 1.0 = 完全没叠。横排按 x 分组、竖排按 y 分组。
     */
    private static List<double[]> pitches(TableLayout L, TableLayout.Zone zone, int seat) {
        Map<String, List<TableLayout.Spot>> lines = new LinkedHashMap<>();
        for (TableLayout.Spot sp : L.spots) {
            if (sp.zone != zone || sp.seat != seat) {
                continue;
            }
            String key = sp.rotated ? "y" + sp.rect.x : "x" + sp.rect.y;
            lines.computeIfAbsent(key, k -> new ArrayList<>()).add(sp);
        }
        List<double[]> out = new ArrayList<>();
        for (List<TableLayout.Spot> line : lines.values()) {
            line.sort((a, b) -> Integer.compare(a.rotated ? a.rect.y : a.rect.x,
                    b.rotated ? b.rect.y : b.rect.x));
            for (int i = 1; i < line.size(); i++) {
                TableLayout.Spot a = line.get(i - 1);
                TableLayout.Spot b = line.get(i);
                int along = a.rotated ? a.rect.h : a.rect.w;
                int step = a.rotated ? b.rect.y - a.rect.y : b.rect.x - a.rect.x;
                if (along > 0) {
                    out.add(new double[] {Math.min(1.0, step / (double) along)});
                }
            }
        }
        return out;
    }

    private static boolean inside(Rect inner, Rect outer) {
        return inner.x >= outer.x - EPS && inner.y >= outer.y - EPS
                && inner.right() <= outer.right() + EPS
                && inner.bottom() <= outer.bottom() + EPS;
    }

    private static int overlap(Rect a, Rect b) {
        int w = Math.min(a.right(), b.right()) - Math.max(a.x, b.x);
        int h = Math.min(a.bottom(), b.bottom()) - Math.max(a.y, b.y);
        return w <= 0 || h <= 0 ? 0 : w * h;
    }
}
