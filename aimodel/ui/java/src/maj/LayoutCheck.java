package maj;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 「布局铁律」的可执行检查（headless，不联网、不开窗口）。
 *
 * <pre>
 *   java -cp out maj.LayoutCheck
 *   java -cp out maj.LayoutCheck --verbose          # 每条尺寸都打一行
 *   java -cp out maj.LayoutCheck --sizes 1000x700   # 只查某几个尺寸
 * </pre>
 *
 * <p>要守的规矩（每一条都是硬性要求）：
 * <ol>
 *   <li>每个座位的四块 —— <b>名牌 / 手牌 / 副露 / 牌河</b> —— 各自占一个矩形，两两不重叠；</li>
 *   <li>这四块都在桌面（绒布）里面，不会被窗口边裁掉；</li>
 *   <li>每一张牌都画在自己的那块矩形里（手牌不越到副露 / 牌河上，反之亦然）；</li>
 *   <li>任何两张牌都不重叠；</li>
 *   <li>中央信息区不压任何座位、也不压任何牌；</li>
 *   <li>顶部横幅（回放 / 局间提示）留出位置后，上面的规矩依然成立。</li>
 * </ol>
 *
 * <p>检查矩阵：{三麻, 四麻} × {空桌, 满副露长牌河, 全亮} × 每个主视角 × 一串窗口尺寸。
 * 全过退出码 0，有问题退出码 1 并打印前若干条冲突。
 */
public final class LayoutCheck {

    /** 一个待查的局面。 */
    private record Case(String name, State state) {
    }

    private static final int MAX_PRINT = 6;

    /** 支持的最小窗口：MainFrame 的最小尺寸是 1080×760，验收线是 1000×700。 */
    private static final int FLOOR_W = 1000;
    private static final int FLOOR_H = 700;

    private LayoutCheck() {
    }

    public static void main(String[] args) throws Json.JsonException {
        boolean verbose = false;
        boolean all = false;
        List<int[]> sizes = defaultSizes();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--verbose", "-v" -> verbose = true;
                case "--all" -> all = true;
                case "--sizes" -> sizes = parseSizes(args[++i]);
                case "--tiers" -> {
                    tiers(Integer.parseInt(args[++i]), args[++i]);
                    return;
                }
                case "--dump" -> {
                    dump(Integer.parseInt(args[++i]), args[++i]);
                    return;
                }
                default -> {
                    System.out.println("用法：maj.LayoutCheck [--verbose] [--all]"
                            + " [--sizes 1000x700,1400x900]");
                    System.exit(2);
                }
            }
        }
        if (all) {
            sizes.addAll(belowFloorSizes());
        }

        System.out.println("═══ 牌桌布局铁律检查（每个座位四块互不重叠 / 牌不出界） ═══");
        System.out.println("支持范围：" + FLOOR_W + "×" + FLOOR_H
                + " 起（窗口最小尺寸 1080×760）");
        int cases = 0;
        int bad = 0;
        int degraded = 0;
        for (int players : new int[] {3, 4}) {
            for (Case c : cases(players)) {
                for (int[] size : sizes) {
                    for (int view = 0; view < players; view++) {
                        cases++;
                        Case positioned = withView(c, view);
                        for (boolean banner : new boolean[] {false, true}) {
                            List<LayoutRules.Violation> vs = checkSupported(size[0], size[1], positioned.state(), banner);
                            String tag = players + " 人 · " + c.name() + " · "
                                    + size[0] + "×" + size[1] + " · 视角 P" + view
                                    + (banner ? " · 带横幅" : "");
                            if (vs.isEmpty()) {
                                if (verbose) {
                                    System.out.println("✓ " + tag);
                                }
                                continue;
                            }
                            // 比支持范围还小的窗口：牌会挤，记一笔降级，不算失败
                            if (size[0] < FLOOR_W || size[1] < FLOOR_H) {
                                degraded++;
                                if (verbose) {
                                    System.out.println("⚠ " + tag + " —— " + vs.size()
                                            + " 处（低于支持范围，已知降级）");
                                }
                                continue;
                            }
                            bad++;
                            System.out.println("✗ " + tag + " —— " + vs.size() + " 处：");
                            for (int k = 0; k < Math.min(MAX_PRINT, vs.size()); k++) {
                                System.out.println(vs.get(k));
                            }
                            if (vs.size() > MAX_PRINT) {
                                System.out.println("    … 还有 " + (vs.size() - MAX_PRINT) + " 处");
                            }
                        }
                    }
                }
            }
        }
        System.out.println();
        Map<String, int[]> smallest = LayoutRules.smallest();
        Map<String, String> smallestAt = LayoutRules.smallestAt();
        Map<String, Double> leastVisible = LayoutRules.leastVisible();
        if (!smallest.isEmpty()) {
            System.out.println("支持范围内各类牌的最小边长（宽×高）：");
            for (Map.Entry<String, int[]> e : smallest.entrySet()) {
                int[] wh = e.getValue();
                Double vis = leastVisible.get(e.getKey());
                System.out.println("  " + e.getKey() + " " + wh[0] + "×" + wh[1]
                        + "　最小出现在 " + smallestAt.get(e.getKey())
                        + (vis == null ? "" : "　叠放时最少露出 " + Math.round(vis * 100) + "%"));
            }
        }
        System.out.println("查了 " + cases + " 个（局面 × 尺寸 × 视角）组合，其中有问题 " + bad + " 个"
                + (degraded > 0 ? "；低于支持范围的降级组合 " + degraded + " 个（--verbose 可见）" : ""));
        System.out.println(bad == 0 ? "布局铁律全部成立 ✓" : "布局有问题 ✗");
        System.exit(bad == 0 ? 0 : 1);
    }

    /** 逐档试布局：每档报「几处违规 + 头一条」，看是哪一档卡住的。 */
    private static void tiers(int players, String size) throws Json.JsonException {
        int[] wh = parseSizes(size).get(0);
        for (int pond : new int[] {0, 12, 18, 24}) {
            State s = Fixtures.stress(players, pond, 4);
            System.out.println("—— " + players + " 人 · 牌河 " + pond + " 张 · " + wh[0] + "×" + wh[1]);
            for (TileImages.Size t : new TileImages.Size[] {TileImages.Size.HAND,
                    TileImages.Size.HAND_M, TileImages.Size.HAND_S, TileImages.Size.HAND_XS}) {
                TableLayout L = TableLayout.buildWith(wh[0], wh[1], s, null, false,
                        new TableLayout.Metrics(t));
                List<LayoutRules.Violation> vs = LayoutRules.check(L, s, t.toString(), false);
                System.out.println("   " + t + " " + t.w + "×" + t.h + " → "
                        + (vs.isEmpty() ? "OK" : vs.size() + " 处：" + vs.get(0)));
                if (!vs.isEmpty() && pond == 12) {
                    for (int slot : L.usedSlots()) {
                        TableLayout.SeatView v = L.view(slot);
                        if (v.seat != 3) {
                            continue;
                        }
                        StringBuilder sb = new StringBuilder();
                        for (TableLayout.Spot sp : L.spots) {
                            if (sp.seat == 3 && sp.zone == TableLayout.Zone.MELD) {
                                sb.append('(').append(sp.rect.x).append(',').append(sp.rect.y)
                                        .append(')');
                            }
                        }
                        System.out.println("      P3 副露块 " + v.meld + " 逐张 " + sb);
                    }
                }
            }
        }
    }

    /** 把一次布局的四个矩形 + 每块画了几张牌打出来（排查布局问题用）。 */
    private static void dump(int players, String size) throws Json.JsonException {
        int[] wh = parseSizes(size).get(0);
        State s = Fixtures.stress(players, 20, 4);
        TableLayout L = TableLayout.build(wh[0], wh[1], s, null, true);
        System.out.println("窗口 " + wh[0] + "×" + wh[1] + " · " + players + " 人 · 桌面 " + L.felt
                + " · 摆牌区 " + L.area);
        System.out.println("中央 " + L.center + " · 宝牌 " + L.centerDora + " · 横幅 " + L.banner);
        for (int slot : L.usedSlots()) {
            TableLayout.SeatView v = L.view(slot);
            int hand = 0;
            int meld = 0;
            int pond = 0;
            for (TableLayout.Spot sp : L.spots) {
                if (sp.seat != v.seat) {
                    continue;
                }
                switch (sp.zone) {
                    case HAND -> hand++;
                    case MELD -> meld++;
                    case POND -> pond++;
                    default -> { }
                }
            }
            System.out.printf("P%d %-5s 名牌%-22s 手牌%-22s(%2d 张) 副露%-22s(%2d 张)"
                            + " 牌河%-22s(%2d 张)%n",
                    v.seat, v.facing, v.plate, v.hand, hand, v.meld, meld, v.pond, pond);
        }
        for (int slot : L.usedSlots()) {
            TableLayout.SeatView v = L.view(slot);
            StringBuilder sb = new StringBuilder();
            for (TableLayout.Spot sp : L.spots) {
                if (sp.seat == v.seat && sp.zone == TableLayout.Zone.HAND) {
                    sb.append('(').append(sp.rect.x).append(',').append(sp.rect.y).append(' ')
                            .append(sp.rect.w).append('x').append(sp.rect.h).append(')');
                }
            }
            System.out.println("P" + v.seat + " 手牌逐张：" + sb);
        }
    }

    // ------------------------------------------------------------ 矩阵

    private static List<Case> cases(int players) throws Json.JsonException {
        List<Case> out = new ArrayList<>();
        out.add(new Case("空桌 13 张", Fixtures.fresh(players)));
        // 一局里真能出现的两种极端：叫得多（副露满）时打出的牌就少；
        // 没人叫时牌河才长。别拿「4 副露 + 24 张牌河」去难为布局 —— 那是打不出来的局面。
        out.add(new Case("满副露+中牌河 12", Fixtures.stress(players, 12, 4)));
        out.add(new Case("长牌河 18+空副露", Fixtures.stress(players, 18, 0)));
        out.add(new Case("全亮（观战/回放）", Fixtures.stress(players, 12, 4, true)));
        State drain = Fixtures.stress(players, 12, 4);
        drain.pending = new State.Pending();
        drain.pending.seat = 0;
        drain.pending.kind = "discard";
        drain.pending.yours = true;
        drain.dangerTile = "9s";
        drain.dangerSeat = 1;
        out.add(new Case("轮到我打牌 + 铳牌标红", drain));
        return out;
    }

    private static Case withView(Case c, int view) throws Json.JsonException {
        State s = State.fromJson(c.state().raw);
        s.you = view;
        return new Case(c.name() + " 视角 P" + view, s);
    }

    private static List<int[]> defaultSizes() {
        List<int[]> out = new ArrayList<>();
        int[][] list = {
            {1000, 700}, {1024, 720}, {1080, 760}, {1200, 800}, {1280, 800},
            {1400, 900}, {1600, 1000}, {1920, 1080}, {2560, 1440},
        };
        for (int[] s : list) {
            out.add(s);
        }
        return out;
    }

    /** 比支持范围还小的窗口（--all 才会查，只做降级观察，不算失败）。 */
    private static List<int[]> belowFloorSizes() {
        List<int[]> out = new ArrayList<>();
        int[][] list = {{900, 640}, {820, 600}, {760, 560}, {640, 480}};
        for (int[] s : list) {
            out.add(s);
        }
        return out;
    }

    private static List<int[]> parseSizes(String spec) {
        List<int[]> out = new ArrayList<>();
        for (String one : spec.split(",")) {
            String[] wh = one.trim().toLowerCase().split("x");
            out.add(new int[] {Integer.parseInt(wh[0]), Integer.parseInt(wh[1])});
        }
        return out;
    }

    // ------------------------------------------------------------ 规则

    /** 查一次布局（算出来 + 用 {@link LayoutRules} 查一遍）。 */
    public static List<LayoutRules.Violation> check(int w, int h, State s, boolean reserveBanner) {
        return LayoutRules.check(TableLayout.build(w, h, s, null, reserveBanner), s,
                w + "×" + h, w >= FLOOR_W && h >= FLOOR_H);
    }

    /** 检查并记统计（支持范围内才记，免得小窗口把「最小牌」的账算歪）。 */
    private static List<LayoutRules.Violation> checkSupported(int w, int h, State s,
            boolean banner) {
        return LayoutRules.check(TableLayout.build(w, h, s, null, banner), s, w + "×" + h, true);
    }
}
