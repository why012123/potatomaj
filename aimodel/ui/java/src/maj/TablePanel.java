package maj;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.TexturePaint;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;
import javax.swing.ImageIcon;
import javax.swing.JPanel;
import javax.swing.Timer;
import javax.swing.ToolTipManager;

/**
 * 牌桌本体：整块画布自绘 —— 木框 + 绒布桌面 + 每家一块信息牌 + 牌河 / 副露 / 手牌。
 *
 * <p>自己的手牌**只画这一遍**（大图、可点、刚摸的留缝、可点的牌带数字键角标），
 * 别人是同样大小的牌背。牌图没加载好时画占位牌，不会白屏。
 *
 * <p>动画：不做任何时序假设，只用<b>前后两份 /api/state 的差异</b>触发
 * （牌河多了几张 = 谁打了牌、手牌多了几张 = 谁摸了牌、hand_no / seed 变了 = 开新局发牌）。
 * 动画只是画在终态上面的一层「幻影」，轮询一到就跳到终态，丢帧也不会错位；
 * 菜单里可以整个关掉。刷新由 {@link #ticker} 驱动（只有真在飞的时候才跑）。
 */
public final class TablePanel extends JPanel {

    /** 点了某张手牌就交给外面去 POST。 */
    public interface ActSink {
        void submitOption(int optionIndex, String why);
    }

    private static final long serialVersionUID = 1L;

    private final TileImages images;
    private final ActSink sink;
    private State state;
    private TableLayout cached;
    private int cachedW = -1;
    private int cachedH = -1;
    private State cachedFor;
    private TableLayout.Spot hover;
    private String banner;
    /** 动画开关（菜单里能关）。 */
    private volatile boolean animOn = true;
    /** 正在飞的牌：出牌 / 摸牌 / 发牌。 */
    private final List<Anim> anims = new ArrayList<>();
    /** 只有「有牌在飞」的时候才跑的刷新器（约 60fps）。 */
    private final Timer ticker;

    /** 一段动画最多留几张幻影牌，防止连点/断线重连之后堆一堆。 */
    private static final int MAX_ANIMS = 60;
    private static final int TICK_MS = 16;

    /** 立直宣言牌跟踪：客户端自己记（后端没给这个信息）。 */
    private final boolean[] wasRiichi = new boolean[6];
    private final int[] riichiTile = {-1, -1, -1, -1, -1, -1};
    private String trackerKey = "";

    public TablePanel(TileImages images, ActSink sink) {
        this.images = images;
        this.sink = sink;
        setOpaque(true);
        setBackground(Palette.WOOD_LO);
        ToolTipManager.sharedInstance().registerComponent(this);
        MouseAdapter mouse = new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                updateHover(e.getX(), e.getY());
            }

            @Override
            public void mouseExited(MouseEvent e) {
                if (hover != null) {
                    hover = null;
                    setCursor(Cursor.getDefaultCursor());
                    repaint();
                }
            }

            @Override
            public void mouseClicked(MouseEvent e) {
                if (!javax.swing.SwingUtilities.isLeftMouseButton(e)) {
                    return;
                }
                TableLayout L = computeLayout();
                TableLayout.Spot sp = L == null ? null : L.hitOption(e.getX(), e.getY());
                if (sp != null && sink != null) {
                    sink.submitOption(sp.optionIndex, "点手牌 " + Tile.cn(sp.tileId));
                }
            }
        };
        addMouseListener(mouse);
        addMouseMotionListener(mouse);
        ticker = new Timer(TICK_MS, e -> {
            if (anims.isEmpty() || !isShowing()) {
                tickerStop();
                return;
            }
            repaint();
        });
        ticker.setCoalesce(true);
    }

    /** 有牌要飞就开刷新器（窗口没显示时不开，省电）。 */
    private void tickerKick() {
        if (!anims.isEmpty() && !ticker.isRunning() && isShowing()) {
            ticker.start();
        }
    }

    private void tickerStop() {
        if (ticker.isRunning()) {
            ticker.stop();
        }
    }

    @Override
    public void removeNotify() {
        tickerStop();
        super.removeNotify();
    }

    /** 换一份局面（EDT 上调）。顺手对比上一份，决定要不要放动画。 */
    public void setState(State s) {
        State before = state;
        this.state = s;
        this.hover = null;
        if (s == null || before == null || !animOn || s.replay) {
            anims.clear();
        } else {
            triggerAnims(before, s);
        }
        trackRiichi(s);
        repaint();
    }

    /** 动画开 / 关。 */
    public void setAnimations(boolean on) {
        this.animOn = on;
        if (!on) {
            anims.clear();
            tickerStop();
            repaint();
        }
    }

    /** 现在有几张牌在飞（自检 / 出图用）。 */
    public int flyingCount() {
        long now = System.currentTimeMillis();
        int n = 0;
        for (Anim a : anims) {
            if (!a.done(now)) {
                n++;
            }
        }
        return n;
    }

    public boolean animationsOn() {
        return animOn;
    }

    /**
     * 用前后两次 {@code /api/state} 的差异触发动画（不做任何时序假设）：
     * <ul>
     *   <li>谁牌河多了几张 → 那几张从他自己手牌的位置滑进牌河（出牌）；</li>
     *   <li>谁手牌多了几张 → 从桌心滑进来（摸牌，别家画牌背）；</li>
     *   <li>hand_no / seed 变了（或者手牌从 0 张变成一把）→ 一圈发牌。</li>
     * </ul>
     */
    private void triggerAnims(State before, State after) {
        TableLayout L = TableLayout.build(Math.max(1, getWidth()), Math.max(1, getHeight()),
                after, riichiTile, bannerReserved(after));
        long now = System.currentTimeMillis();
        int n = Math.min(before.players(), after.players());
        boolean newHand = before.handNo != after.handNo || before.seed != after.seed;
        boolean dealt = newHand;
        for (int seat = 0; seat < n && !dealt; seat++) {
            dealt = handCount(before, seat) == 0 && handCount(after, seat) > 0;
        }
        if (dealt) {
            dealAnims(L, now, n);
            return;
        }
        boolean anyDiscard = false;
        for (int seat = 0; seat < n; seat++) {
            State.Player pb = before.player(seat);
            State.Player pa = after.player(seat);
            if (pb == null || pa == null) {
                continue;
            }
            int grew = pa.pond.size() - pb.pond.size();
            if (grew <= 0 || grew > 4) {
                continue;
            }
            anyDiscard = true;
            List<TableLayout.Spot> pond = pondOf(L, seat);
            TableLayout.SeatView v = viewOf(L, seat);
            Rect from = v == null ? L.felt.inset(L.felt.w / 3)
                    : new Rect(v.hand.centerX() - 20, v.hand.centerY() - 20, 40, 40);
            for (int k = Math.max(0, pond.size() - grew); k < pond.size(); k++) {
                Rect to = pond.get(k).rect;
                anims.add(new Anim(pond.get(k).tileId, seat, to,
                        new Rect(from.centerX() - to.w / 2, from.centerY() - to.h / 2, to.w, to.h),
                        now, 190, 0, pond.get(k).size));
            }
        }
        // 摸牌：每家手牌多了 1~2 张就从桌心滑进来
        for (int seat = 0; seat < n; seat++) {
            int oldCount = handCount(before, seat);
            int newCount = handCount(after, seat);
            int grew = newCount - oldCount;
            if (grew <= 0 || grew > 2) {
                continue;
            }
            List<TableLayout.Spot> hand = handOf(L, seat);
            Rect from = flyFrom(L, seat);
            for (int k = Math.max(0, hand.size() - grew); k < hand.size(); k++) {
                TableLayout.Spot sp = hand.get(k);
                anims.add(new Anim(sp.tileId, seat, sp.rect,
                        new Rect(from.x, from.y, sp.rect.w, sp.rect.h), now, 210,
                        anyDiscard ? 120 : 0, sp.size));
            }
        }
        trimAnims();
    }

    /** 一圈发牌：每家手牌一张张从桌心飞到位（自己的看得见牌面，别家是牌背）。 */
    private void dealAnims(TableLayout L, long now, int players) {
        for (int seat = 0; seat < players; seat++) {
            List<TableLayout.Spot> hand = handOf(L, seat);
            Rect from = flyFrom(L, seat);
            for (int i = 0; i < hand.size(); i++) {
                TableLayout.Spot sp = hand.get(i);
                long delay = (long) i * 38 + seat * 12L;
                if (sp.drawn) {
                    delay += 200;                 // 刚摸的那张最后到
                }
                anims.add(new Anim(sp.tileId, seat, sp.rect,
                        new Rect(from.x, from.y, sp.rect.w, sp.rect.h), now, 240, delay, sp.size));
            }
        }
        trimAnims();
    }

    /** 飞牌起点：桌心（稍微偏上一点，别压着宝牌）。 */
    private static Rect flyFrom(TableLayout L, int seat) {
        return new Rect(L.felt.centerX() - 18, L.felt.centerY() - 60, 36, 36);
    }

    private static int handCount(State s, int seat) {
        State.Player p = s.player(seat);
        if (p == null) {
            return 0;
        }
        return p.hand != null && !p.hand.isEmpty() ? p.hand.size() : p.handCount;
    }

    private static List<TableLayout.Spot> handOf(TableLayout L, int seat) {
        return zoneOf(L, TableLayout.Zone.HAND, seat);
    }

    private static List<TableLayout.Spot> pondOf(TableLayout L, int seat) {
        return zoneOf(L, TableLayout.Zone.POND, seat);
    }

    private static List<TableLayout.Spot> zoneOf(TableLayout L, TableLayout.Zone zone, int seat) {
        List<TableLayout.Spot> out = new ArrayList<>();
        for (TableLayout.Spot sp : L.spots) {
            if (sp.zone == zone && sp.seat == seat) {
                out.add(sp);
            }
        }
        return out;
    }

    private void trimAnims() {
        if (anims.size() > MAX_ANIMS) {
            anims.subList(0, anims.size() - MAX_ANIMS).clear();
        }
        tickerKick();
    }

    private static boolean bannerReserved(State s) {
        return (s.bannerText != null && !s.bannerText.isBlank()) || s.awaitingNext;
    }

    private TableLayout.SeatView viewOf(TableLayout L, int seat) {
        for (int slot : L.usedSlots()) {
            TableLayout.SeatView v = L.view(slot);
            if (v != null && v.seat == seat) {
                return v;
            }
        }
        return null;
    }

    /** 一张正在飞的牌。 */
    private static final class Anim {
        final String tileId;
        final int seat;
        final Rect to;
        final Rect from;
        final long start;
        final long dur;
        final long delay;
        final TileImages.Size size;

        Anim(String tileId, int seat, Rect to, Rect from, long start, long dur, long delay,
                TileImages.Size size) {
            this.size = size;
            this.tileId = tileId;
            this.seat = seat;
            this.to = to;
            this.from = from;
            this.start = start;
            this.dur = dur;
            this.delay = delay;
        }

        float progress(long now) {
            long t = now - start - delay;
            if (t <= 0) {
                return 0f;
            }
            return Math.min(1f, (float) t / dur);
        }

        boolean done(long now) {
            return progress(now) >= 1f;
        }
    }

    public State state() {
        return state;
    }

    /** 顶部横幅（连接断开之类），null = 不显示。 */
    public void setBanner(String text) {
        if (text == null ? banner != null : !text.equals(banner)) {
            banner = text;
            repaint();
        }
    }

    public List<TableLayout.Spot> clickable() {
        TableLayout L = computeLayout();
        return L == null ? List.of() : L.clickable();
    }

    // ------------------------------------------------------------ 立直跟踪

    /** 谁刚立直，谁牌河最后一张就是立直宣言牌（横放标出来）。 */
    private void trackRiichi(State s) {
        if (s == null) {
            return;
        }
        String key = s.seed + "/" + s.handNo + "/" + s.players();
        if (!key.equals(trackerKey)) {
            trackerKey = key;
            for (int i = 0; i < wasRiichi.length; i++) {
                wasRiichi[i] = false;
                riichiTile[i] = -1;
            }
        }
        for (int i = 0; i < s.players() && i < wasRiichi.length; i++) {
            State.Player p = s.player(i);
            if (p == null) {
                continue;
            }
            if (p.riichi && !wasRiichi[i] && !p.pond.isEmpty()) {
                riichiTile[i] = p.pond.size() - 1;
            }
            wasRiichi[i] = p.riichi;
        }
    }

    private void updateHover(int x, int y) {
        TableLayout L = computeLayout();
        TableLayout.Spot sp = L == null ? null : L.hitOption(x, y);
        if (sp != hover) {
            hover = sp;
            setCursor(Cursor.getPredefinedCursor(
                    sp != null ? Cursor.HAND_CURSOR : Cursor.DEFAULT_CURSOR));
            repaint();
        }
    }

    @Override
    public String getToolTipText(MouseEvent e) {
        TableLayout L = computeLayout();
        if (L == null) {
            return null;
        }
        for (TableLayout.Spot sp : L.spots) {
            if (sp.rect.hits(e.getX(), e.getY(), 1) && sp.tileId != null) {
                StringBuilder sb = new StringBuilder(Tile.cnFull(sp.tileId));
                switch (sp.zone) {
                    case MELD -> sb.append("（副露）");
                    case POND -> sb.append("（牌河）");
                    case HAND -> sb.append("（手牌）");
                    default -> sb.append("（宝牌指示）");
                }
                if (sp.clickable()) {
                    sb.append(" — 点它打出（键 ").append(sp.optionIndex + 1).append("）");
                }
                return sb.toString();
            }
        }
        return null;
    }

    private TableLayout computeLayout() {
        State s = state;
        if (s == null) {
            return null;
        }
        if (cached == null || cachedW != getWidth() || cachedH != getHeight() || cachedFor != s) {
            boolean bannerUp = (s.bannerText != null && !s.bannerText.isBlank())
                    || s.awaitingNext;
            cached = TableLayout.build(getWidth(), getHeight(), s, riichiTile, bannerUp);
            cachedW = getWidth();
            cachedH = getHeight();
            cachedFor = s;
        }
        return cached;
    }

    // ------------------------------------------------------------ 绘制

    @Override
    protected void paintComponent(Graphics g0) {
        super.paintComponent(g0);
        Graphics2D g = (Graphics2D) g0.create();
        try {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
                    RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            g.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL,
                    RenderingHints.VALUE_STROKE_PURE);
            paintWood(g);
            TableLayout L = computeLayout();
            if (L == null || L.felt.w <= 0) {
                paintPlaceholder(g, "正在连接后端…");
                return;
            }
            paintFelt(g, L.felt);
            for (int slot : L.usedSlots()) {
                paintPlate(g, L, slot);
            }
            long now = System.currentTimeMillis();
            anims.removeIf(a -> a.done(now));
            for (TableLayout.Spot sp : L.spots) {
                if (isFlying(sp, now)) {
                    continue;   // 这张正在飞，画在下面
                }
                paintSpot(g, sp);
            }
            paintAnims(g, now);
            for (TableLayout.Text t : L.texts) {
                paintText(g, t);
            }
            paintCenter(g, L);
            paintBrand(g, L);
            paintTopBanner(g, L);
            if (banner != null) {
                paintBanner(g, banner);
            }
        } finally {
            g.dispose();
        }
    }

    private void paintWood(Graphics2D g) {
        int w = getWidth();
        int h = getHeight();
        g.setPaint(new GradientPaint(0, 0, Palette.WOOD_HI, w, h, Palette.WOOD_LO));
        g.fillRect(0, 0, w, h);
        // 木纹：几条很淡的斜线，别太抢戏
        g.setColor(new Color(255, 235, 205, 12));
        g.setStroke(new BasicStroke(1f));
        for (int i = -h; i < w; i += 9) {
            g.drawLine(i, 0, i + h, h);
        }
        g.setPaint(new GradientPaint(0, 0, new Color(255, 255, 255, 18),
                w, h, new Color(0, 0, 0, 55)));
        g.fillRect(0, 0, w, h);
    }

    /** 这张牌此刻正被动画接管吗。 */
    private boolean isFlying(TableLayout.Spot sp, long now) {
        for (Anim a : anims) {
            if (a.seat == sp.seat && sameRect(a.to, sp.rect) && !a.done(now)) {
                return true;
            }
        }
        return false;
    }

    private static boolean sameRect(Rect a, Rect b) {
        return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h;
    }

    /** 画所有在飞的牌（缓出 + 淡入），动画不阻塞轮询，新状态一到就跳终态。 */
    private void paintAnims(Graphics2D g, long now) {
        for (Anim a : anims) {
            float t = a.progress(now);
            if (t <= 0f || t >= 1f) {
                continue;
            }
            float e = 1f - (1f - t) * (1f - t);          // ease-out
            int x = Math.round(a.from.x + (a.to.x - a.from.x) * e);
            int y = Math.round(a.from.y + (a.to.y - a.from.y) * e);
            java.awt.Composite old = g.getComposite();
            g.setComposite(java.awt.AlphaComposite.getInstance(
                    java.awt.AlphaComposite.SRC_OVER, 0.35f + 0.65f * t));
            TableLayout.Spot ghost = new TableLayout.Spot(new Rect(x, y, a.to.w, a.to.h),
                    a.tileId, TableLayout.Zone.HAND, a.seat, a.size,
                    a.to.w > a.to.h, a.tileId == null, false, false, false, -1, null);
            paintSpot(g, ghost);
            g.setComposite(old);
        }
    }

    private void paintFelt(Graphics2D g, Rect f) {
        java.awt.geom.RoundRectangle2D round = new java.awt.geom.RoundRectangle2D.Float(
                f.x, f.y, f.w, f.h, 26, 26);
        g.setPaint(new GradientPaint(f.x, f.y, Palette.FELT_HI, f.x, f.bottom(), Palette.FELT_LO));
        g.fill(round);
        // 桌布质感：噪点平铺 + 中间一圈柔光
        java.awt.TexturePaint tex = new TexturePaint(Palette.noiseTile(),
                new Rectangle2D.Float(0, 0, 48, 48));
        g.setPaint(tex);
        g.fill(round);
        g.setPaint(new java.awt.RadialGradientPaint(
                new java.awt.geom.Point2D.Float(f.centerX(), f.centerY()),
                Math.max(f.w, f.h) * 0.62f,
                new float[] {0f, 1f},
                new Color[] {new Color(255, 255, 255, 16), new Color(0, 0, 0, 60)}));
        g.fill(round);
        g.setColor(new Color(0, 0, 0, 90));
        g.setStroke(new BasicStroke(3f));
        g.draw(round);
    }

    private void paintPlaceholder(Graphics2D g, String text) {
        g.setFont(Palette.bold(18));
        g.setColor(Palette.DIM);
        FontMetrics fm = g.getFontMetrics();
        g.drawString(text, (getWidth() - fm.stringWidth(text)) / 2, getHeight() / 2);
    }

    /** 每家一块信息牌：名字 / 点数 / 自风 / 亲 / 立直 / 一发 / 拔北 / 门清 / 张数。 */
    private void paintPlate(Graphics2D g, TableLayout L, int slot) {
        TableLayout.SeatView v = L.view(slot);
        if (v == null || v.plate.w < 40) {
            return;
        }
        State s = state;
        int seat = v.seat;
        State.Player p = s.player(seat);
        State.Seat info = s.seat(seat);
        boolean me = seat == s.you;
        boolean thinking = s.pending != null && s.pending.seat == seat && !s.pending.yours;
        Rect box = v.plate;
        int h = 24;
        List<Object[]> chips = new ArrayList<>();
        chips.add(new Object[] {info == null ? "P" + seat : shortName(info.name),
                me ? Palette.GOLD : Palette.PANEL_SOFT, me ? Color.BLACK : Palette.INK});
        if (p != null) {
            chips.add(new Object[] {String.valueOf(p.score), new Color(0, 0, 0, 110),
                    scoreColor(p.score, s.players())});
            if (!p.seatWind.isEmpty()) {
                chips.add(new Object[] {p.seatWind + "家", new Color(0, 0, 0, 110), Palette.INK});
            }
            if (p.isDealer) {
                chips.add(new Object[] {"亲", Palette.RED, Color.WHITE});
            }
            if (p.riichi) {
                chips.add(new Object[] {"立直", Palette.RED, Color.WHITE});
            }
            if (p.ippatsu) {
                chips.add(new Object[] {"一发", Palette.GOLD, Color.BLACK});
            }
            if (p.kita > 0) {
                chips.add(new Object[] {"拔北" + p.kita, new Color(0x2C, 0x6E, 0x5A),
                        Color.WHITE});
            }
            chips.add(new Object[] {p.menzen() ? "门前" : "副露" + p.melds.size(),
                    p.menzen() ? new Color(0, 0, 0, 110) : new Color(0x2A, 0x4A, 0x74),
                    p.menzen() ? Palette.DIM : Color.WHITE});
            chips.add(new Object[] {p.handCount + "张", new Color(0, 0, 0, 90), Palette.DIM});
            if (thinking) {
                chips.add(new Object[] {"思考中…", Palette.RED, Color.WHITE});
            }
        }
        // 先量一遍宽度，再决定信息牌多大（免得字被牌子切掉）
        g.setFont(Palette.font(11));
        FontMetrics fm = g.getFontMetrics();
        int need = 12;
        for (Object[] c : chips) {
            need += fm.stringWidth((String) c[0]) + 14;
        }
        Rect plate = new Rect(box.x, box.y, Math.min(Math.max(60, need), Math.max(60, box.w)), h);
        g.setColor(Palette.PLATE);
        g.fillRoundRect(plate.x, plate.y, plate.w, plate.h, 12, 12);
        g.setColor(me ? Palette.GOLD : thinking ? Palette.RED : Palette.LINE);
        g.setStroke(new BasicStroke(me || thinking ? 2f : 1f));
        g.drawRoundRect(plate.x, plate.y, plate.w, plate.h, 12, 12);

        java.awt.Shape old = g.getClip();
        g.clipRect(plate.x + 1, plate.y, plate.w - 2, plate.h);
        int x = plate.x + 6;
        int y = plate.y + 4;
        for (Object[] c : chips) {
            // 放不下就到此为止：宁可少一个徽标，也不要画半个出来
            int w = fm.stringWidth((String) c[0]) + 10;
            if (x + w > plate.right() - 4) {
                break;
            }
            x = chip(g, x, y, (String) c[0], (Color) c[1], (Color) c[2], h - 8);
        }
        g.setClip(old);
    }

    private static Color scoreColor(int score, int players) {
        int base = players >= 4 ? 25000 : 35000;
        if (score >= base + 5000) {
            return Palette.GREEN;
        }
        if (score < base - 8000) {
            return Palette.RED;
        }
        return Palette.INK;
    }

    private static String shortName(String name) {
        int i = name.indexOf('（');
        String n = i > 0 ? name.substring(0, i) : name;
        return n.length() > 8 ? n.substring(0, 8) : n;
    }

    private int chip(Graphics2D g, int x, int y, String text, Color bg, Color fg, int h) {
        g.setFont(Palette.font(11));
        FontMetrics fm = g.getFontMetrics();
        int w = fm.stringWidth(text) + 10;
        g.setColor(bg);
        g.fillRoundRect(x, y, w, h, h, h);
        g.setColor(fg);
        g.drawString(text, x + 5, y + h - 4);
        return x + w + 4;
    }

    // ------------------------------------------------------------ 牌

    private void paintSpot(Graphics2D g, TableLayout.Spot sp) {
        Rect r = sp.rect;
        boolean hovered = sp == hover;
        if (sp.back) {
            drawBack(g, r, sp.rotated);
        } else {
            ImageIcon ic = sp.tileId == null ? null : images.get(sp.tileId, sp.size, sp.rotated);
            if (ic != null) {
                g.setColor(Palette.SHADOW);
                g.fillRoundRect(r.x + 2, r.y + 3, r.w, r.h, 6, 6);
                g.drawImage(ic.getImage(), r.x, r.y, r.w, r.h, null);
            } else {
                drawFakeTile(g, r, sp.tileId);
            }
        }
        if (sp.danger) {                    // 铳牌：红底 + 红框，跟立直的金框各画各的
            g.setColor(new Color(0xE0, 0x57, 0x4A, 90));
            g.fillRoundRect(r.x - 2, r.y - 2, r.w + 4, r.h + 4, 8, 8);
            g.setColor(new Color(0xFF, 0x3B, 0x30));
            g.setStroke(new BasicStroke(3.5f));
            g.drawRoundRect(r.x - 2, r.y - 2, r.w + 4, r.h + 4, 8, 8);
        }
        if (sp.highlight) {                 // 被鸣 / 被荣的那张
            g.setColor(new Color(0xE0, 0x57, 0x4A, 70));
            g.fillRoundRect(r.x - 3, r.y - 3, r.w + 6, r.h + 6, 8, 8);
            g.setColor(Palette.RED);
            g.setStroke(new BasicStroke(3f));
            g.drawRoundRect(r.x - 1, r.y - 1, r.w + 2, r.h + 2, 7, 7);
        } else if (hovered) {
            g.setColor(new Color(255, 255, 255, 70));
            g.fillRoundRect(r.x, r.y, r.w, r.h, 6, 6);
            g.setColor(Palette.GOLD);
            g.setStroke(new BasicStroke(3f));
            g.drawRoundRect(r.x, r.y, r.w, r.h, 6, 6);
        } else if (sp.clickable()) {         // 可以打出的牌：绿框 + 数字键角标
            g.setColor(new Color(0x6A, 0xCE, 0x96, 40));
            g.fillRoundRect(r.x, r.y, r.w, r.h, 6, 6);
            g.setColor(Palette.GREEN);
            g.setStroke(new BasicStroke(2f));
            g.drawRoundRect(r.x + 1, r.y + 1, r.w - 2, r.h - 2, 6, 6);
            int key = sp.optionIndex + 1;
            if (key <= 9) {
                int d = 18;
                g.setColor(new Color(0x0B, 0x22, 0x1A, 230));
                g.fillOval(r.x + 3, r.y + 3, d, d);
                g.setColor(Palette.GREEN);
                g.setStroke(new BasicStroke(1.4f));
                g.drawOval(r.x + 3, r.y + 3, d, d);
                g.setFont(Palette.bold(12));
                g.setColor(Palette.INK);
                String k = String.valueOf(key);
                FontMetrics fm = g.getFontMetrics();
                g.drawString(k, r.x + 3 + (d - fm.stringWidth(k)) / 2, r.y + 3 + d - 5);
            }
        }
        if (sp.note != null && !sp.note.isEmpty()) {
            if (TableLayout.MARK_LAST.equals(sp.note)) {
                g.setColor(new Color(0xE8, 0xC4, 0x6A, 190));
                g.setStroke(new BasicStroke(1.6f));
                g.drawRoundRect(r.x, r.y, r.w - 1, r.h - 1, 6, 6);
            } else {
                boolean red = TableLayout.MARK_CALLED.equals(sp.note)
                        || TableLayout.MARK_DANGER.equals(sp.note);
                g.setFont(Palette.bold(10));
                FontMetrics fm = g.getFontMetrics();
                int w = fm.stringWidth(sp.note) + 8;
                int bx = r.right() - w - 1;
                int by = r.bottom() - 14;
                g.setColor(red ? new Color(0x8C, 0x24, 0x1C, 235) : new Color(0, 0, 0, 200));
                g.fillRoundRect(bx, by, w, 13, 6, 6);
                g.setColor(red ? Color.WHITE : Palette.GOLD);
                g.drawString(sp.note, bx + 4, by + 10);
            }
        }
    }

    private void drawBack(Graphics2D g, Rect r, boolean rotated) {
        g.setColor(Palette.SHADOW);
        g.fillRoundRect(r.x + 2, r.y + 3, r.w, r.h, 6, 6);
        g.setPaint(new GradientPaint(r.x, r.y, Palette.TILE_BACK, r.x, r.bottom(),
                Palette.TILE_BACK_LO));
        g.fillRoundRect(r.x, r.y, r.w, r.h, 6, 6);
        g.setColor(Palette.TILE_BACK_EDGE);
        g.setStroke(new BasicStroke(1.2f));
        g.drawRoundRect(r.x, r.y, r.w - 1, r.h - 1, 6, 6);
        // 背面花纹：斜向细纹
        g.setColor(new Color(255, 255, 255, 26));
        int step = rotated ? 7 : 6;
        for (int i = -r.h; i < r.w; i += step) {
            g.drawLine(r.x + Math.max(0, i), r.y + Math.max(0, -i),
                    r.x + Math.min(r.w, i + r.h), r.y + Math.min(r.h, r.h - Math.max(0, i + r.h - r.w)));
        }
        g.setColor(new Color(255, 255, 255, 20));
        g.drawRoundRect(r.x + 3, r.y + 3, r.w - 7, r.h - 7, 4, 4);
    }

    private void drawFakeTile(Graphics2D g, Rect r, String id) {
        g.setColor(Palette.SHADOW);
        g.fillRoundRect(r.x + 2, r.y + 3, r.w, r.h, 6, 6);
        g.setColor(Palette.TILE_FACE);
        g.fillRoundRect(r.x, r.y, r.w, r.h, 6, 6);
        g.setColor(Palette.TILE_EDGE);
        g.drawRoundRect(r.x, r.y, r.w - 1, r.h - 1, 6, 6);
        String text = id == null ? "?" : Tile.cn(id);
        g.setColor(id != null && Tile.isRedId(id) ? Palette.RED : new Color(0x22, 0x22, 0x22));
        g.setFont(Palette.bold(Math.max(10, Math.min(r.w, r.h) / 3)));
        FontMetrics fm = g.getFontMetrics();
        g.drawString(text, r.x + (r.w - fm.stringWidth(text)) / 2,
                r.y + (r.h + fm.getAscent()) / 2 - 2);
    }

    private void paintText(Graphics2D g, TableLayout.Text t) {
        if (t.rect.w <= 0 || t.text.isEmpty()) {
            return;
        }
        g.setFont(Palette.font(12));
        g.setColor(switch (t.style) {
            case 1 -> Palette.DIM;
            case 2 -> Palette.GOLD;
            case 3 -> Palette.RED;
            case 4 -> Palette.BLUE;
            default -> Palette.INK;
        });
        FontMetrics fm = g.getFontMetrics();
        int x = t.center ? t.rect.centerX() - fm.stringWidth(t.text) / 2 : t.rect.x;
        g.drawString(t.text, x, t.rect.y + fm.getAscent());
    }

    // ------------------------------------------------------------ 中央信息区

    private void paintCenter(Graphics2D g, TableLayout L) {
        State s = state;
        Rect c = L.center;
        if (c.w < 60 || c.h < 40) {
            return;
        }
        g.setColor(new Color(0, 0, 0, 55));
        g.fillRoundRect(c.x, c.y, c.w, c.h, 18, 18);
        g.setColor(new Color(0x2E, 0x6E, 0x52));
        g.setStroke(new BasicStroke(1.4f));
        g.drawRoundRect(c.x, c.y, c.w, c.h, 18, 18);

        g.setFont(Palette.font(12));
        g.setColor(Palette.DIM);
        drawCentered(g, "宝牌指示", c.centerX(), c.y + 18);

        Rect t = L.centerText;
        if (t.w <= 20) {
            return;
        }
        int limit = c.bottom() - 8;          // 卡片下沿，别把字画到牌上去
        int line = t.h >= 84 ? 21 : (t.h >= 60 ? 18 : 16);   // 地方小就把行距收紧
        int y = t.y + 14;
        if (y > limit) {
            return;
        }
        g.setFont(Palette.bold(t.h >= 84 ? 16 : 15));
        g.setColor(Palette.INK);
        drawCentered(g, "剩余 " + s.wallLeft + " 张 · 第 " + s.turn + " 巡", c.centerX(), y);
        y += line;

        // 第二行就是「该谁动」——地方小的时候也不能丢
        String turnText;
        Color turnColor;
        if (s.replay) {
            turnText = "回放中 · 第 " + (s.replayIndex + 1) + " / " + s.replayFrames + " 手";
            turnColor = Palette.GOLD;
        } else if (s.awaitingNext) {
            turnText = "本局结束，等确认";
            turnColor = Palette.GREEN;
        } else if (s.finished) {
            turnText = "对局结束";
            turnColor = Palette.GOLD;
        } else if (s.pending != null && s.pending.yours) {
            turnText = "轮到你：" + s.pending.kindCn;
            turnColor = Palette.GOLD;
        } else if (s.pending != null) {
            State.Seat st = s.seat(s.pending.seat);
            turnText = (st == null ? "P" + s.pending.seat : shortName(st.name)) + " 正在"
                    + s.pending.kindCn;
            turnColor = Palette.RED;
        } else {
            turnText = s.waitingFor.isEmpty() ? "等待中…" : "等待 " + s.waitingFor;
            turnColor = Palette.DIM;
        }
        if (y <= limit) {
            g.setFont(Palette.bold(t.h >= 84 ? 18 : 16));
            g.setColor(turnColor);
            drawCentered(g, turnText, c.centerX(), y);
            y += line;
        }

        if (y <= limit) {
            g.setFont(Palette.font(t.h >= 84 ? 13 : 12));
            g.setColor(Palette.DIM);
            State.Seat dealer = s.seat(s.dealer);
            drawCentered(g, s.roundWind + "场 " + s.handNo + " 局 · 亲 "
                    + (dealer == null ? "P" + s.dealer : shortName(dealer.name)), c.centerX(), y);
            y += line;
        }
        if (s.pending != null && s.pending.tile != null && y <= limit) {
            g.setFont(Palette.bold(13));
            g.setColor(Palette.RED);
            drawCentered(g, "被鸣的牌：" + (s.pending.tileCn.isEmpty()
                    ? Tile.cn(s.pending.tile) : s.pending.tileCn), c.centerX(), y);
            y += line;
        }
        if (s.pending != null && s.pending.yours && s.pending.waited > 0 && y <= limit) {
            g.setFont(Palette.font(12));
            g.setColor(Palette.DIM);
            drawCentered(g, String.format("已经等了 %.1f 秒", s.pending.waited), c.centerX(), y);
            y += line;
        }
        if (s.hint != null && y <= limit) {
            g.setFont(Palette.font(12));
            g.setColor(Palette.BLUE);
            drawCentered(g, "我的牌效：向听 " + s.hint.shanten + " · 进张 " + s.hint.ukeire
                    + " 种 · " + (s.hint.menzen ? "门前清" : "副露"), c.centerX(), y);
            y += line;
        }
        if (y <= limit) {
            g.setFont(Palette.font(11));
            g.setColor(Palette.DIM);
            drawCentered(g, s.players() + " 人 · 种子 " + s.seed + " · 节拍 "
                    + String.format("%.1fs", s.delay), c.centerX(), y);
        }
    }

    /** 牌桌上方那条横幅：回放时显示「这一手在干嘛」，局间显示「第 N 局结束」。 */
    private void paintTopBanner(Graphics2D g, TableLayout L) {
        State s = state;
        if (s == null || L.banner.w <= 0) {
            return;
        }
        String text = s.bannerText == null ? "" : s.bannerText.trim();
        boolean replay = s.replay;
        if (text.isEmpty()) {
            if (s.awaitingNext) {
                text = s.lastHandNote.isEmpty()
                        ? "第 " + s.handNo + " 局结束 —— 点「下一局 ▶」继续"
                        : s.lastHandNote + " —— 点「下一局 ▶」继续";
                replay = false;
            } else {
                return;
            }
        }
        boolean isNext = !replay && s.awaitingNext;
        int hgt = 36;
        g.setFont(Palette.bold(17));
        FontMetrics fm = g.getFontMetrics();
        int prefix = replay ? 70 : 0;
        String show = text;
        int maxW = L.felt.w - 24;
        while (fm.stringWidth(show) + prefix + 34 > maxW && show.length() > 6) {
            show = show.substring(0, show.length() - 2);
        }
        int w = Math.min(maxW, fm.stringWidth(show) + prefix + 34);
        int x = L.felt.centerX() - w / 2;
        int y = L.banner.y;
        g.setColor(replay ? new Color(0x0B, 0x22, 0x1A, 238) : new Color(0x2A, 0x2A, 0x10, 242));
        g.fillRoundRect(x, y, w, hgt, 18, 18);
        g.setColor(replay ? Palette.GOLD : (isNext ? Palette.GREEN : Palette.GOLD));
        g.setStroke(new BasicStroke(2f));
        g.drawRoundRect(x, y, w, hgt, 18, 18);
        if (replay) {
            g.setFont(Palette.bold(13));
            g.setColor(Palette.GOLD);
            g.drawString("第 " + (s.replayIndex + 1) + " 手", x + 14, y + 24);
        }
        g.setFont(Palette.bold(17));
        g.setColor(Palette.INK);
        g.drawString(show, x + (replay ? prefix : 17), y + 24);
    }

    /** 右下角小小的模式水印，让人一眼看出三麻 / 四麻。 */
    private void paintBrand(Graphics2D g, TableLayout L) {
        State s = state;
        String text = s.players() >= 4 ? "四麻" : "三麻";
        g.setFont(Palette.bold(13));
        g.setColor(new Color(255, 255, 255, 34));
        FontMetrics fm = g.getFontMetrics();
        g.drawString(text, L.felt.right() - fm.stringWidth(text) - 14, L.felt.bottom() - 12);
    }

    private void drawCentered(Graphics2D g, String text, int cx, int baseline) {
        FontMetrics fm = g.getFontMetrics();
        g.drawString(text, cx - fm.stringWidth(text) / 2, baseline);
    }

    /** 连接断了 / 出问题时的横幅。 */
    private void paintBanner(Graphics2D g, String text) {
        g.setFont(Palette.bold(15));
        FontMetrics fm = g.getFontMetrics();
        int w = fm.stringWidth(text) + 34;
        int x = (getWidth() - w) / 2;
        g.setColor(new Color(0x40, 0x12, 0x0C, 235));
        g.fillRoundRect(x, 14, w, 32, 14, 14);
        g.setColor(Palette.RED);
        g.setStroke(new BasicStroke(1.6f));
        g.drawRoundRect(x, 14, w, 32, 14, 14);
        g.setColor(Color.WHITE);
        g.drawString(text, x + 17, 36);
    }

    /** 给外部（自检）用：当前画了几个座位。 */
    public int seatCount() {
        TableLayout L = computeLayout();
        if (L == null) {
            return 0;
        }
        int n = 0;
        for (int slot = 0; slot < L.slotSeat.length; slot++) {
            if (L.slotSeat[slot] >= 0) {
                n++;
            }
        }
        return n;
    }

    /** 这一桌目前用的牌尺寸（全桌统一，出图 / 自检用）。 */
    public String tileSizeText() {
        TableLayout L = computeLayout();
        if (L == null) {
            return "";
        }
        for (TableLayout.Spot sp : L.spots) {
            if (sp.zone == TableLayout.Zone.HAND && sp.seat == (state == null ? 0 : state.you)) {
                return "，统一牌尺寸 " + sp.rect.w + "×" + sp.rect.h;
            }
        }
        return "";
    }

    /** 一张牌都没画出来的话返回 true（自检用）。 */
    public boolean isEmpty() {
        TableLayout L = computeLayout();
        return L == null || L.spots.isEmpty();
    }
}
