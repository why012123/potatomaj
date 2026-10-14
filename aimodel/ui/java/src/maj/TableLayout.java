package maj;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * 牌桌几何：俯视视角，四个方向比例统一。
 *
 * <p>几个人怎么坐：
 * <ul>
 *   <li><b>三麻</b>：自己在下，两家在<b>上方左 / 上方右</b>；中间信息区居中。</li>
 *   <li><b>四麻</b>：自己下、下家右、对家上、上家左；左右两家竖排（牌横放）。</li>
 * </ul>
 *
 * <p>每家固定四块，互不重叠：
 * <ol>
 *   <li><b>手牌</b>：四家一样大；自己那家可点、刚摸的留缝；</li>
 *   <li><b>副露</b>：单独一行 / 一列，永远在手牌<b>外侧</b>，绝不压手牌；</li>
 *   <li><b>牌河</b>：按打出顺序排，铳牌标红；</li>
 *   <li>名字条。</li>
 * </ol>
 */
public final class TableLayout {

    /** 牌属于哪个区。 */
    public enum Zone { HAND, MELD, POND, DORA }

    /** 座位朝向：DOWN = 坐在下方（主视角）。 */
    public enum Facing { DOWN, UP, LEFT, RIGHT }

    public static final int SLOT_BOTTOM = 0;
    public static final int SLOT_RIGHT = 1;
    public static final int SLOT_TOP = 2;
    public static final int SLOT_LEFT = 3;
    public static final int SLOT_UPPER_LEFT = 4;
    public static final int SLOT_UPPER_RIGHT = 5;

    public static final String MARK_LAST = "最后";
    public static final String MARK_RIICHI = "立直";
    public static final String MARK_CALLED = "鸣";
    public static final String MARK_DANGER = "铳";

    /** 一张牌（或一张牌背）的位置。 */
    public static final class Spot {
        public final Rect rect;
        public final String tileId;
        public final Zone zone;
        public final int seat;
        public final TileImages.Size size;
        public final boolean rotated;
        public final boolean back;
        public final boolean drawn;
        public final boolean highlight;
        public final boolean danger;
        public final int optionIndex;
        public final String note;

        Spot(Rect rect, String tileId, Zone zone, int seat, TileImages.Size size,
                boolean rotated, boolean back, boolean drawn, boolean highlight, boolean danger,
                int optionIndex, String note) {
            this.rect = rect;
            this.tileId = tileId;
            this.zone = zone;
            this.seat = seat;
            this.size = size;
            this.rotated = rotated;
            this.back = back;
            this.drawn = drawn;
            this.highlight = highlight;
            this.danger = danger;
            this.optionIndex = optionIndex;
            this.note = note;
        }

        public boolean clickable() {
            return optionIndex >= 0;
        }
    }

    /** 一段要画的文字。 */
    public static final class Text {
        public final Rect rect;
        public final String text;
        /** 0 正文 1 暗 2 金 3 红 4 蓝。 */
        public final int style;
        public final boolean center;

        Text(Rect rect, String text, int style, boolean center) {
            this.rect = rect;
            this.text = text;
            this.style = style;
            this.center = center;
        }
    }

    /** 一个座位的位置。 */
    public static final class SeatView {
        public final int seat;
        public final Facing facing;
        public final Rect box;
        public final Rect plate;
        public final Rect hand;
        public final Rect meld;
        public final Rect pond;
        public final TileImages.Size handSize;
        public final boolean handRotated;

        SeatView(int seat, Facing facing, Rect box, Rect plate, Rect hand, Rect meld, Rect pond,
                TileImages.Size handSize, boolean handRotated) {
            this.seat = seat;
            this.facing = facing;
            this.box = box;
            this.plate = plate;
            this.hand = hand;
            this.meld = meld;
            this.pond = pond;
            this.handSize = handSize;
            this.handRotated = handRotated;
        }
    }

    public final List<Spot> spots = new ArrayList<>();
    public final List<Text> texts = new ArrayList<>();
    public final SeatView[] views = new SeatView[6];
    public final int[] slotSeat = {-1, -1, -1, -1, -1, -1};
    public Rect felt = new Rect(0, 0, 0, 0);
    /** 真正用来摆牌的区域（顶部横幅会占掉一条）。 */
    public Rect area = new Rect(0, 0, 0, 0);
    public Rect center = new Rect(0, 0, 0, 0);
    public Rect centerDora = new Rect(0, 0, 0, 0);
    public Rect centerText = new Rect(0, 0, 0, 0);
    /** 顶部横幅（回放那一手在干嘛 / 局间提示）。 */
    public Rect banner = new Rect(0, 0, 0, 0);
    /** 这次布局有没有真的为横幅让出位置（没让的话牌会占满，别拿横幅去判违规）。 */
    public boolean bannerReserved;
    public int width;
    public int height;

    private static final int GAP = 4;
    /** 顶部横幅高度。 */
    static final int BANNER_H = 34;
    /** 中央信息区跟上下两排之间留的空隙。 */
    static final int CENTER_GAP = 10;
    /** 刚摸的那张牌前面留的缝。 */
    static final int DRAWN_GAP = 12;
    /** 副露最多码几行（列）。 */
    static final int MAX_MELD_LINES = 3;
    /** 挤不下时同一行/列最少要露出这么大一截。 */
    static final double MIN_VISIBLE = 0.58;
    /** 牌河放宽一点：它是整片的网格，压到一半也还看得清，牌大更要紧。 */
    static final double POND_MIN_VISIBLE = 0.5;

    /**
     * 全桌<b>只有一个尺寸</b>：自己的手牌、别家的手牌、副露、牌河、宝牌指示全用这一档。
     * 具体是哪一档由 {@link #build} 从大往小试出来（摆得下的最大那一档）。
     */
    static final class Metrics {
        final TileImages.Size tile;

        Metrics(TileImages.Size tile) {
            this.tile = tile;
        }

    }

    private Metrics metrics = new Metrics(TileImages.Size.HAND);

    private TableLayout() {
    }

    /**
     * 算一次布局。
     *
     * @param riichiTiles 每个座位「立直宣言牌」在牌河里的下标（不知道给 -1）
     */
    public static TableLayout build(int w, int h, State s, int[] riichiTiles) {
        return build(w, h, s, riichiTiles, false);
    }

    /**
     * @param reserveBanner 顶上要不要留一条横幅的位置（回放那一手 / 局间提示），
     *                      留了就把牌桌整体往下挪，横幅永远不会压到人家的信息牌
     */
    public static TableLayout build(int w, int h, State s, int[] riichiTiles,
            boolean reserveBanner) {
        // 全桌统一一个尺寸：从大往小试，**第一档真的摆得下的就用它**。
        // 判据是 {@link LayoutRules}（跟自检工具同一套规则），不会「自检过了界面没过」。
        TileImages.Size[] ladder = {TileImages.Size.HAND, TileImages.Size.HAND_M,
            TileImages.Size.HAND_S, TileImages.Size.HAND_XS};
        TableLayout fallback = null;
        for (TileImages.Size t : ladder) {
            TableLayout L = buildWith(w, h, s, riichiTiles, reserveBanner, new Metrics(t));
            if (fallback == null) {
                fallback = L;
            }
            if (L.felt.w <= 0 || LayoutRules.ok(L, s)) {
                return L;
            }
        }
        return fallback;
    }

    static TableLayout buildWith(int w, int h, State s, int[] riichiTiles,
            boolean reserveBanner, Metrics m) {
        TableLayout L = new TableLayout();
        L.width = w;
        L.height = h;
        if (w < 360 || h < 300 || s == null) {
            return L;
        }
        int frame = Math.max(8, Math.min(w, h) / 46);
        L.felt = new Rect(frame, frame, w - 2 * frame, h - 2 * frame);
        int top = reserveBanner ? BANNER_H + 6 : 0;
        L.area = new Rect(L.felt.x, L.felt.y + top, L.felt.w, L.felt.h - top);
        L.bannerReserved = reserveBanner;
        L.metrics = m;
        if (s.players() >= 4) {
            L.buildFour(s, riichiTiles);
        } else {
            L.buildThree(s, riichiTiles);
        }
        L.buildBanner();
        L.buildCenterDora(s);
        return L;
    }

    // ------------------------------------------------------------ 座位分配

    private void buildFour(State s, int[] riichiTiles) {
        Rect f = area;
        int view = clampSeat(s.you, 4);
        slotSeat[SLOT_BOTTOM] = view;
        slotSeat[SLOT_RIGHT] = (view + 1) % 4;
        slotSeat[SLOT_TOP] = (view + 2) % 4;
        slotSeat[SLOT_LEFT] = (view + 3) % 4;

        Rect top = rect(f, 0.205, 0.010, 0.590, 0.330);
        Rect bottom = rect(f, 0.205, 0.600, 0.590, 0.390);
        placeUp(top, SLOT_TOP, riichiTiles, s);
        placeSide(rect(f, 0.004, 0.010, 0.186, 0.975), SLOT_LEFT, true, riichiTiles, s);
        placeSide(rect(f, 0.810, 0.010, 0.186, 0.975), SLOT_RIGHT, false, riichiTiles, s);
        placeDown(bottom, SLOT_BOTTOM, riichiTiles, s);
        // 中央信息区严格卡在上下两排之间的空隙里 —— 永远压不到任何一家的牌河
        center = between(rect(f, 0.300, 0, 0.400, 0), top, bottom);
    }

    private void buildThree(State s, int[] riichiTiles) {
        Rect f = area;
        int view = clampSeat(s.you, 3);
        slotSeat[SLOT_BOTTOM] = view;
        slotSeat[SLOT_UPPER_RIGHT] = (view + 1) % 3;   // 下家 → 上方右
        slotSeat[SLOT_UPPER_LEFT] = (view + 2) % 3;    // 上家 → 上方左

        Rect left = rect(f, 0.006, 0.010, 0.494, 0.350);
        Rect right = rect(f, 0.500, 0.010, 0.494, 0.350);
        Rect bottom = rect(f, 0.060, 0.580, 0.880, 0.410);
        placeUp(left, SLOT_UPPER_LEFT, riichiTiles, s);
        placeUp(right, SLOT_UPPER_RIGHT, riichiTiles, s);
        placeDown(bottom, SLOT_BOTTOM, riichiTiles, s);
        // 上面两家占满整个宽度，所以中央区只能靠纵向的空隙避让
        center = between(rect(f, 0.315, 0, 0.370, 0), left, bottom);
    }

    /** 上下两排之间的那条空隙（横向沿用 {@code band} 的比例），再往里收一圈。 */
    private static Rect between(Rect band, Rect topRow, Rect bottomBox) {
        int y = topRow.bottom() + CENTER_GAP;
        int h = Math.max(0, bottomBox.y - CENTER_GAP - y);
        return new Rect(band.x, y, band.w, h);
    }

    // ------------------------------------------------------------ 三种朝向

    /**
     * 下方（主视角）：名字条 → 牌河 → 副露一行 → 手牌贴下边缘。
     * 三块各有固定区域：先扣掉手牌和「至少一行牌河」，剩下的才给副露，
     * 副露再按这块地方自动换行 / 换小一档，绝不越界压到手上。
     */
    private void placeDown(Rect box, int slot, int[] riichiTiles, State s) {
        int seat = slotSeat[slot];
        TileImages.Size hand = metrics.tile;
        TileImages.Size pond = metrics.tile;
        int plateH = 24;
        int handH = hand.h + 6;
        Rect plate = new Rect(box.x, box.y, (int) (box.w * 0.60), plateH);
        Rect handArea = new Rect(box.x, box.bottom() - hand.h - 2, box.w, handH);
        int room = box.h - plateH - handH;
        // 副露先按「放得下就行」的最小占位拿地方（它是一行行码的，压缩空间小），
        // 剩下的全给牌河 —— 牌河张数最多，能多一行就少一次叠牌。
        MeldFit fit = fitMelds(s.player(seat), box.w, Math.max(0, room - (pond.h + 2) - 6),
                false, metrics.tile);
        int meldH = fit.lines == 0 ? 0 : meldThickness(fit.lines, fit.size, false);
        Rect meldArea = new Rect(box.x, handArea.y - meldH - 6, box.w, meldH);
        Rect pondArea = new Rect(box.x + (int) (box.w * 0.02), box.y + plateH + 4,
                (int) (box.w * 0.96), Math.max(0, (meldArea.y > 0 ? meldArea.y : handArea.y)
                        - box.y - plateH - 10));
        views[slot] = new SeatView(seat, Facing.DOWN, box, plate, handArea, meldArea, pondArea,
                hand, false);

        placeHand(handArea, seat, hand, false, true, s);
        placeMelds(meldArea, seat, fit.size, false, s, fit.lines, 1);   // 横排：居中
        placePond(pondArea, seat, pond, false, 1, 1, true, false, riichiTiles, s);
    }

    /** 上方（对家）：名字条 → 手牌一行 → 副露一行 → 牌河（贴中间）。 */
    private void placeUp(Rect box, int slot, int[] riichiTiles, State s) {
        int seat = slotSeat[slot];
        TileImages.Size hand = metrics.tile;
        TileImages.Size pond = metrics.tile;
        int plateH = 24;
        Rect handArea = new Rect(box.x, box.y + plateH + 2, box.w, hand.h + 6);
        Rect plate = new Rect(box.x, box.y, Math.min((int) (box.w * 0.60), handArea.w), plateH);
        int room = box.h - plateH - handArea.h;
        MeldFit fit = fitMelds(s.player(seat), box.w, Math.max(0, room - (pond.h + 2) - 6),
                false, metrics.tile);
        int meldH = fit.lines == 0 ? 0 : meldThickness(fit.lines, fit.size, false);
        Rect meldArea = new Rect(box.x, handArea.bottom() + 4, box.w, meldH);
        Rect pondArea = new Rect(box.x + (int) (box.w * 0.02), (meldH > 0 ? meldArea.bottom()
                : handArea.bottom()) + 6, (int) (box.w * 0.96),
                Math.max(0, box.bottom() - (meldH > 0 ? meldArea.bottom() : handArea.bottom())
                        - 10));
        views[slot] = new SeatView(seat, Facing.UP, box, plate, handArea, meldArea, pondArea,
                hand, false);

        placeHand(handArea, seat, hand, false, false, s);
        placeMelds(meldArea, seat, fit.size, false, s, fit.lines, 1);   // 横排：居中
        placePond(pondArea, seat, pond, false, 1, -1, true, false, riichiTiles, s);
    }

    /** 左右两家：竖排（牌横放），手牌贴外侧，副露/牌河在内侧一列。 */
    private void placeSide(Rect box, int slot, boolean left, int[] riichiTiles, State s) {
        int seat = slotSeat[slot];
        TileImages.Size hand = metrics.tile;
        // 横放后的牌是 w(true) × h(true)（宽 = 原来的高），别拿 h(true) 当列宽 ——
        // 那样整列比牌还窄，牌会左右各探出去一截，压到副露上。
        int handW = hand.w(true);
        int handX = left ? box.x : box.right() - handW;
        Rect handArea = new Rect(handX, box.y + 30, handW, box.h - 36);
        int innerX = left ? box.x + handW + 8 : box.x;
        int innerW = Math.max(0, box.w - handW - 8);
        // 名牌在牌的上方，可以用整块宽度 —— 不然名字 / 点数会被切掉半截
        Rect plate = new Rect(box.x, box.y, box.w, 26);
        int innerH = box.h - 40;
        // 副露沿纵向码、牌河也在下面，先给牌河留一份保底（至少一档牌高），
        // 剩下的才是副露的地方；副露挤不下就在自己那块里挨紧（牌的大小不变）。
        int pondReserve = Math.max(metrics.tile.w(true), (int) (innerH * 0.45));
        int meldBudget = Math.max(metrics.tile.w(true), innerH - pondReserve - 8);
        int maxCols = Math.max(1, (innerW + 10) / (metrics.tile.w(true) + 10));
        MeldFit fit = fitMelds(s.player(seat), meldBudget,
                Math.max(1, Math.min(innerW, maxCols * (metrics.tile.w(true) + 10) - 10)),
                true, metrics.tile);
        int meldH = fit.lines == 0 ? 0
                : Math.min(meldBudget, meldPack(s.player(seat), fit.size, true, meldBudget,
                        MAX_MELD_LINES).total());
        Rect meldArea = new Rect(innerX, box.y + 32, innerW, meldH);
        Rect pondArea = new Rect(innerX, meldArea.bottom() + 8, innerW,
                Math.max(0, box.bottom() - meldArea.bottom() - 12));
        views[slot] = new SeatView(seat, left ? Facing.LEFT : Facing.RIGHT, box, plate, handArea,
                meldArea, pondArea, hand, true);

        placeHand(handArea, seat, hand, true, false, s);
        placeMelds(meldArea, seat, fit.size, true, s, fit.lines, 0);    // 竖排：贴外侧
        placePond(pondArea, seat, metrics.tile, true, left ? -1 : 1, 1, false, true,
                riichiTiles, s);
    }

    // ------------------------------------------------------------ 手牌

    /**
     * 手牌。回放 / 观战全亮时每家的牌都是正面；否则只有主视角那家是正面。
     *
     * @return 实际占用的矩形（副露要靠它来避让）
     */
    private Rect placeHand(Rect area, int seat, TileImages.Size size, boolean rotated,
            boolean isViewer, State s) {
        boolean mine = isViewer && seat == s.you;
        State.Player pl = s.player(seat);
        // 后端给了别家的 hand（回放 / 观战 reveal）就亮着画，没给就是牌背
        boolean faceUp = mine || s.revealAll || (pl != null && pl.hand != null);
        List<String> ids = new ArrayList<>();
        List<Integer> options = new ArrayList<>();
        List<Boolean> drawn = new ArrayList<>();
        State.Player p = s.player(seat);

        if (mine) {
            List<String> disp = new ArrayList<>();
            for (String id : s.myHand()) {
                disp.add(Tile.isValid(id) ? id : Tile.normalize(id));
            }
            for (String red : s.myRed()) {            // 赤 5 换成赤牌图
                String redId = Tile.redIdOf(Tile.normalize(red));
                if (redId == null) {
                    continue;
                }
                for (int i = 0; i < disp.size(); i++) {
                    if (disp.get(i).equals(Tile.normalize(red))) {
                        disp.set(i, redId);
                        break;
                    }
                }
            }
            disp.sort(Comparator.comparingInt(Tile::sortKey));
            State.Pending pend = s.pending;
            boolean canPick = pend != null && pend.yours && pend.isDiscard() && !s.replay;
            String drawnId = null;
            if (pend != null && pend.drawn != null && Tile.isValid(pend.drawn)) {
                String want = Tile.normalize(pend.drawn);
                for (int i = disp.size() - 1; i >= 0; i--) {
                    if (Tile.normalize(disp.get(i)).equals(want)) {
                        drawnId = disp.remove(i);
                        break;
                    }
                }
            }
            for (String id : disp) {
                ids.add(id);
                options.add(canPick ? pend.indexOfLabel(id) : -1);
                drawn.add(false);
            }
            if (drawnId != null) {
                ids.add(drawnId);
                options.add(canPick ? pend.indexOfLabel(drawnId) : -1);
                drawn.add(true);
            }
        } else {
            List<String> handIds = p == null ? null : p.hand;
            if (handIds != null && !handIds.isEmpty()) {
                List<String> sorted = new ArrayList<>(handIds);
                sorted.sort(Comparator.comparingInt(Tile::sortKey));
                for (String id : sorted) {
                    ids.add(id);
                    options.add(-1);
                    drawn.add(false);
                }
            } else {
                int count = p == null ? 13 : p.handCount;
                for (int i = 0; i < count; i++) {
                    ids.add(null);
                    options.add(-1);
                    drawn.add(false);
                }
            }
        }
        if (ids.isEmpty()) {
            return new Rect(area.x, area.y, 0, 0);
        }

        int tw = size.w(rotated);
        int th = size.h(rotated);
        // 沿手牌方向的边长（横放时是 th，竖放时是 tw）—— 这里以前写死 tw，
        // 左右两家的手牌会算错宽度，整列歪出自己那块。
        int along = rotated ? th : tw;
        int count = ids.size();
        int span = rotated ? area.h : area.w;
        boolean hasDrawn = drawn.contains(Boolean.TRUE);
        int extra = hasDrawn ? DRAWN_GAP : 0;
        // 步进：理想是「牌宽 + 4」；挤不下就先贴紧，再挤不下就允许相邻牌叠一点。
        // 无论窗口多小，整行都落在自己这块矩形里 —— 不越界、不压别人。
        int step = along + 4;
        if (count > 1 && (long) (count - 1) * step + along > span - extra) {
            step = along;
        }
        if (count > 1 && (long) (count - 1) * step + along > span - extra) {
            step = Math.max(1, (span - extra - along) / (count - 1));
        }
        int total = (count - 1) * step + along + extra;
        int usedW = rotated ? tw : total;
        int usedH = rotated ? total : th;
        if (rotated) {
            int y = area.y + Math.max(0, (area.h - total) / 2);
            for (int i = 0; i < count; i++) {
                if (drawn.get(i)) {
                    y += extra;
                }
                spots.add(new Spot(new Rect(area.x + Math.max(0, (area.w - tw) / 2), y, tw, th),
                        ids.get(i), Zone.HAND, seat, size, true, !faceUp, drawn.get(i), false,
                        false, options.get(i), null));
                y += step;
            }
            return new Rect(area.x, area.y, usedW, usedH);
        }
        int x = area.x + Math.max(0, (area.w - total) / 2);
        int x0 = x;
        for (int i = 0; i < count; i++) {
            if (drawn.get(i)) {
                x += extra;
            }
            spots.add(new Spot(new Rect(x, area.y + (area.h - th) / 2, tw, th), ids.get(i),
                    Zone.HAND, seat, size, false, !faceUp, drawn.get(i), false, false,
                    options.get(i), null));
            x += step;
        }
        return new Rect(x0, area.y + (area.h - th) / 2, usedW, usedH);
    }

    // ------------------------------------------------------------ 副露

    /**
     * 副露：独立一块，永远和手牌分开（自己在手牌上方一行、对家在手牌下方一行、
     * 左右两家在手牌内侧一列）。摆不下自动换行 / 换列，再不行换小一档；
     * 实在换不下就<b>叠着放</b>——宁可挤一点，也不许把牌吞掉。
     *
     * @param alignEnd true = 靠右（或靠下）对齐
     */
    private void placeMelds(Rect area, int seat, TileImages.Size size, boolean rotated,
            State s, int maxLines, int align) {
        State.Player p = s.player(seat);
        if (p == null || p.melds.isEmpty() || area.w < 12 || area.h < 12) {
            return;
        }
        int tw = size.w(rotated);
        int th = size.h(rotated);
        int alongTile = rotated ? th : tw;
        int gap = 3;
        int groupGap = 10;
        int alongRoom = Math.max(1, rotated ? area.h : area.w);
        List<State.Meld> melds = p.melds;

        // 1) 先按「自然长度」把几组分成若干行（列）
        int[] glen = new int[melds.size()];
        for (int i = 0; i < melds.size(); i++) {
            int n = melds.get(i).tiles().size();
            glen[i] = n * alongTile + gap * Math.max(0, n - 1);
        }
        List<List<Integer>> lines = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        int used = 0;
        for (int i = 0; i < melds.size(); i++) {
            int add = glen[i] + (cur.isEmpty() ? 0 : groupGap);
            // 还能再起一行才算换行（lines.size() 是已经收起来的行数）
            if (!cur.isEmpty() && used + add > alongRoom
                    && lines.size() + 1 < Math.max(1, maxLines)) {
                lines.add(cur);
                cur = new ArrayList<>();
                used = 0;
                add = glen[i];
            }
            cur.add(i);
            used += add;
        }
        lines.add(cur);

        // 2) 每一行量出总长，超过就**整体等比压紧**（牌的大小不变，只是挨紧甚至叠一点）。
        //    坐标是「自然位置 × scale」再取整 —— 不逐张累加，免得取整误差攒出界。
        int line = 0;
        for (List<Integer> group : lines) {
            int total = 0;
            for (int i = 0; i < group.size(); i++) {
                total += glen[group.get(i)] + (i > 0 ? groupGap : 0);
            }
            // 最后一张牌是实打实的牌宽，不跟着缩 —— 压紧的只是位置
            double scale = total > alongRoom
                    ? Math.max(0.05, (double) (alongRoom - alongTile)
                            / Math.max(1, total - alongTile))
                    : 1.0;
            // 整行的实际长度：最后一张是整张牌，不跟着缩
            int extent = (int) Math.round((total - alongTile) * scale) + alongTile;
            int offset = switch (align) {
                case 1 -> Math.max(0, (alongRoom - extent) / 2);        // 居中
                case 2 -> Math.max(0, alongRoom - extent);              // 靠右 / 靠下
                default -> 0;
            };
            int start = (rotated ? area.y : area.x) + offset;
            int cross = rotated ? area.x + line * (tw + 10) : area.y + line * (th + 6);
            double nat = 0;
            for (int gi : group) {
                State.Meld m = melds.get(gi);
                List<String> tiles = m.tiles();
                int n = tiles.size();
                for (int i = 0; i < n; i++) {
                    int alongPos = start + (int) Math.round(nat * scale);
                    boolean hidden = m.isClosedKan() && (i == 1 || i == 2);
                    Rect r = rotated
                            ? new Rect(cross, alongPos, tw, th)
                            : new Rect(alongPos, cross, tw, th);
                    spots.add(new Spot(r, tiles.get(i), Zone.MELD, seat, size, rotated, hidden,
                            false, false, false, -1, i == n - 1 ? m.kindCn() : null));
                    if (i < n - 1) {
                        nat += alongTile + gap;      // 组内间距
                    }
                }
                nat += alongTile + groupGap;         // 组尾 + 组间距
            }
            line++;
        }
    }

    /** 一次副露折行的结果：行数 + 单行最长占用。 */
    private record Pack(int lines, int total) {
    }

    /** 副露选中的档位 + 要码几行。 */
    private record MeldFit(TileImages.Size size, int lines) {
    }

    /**
     * 副露块要多厚：横排看高度、竖排看宽度。
     * 跟 {@link #placeMelds} 实际用的「换行时 y += th + 6 / 换列时 x += tw + 10」对齐。
     */
    private static int meldThickness(int lines, TileImages.Size size, boolean rotated) {
        int n = Math.max(1, lines);
        // 第 k 行从 y + k*(牌高+6) 开始，所以总高是 n*牌高 + (n-1)*间距（不是 n*(牌高+6)）
        return rotated ? n * size.w(true) + (n - 1) * 10 : n * size.h(false) + (n - 1) * 6;
    }

    /**
     * 挑一档副露：从 preferred 往下试，哪一档「行数 × 厚度」塞得进预算就用哪一档。
     * 一档都塞不进就返回最小那档 —— {@link #placeMelds} 会叠着放，绝不吞牌。
     */
    private static MeldFit fitMelds(State.Player p, int span, int budget, boolean rotated,
            TileImages.Size size) {
        // 尺寸全桌统一，不能靠换小牌省地方：只压行数 / 列数，剩下的交给 placeMelds
        // 在自己那块里挨紧（牌数一张不少）。
        int cap = MAX_MELD_LINES;
        if (budget > 0) {
            int per = rotated ? size.w(true) + 10 : size.h(false) + 6;
            int slack = rotated ? 10 : 6;
            cap = Math.min(cap, Math.max(1, (budget + slack) / per));
        }
        int lines = meldLines(p, size, rotated, Math.max(1, span), cap);
        return new MeldFit(size, lines);
    }

    /**
     * 副露折行：<b>跟 {@link #placeMelds} 用的是同一套贪心规则</b>
     * （组内 3px 间距、组尾留 10px、换行后加 6px），所以「估几行」和「实际摆几行」
     * 永远一致 —— 不然估一行摆两行，第二行就压到隔壁块上去了。
     */
    private static Pack meldPack(State.Player p, TileImages.Size size, boolean rotated, int span,
            int maxLines) {
        if (p == null || p.melds.isEmpty() || span <= 0) {
            return new Pack(0, 0);
        }
        int alongTile = rotated ? size.h(true) : size.w(false);
        int used = 0;
        int lines = 1;
        int longest = 0;
        for (State.Meld m : p.melds) {
            int n = m.tiles().size();
            int gl = alongTile * n + 3 * (n - 1);
            int add = gl + (used == 0 ? 0 : 10);
            if (used > 0 && used + add > span && lines < Math.max(1, maxLines)) {
                lines++;
                longest = Math.max(longest, used);
                used = gl;
            } else {
                used += add;
            }
        }
        longest = Math.max(longest, used);
        return new Pack(lines, Math.min(longest, span));
    }

    /** 这副副露在这块地方要占几行（列）：给区域定尺寸用。 */
    private static int meldLines(State.Player p, TileImages.Size size, boolean rotated,
            int span, int maxLines) {
        return meldPack(p, size, rotated, span, maxLines).lines();
    }

    // ------------------------------------------------------------ 牌河

    /**
     * 牌河：按打出顺序排。
     *
     * @param colDir  列方向（+1 左→右，-1 右→左）
     * @param rowDir  行方向（+1 上→下，-1 下→上）
     * @param centerX 整块横向居中
     * @param centerY 整块纵向居中
     */
    private void placePond(Rect area, int seat, TileImages.Size size, boolean rotated,
            int colDir, int rowDir, boolean centerX, boolean centerY,
            int[] riichiTiles, State s) {
        State.Player p = s.player(seat);
        if (p == null || p.pond.isEmpty()) {
            return;
        }
        List<String> ids = p.pond;
        int n = ids.size();
        int tw = size.w(rotated);
        int th = size.h(rotated);
        if (area.w < tw || area.h < th) {
            return;
        }
        // 牌河能长到二十几张，一张都不许悄悄吞掉（以前这里 min() 一夹就丢牌）。
        // 尺寸全桌统一，这里只调「排几行 / 挤多紧」：优先少行多列（横向挨紧），
        // 横向实在挤不下才加行 —— 行数越少，牌河越扁、越不占地方。
        int bestRows = 1;
        double bestScore = -1;
        for (int rows = 1; rows <= n; rows++) {
            int c = (n + rows - 1) / rows;
            double px = c > 1 ? (double) (area.w - tw) / (c - 1) : tw + 4;
            double py = rows > 1 ? (double) (area.h - th) / (rows - 1) : th + 4;
            double score = Math.min(Math.min(1.0, px / tw), Math.min(1.0, py / th));
            if (score >= POND_MIN_VISIBLE) {
                bestRows = rows;
                bestScore = score;
                break;
            }
            if (score > bestScore) {
                bestScore = score;
                bestRows = rows;
            }
        }
        int rows = bestRows;
        int cols = Math.max(1, Math.min(n, (n + rows - 1) / rows));
        rows = (n + cols - 1) / cols;
        int pitchX = cols > 1 ? Math.max(1, (area.w - tw) / (cols - 1)) : tw + 4;
        int pitchY = rows > 1 ? Math.max(1, (area.h - th) / (rows - 1)) : th + 4;
        if (rows == 1) {
            pitchX = Math.min(pitchX, tw + 4);
        }
        if (cols == 1) {
            pitchY = Math.min(pitchY, th + 4);
        }
        int blockW = (cols - 1) * pitchX + tw;
        int blockH = (rows - 1) * pitchY + th;
        int ox = centerX ? area.x + Math.max(0, (area.w - blockW) / 2)
                : (colDir > 0 ? area.x : area.right() - blockW);
        int oy = centerY ? area.y + Math.max(0, (area.h - blockH) / 2)
                : (rowDir > 0 ? area.y : area.bottom() - blockH);
        int riichi = riichiTiles != null && seat < riichiTiles.length ? riichiTiles[seat] : -1;
        String called = s.pending == null || s.pending.tile == null
                ? null : Tile.normalize(s.pending.tile);
        String danger = s.dangerTile == null ? null : Tile.normalize(s.dangerTile);
        boolean dangerHere = danger != null && (s.dangerSeat < 0 || s.dangerSeat == seat);
        for (int i = 0; i < n; i++) {
            int c = i % cols;
            int r = i / cols;
            int sx = colDir > 0 ? ox + c * pitchX : ox + blockW - tw - c * pitchX;
            int sy = rowDir > 0 ? oy + r * pitchY : oy + blockH - th - r * pitchY;
            String id = ids.get(i);
            boolean isCalled = called != null && called.equals(Tile.normalize(id));
            boolean isDanger = dangerHere && danger.equals(Tile.normalize(id)) && i == n - 1;
            boolean isRiichi = i == riichi;
            String note = isDanger ? MARK_DANGER
                    : isRiichi ? MARK_RIICHI
                    : isCalled ? MARK_CALLED
                    : (i == n - 1 ? MARK_LAST : null);
            spots.add(new Spot(new Rect(sx, sy, tw, th), id, Zone.POND, seat, size,
                    rotated ^ isRiichi, false, false, isCalled, isDanger, -1, note));
        }
    }

    // ------------------------------------------------------------ 横幅 / 中央

    private void buildBanner() {
        int bw = (int) (felt.w * 0.62);
        banner = new Rect(felt.centerX() - bw / 2, felt.y + 4, bw, BANNER_H);
    }

    private void buildCenterDora(State s) {
        if (center.w <= 40 || center.h <= 40) {
            return;
        }
        Rect area = center.inset(12);
        // 宝牌也跟着窗口缩，省下的高度留给「剩余 / 巡目 / 该谁动」
        TileImages.Size ds = metrics.tile;
        Rect dora = new Rect(area.x, area.y + 14, area.w, ds.h);
        centerDora = dora;
        int tw = ds.w;
        int gap = 6;
        int total = s.dora.size() * (tw + gap) - gap;
        int x = dora.x + Math.max(0, (dora.w - total) / 2);
        for (String d : s.dora) {
            spots.add(new Spot(new Rect(x, dora.y, tw, ds.h), d, Zone.DORA, -1, ds, false, false,
                    false, false, false, -1, null));
            x += tw + gap;
        }
        centerText = new Rect(area.x, dora.bottom() + 10, area.w,
                Math.max(0, area.bottom() - dora.bottom() - 10));
    }

    // ------------------------------------------------------------ 命中

    public Spot hitOption(int x, int y) {
        Spot best = null;
        for (Spot sp : spots) {
            if (sp.clickable() && sp.rect.hits(x, y, 3)) {
                best = sp;
            }
        }
        return best;
    }

    public List<Spot> clickable() {
        List<Spot> out = new ArrayList<>();
        for (Spot sp : spots) {
            if (sp.clickable()) {
                out.add(sp);
            }
        }
        return out;
    }

    public SeatView view(int slot) {
        return slot >= 0 && slot < views.length ? views[slot] : null;
    }

    public List<Integer> usedSlots() {
        List<Integer> out = new ArrayList<>();
        for (int slot = 0; slot < slotSeat.length; slot++) {
            if (slotSeat[slot] >= 0) {
                out.add(slot);
            }
        }
        return out;
    }

    private static int clampSeat(int seat, int players) {
        return seat < 0 || seat >= players ? 0 : seat;
    }

    private static Rect rect(Rect base, double x, double y, double w, double h) {
        return new Rect(base.x + (int) Math.round(base.w * x),
                base.y + (int) Math.round(base.h * y),
                (int) Math.round(base.w * w), (int) Math.round(base.h * h));
    }
}
