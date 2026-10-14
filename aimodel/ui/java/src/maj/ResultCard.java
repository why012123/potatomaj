package maj;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.RenderingHints;
import java.util.ArrayList;
import java.util.List;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JViewport;
import javax.swing.ScrollPaneConstants;
import javax.swing.Scrollable;

/**
 * 本局结算页（单独一张卡，不是弹个小框）：用 {@code /api/state} 的 {@code hand_result}。
 *
 * <p>左边把赢家的牌**完整画成 14 张**：副露 + 暗牌 + 和牌那张（单独留缝、金边），
 * 右边是役种列表（真役 / 宝牌分开配色）、番符（役满就写 N 倍役满）、点数，
 * 下面是每家的增减（放铳的那家标红）和结算后点数。流局则显示谁听谁没听 + 不听罚符。
 *
 * <p>版面跟着窗口缩：牌型那条自己挑档位（摆不下就相邻叠一点），役种那张自己滚，
 * 整页再套一层纵向滚动 —— 1000×700 也不会把字裁掉半截。
 */
public final class ResultCard extends JPanel {

    /** 结算页上的操作。 */
    public interface Sink {
        /** POST /api/next 继续下一局。 */
        void nextHand();

        /** 跳到回放，并定位到这一局最后一帧。 */
        void replayThisHand();

        /** 回牌桌看看。 */
        void backToTable();
    }

    /** 右列（役种 / 番数）的最小宽度：再窄就整页滚，而不是把字切掉。 */
    private static final int RIGHT_MIN = 300;

    private final Sink sink;
    private final TileImages images;
    private final JLabel title = new JLabel("本局结算");
    private final JLabel subtitle = new JLabel(" ");
    private final HandStrip handStrip;
    private final JPanel yakuBox = new JPanel();
    private final JLabel hanFu = new JLabel(" ");
    private final JLabel points = new JLabel(" ");
    private final JPanel scoreBox = new JPanel(new GridBagLayout());
    private final JButton nextBtn = new JButton("下一局 ▶");
    private final JButton replayBtn = new JButton("看这一局回放");

    public ResultCard(TileImages images, Sink sink) {
        super(new BorderLayout(0, 10));
        this.images = images;
        this.sink = sink;
        setBackground(Palette.PANEL);
        setBorder(BorderFactory.createEmptyBorder(14, 20, 12, 20));

        JPanel head = new JPanel(new BorderLayout(0, 2));
        head.setOpaque(false);
        title.setFont(Palette.bold(26));
        title.setForeground(Palette.GOLD);
        subtitle.setFont(Palette.font(15));
        subtitle.setForeground(Palette.INK);
        head.add(title, BorderLayout.NORTH);
        head.add(subtitle, BorderLayout.SOUTH);
        add(head, BorderLayout.NORTH);

        // 第一行：赢家的牌（左，跟着窗口伸缩） + 役种 / 番数（右，给个下限）
        handStrip = new HandStrip();
        JPanel left = card("赢家的牌");
        left.add(handStrip, BorderLayout.CENTER);
        left.setMinimumSize(new Dimension(280, 170));

        yakuBox.setOpaque(false);
        yakuBox.setLayout(new BoxLayout(yakuBox, BoxLayout.Y_AXIS));
        JPanel yak = card("役种");
        JScrollPane yakScroll = new JScrollPane(yakuBox,
                ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED,
                ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
        yakScroll.setBorder(null);
        yakScroll.getViewport().setOpaque(false);
        yakScroll.setOpaque(false);
        yakScroll.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
        yak.add(yakScroll, BorderLayout.CENTER);

        hanFu.setFont(Palette.bold(20));
        hanFu.setForeground(Palette.INK);
        points.setFont(Palette.bold(22));
        points.setForeground(Palette.GREEN);
        JPanel numsCol = new JPanel();
        numsCol.setOpaque(false);
        numsCol.setLayout(new BoxLayout(numsCol, BoxLayout.Y_AXIS));
        numsCol.add(hanFu);
        numsCol.add(Box.createVerticalStrut(4));
        numsCol.add(points);
        JPanel nums = card("番数 / 点数");
        nums.add(numsCol, BorderLayout.CENTER);

        JPanel right = new JPanel(new BorderLayout(0, 8));
        right.setOpaque(false);
        right.add(yak, BorderLayout.CENTER);
        right.add(nums, BorderLayout.SOUTH);
        right.setMinimumSize(new Dimension(RIGHT_MIN, 140));
        right.setPreferredSize(new Dimension(360, 250));

        JPanel row1 = new JPanel(new GridBagLayout());
        row1.setOpaque(false);
        GridBagConstraints gc = new GridBagConstraints();
        gc.gridy = 0;
        gc.fill = GridBagConstraints.BOTH;
        gc.weighty = 1;
        gc.insets = new Insets(0, 0, 0, 12);
        gc.gridx = 0;
        gc.weightx = 0.72;                  // 牌型那条要地方，牌才画得大
        row1.add(left, gc);
        gc.gridx = 1;
        gc.weightx = 0.28;
        gc.insets = new Insets(0, 0, 0, 0);
        row1.add(right, gc);
        row1.setAlignmentX(LEFT_ALIGNMENT);
        row1.setMaximumSize(new Dimension(Integer.MAX_VALUE, 400));

        scoreBox.setOpaque(false);
        JPanel scoreCard = card("点数增减");
        scoreCard.add(scoreBox, BorderLayout.CENTER);
        scoreCard.setAlignmentX(LEFT_ALIGNMENT);
        scoreCard.setMaximumSize(new Dimension(Integer.MAX_VALUE, 1400));

        JPanel content = new ScrollWidePanel();
        content.setOpaque(false);
        content.setLayout(new BoxLayout(content, BoxLayout.Y_AXIS));
        content.add(row1);
        content.add(Box.createVerticalStrut(10));
        content.add(scoreCard);

        // 窗口再小也不裁切：整页纵向可滚，动作条永远贴在底下
        JScrollPane scroll = new JScrollPane(content,
                ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED,
                ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
        scroll.setBorder(null);
        scroll.getViewport().setOpaque(false);
        scroll.setOpaque(false);
        scroll.getVerticalScrollBar().setUnitIncrement(18);
        add(scroll, BorderLayout.CENTER);

        nextBtn.setFont(Palette.bold(20));
        nextBtn.setBackground(Palette.GOLD);
        nextBtn.setForeground(Color.BLACK);
        nextBtn.setOpaque(true);
        nextBtn.setFocusPainted(false);
        nextBtn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        nextBtn.setMargin(new Insets(10, 26, 10, 26));
        nextBtn.addActionListener(e -> {
            if (sink != null) {
                sink.nextHand();
            }
        });
        replayBtn.setFont(Palette.font(15));
        replayBtn.addActionListener(e -> {
            if (sink != null) {
                sink.replayThisHand();
            }
        });
        JButton back = new JButton("返回牌桌");
        back.setFont(Palette.font(15));
        back.addActionListener(e -> {
            if (sink != null) {
                sink.backToTable();
            }
        });
        JPanel actions = new JPanel(new FlowLayout(FlowLayout.RIGHT, 10, 0));
        actions.setOpaque(false);
        actions.add(back);
        actions.add(replayBtn);
        actions.add(nextBtn);
        add(actions, BorderLayout.SOUTH);
    }

    /**
     * 宽度跟着视口走的内容面板。
     *
     * <p>普通 JPanel 放进 {@link JScrollPane} 时会被按「首选宽度」摆放，
     * 窗口一窄右边的役种 / 番数就被裁掉了；实现 {@link Scrollable} 并让
     * {@code getScrollableTracksViewportWidth()} 返回 true，内容就会缩到视口宽度。
     */
    private static final class ScrollWidePanel extends JPanel implements Scrollable {
        private static final long serialVersionUID = 1L;

        @Override
        public Dimension getPreferredScrollableViewportSize() {
            return getPreferredSize();
        }

        @Override
        public int getScrollableUnitIncrement(java.awt.Rectangle visible, int orientation,
                int direction) {
            return 18;
        }

        @Override
        public int getScrollableBlockIncrement(java.awt.Rectangle visible, int orientation,
                int direction) {
            return Math.max(24, visible.height - 24);
        }

        @Override
        public boolean getScrollableTracksViewportWidth() {
            return true;
        }

        /** 内容比视口矮的时候撑满视口（多余的纵向空间给「赢家的牌 / 役种」那一行）。 */
        @Override
        public boolean getScrollableTracksViewportHeight() {
            java.awt.Container p = getParent();
            return p instanceof JViewport
                    && getPreferredSize().height <= p.getHeight();
        }
    }

    private static JPanel card(String t) {
        JPanel card = new JPanel(new BorderLayout(0, 6));
        card.setBackground(Palette.PANEL_SOFT);
        card.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(Palette.LINE, 1),
                BorderFactory.createEmptyBorder(10, 14, 12, 14)));
        JLabel lab = new JLabel(t);
        lab.setFont(Palette.bold(15));
        lab.setForeground(Palette.GOLD);
        card.add(lab, BorderLayout.NORTH);
        return card;
    }

    // ------------------------------------------------------------ 刷新

    /** 用当前局面刷新这一页（必须 EDT）。 */
    public void show(State s) {
        State.RoundResult r = s == null ? null : s.handResult;
        if (r == null) {
            title.setText("本局结算");
            subtitle.setText("还没有结算数据（后端还没给 hand_result）");
            handStrip.set(null, null, null, null);
            yakuBox.removeAll();
            hanFu.setText(" ");
            points.setText(" ");
            scoreBox.removeAll();
            nextBtn.setEnabled(false);
            revalidate();
            repaint();
            return;
        }
        String who = name(s, r.winner);
        if (r.isDraw()) {
            title.setText("第 " + r.handNo + " 局 · " + (r.label.isEmpty() ? "流局" : r.label));
            subtitle.setText("流局 —— " + tenpaiText(s, r));
        } else {
            title.setText("第 " + r.handNo + " 局 · " + (r.label.isEmpty() ? "和了" : r.label)
                    + "（" + who + "）");
            subtitle.setText(r.tsumo
                    ? who + " 自摸" + (r.winTile.isEmpty() ? "" : "　和牌张 " + Tile.cn(r.winTile))
                    : who + " 荣和 ← " + name(s, r.loser) + " 放铳"
                      + (r.winTile.isEmpty() ? "" : "　和牌张 " + Tile.cn(r.winTile)));
        }
        handStrip.set(r.isDraw() ? new ArrayList<>() : r.hand, r.melds, r.winTile, images);
        replayBtn.setEnabled(true);
        nextBtn.setEnabled(s.awaitingNext || s.confirmBetween);

        yakuBox.removeAll();
        if (r.isDraw()) {
            yakuBox.add(line("流局，没有役", Palette.DIM, 14));
        } else {
            for (State.Yaku y : r.realYaku()) {
                yakuBox.add(line(y.name + "　" + y.han + " 番", Palette.INK, 15));
            }
            List<State.Yaku> bonus = r.bonusYaku();
            if (!bonus.isEmpty()) {
                yakuBox.add(Box.createVerticalStrut(6));
                JLabel tip = new JLabel("宝牌类");
                tip.setFont(Palette.font(12));
                tip.setForeground(Palette.DIM);
                tip.setAlignmentX(LEFT_ALIGNMENT);
                yakuBox.add(tip);
                for (State.Yaku y : bonus) {
                    yakuBox.add(line(y.name + "　" + y.han + " 番", Palette.BLUE, 15));
                }
            }
            if (r.realYaku().isEmpty() && bonus.isEmpty()) {
                yakuBox.add(line("（没有役种数据）", Palette.DIM, 14));
            }
        }
        if (r.yakuman > 0) {
            hanFu.setText(r.yakuman + " 倍役满");
            hanFu.setForeground(Palette.RED);
        } else if (r.isDraw()) {
            hanFu.setText("流局");
            hanFu.setForeground(Palette.DIM);
        } else {
            hanFu.setText(r.totalHan + " 番 " + r.fu + " 符"
                    + (r.doraHan > 0 ? "（宝牌 " + r.doraHan + " 番）" : ""));
            hanFu.setForeground(Palette.INK);
        }
        points.setText(r.isDraw() ? "不听罚符见下" : "+" + Math.max(0, r.points) + " 点");

        scoreBox.removeAll();
        int rows = Math.max(r.deltas.size(), Math.max(r.scores.size(), r.players));
        for (int i = 0; i < rows; i++) {
            int d = i < r.deltas.size() ? r.deltas.get(i) : 0;
            int sc = i < r.scores.size() ? r.scores.get(i) : 0;
            boolean isWinner = !r.isDraw() && i == r.winner;
            boolean isLoser = !r.isDraw() && i == r.loser;
            String tag = isWinner ? "和" : isLoser ? "铳" : "";
            JLabel lab = new JLabel(name(s, i) + (tag.isEmpty() ? "" : "（" + tag + "）"));
            lab.setFont(Palette.font(14));
            lab.setForeground(isWinner ? Palette.GOLD : isLoser ? Palette.RED : Palette.INK);
            JLabel dl = new JLabel((d > 0 ? "+" : "") + d);
            dl.setFont(Palette.bold(17));
            dl.setForeground(d > 0 ? Palette.GREEN : d < 0 ? Palette.RED : Palette.DIM);
            JLabel sl = new JLabel(String.valueOf(sc));
            sl.setFont(Palette.font(15));
            sl.setForeground(Palette.INK);
            JLabel tp = new JLabel(r.isDraw() && i < r.tenpai.size()
                    ? (r.tenpai.get(i) ? "听牌" : "不听") : " ");
            tp.setFont(Palette.font(13));
            tp.setForeground(Palette.DIM);
            GridBagConstraints c = new GridBagConstraints();
            c.gridy = i;
            c.insets = new Insets(2, 6, 2, 16);
            c.anchor = GridBagConstraints.LINE_START;
            c.gridx = 0;
            scoreBox.add(lab, c);
            c.gridx = 1;
            scoreBox.add(dl, c);
            c.gridx = 2;
            scoreBox.add(sl, c);
            c.gridx = 3;
            c.insets = new Insets(2, 0, 2, 6);
            scoreBox.add(tp, c);
        }
        // 右下角放一个能伸能缩的空格：多出来的地方全给它，表格就永远贴在左上角
        GridBagConstraints filler = new GridBagConstraints();
        filler.gridx = 4;
        filler.gridy = rows;
        filler.weightx = 1;
        filler.weighty = 1;
        filler.fill = GridBagConstraints.BOTH;
        scoreBox.add(Box.createGlue(), filler);
        revalidate();
        repaint();
    }

    private String tenpaiText(State s, State.RoundResult r) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < r.tenpai.size(); i++) {
            if (r.tenpai.get(i)) {
                sb.append(sb.length() > 0 ? "、" : "").append(name(s, i)).append(" 听牌");
            }
        }
        return sb.length() == 0 ? "全都不听" : sb.toString();
    }

    private JLabel line(String text, Color color, int size) {
        JLabel lab = new JLabel(text);
        lab.setFont(Palette.font(size));
        lab.setForeground(color);
        lab.setAlignmentX(LEFT_ALIGNMENT);
        return lab;
    }

    private static String name(State s, int seat) {
        if (seat < 0) {
            return "？";
        }
        State.Seat st = s.seat(seat);
        if (st == null) {
            return "P" + seat;
        }
        String n = st.name;
        int i = n.indexOf('（');
        return i > 0 ? n.substring(0, i) : n;
    }

    /** 给自检 / 出图用。 */
    public JButton nextButton() {
        return nextBtn;
    }

    public HandStrip strip() {
        return handStrip;
    }

    // ------------------------------------------------------------ 牌型

    /**
     * 赢家的牌：副露 + 暗牌 + 和牌那张（留缝 + 金边）。
     * 牌宽跟着面板宽度挑档位，小窗口下不会画到面板外面去。
     */
    public final class HandStrip extends JPanel {
        private static final long serialVersionUID = 1L;
        private static final int GAP = 3;
        private static final int GROUP_GAP = 10;
        private static final int WIN_GAP = 14;
        private static final int PAD = 12;
        /** 从大到小的牌档（结算页只用「手牌」这一系）。 */
        private static final TileImages.Size[] LADDER = {TileImages.Size.HAND,
            TileImages.Size.HAND_M, TileImages.Size.HAND_S, TileImages.Size.MINI_M,
            TileImages.Size.MINI_S};
        /** 挤不下时最少露出这么大一截，否则退一档。 */
        private static final float MIN_VISIBLE = 0.62f;
        private List<String> concealed = new ArrayList<>();
        private List<State.Meld> melds = new ArrayList<>();
        private String winTile;
        private TileImages art;

        HandStrip() {
            setOpaque(false);
        }

        void set(List<String> concealed, List<State.Meld> melds, String winTile,
                TileImages art) {
            this.concealed = concealed == null ? new ArrayList<>() : concealed;
            this.melds = melds == null ? new ArrayList<>() : melds;
            // 流局时后端给的是空串，按「没有和牌张」处理
            this.winTile = winTile == null || winTile.isBlank() || !Tile.isValid(winTile)
                    ? null : winTile;
            this.art = art;
            revalidate();
            repaint();
        }

        /** 一共要画几张（含和牌那张）。 */
        private int slotCount() {
            int n = concealed.size() + (winTile == null ? 0 : 1);
            for (State.Meld m : melds) {
                n += m.tiles().size();
            }
            return n;
        }

        @Override
        public Dimension getPreferredSize() {
            int n = Math.max(1, slotCount());
            return new Dimension(n * (TileImages.Size.HAND.w + GAP) + 80,
                    TileImages.Size.HAND.h + 46);
        }

        @Override
        public Dimension getMinimumSize() {
            return new Dimension(200, TileImages.Size.MINI_S.h + 40);
        }



        @Override
        protected void paintComponent(Graphics g0) {
            super.paintComponent(g0);
            Graphics2D g = (Graphics2D) g0.create();
            try {
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                        RenderingHints.VALUE_ANTIALIAS_ON);
                g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
                        RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                int count = slotCount();
                if (count == 0) {
                    g.setColor(Palette.DIM);
                    g.setFont(Palette.font(14));
                    g.drawString("（流局：没有和牌牌型）", 12, getHeight() / 2);
                    return;
                }
                int width = Math.max(120, getWidth());
                int fixed = GROUP_GAP * melds.size() + (winTile == null ? 0 : WIN_GAP);
                int room = Math.max(40, width - PAD - fixed);
                // 先挑「画出来最大、还不至于糊成一片」的那一档：宽度不够就相邻叠一点，
                // 至少露 MIN_VISIBLE 一截 —— 结算页的牌也不能小成芝麻。
                TileImages.Size artSize = TileImages.Size.HAND;
                for (TileImages.Size cand : LADDER) {
                    artSize = cand;
                    int step0 = Math.max(1, (room - cand.w) / Math.max(1, count - 1));
                    if ((long) count * (cand.w + GAP) <= room || step0 >= cand.w * MIN_VISIBLE) {
                        break;
                    }
                }
                int tw = artSize.w;
                int th = artSize.h;
                int step = tw + GAP;
                if ((long) count * step - GAP > room && count > 1) {
                    step = Math.max(1, (room - tw) / (count - 1));
                }
                int total = (count - 1) * step + tw;
                int y = Math.max(4, (getHeight() - th) / 2 - 4);
                int x = Math.max(6, (width - total) / 2);

                for (State.Meld m : melds) {
                    for (String id : m.tiles()) {
                        drawTile(g, x, y, tw, th, artSize, id);
                        x += step;
                    }
                    g.setColor(Palette.DIM);
                    g.setFont(Palette.font(11));
                    g.drawString(m.kindCn(), x - step + 2, y + th + 13);
                    x += GROUP_GAP;
                }
                for (String id : concealed) {
                    drawTile(g, x, y, tw, th, artSize, id);
                    x += step;
                }
                if (winTile != null) {
                    x += Math.max(0, WIN_GAP - GAP);
                    int wy = Math.max(2, y - 5);
                    g.setColor(Palette.GOLD);
                    g.setStroke(new java.awt.BasicStroke(2.5f));
                    g.drawRoundRect(x - 3, wy - 3, tw + 6, th + 6, 8, 8);
                    drawTile(g, x, wy, tw, th, artSize, winTile);
                    g.setFont(Palette.bold(12));
                    FontMetrics fm = g.getFontMetrics();
                    String tag = "和牌";
                    g.setColor(Palette.GOLD);
                    g.drawString(tag, x + (tw - fm.stringWidth(tag)) / 2, wy + th + 18);
                }
            } finally {
                g.dispose();
            }
        }

        private void drawTile(Graphics2D g, int x, int y, int w, int h, TileImages.Size artSize,
                String id) {
            g.setColor(Palette.SHADOW);
            g.fillRoundRect(x + 2, y + 3, w, h, 7, 7);
            ImageIcon ic = art == null ? null : art.get(id, artSize, false);
            if (ic != null) {
                g.drawImage(ic.getImage(), x, y, w, h, null);   // 插值在 paintComponent 里设好
                return;
            }
            g.setColor(Palette.TILE_FACE);
            g.fillRoundRect(x, y, w, h, 7, 7);
            g.setColor(Palette.TILE_EDGE);
            g.drawRoundRect(x, y, w - 1, h - 1, 7, 7);
            String text = id == null ? "?" : Tile.cn(id);
            g.setFont(Palette.bold(Math.max(10, Math.min(w, h) / 3)));
            g.setColor(Tile.isRedId(id) ? Palette.RED : new Color(0x22, 0x22, 0x22));
            FontMetrics fm = g.getFontMetrics();
            g.drawString(text, x + (w - fm.stringWidth(text)) / 2,
                    y + (h + fm.getAscent()) / 2 - 3);
        }
    }
}
