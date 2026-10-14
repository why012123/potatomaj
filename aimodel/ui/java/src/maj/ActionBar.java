package maj;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 * 底部动作条。
 *
 * <p>打牌的时候**这里不再画牌**（手牌只在自己那一行画一遍，见 {@link TablePanel}）：
 * 这里只说一句「点手牌 / 按数字键」，可点的牌上已经标了键位角标。
 * 其它决策（碰 / 杠 / 荣和 / 立直 / 拔北 …）才在这里摆按钮，文案用 {@code options[i].cn}。
 */
public final class ActionBar extends JPanel {

    /** 按钮点了以后交给外面去 POST /api/act。 */
    public interface Sink {
        void submitOption(int index, String why);

        /** 局间确认：POST /api/next 继续下一局。 */
        void nextHand();
    }

    private static final long serialVersionUID = 1L;

    private final Sink sink;
    private final JLabel status = new JLabel();
    private final JLabel hint = new JLabel();
    private final JPanel buttons = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
    private String signature = "\u0000";

    public ActionBar(Sink sink) {
        super(new BorderLayout(10, 0));
        this.sink = sink;
        setBackground(Palette.PANEL);
        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(1, 0, 0, 0, Palette.LINE),
                BorderFactory.createEmptyBorder(8, 12, 8, 12)));
        setPreferredSize(new Dimension(200, 72));

        status.setFont(Palette.bold(14));
        status.setForeground(Palette.DIM);
        hint.setFont(Palette.font(12));
        hint.setForeground(Palette.DIM);

        JPanel left = new JPanel(new BorderLayout(0, 2));
        left.setOpaque(false);
        left.add(status, BorderLayout.NORTH);
        left.add(hint, BorderLayout.SOUTH);
        left.setPreferredSize(new Dimension(392, 52));
        buttons.setOpaque(false);
        add(left, BorderLayout.WEST);
        add(buttons, BorderLayout.CENTER);
    }

    /** 每 500ms 调一次：内容变了才重建按钮。 */
    public void refresh(State s) {
        State.Pending p = s == null ? null : s.pending;
        boolean between = s != null && !s.replay && s.awaitingNext;
        String sig = between ? ("next:" + s.handNo + ":" + s.lastHandNote)
                : (p == null ? "none" : p.signature());
        if (sig.equals(signature)) {
            return;
        }
        signature = sig;
        buttons.removeAll();
        if (between) {
            status.setForeground(Palette.GREEN);
            String note = s.lastHandNote.isEmpty()
                    ? "第 " + s.handNo + " 局结束" : s.lastHandNote;
            status.setText(note);
            hint.setText("准备好了就点右边继续（后端在等你确认）");
            JButton next = new JButton("下一局 ▶");
            next.setFont(Palette.bold(17));
            next.setBackground(Palette.GOLD);
            next.setForeground(Color.BLACK);
            next.setOpaque(true);
            next.setFocusPainted(false);
            next.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            next.setMargin(new java.awt.Insets(8, 26, 8, 26));
            next.addActionListener(e -> {
                if (sink != null) {
                    sink.nextHand();
                }
            });
            buttons.add(next);
            revalidate();
            repaint();
            return;
        }
        if (p == null) {
            status.setForeground(Palette.DIM);
            status.setText(s == null ? "未连接" : "等待中…");
            hint.setText(s != null && !s.waitingFor.isEmpty() ? "等 " + s.waitingFor : " ");
        } else if (!p.yours) {
            State.Seat st = s.seat(p.seat);
            status.setForeground(Palette.DIM);
            status.setText("等 " + (st == null ? "P" + p.seat : shortName(st.name)) + " " + p.kindCn
                    + " …");
            hint.setText("轮到别人，先看着");
        } else if (p.isDiscard()) {
            status.setForeground(Palette.GOLD);
            status.setText("轮到你出牌：" + p.kindCn + "（点手牌，或按牌上的数字键）");
            hint.setText(discardHint(s, p));
        } else {
            status.setForeground(Palette.GOLD);
            status.setText("轮到你：" + p.kindCn
                    + (p.tileCn.isEmpty() ? "" : "（" + p.tileCn + "）"));
            hint.setText("按按钮，或直接按数字键");
            int n = 0;
            for (State.Option o : p.options) {
                n++;
                JButton b = textButton(o, n);
                final int idx = o.index;
                final String why = "按钮 " + (o.cn.isEmpty() ? o.label : o.cn);
                b.addActionListener(e -> {
                    if (sink != null) {
                        sink.submitOption(idx, why);
                    }
                });
                buttons.add(b);
            }
        }
        revalidate();
        repaint();
    }

    private String discardHint(State s, State.Pending p) {
        StringBuilder sb = new StringBuilder();
        if (p.drawn != null) {
            sb.append("刚摸 ").append(Tile.cn(p.drawn)).append(" · ");
        }
        sb.append("可打 ").append(p.options.size()).append(" 种");
        if (s.hint != null) {
            sb.append(" · 向听 ").append(s.hint.shanten).append(" · 进张 ").append(s.hint.ukeire);
        }
        return sb.toString();
    }

    private JButton textButton(State.Option o, int key) {
        String text = o.cn == null || o.cn.isEmpty() ? o.label : o.cn;
        JButton b = new JButton(key + "  " + text);
        b.setToolTipText("提交 " + o.label);
        b.setFont(Palette.bold(15));
        b.setForeground(Color.BLACK);
        b.setBackground(accent(o));
        b.setOpaque(true);
        b.setFocusPainted(false);
        b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        b.setMargin(new java.awt.Insets(6, 14, 6, 14));
        return b;
    }

    private static Color accent(State.Option o) {
        String l = o.label == null ? "" : o.label;
        return switch (l) {
            case "pass", "no" -> new Color(0xD8, 0xD4, 0xC8);
            case "ron", "tsumo", "yes" -> Palette.GOLD;
            case "riichi" -> Palette.RED;
            default -> new Color(0xEC, 0xE8, 0xDC);
        };
    }

    /** 已经提交、还没等到新局面：把按钮收起来，免得手抖点两下。 */
    public void showSubmitted(String why) {
        signature = "\u0000submitted";
        buttons.removeAll();
        status.setForeground(Palette.BLUE);
        status.setText("已提交：" + why + "，等后端回话…");
        hint.setText(" ");
        revalidate();
        repaint();
    }

    private static String shortName(String name) {
        int i = name.indexOf('（');
        String n = i > 0 ? name.substring(0, i) : name;
        return n.length() > 8 ? n.substring(0, 8) : n;
    }
}
