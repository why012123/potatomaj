package maj;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Window;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

/**
 * 结算面板：每家点数 / 顺位 / 段位分，以及每一局谁和了什么。
 * 关掉或者点「再来一局」都可以，后者会 POST /api/new。
 */
public final class ResultDialog extends JDialog {

    private static final long serialVersionUID = 1L;

    private boolean again;

    public ResultDialog(Window owner, State s) {
        super(owner, "对局结束", ModalityType.APPLICATION_MODAL);
        State.Result r = s.result;

        JPanel root = new JPanel(new BorderLayout(10, 10));
        root.setBackground(Palette.PANEL);
        root.setBorder(BorderFactory.createEmptyBorder(14, 16, 14, 16));

        JLabel head = new JLabel(r == null ? "对局结束（没有拿到结算数据）"
                : s.players() + " 人 · 种子 " + s.seed + " · 共 " + r.hands.size() + " 局");
        head.setFont(Palette.bold(16));
        head.setForeground(Palette.GOLD);
        root.add(head, BorderLayout.NORTH);

        JPanel mid = new JPanel(new BorderLayout(8, 8));
        mid.setOpaque(false);
        if (r != null) {
            mid.add(scoreTable(s, r), BorderLayout.NORTH);
            JTextArea area = new JTextArea(handsText(s, r));
            area.setEditable(false);
            area.setFont(Palette.font(13));
            area.setBackground(Palette.PANEL_SOFT);
            area.setForeground(Palette.INK);
            area.setBorder(BorderFactory.createEmptyBorder(8, 10, 8, 10));
            JScrollPane sp = new JScrollPane(area);
            sp.setBorder(BorderFactory.createLineBorder(Palette.LINE));
            sp.setPreferredSize(new Dimension(560, 240));
            mid.add(sp, BorderLayout.CENTER);
        }
        root.add(mid, BorderLayout.CENTER);

        JButton againBtn = new JButton("再来一局");
        againBtn.addActionListener(e -> {
            again = true;
            dispose();
        });
        JButton close = new JButton("关闭");
        close.addActionListener(e -> dispose());
        JPanel bottom = new JPanel(new java.awt.FlowLayout(java.awt.FlowLayout.RIGHT, 8, 0));
        bottom.setOpaque(false);
        bottom.add(close);
        bottom.add(againBtn);
        root.add(bottom, BorderLayout.SOUTH);
        getRootPane().setDefaultButton(againBtn);

        setContentPane(root);
        pack();
        setLocationRelativeTo(owner);
    }

    /** 用户是不是点了「再来一局」。 */
    public boolean againClicked() {
        return again;
    }

    private JPanel scoreTable(State s, State.Result r) {
        JPanel p = new JPanel(new GridBagLayout());
        p.setOpaque(false);
        String[] cols = {"座位", "点数", "顺位", "段位分", "本局增减"};
        for (int c = 0; c < cols.length; c++) {
            JLabel lab = new JLabel(cols[c]);
            lab.setFont(Palette.bold(13));
            lab.setForeground(Palette.DIM);
            GridBagConstraints gc = new GridBagConstraints();
            gc.gridx = c;
            gc.gridy = 0;
            gc.insets = new Insets(2, 8, 6, 8);
            gc.anchor = GridBagConstraints.LINE_START;
            p.add(lab, gc);
        }
        for (int i = 0; i < s.players(); i++) {
            State.Seat info = s.seat(i);
            String name = info == null ? "P" + i : info.name;
            int score = i < r.scores.size() ? r.scores.get(i) : 0;
            int rank = i < r.ranks.size() ? r.ranks.get(i) : 0;
            double rp = i < r.rankPoints.size() ? r.rankPoints.get(i) : 0;
            int delta = i < r.deltas.size() ? r.deltas.get(i) : 0;
            String[] cells = {
                name,
                String.valueOf(score),
                (rank + 1) + " 位",
                String.format("%+.1f", rp),
                String.format("%+d", delta),
            };
            for (int c = 0; c < cells.length; c++) {
                JLabel lab = new JLabel(cells[c]);
                lab.setFont(c == 1 ? Palette.bold(14) : Palette.font(13));
                boolean win = rank == 0;
                lab.setForeground(c == 1 ? (win ? Palette.GOLD : Palette.INK)
                        : c == 3 ? (rp >= 0 ? Palette.GREEN : Palette.RED) : Palette.INK);
                GridBagConstraints gc = new GridBagConstraints();
                gc.gridx = c;
                gc.gridy = i + 1;
                gc.insets = new Insets(2, 8, 2, 8);
                gc.anchor = GridBagConstraints.LINE_START;
                p.add(lab, gc);
            }
        }
        return p;
    }

    private String handsText(State s, State.Result r) {
        StringBuilder sb = new StringBuilder();
        sb.append("每一局\n");
        int no = 1;
        for (State.HandResult h : r.hands) {
            sb.append("  ").append(no++).append(". ").append(h.text);
            if (h.turns > 0) {
                sb.append("（第 ").append(h.turns).append(" 巡）");
            }
            if (!h.deltas.isEmpty()) {
                sb.append("  ");
                for (int i = 0; i < h.deltas.size(); i++) {
                    int d = h.deltas.get(i);
                    if (d != 0) {
                        sb.append('P').append(i).append(String.format("%+d ", d));
                    }
                }
            }
            sb.append('\n');
        }
        if (r.hands.isEmpty()) {
            sb.append("  （流局 / 没有和牌记录）\n");
        }
        sb.append("\n数据统计\n");
        String[] keys = {"和了率", "放铳率", "立直率", "副露率", "平均翻数"};
        sb.append("  座位    ");
        for (String k : keys) {
            sb.append(String.format("%-8s", k));
        }
        sb.append('\n');
        for (int i = 0; i < r.stats.size(); i++) {
            State.Seat info = s.seat(i);
            sb.append(String.format("  %-6s", info == null ? "P" + i : "P" + i));
            for (String k : keys) {
                Double v = r.stats.get(i).get(k);
                sb.append(String.format("%-8s", v == null ? "-" : String.format("%.2f", v)));
            }
            sb.append('\n');
        }
        return sb.toString();
    }
}
