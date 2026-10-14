package maj;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.util.List;
import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

/** 右侧牌谱：显示 {@code /api/state} 里的 log，自动滚到底。 */
public final class LogPanel extends JPanel {

    private static final long serialVersionUID = 1L;
    private static final java.awt.Color LOG_BG = new java.awt.Color(0x0B, 0x16, 0x13);

    private final JTextArea area = new JTextArea();
    private String shown = "";
    private boolean stickToBottom = true;

    public LogPanel() {
        super(new BorderLayout());
        setBackground(Palette.PANEL);
        JLabel title = new JLabel(" 牌谱");
        title.setFont(Palette.bold(13));
        title.setForeground(Palette.INK);
        title.setBorder(BorderFactory.createEmptyBorder(6, 4, 6, 4));
        add(title, BorderLayout.NORTH);

        area.setEditable(false);
        area.setLineWrap(true);
        area.setWrapStyleWord(false);
        area.setFont(Palette.font(12));
        area.setBackground(LOG_BG);
        area.setForeground(Palette.INK);
        area.setCaretColor(Palette.INK);
        area.setBorder(BorderFactory.createEmptyBorder(4, 8, 8, 8));

        JScrollPane sp = new JScrollPane(area);
        sp.setBorder(BorderFactory.createMatteBorder(1, 1, 0, 0, Palette.LINE));
        sp.getVerticalScrollBar().setUnitIncrement(16);
        // 用户自己往上翻了就暂时别硬拽到底
        sp.getVerticalScrollBar().addAdjustmentListener(e -> {
            var bar = e.getAdjustable();
            stickToBottom = bar.getValue() >= bar.getMaximum() - bar.getVisibleAmount() - 24;
        });
        add(sp, BorderLayout.CENTER);
        setPreferredSize(new Dimension(360, 100));
    }

    /** 有新日志就换文本（没变就不动，省得滚动条每 500ms 跳一下）。 */
    public void update(List<String> log) {
        StringBuilder sb = new StringBuilder();
        for (String line : log) {
            sb.append(line).append('\n');
        }
        String text = sb.toString();
        if (text.equals(shown)) {
            return;
        }
        shown = text;
        area.setText(text);
        if (stickToBottom) {
            area.setCaretPosition(area.getDocument().getLength());
        }
    }
}
