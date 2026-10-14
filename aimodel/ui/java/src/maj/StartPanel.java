package maj;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.util.ArrayList;
import java.util.List;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

/**
 * 开局选择页：**打开窗口第一眼就是它**。
 *
 * <p>默认值就是**模型**：三麻填 {@code suggest["3"]}（maj-flash / maj-huge / maj-mountain），
 * 四麻填 {@code suggest["4"]}（maj-flash4 / maj-mountain4 / maj-normal4），
 * 切换人数会重新填一遍。下拉里模型显示成 {@code maj-flash · 速攻}，并且按当前人数过滤。
 * 开局前会检查「模型在不在、人数配不配」，不会让座位偷偷退化成教师。
 */
public final class StartPanel extends JPanel {

    /** 选好了以后交给 MainFrame 去 POST /api/config。 */
    public interface Sink {
        void startGame(MajClient.Config cfg);

        void startBackend();
    }

    private static final long serialVersionUID = 1L;

    /** 下拉框里的一项。 */
    private record Choice(String value, String label, boolean model) {
        @Override
        public String toString() {
            return label;
        }
    }

    private final Sink sink;
    private final JRadioButton sanma = new JRadioButton("三麻");
    private final JRadioButton yonma = new JRadioButton("四麻");
    private final JLabel modeNote = new JLabel();
    private final JPanel seatRows = new JPanel(new GridBagLayout());
    private final List<JComboBox<Choice>> combos = new ArrayList<>();
    private final JSlider delay = new JSlider(0, 30, 10);
    private final JLabel delayText = new JLabel("1.0 秒");
    private final JTextField seed = new JTextField("1", 8);
    private final JLabel backendState = new JLabel("后端：检查中…");
    private final JLabel warn = new JLabel(" ");
    private final JButton startBackend = new JButton("启动后端");
    private final JCheckBox stopOnExit = new JCheckBox("关窗口时停掉后端", true);
    private MajClient.Models models = new MajClient.Models();
    private String modelsSig = "";
    private String lastSeats = "";
    private boolean building;
    /** 用户自己动过下拉 / 点过一键按钮没有（动过就别用默认值覆盖他）。 */
    private boolean userTouched;

    public StartPanel(Sink sink) {
        super(new BorderLayout(0, 14));
        this.sink = sink;
        setBackground(Palette.PANEL);
        setBorder(BorderFactory.createEmptyBorder(22, 34, 20, 34));

        add(header(), BorderLayout.NORTH);
        add(body(), BorderLayout.CENTER);
        add(footer(), BorderLayout.SOUTH);

        sanma.setSelected(true);
        ButtonGroup bg = new ButtonGroup();
        bg.add(sanma);
        bg.add(yonma);
        sanma.addActionListener(e -> modeChanged());
        yonma.addActionListener(e -> modeChanged());
        rebuildSeats(defaultSeats(players()));
        updateModeNote();
    }

    // ------------------------------------------------------------ 组件

    private JPanel header() {
        JPanel p = new JPanel(new BorderLayout(0, 4));
        p.setOpaque(false);
        JLabel title = new JLabel("土豆麻将 · Java 牌桌");
        title.setFont(Palette.bold(30));
        title.setForeground(Palette.GOLD);
        JLabel sub = new JLabel("先选好这一桌：默认已经填好当前人数的模型，直接开打或改成人类自己上。");
        sub.setFont(Palette.font(14));
        sub.setForeground(Palette.DIM);
        p.add(title, BorderLayout.NORTH);
        p.add(sub, BorderLayout.SOUTH);
        return p;
    }

    private JPanel body() {
        JPanel p = new JPanel(new GridBagLayout());
        p.setOpaque(false);
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = 0;
        c.weightx = 1;
        c.fill = GridBagConstraints.HORIZONTAL;
        c.insets = new Insets(0, 0, 14, 0);
        p.add(modeCard(), c);
        c.gridy = 1;
        c.insets = new Insets(0, 0, 0, 0);
        p.add(seatCard(), c);
        GridBagConstraints glue = new GridBagConstraints();
        glue.gridx = 0;
        glue.gridy = 2;
        glue.weighty = 1;
        glue.fill = GridBagConstraints.BOTH;
        p.add(Box.createGlue(), glue);
        return p;
    }

    private JPanel modeCard() {
        JPanel card = card("对局模式");
        JPanel col = new JPanel(new GridBagLayout());
        col.setOpaque(false);
        GridBagConstraints gc = new GridBagConstraints();
        gc.gridx = 0;
        gc.weightx = 1;
        gc.fill = GridBagConstraints.HORIZONTAL;
        gc.anchor = GridBagConstraints.NORTHWEST;
        int row = 0;

        JPanel modes = new JPanel(new java.awt.FlowLayout(java.awt.FlowLayout.LEFT, 10, 0));
        modes.setOpaque(false);
        style(sanma, 22);
        style(yonma, 22);
        modes.add(sanma);
        modes.add(yonma);
        gc.gridy = row++;
        gc.insets = new Insets(0, 0, 4, 0);
        col.add(modes, gc);

        modeNote.setFont(Palette.font(12));
        modeNote.setForeground(Palette.DIM);
        gc.gridy = row++;
        gc.insets = new Insets(0, 0, 16, 0);
        col.add(modeNote, gc);

        JLabel dl = new JLabel("AI 节拍（每手停顿）");
        dl.setFont(Palette.bold(14));
        dl.setForeground(Palette.INK);
        gc.gridy = row++;
        gc.insets = new Insets(0, 0, 2, 0);
        col.add(dl, gc);
        delay.setOpaque(false);
        delay.setMajorTickSpacing(10);
        delay.setMinorTickSpacing(5);
        delay.setPaintTicks(true);
        delay.setPaintLabels(false);
        delay.setPreferredSize(new Dimension(320, 44));
        delay.addChangeListener(e -> delayText.setText(
                String.format("%.1f 秒", delay.getValue() / 10.0)));
        delayText.setText("1.0 秒");
        delayText.setFont(Palette.bold(15));
        delayText.setForeground(Palette.GOLD);
        gc.gridy = row++;
        gc.insets = new Insets(0, 0, 0, 0);
        gc.fill = GridBagConstraints.NONE;
        gc.anchor = GridBagConstraints.NORTHWEST;
        col.add(delay, gc);
        gc.gridy = row++;
        gc.insets = new Insets(0, 0, 14, 0);
        col.add(delayText, gc);

        JLabel sl = new JLabel("随机种子（同一套配置同一种子可复现）");
        sl.setFont(Palette.bold(14));
        sl.setForeground(Palette.INK);
        gc.gridy = row++;
        gc.insets = new Insets(0, 0, 2, 0);
        col.add(sl, gc);
        seed.setFont(Palette.font(15));
        seed.setPreferredSize(new Dimension(150, 30));
        gc.gridy = row++;
        gc.insets = new Insets(0, 0, 0, 0);
        col.add(seed, gc);

        JLabel usage = new JLabel("玩起来：点手牌出牌 / 键盘 1~9 选第 N 项 / "
                + "Cmd·Ctrl+R 回这一页 / Cmd·Ctrl+N 换一批");
        usage.setFont(Palette.font(12));
        usage.setForeground(Palette.DIM);
        gc.gridy = row++;
        gc.insets = new Insets(14, 0, 0, 0);
        col.add(usage, gc);

        GridBagConstraints glue = new GridBagConstraints();
        glue.gridx = 0;
        glue.gridy = row;
        glue.weighty = 1;
        glue.fill = GridBagConstraints.BOTH;
        col.add(Box.createGlue(), glue);

        JPanel wrap = new JPanel(new BorderLayout());
        wrap.setOpaque(false);
        wrap.add(col, BorderLayout.NORTH);
        card.add(wrap, BorderLayout.CENTER);
        return card;
    }

    private JPanel seatCard() {
        JPanel card = card("每个座位派谁（默认已经是模型，可换成人类自己打）");
        seatRows.setOpaque(false);
        JPanel wrap = new JPanel(new BorderLayout());
        wrap.setOpaque(false);
        wrap.add(seatRows, BorderLayout.NORTH);
        card.add(wrap, BorderLayout.CENTER);
        warn.setFont(Palette.font(12));
        warn.setForeground(Palette.RED);
        card.add(warn, BorderLayout.SOUTH);
        return card;
    }

    private JPanel footer() {
        JPanel p = new JPanel(new BorderLayout(12, 0));
        p.setOpaque(false);

        JPanel left = new JPanel();
        left.setOpaque(false);
        left.setLayout(new javax.swing.BoxLayout(left, javax.swing.BoxLayout.Y_AXIS));
        backendState.setFont(Palette.font(13));
        backendState.setForeground(Palette.DIM);
        backendState.setAlignmentX(LEFT_ALIGNMENT);
        stopOnExit.setOpaque(false);
        stopOnExit.setFont(Palette.font(13));
        stopOnExit.setForeground(Palette.DIM);
        stopOnExit.setAlignmentX(LEFT_ALIGNMENT);
        left.add(backendState);
        left.add(Box.createVerticalStrut(4));
        left.add(stopOnExit);
        p.add(left, BorderLayout.WEST);

        JButton versus = new JButton("模型对战：我 vs 模型");
        versus.setFont(Palette.bold(15));
        versus.setToolTipText("P0 留给你自己，其余座位填当前人数的推荐模型");
        versus.addActionListener(e -> modelVersusHuman());
        JButton spectate = new JButton("观战：全是模型");
        spectate.setFont(Palette.bold(15));
        spectate.setToolTipText("所有座位都交给模型，你只看");
        spectate.addActionListener(e -> spectateAll());
        startBackend.setFont(Palette.font(15));
        startBackend.addActionListener(e -> {
            if (sink != null) {
                sink.startBackend();
            }
        });
        JButton start = new JButton("开始对局  ▶");
        start.setFont(Palette.bold(20));
        start.setBackground(Palette.GOLD);
        start.setForeground(Color.BLACK);
        start.setOpaque(true);
        start.setFocusPainted(false);
        start.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        start.setMargin(new Insets(10, 26, 10, 26));
        start.addActionListener(e -> tryStart());

        JPanel right = new JPanel(new java.awt.FlowLayout(java.awt.FlowLayout.RIGHT, 10, 0));
        right.setOpaque(false);
        right.add(versus);
        right.add(spectate);
        right.add(startBackend);
        right.add(start);
        p.add(right, BorderLayout.EAST);
        return p;
    }

    private static void style(JRadioButton b, int size) {
        b.setOpaque(false);
        b.setFont(Palette.bold(size));
        b.setForeground(Palette.INK);
    }

    private static JPanel card(String title) {
        JPanel card = new JPanel(new BorderLayout(0, 10));
        card.setBackground(Palette.PANEL_SOFT);
        card.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(Palette.LINE, 1),
                BorderFactory.createEmptyBorder(16, 18, 16, 18)));
        JLabel t = new JLabel(title);
        t.setFont(Palette.bold(16));
        t.setForeground(Palette.GOLD);
        t.setHorizontalAlignment(SwingConstants.LEFT);
        card.add(t, BorderLayout.NORTH);
        return card;
    }

    // ------------------------------------------------------------ 模式 / 默认值

    private int players() {
        return sanma.isSelected() ? 3 : 4;
    }

    private void modeChanged() {
        rebuildSeats(defaultSeats(players()));
        updateModeNote();
        refreshWarn();
    }

    private void updateModeNote() {
        List<String> pool = models.modelsFor(players());
        String poolText = pool.isEmpty() ? "（这个人数暂时没有模型，会退回教师）"
                : "（可选模型：" + String.join(" / ", shortNames(pool)) + "）";
        modeNote.setText(sanma.isSelected()
                ? "3 人：没有吃、有拔北，两家在上方左右 " + poolText
                : "4 人：有吃、无拔北，四家围一圈 " + poolText);
    }

    private static List<String> shortNames(List<String> in) {
        List<String> out = new ArrayList<>();
        for (String s : in) {
            out.add(s);
        }
        return out;
    }

    /** 这个人数的默认座位表：P0=人类，其余按推荐模型填。 */
    public List<String> defaultSeats(int players) {
        return models.defaultSeats(players);
    }

    /** 模型对战：P0=人类，其余座位填模型。 */
    public void modelVersusHuman() {
        userTouched = true;
        applySeats(models.defaultSeats(players()));
        delay.setValue(Math.max(delay.getValue(), 10));
        refreshWarn();
    }

    /** 观战：全是模型。 */
    public void spectateAll() {
        userTouched = true;
        applySeats(models.spectateSeats(players()));
        delay.setValue(10);
        refreshWarn();
    }

    private void applySeats(List<String> seats) {
        if (seats.isEmpty()) {
            return;
        }
        for (int i = 0; i < combos.size(); i++) {
            select(combos.get(i), seats.get(i % seats.size()));
        }
    }

    // ------------------------------------------------------------ 数据

    /** 模型列表（MainFrame 拿到 /api/models 后回调；列表真变了才重建下拉）。 */
    public void setModels(MajClient.Models m) {
        if (m == null) {
            return;
        }
        String sig = String.join(",", m.models) + "|" + m.info.keySet() + "|" + m.suggest;
        if (sig.equals(modelsSig)) {
            return;                       // 内容没变就别动下拉（免得把用户正在选的框关掉）
        }
        this.models = m;
        this.modelsSig = sig;
        List<String> keep = currentSeats();
        // 只有「用户自己选过」才保留；否则（比如刚启动、后端还没起时列表是空的）
        // 一旦拿到模型列表就重填一遍默认值 —— 不然第一次开桌会全是教师。
        boolean keepOk = userTouched && keep.size() == players();
        rebuildSeats(keepOk ? keep : defaultSeats(players()));
        updateModeNote();
    }

    /** 后端状态显示。 */
    public void setBackendState(boolean connected, boolean owned, String url) {
        if (connected) {
            backendState.setForeground(Palette.GREEN);
            backendState.setText("后端：" + url + " ✅ 已连接"
                    + (owned ? "（本窗口启动，关窗口会一起停）" : "（外面起的，不归我管）"));
            startBackend.setText("后端在跑");
            startBackend.setEnabled(true);
            stopOnExit.setEnabled(owned);
            stopOnExit.setSelected(owned);
        } else {
            backendState.setForeground(Palette.RED);
            backendState.setText("后端：" + url + " ✗ 连不上（点「开始对局」会自动拉起）");
            startBackend.setText("启动后端");
            startBackend.setEnabled(true);
            stopOnExit.setEnabled(true);
        }
    }

    public boolean stopBackendOnExit() {
        return stopOnExit.isSelected();
    }

    private List<String> choices() {
        return models.seatChoices(players());
    }

    private static void select(JComboBox<Choice> box, String value) {
        for (int i = 0; i < box.getItemCount(); i++) {
            if (box.getItemAt(i).value().equals(value)) {
                box.setSelectedIndex(i);
                return;
            }
        }
        if (box.getItemCount() > 0) {
            box.setSelectedIndex(0);
        }
    }

    private void rebuildSeats(List<String> want) {
        building = true;
        seatRows.removeAll();
        combos.clear();
        int players = players();
        String[] four = {"P0 · 你（画面下方）", "P1 · 下家", "P2 · 对家", "P3 · 上家"};
        String[] three = {"P0 · 你（画面下方）", "P1 · 上方右", "P2 · 上方左"};
        String[] names = players == 3 ? three : four;
        List<String> options = choices();
        for (int i = 0; i < players; i++) {
            GridBagConstraints c = new GridBagConstraints();
            c.gridx = 0;
            c.gridy = i;
            c.insets = new Insets(5, 2, 5, 12);
            c.anchor = GridBagConstraints.LINE_END;
            JLabel lab = new JLabel(names[i] + "：");
            lab.setFont(Palette.font(14));
            lab.setForeground(Palette.INK);
            seatRows.add(lab, c);

            JComboBox<Choice> box = new JComboBox<>();
            box.setFont(Palette.font(14));
            for (String v : options) {
                box.addItem(new Choice(v, models.label(v), models.isModel(v)));
            }
            String value = i < want.size() ? want.get(i) : "teacher";
            if (!options.contains(value)) {
                // 这个值不在当前人数的候选里（老配置 / 模型被删了）：
                // 照样列出来并标出来，让用户看见，而不是偷偷换成别的
                box.addItem(new Choice(value, badLabel(value), models.isModel(value)));
            }
            select(box, value);
            box.setPreferredSize(new Dimension(300, 30));
            box.setToolTipText(models.isModel(value) ? tipFor(value) : null);
            box.addActionListener(e -> {
                if (!building) {
                    Choice ch = (Choice) box.getSelectedItem();
                    box.setToolTipText(ch != null && ch.model() ? tipFor(ch.value()) : null);
                    userTouched = true;
                    refreshWarn();
                }
            });
            c = new GridBagConstraints();
            c.gridx = 1;
            c.gridy = i;
            c.insets = new Insets(5, 2, 5, 12);
            c.anchor = GridBagConstraints.LINE_START;
            seatRows.add(box, c);
            combos.add(box);

            GridBagConstraints filler = new GridBagConstraints();
            filler.gridx = 2;
            filler.gridy = i;
            filler.weightx = 1;
            filler.fill = GridBagConstraints.HORIZONTAL;
            seatRows.add(Box.createGlue(), filler);
        }
        building = false;
        updateModeNote();
        refreshWarn();
        seatRows.revalidate();
        seatRows.repaint();
    }

    /** 把当前座位表的问题实时写在卡片底下（没问题就清空）。 */
    private void refreshWarn() {
        if (building) {
            return;
        }
        List<String> problems = checkSeats();
        if (problems.isEmpty()) {
            List<String> ms = modelSeats();
            warn.setForeground(ms.isEmpty() ? Palette.RED : Palette.DIM);
            warn.setText(ms.isEmpty()
                    ? "⚠ 这一桌一个模型都没有，全是内置 AI（教师 / 随机）"
                    : "当前有 " + ms.size() + " 个座位用模型：" + String.join("、", ms));
        } else {
            warn.setForeground(Palette.RED);
            warn.setText("⚠ " + problems.get(0)
                    + (problems.size() > 1 ? "（还有 " + (problems.size() - 1) + " 条）" : ""));
        }
    }

    /** 不匹配 / 未知的座位值在下拉里长这样。 */
    private String badLabel(String value) {
        if (!models.known(value)) {
            return value + " ⚠ 不认识（weights 里没有）";
        }
        int mp = models.playersOf(value);
        if (mp > 0 && mp != players()) {
            return value + " ⚠ " + mp + " 人模型（当前 " + players() + " 人）";
        }
        if (mp == 0 && !models.models.contains(value)) {
            return value + " ⚠ 通用模型";
        }
        return value;
    }

    private String tipFor(String model) {
        MajClient.Models.ModelInfo mi = models.info.get(model);
        if (mi == null) {
            return model;
        }
        StringBuilder sb = new StringBuilder(model);
        if (!mi.cn.isEmpty()) {
            sb.append(" · ").append(mi.cn);
        }
        if (!mi.style.isEmpty()) {
            sb.append(" · 风格 ").append(mi.style);
        }
        sb.append(mi.players == 0 ? " · 通用" : " · 只用于 " + mi.players + " 人");
        if (!mi.reward.isEmpty()) {
            sb.append(" · reward=").append(mi.reward);
        }
        return sb.toString();
    }

    /** 当前座位表。 */
    public List<String> currentSeats() {
        List<String> out = new ArrayList<>();
        for (JComboBox<Choice> box : combos) {
            Choice c = (Choice) box.getSelectedItem();
            out.add(c == null ? "teacher" : c.value());
        }
        lastSeats = String.join(",", out);
        return out;
    }

    public String lastSeats() {
        return lastSeats;
    }

    // ------------------------------------------------------------ 校验 / 开局

    /** 开局前检查：模型在不在、人数配不配。返回问题列表（空 = 没问题）。 */
    public List<String> checkSeats() {
        List<String> problems = new ArrayList<>();
        List<String> seats = currentSeats();
        int players = players();
        if (models.models.isEmpty()) {
            problems.add("后端没给出任何模型（是不是没连上？）—— 所有座位都会退化成教师");
        }
        for (int i = 0; i < seats.size(); i++) {
            String v = seats.get(i);
            if (!models.known(v)) {
                problems.add("P" + i + " 的「" + v + "」不是已知座位 / 模型（weights 里没有）");
                continue;
            }
            int mp = models.playersOf(v);
            if (mp > 0 && mp != players) {
                problems.add("P" + i + " 选了 " + mp + " 人模型「" + v + "」，当前是 " + players
                        + " 人");
            }
        }
        return problems;
    }

    /** 当前有哪些座位是模型（给状态栏 / 自检看）。 */
    public List<String> modelSeats() {
        List<String> out = new ArrayList<>();
        for (String v : currentSeats()) {
            if (models.isModel(v)) {
                out.add(v);
            }
        }
        return out;
    }

    private void tryStart() {
        List<String> problems = checkSeats();
        if (!problems.isEmpty()) {
            StringBuilder sb = new StringBuilder("开始前先确认一下：\n");
            for (String p : problems) {
                sb.append("  · ").append(p).append('\n');
            }
            sb.append("\n如果继续，后端会把认不出的座位退化成教师（你不会知道是不是模型在打）。");
            warn.setText("⚠ " + problems.get(0) + (problems.size() > 1
                    ? "（还有 " + (problems.size() - 1) + " 条）" : ""));
            Object[] options = {"返回修改", "仍然开始"};
            int r = JOptionPane.showOptionDialog(this, sb.toString(), "座位配置有问题",
                    JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE, null, options,
                    options[0]);
            if (r != 1) {
                return;
            }
        }
        MajClient.Config cfg = collect();
        if (cfg != null && sink != null) {
            sink.startGame(cfg);
        }
    }

    /** 收集成一份配置（座位表非法也照收，调用方一般先用 {@link #checkSeats()}）。 */
    public MajClient.Config collect() {
        List<String> seats = currentSeats();
        long sd;
        try {
            sd = Long.parseLong(seed.getText().trim());
        } catch (NumberFormatException e) {
            sd = System.currentTimeMillis() % 100000;
            seed.setText(String.valueOf(sd));
        }
        return new MajClient.Config(players(), seats, delay.getValue() / 10.0, sd);
    }

    /** 牌桌里改了配置回到这一页时同步界面（保留原样，不覆盖成默认值）。 */
    public void apply(MajClient.Config cfg) {
        if (cfg == null) {
            return;
        }
        userTouched = true;
        if (cfg.players >= 4) {
            yonma.setSelected(true);
        } else {
            sanma.setSelected(true);
        }
        delay.setValue((int) Math.round(cfg.delay * 10));
        seed.setText(String.valueOf(cfg.seed));
        if (!cfg.seats.isEmpty()) {
            rebuildSeats(cfg.seats);
        }
    }

    /** 自检 / 出图用：切到某个人数并重新填默认值。 */
    void debugSelectMode(int players) {
        if (players >= 4) {
            yonma.setSelected(true);
        } else {
            sanma.setSelected(true);
        }
        modeChanged();
    }

    public Font titleFont() {
        return Palette.bold(30);
    }
}
