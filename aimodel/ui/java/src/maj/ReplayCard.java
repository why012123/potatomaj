package maj;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

/**
 * 回放模式：一手一手地看整局牌谱，**四家手牌全亮**。
 *
 * <p>数据源是 {@code /api/replay}：先拿元信息（总帧数 / 人数 / 座位名），
 * 再**分批预取**（一次 {@value #BATCH} 帧，快翻到底就接着预取下一批），
 * 所以拖动滑块不会一帧一个请求。
 *
 * <p>控件：⏮ 首帧 / ◀ 上一手 / ▶ 播放暂停 / ⏭ 下一手 / 末帧 + 滑块 + 速度。
 * 键盘：←/→ 翻帧、空格播放暂停、Home/End 首尾、Esc 退出、Tab 与 1~4 换主视角。
 */
public final class ReplayCard extends JPanel {

    /** 回放卡要跟外面说的话。 */
    public interface Sink {
        /** 退出回放，回到实时牌桌。 */
        void exitReplay();

        /** 主视角变了（回放里只影响怎么画，不重新拉数据）。 */
        void replayViewChanged(int seat);
    }

    /** 一次预取多少帧。 */
    static final int BATCH = 200;
    /** 缓存最多留多少帧（超出就把离当前帧最远的丢掉）。 */
    private static final int CACHE_MAX = 900;

    private final MajClient client;
    private final Sink sink;
    private final TablePanel table;
    private final ExecutorService net;
    private final JLabel info = new JLabel("回放：正在读取…");
    private final JLabel hint = new JLabel(" ");
    private final JSlider slider = new JSlider(0, 0, 0);
    private final JButton playBtn = new JButton("▶ 播放");
    private final JButton speedBtn = new JButton("0.5s");
    private final Timer player;

    private final Map<Integer, Map<String, Object>> cache =
            new LinkedHashMap<>(16, 0.75f, true) {
                private static final long serialVersionUID = 1L;

                @Override
                protected boolean removeEldestEntry(Map.Entry<Integer, Map<String, Object>> e) {
                    return size() > CACHE_MAX;
                }
            };

    private MajClient.ReplayMeta meta;
    private int index;
    private int viewSeat;
    private boolean pendingFetch;
    private int fetchFrom = -1;
    private int fetchTo = -1;
    private int playMs = 500;
    private boolean loading;

    public ReplayCard(MajClient client, TileImages images, ExecutorService net, Sink sink) {
        super(new BorderLayout());
        this.client = client;
        this.net = net;
        this.sink = sink;
        setBackground(Palette.PANEL);

        table = new TablePanel(images, null);
        add(table, BorderLayout.CENTER);
        add(controls(), BorderLayout.SOUTH);

        addComponentListener(new java.awt.event.ComponentAdapter() {
            @Override
            public void componentResized(java.awt.event.ComponentEvent e) {
                relayoutControls();
            }
        });
        player = new Timer(playMs, e -> {
            if (meta == null || index >= meta.frames() - 1) {
                setPlaying(false);
                return;
            }
            seek(index + 1);
        });
        player.setCoalesce(true);
    }

    private JPanel controls() {
        bar = new JPanel(new BorderLayout(10, 0));
        bar.setBackground(Palette.PANEL);
        bar.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(1, 0, 0, 0, Palette.LINE),
                BorderFactory.createEmptyBorder(8, 12, 8, 12)));
        bar.setPreferredSize(new Dimension(200, 96));

        JPanel left = new JPanel(new BorderLayout(0, 2));
        left.setOpaque(false);
        info.setFont(Palette.bold(13));
        info.setForeground(Palette.GOLD);
        hint.setFont(Palette.font(12));
        hint.setForeground(Palette.DIM);
        left.add(info, BorderLayout.NORTH);
        left.add(hint, BorderLayout.SOUTH);
        left.setPreferredSize(new Dimension(360, 56));

        infoHolder = left;
        JButton first = iconButton("⏮ 首帧", "回放开头（Home）");
        first.addActionListener(e -> seek(0));
        JButton prev = iconButton("◀ 上一手", "上一手（←）");
        prev.addActionListener(e -> seek(index - 1));
        playBtn.setFont(Palette.bold(15));
        playBtn.setPreferredSize(new Dimension(104, 40));
        playBtn.setToolTipText("播放 / 暂停（空格）");
        playBtn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        playBtn.addActionListener(e -> togglePlay());
        JButton next = iconButton("下一手 ▶", "下一手（→）");
        next.addActionListener(e -> seek(index + 1));
        JButton last = iconButton("末帧 ⏭", "回放结尾（End）");
        last.addActionListener(e -> seek(meta == null ? 0 : meta.frames() - 1));
        speedBtn.setFont(Palette.font(13));
        speedBtn.setPreferredSize(new Dimension(72, 40));
        speedBtn.setToolTipText("自动播放的间隔，点一下换一档");
        speedBtn.addActionListener(e -> cycleSpeed());
        JButton back = iconButton("Esc 退出回放", "回到实时牌桌");
        back.addActionListener(e -> {
            if (sink != null) {
                sink.exitReplay();
            }
        });

        JPanel btns = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        btns.setOpaque(false);
        btnsFlow = btns;
        btns.add(first);
        btns.add(prev);
        btns.add(playBtn);
        btns.add(next);
        btns.add(last);
        btns.add(speedBtn);
        btns.add(back);
        fullButtons.add(first);
        fullButtons.add(prev);
        fullButtons.add(playBtn);
        fullButtons.add(next);
        fullButtons.add(last);
        fullButtons.add(speedBtn);
        fullButtons.add(back);
        fullLabels.add("⏮ 首帧");
        fullLabels.add("◀ 上一手");
        fullLabels.add("▶ 播放");
        fullLabels.add("下一手 ▶");
        fullLabels.add("末帧 ⏭");
        fullLabels.add("0.5s");
        fullLabels.add("Esc 退出回放");
        shortLabels.add("⏮");
        shortLabels.add("◀");
        shortLabels.add("▶");
        shortLabels.add("▶");
        shortLabels.add("⏭");
        shortLabels.add("0.5s");
        shortLabels.add("✕");

        slider.setOpaque(false);
        slider.setPreferredSize(new Dimension(200, 30));
        slider.setToolTipText("拖动跳到任意一手（0 ~ 最后一手）");
        slider.addChangeListener(e -> {
            if (!slider.getValueIsAdjusting()) {
                seek(slider.getValue());
            } else {
                showPreview(slider.getValue());
            }
        });
        JPanel mid = new JPanel(new BorderLayout(0, 4));
        mid.setOpaque(false);
        mid.add(btns, BorderLayout.NORTH);
        mid.add(slider, BorderLayout.CENTER);

        viewLabel.setFont(Palette.font(12));
        viewLabel.setForeground(Palette.DIM);
        viewLabel.setBorder(BorderFactory.createEmptyBorder(0, 10, 0, 0));
        btns.add(viewLabel);

        bar.add(left, BorderLayout.WEST);
        bar.add(mid, BorderLayout.CENTER);
        return bar;
    }

    private final java.util.List<JButton> seatButtons = new java.util.ArrayList<>();
    private final JLabel viewLabel = new JLabel("主视角");
    private JPanel btnsFlow;
    private JPanel bar;
    private JPanel seatHolder;
    private JPanel infoHolder;
    private boolean compactMode;
    private final java.util.List<JButton> fullButtons = new java.util.ArrayList<>();
    private final java.util.List<String> fullLabels = new java.util.ArrayList<>();
    private final java.util.List<String> shortLabels = new java.util.ArrayList<>();

    /** 窄了就把按钮文字缩成图标（工具提示里还有全称），并让整条按行数长高。 */
    private void relayoutControls() {
        int w = getWidth();
        boolean compact = w > 0 && w < 1180;
        if (compact != compactMode) {
            compactMode = compact;
            if (fullButtons.size() == shortLabels.size()) {
                for (int i = 0; i < fullButtons.size(); i++) {
                    JButton b = fullButtons.get(i);
                    String text = compact ? shortLabels.get(i) : fullLabels.get(i);
                    b.setText(text);
                    b.setPreferredSize(new Dimension(
                            Math.max(compact ? 46 : 96, text.length() * 15 + 22), 40));
                }
                info.setPreferredSize(new Dimension(compact ? 232 : 360, 56));
            }
            viewLabel.setText(compact ? "视角" : "主视角");
            for (JButton b : seatButtons) {
                b.setPreferredSize(new Dimension(compact ? 40 : 46, 40));
            }
            btnsFlow.revalidate();
            revalidate();
        }
        int avail = Math.max(240, w - 24);
        int rows = Math.max(1, (int) Math.ceil(buttonsWidth() / (double) avail));
        int h = Math.max(96, rows * 46 + 44);
        if (bar.getPreferredSize().height != h) {
            bar.setPreferredSize(new Dimension(200, h));
            revalidate();
        }
    }

    /** 布局时按当前宽度调一次（离屏出图 / 拖动窗口都走这里，幂等）。 */
    @Override
    public void doLayout() {
        relayoutControls();
        super.doLayout();
    }

    private int buttonsWidth() {
        int total = 24;
        for (JButton b : fullButtons) {
            total += b.getPreferredSize().width + 6;
        }
        for (JButton b : seatButtons) {
            total += b.getPreferredSize().width + 6;
        }
        return total + viewLabel.getPreferredSize().width + 10;
    }

    private JButton iconButton(String text, String tip) {
        JButton b = new JButton(text);
        b.setFont(Palette.font(14));
        b.setToolTipText(tip);
        b.setPreferredSize(new Dimension(text.length() * 15 + 22, 40));
        b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        return b;
    }

    // ------------------------------------------------------------ 打开 / 关闭

    /** 进回放：拉元信息 + 第一批帧。 */
    public void open() {
        info.setText("回放：正在读取牌谱…");
        hint.setText(" ");
        cache.clear();
        meta = null;
        index = 0;
        slider.setValue(0);
        slider.setMaximum(0);
        loading = true;
        net.submit(() -> {
            try {
                MajClient.ReplayMeta m = client.replayMeta();
                SwingUtilities.invokeLater(() -> {
                    meta = m;
                    slider.setMaximum(Math.max(0, m.frames() - 1));
                    buildSeatButtons(m.players());
                    loading = false;
                    if (viewSeat >= m.players()) {
                        viewSeat = 0;
                    }
                    seek(0);
                    ensureAround(0);
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    loading = false;
                    info.setForeground(Palette.RED);
                    info.setText("读不到回放：" + e.getMessage());
                });
            }
        });
    }

    /** 退出回放：停自动播放。 */
    public void close() {
        setPlaying(false);
    }

    private void buildSeatButtons(int players) {
        if (btnsFlow == null) {
            return;
        }
        for (JButton old : seatButtons) {
            btnsFlow.remove(old);
        }
        seatButtons.clear();
        int at = btnsFlow.getComponentCount();   // 放在最后
        for (int i = 0; i < players; i++) {
            final int seat = i;
            JButton b = new JButton("P" + i);
            b.setFont(Palette.font(12));
            b.setMargin(new java.awt.Insets(2, 10, 2, 10));
            b.setToolTipText("把 P" + i + " 摆到画面下方（回放里按 1~" + players + " 或 Tab）");
            b.addActionListener(e -> setViewSeat(seat));
            b.setPreferredSize(new Dimension(46, 40));
            btnsFlow.add(b, Math.min(at + i, btnsFlow.getComponentCount()));
            seatButtons.add(b);
        }
        btnsFlow.revalidate();
        btnsFlow.repaint();
    }

    // ------------------------------------------------------------ 帧

    /** 主视角（回放里只影响怎么画）。 */
    public void setViewSeat(int seat) {
        if (meta != null && (seat < 0 || seat >= meta.players())) {
            return;
        }
        viewSeat = Math.max(0, seat);
        render();
        if (sink != null) {
            sink.replayViewChanged(viewSeat);
        }
    }

    public int viewSeat() {
        return viewSeat;
    }

    /** 跳到第 i 帧（超范围会夹住）。 */
    public void seek(int i) {
        if (meta == null) {
            return;
        }
        int n = Math.max(0, meta.frames() - 1);
        int want = Math.max(0, Math.min(n, i));
        if (want == index && cache.containsKey(want) && table.state() != null) {
            return;
        }
        index = want;
        if (slider.getValue() != want) {
            slider.setValue(want);
        }
        ensureAround(want);
        render();
    }

    public int index() {
        return index;
    }

    public int frames() {
        return meta == null ? 0 : meta.frames();
    }

    public boolean playing() {
        return player.isRunning();
    }

    public void step(int delta) {
        seek(index + delta);
    }

    public void togglePlay() {
        setPlaying(!player.isRunning());
    }

    private void setPlaying(boolean playing) {
        if (playing) {
            if (meta == null || meta.frames() <= 1) {
                return;
            }
            if (index >= meta.frames() - 1) {
                index = 0;
            }
            player.setDelay(playMs);
            player.start();
            playBtn.setText(compactMode ? "⏸" : "⏸ 暂停");
        } else {
            player.stop();
            playBtn.setText(compactMode ? "▶" : "▶ 播放");
        }
        updateInfo();
    }

    private void cycleSpeed() {
        int[] options = {200, 500, 1000, 2000};
        int cur = 1;
        for (int i = 0; i < options.length; i++) {
            if (options[i] == playMs) {
                cur = i;
            }
        }
        playMs = options[(cur + 1) % options.length];
        speedBtn.setText(String.format("%.1fs", playMs / 1000.0));
        if (player.isRunning()) {
            player.setDelay(playMs);
        }
        updateInfo();
    }

    private void showPreview(int i) {
        info.setText("第 " + (i + 1) + " / " + (meta == null ? 0 : meta.frames())
                + " 手（松手就跳过去）");
    }

    private void updateInfo() {
        if (meta == null) {
            return;
        }
        Map<String, Object> f = cache.get(index);
        String txt = f == null ? "" : Json.str(f.get("text"), "");
        info.setText("第 " + (index + 1) + " / " + meta.frames() + " 手"
                + (player.isRunning() ? " · 播放中 " + String.format("%.1fs", playMs / 1000.0) : "")
                + (txt.isEmpty() ? "" : "　" + txt));
    }

    // ------------------------------------------------------------ 预取 + 渲染

    /** 缓存里没有这一帧就去拉一批（后台线程，一次 BATCH 帧）。 */
    private void ensureAround(int i) {
        if (meta == null) {
            return;
        }
        int total = meta.frames();
        if (total <= 0) {
            return;
        }
        int lo = Math.max(0, i - 40);
        int hi = Math.min(total - 1, lo + BATCH - 1);
        lo = Math.max(0, hi - BATCH + 1);
        boolean have = true;
        for (int k = i - 10; k <= i + 60; k++) {
            if (k >= 0 && k < total && !cache.containsKey(k)) {
                have = false;
                break;
            }
        }
        if (have || pendingFetch) {
            return;
        }
        pendingFetch = true;
        fetchFrom = lo;
        fetchTo = hi;
        final int f0 = lo;
        final int f1 = hi;
        net.submit(() -> {
            List<Map<String, Object>> got = new ArrayList<>();
            String err = null;
            try {
                got = client.replayFrames(f0, f1);
            } catch (Exception e) {
                err = e.getMessage();
            }
            final List<Map<String, Object>> frames = got;
            final String error = err;
            SwingUtilities.invokeLater(() -> {
                pendingFetch = false;
                if (error != null) {
                    info.setForeground(Palette.RED);
                    info.setText("预取 " + f0 + "~" + f1 + " 失败：" + error);
                    return;
                }
                for (Map<String, Object> fr : frames) {
                    cache.put(Json.i(fr.get("i"), 0), fr);
                }
                hint.setText("已缓存 " + cache.size() + " 手（一次预取 " + BATCH + " 手）");
                render();
            });
        });
    }

    /** 拿当前帧画牌桌。 */
    private void render() {
        if (meta == null) {
            return;
        }
        Map<String, Object> f = cache.get(index);
        if (f == null) {
            table.setState(null);
            table.setBanner("正在取第 " + (index + 1) + " 手…");
            info.setText("第 " + (index + 1) + " / " + meta.frames() + " 手 · 取帧中…");
            return;
        }
        State st = State.fromReplayFrame(f, meta.seats(), viewSeat);
        st.replayFrames = meta.frames();
        st.spectatorOk = meta.spectatorOk();
        table.setState(st);
        updateInfo();
    }

    /** 打开回放并定位到第 handNo 局的最后一帧（hand_no 单调，二分找）。 */
    public void openAtHandEnd(int handNo) {
        open();
        net.submit(() -> {
            try {
                MajClient.ReplayMeta m = client.replayMeta();
                int lo = 0;
                int hi = Math.max(0, m.frames() - 1);
                int ans = 0;
                while (lo <= hi) {
                    int mid = (lo + hi) / 2;
                    Map<String, Object> f = client.replayFrame(mid);
                    cache.put(mid, f);
                    int h = Json.i(f.get("hand_no"), 1);
                    if (h <= handNo) {
                        ans = mid;
                        lo = mid + 1;
                    } else {
                        hi = mid - 1;
                    }
                }
                final int at = ans;
                SwingUtilities.invokeLater(() -> {
                    seek(at);
                    hint.setText("已定位到第 " + handNo + " 局最后一手（第 " + (at + 1) + " 帧）");
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> hint.setText("定位失败：" + e.getMessage()));
            }
        });
    }

    /** 给外面（自检 / 出图）用：当前这一帧。 */
    public Map<String, Object> currentFrame() {
        return cache.get(index);
    }

    /** 给外面（自检 / 出图）用：直接塞一批帧进来，省得联网。 */
    public void debugLoad(MajClient.ReplayMeta m, List<Map<String, Object>> frames, int at) {
        this.meta = m;
        for (Map<String, Object> f : frames) {
            cache.put(Json.i(f.get("i"), 0), f);
        }
        slider.setMaximum(Math.max(0, m.frames() - 1));
        buildSeatButtons(m.players());
        loading = false;
        seek(at);
    }

    public TablePanel table() {
        return table;
    }

    /** 回放卡上的牌桌（出图 / 自检用）。 */
    public boolean isReady() {
        return meta != null && !loading;
    }
}
