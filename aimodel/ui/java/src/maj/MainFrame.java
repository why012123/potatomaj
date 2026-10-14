package maj;

import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.JSplitPane;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

/**
 * 主窗口：开局选择页 ↔ 牌桌两张卡，外加轮询、提交动作、菜单、退出清理。
 *
 * <p>线程模型：{@link Timer} 每 500ms 在 EDT 上「要不要拉一次」，真正的 HTTP
 * 全在 {@code net} 线程池；回来用 {@link SwingUtilities#invokeLater} 更新界面。
 * EDT 上永远不碰网络。
 *
 * <p>后端是 {@link Backend} 用 ProcessBuilder 起的，句柄在本窗口手里，
 * 关窗口会一起停掉（外面起的服务不动，状态栏会写清楚）。
 */
public final class MainFrame extends JFrame
        implements TablePanel.ActSink, ActionBar.Sink, StartPanel.Sink {

    private static final long serialVersionUID = 1L;
    private static final int POLL_MS = 500;
    private static final String CARD_START = "start";
    private static final String CARD_TABLE = "table";
    private static final String CARD_RESULT = "result";
    private static final String CARD_REPLAY = "replay";

    private final MajClient client;
    private final Backend backend;
    private final TileImages images;
    private final CardLayout cards = new CardLayout();
    private final JPanel root = new JPanel(cards);

    private final StartPanel startPanel;
    private final ResultCard resultCard;
    private final ReplayCard replayCard;
    private final TablePanel table;
    private final ActionBar actions;
    private final LogPanel logPanel;
    private final JSplitPane split;
    private final JLabel status = new JLabel(" ");
    private final JMenu viewMenu = new JMenu("视角");

    private final Timer timer;
    private final ExecutorService net;
    private final AtomicBoolean inFlight = new AtomicBoolean();
    private final boolean serverAllowed;

    private volatile State state;
    private volatile MajClient.Models models = new MajClient.Models();
    private int viewSeat;
    private int failCount;
    private long nextTryAt;
    private long actedAt;
    private String actedSignature = "";
    private String viewMenuSignature = "";
    private boolean resultShown;
    /** 已经给哪一手弹过结算页（seed/handNo）。 */
    private String shownRound = "";
    private boolean preloadedAll;
    private long lastLatency = -1;
    private long lastSeenSeed = Long.MIN_VALUE;
    /** 桌上没人（后端 spectator_ok）→ 可以带 reveal=1 全亮。 */
    private volatile boolean revealOk;
    private boolean shuttingDown;
    private boolean onStartPage = true;
    private int startTicks;
    private boolean backendKnown;
    private MajClient.Config lastConfig;
    private boolean stopBackendOnExit = true;

    public MainFrame(MajClient client, Backend backend, boolean serverAllowed, int seat,
            boolean showLog) {
        super("土豆麻将 · Java 牌桌");
        this.client = client;
        this.backend = backend;
        this.serverAllowed = serverAllowed;
        this.viewSeat = Math.max(0, seat);
        this.images = new TileImages(client, java.nio.file.Path.of(".tilecache"));
        this.net = Executors.newFixedThreadPool(2, r -> {
            Thread t = new Thread(r, "maj-net");
            t.setDaemon(true);
            return t;
        });

        table = new TablePanel(images, this);
        actions = new ActionBar(this);
        logPanel = new LogPanel();
        startPanel = new StartPanel(this);
        replayCard = new ReplayCard(client, images, net, new ReplayCard.Sink() {
            @Override
            public void exitReplay() {
                replayCard.close();
                showCard(CARD_TABLE);
            }

            @Override
            public void replayViewChanged(int seat) {
                status.setText(" 回放主视角：P" + seat);
            }
        });
        resultCard = new ResultCard(images, new ResultCard.Sink() {
            @Override
            public void nextHand() {
                MainFrame.this.nextHand();
                showCard(CARD_TABLE);
            }

            @Override
            public void replayThisHand() {
                openReplay();
            }

            @Override
            public void backToTable() {
                showCard(CARD_TABLE);
            }
        });
        images.setOnLoaded(() -> {
            table.repaint();
            actions.refresh(state);
        });

        split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, table, logPanel);
        split.setResizeWeight(1.0);
        split.setDividerSize(6);
        split.setBorder(null);
        split.setDividerLocation(0.76);

        JPanel tableCard = new JPanel(new BorderLayout());
        tableCard.setBackground(Palette.PANEL);
        tableCard.add(split, BorderLayout.CENTER);
        JPanel south = new JPanel(new BorderLayout());
        south.setBackground(Palette.PANEL);
        south.add(actions, BorderLayout.NORTH);
        south.add(statusBar(), BorderLayout.SOUTH);
        tableCard.add(south, BorderLayout.SOUTH);
        if (!showLog) {
            logPanel.setVisible(false);
            split.setDividerLocation(1.0);
        }

        root.setBackground(Palette.PANEL);
        root.add(startPanel, CARD_START);
        root.add(tableCard, CARD_TABLE);
        root.add(resultCard, CARD_RESULT);
        root.add(replayCard, CARD_REPLAY);
        setLayout(new BorderLayout());
        add(root, BorderLayout.CENTER);
        setJMenuBar(buildMenu());

        bindKeys();
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                shutdown();
            }
        });
        setMinimumSize(new Dimension(1080, 760));
        setSize(1400, 900);
        setLocationRelativeTo(null);

        // JVM 被外部杀掉时也把后端带走
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (backend != null && backend.owned()) {
                backend.stop();
            }
        }, "maj-backend-reaper"));

        timer = new Timer(POLL_MS, e -> tick());
        timer.setCoalesce(true);
        timer.start();
        showCard(CARD_START);
        fetchModels();
        tick();
    }

    /** 进回放模式。 */
    void openReplay() {
        replayCard.open();
        showCard(CARD_REPLAY);
        status.setForeground(Palette.GOLD);
        status.setText(" 回放模式：←/→ 翻帧 · 空格播放暂停 · Home/End 首尾 · Tab 换主视角 · Esc 退出");
    }

    /** 看这一局的回放（从结算页跳过来）。 */
    void replayThisHand() {
        State s = state;
        int handNo = s != null && s.handResult != null ? s.handResult.handNo : 1;
        replayCard.openAtHandEnd(handNo);
        showCard(CARD_REPLAY);
    }

    /** 换主视角：实时桌子按这个座位重新拉 state，回放里只换个方向画。 */
    void cycleViewSeat() {
        State s = state;
        int n = s == null ? 4 : s.players();
        if (replayCard.isShowing()) {
            replayCard.setViewSeat((replayCard.viewSeat() + 1) % Math.max(1, n));
        } else {
            switchSeat((viewSeat + 1) % Math.max(1, n));
        }
    }

    private JPanel statusBar() {
        JPanel bar = new JPanel(new BorderLayout(8, 0));
        bar.setBackground(Palette.PANEL);
        bar.setBorder(javax.swing.BorderFactory.createEmptyBorder(3, 10, 4, 10));
        status.setFont(Palette.font(12));
        status.setForeground(Palette.DIM);
        bar.add(status, BorderLayout.CENTER);
        JButton reconfig = new JButton("⚙ 重开一局…");
        reconfig.setFont(Palette.font(12));
        reconfig.setMargin(new java.awt.Insets(2, 10, 2, 10));
        reconfig.setToolTipText("回到开局选择页（Cmd/Ctrl+R）");
        reconfig.addActionListener(e -> showCard(CARD_START));
        bar.add(reconfig, BorderLayout.EAST);
        return bar;
    }

    // ------------------------------------------------------------ 页面切换

    private void showCard(String card) {
        onStartPage = CARD_START.equals(card);
        cards.show(root, card);
        if (onStartPage) {
            startTicks = 99;
            if (lastConfig != null) {
                startPanel.apply(lastConfig);
            }
            tick();
        } else {
            table.repaint();
        }
    }

    // ------------------------------------------------------------ 轮询

    private void tick() {
        if (shuttingDown || inFlight.get() || System.currentTimeMillis() < nextTryAt) {
            return;
        }
        inFlight.set(true);
        if (onStartPage) {
            net.submit(() -> {
                try {
                    boolean ok = client.ping();
                    MajClient.Models m = ok ? client.models() : null;
                    SwingUtilities.invokeLater(() -> {
                        backendKnown = ok;
                        startPanel.setBackendState(ok, backend != null && backend.owned(),
                                client.base());
                        if (m != null) {
                            models = m;
                            startPanel.setModels(m);
                        }
                        if (!ok) {
                            status.setForeground(Palette.RED);
                            status.setText(" 后端 " + client.base()
                                    + " 连不上 —— 点「开始对局」会自动拉起，或点「启动后端」");
                        } else {
                            status.setForeground(Palette.DIM);
                            status.setText(" 后端 " + client.base() + " 已连接"
                                    + (backend != null && backend.owned()
                                            ? " · 本窗口启动（关窗口会停）" : " · 外部服务"));
                        }
                    });
                } catch (Exception e) {
                    SwingUtilities.invokeLater(() -> startPanel.setBackendState(false, false,
                            client.base()));
                } finally {
                    inFlight.set(false);
                }
            });
            return;
        }
        final int seat = viewSeat;
        final long t0 = System.nanoTime();
        net.submit(() -> {
            try {
                State s = client.state(seat, revealOk);
                long ms = (System.nanoTime() - t0) / 1_000_000;
                SwingUtilities.invokeLater(() -> {
                    lastLatency = ms;
                    applyState(s);
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> applyFailure(e));
            } finally {
                inFlight.set(false);
            }
        });
    }

    private void applyState(State s) {
        if (shuttingDown) {
            return;
        }
        failCount = 0;
        nextTryAt = 0;
        table.setBanner(null);
        if (viewSeat >= s.players() && s.players() > 0) {
            viewSeat = 0;
        }
        if (s.seed != lastSeenSeed) {
            lastSeenSeed = s.seed;
            resultShown = false;
        }
        revealOk = s.spectatorOk;
        state = s;
        table.setState(s);
        actions.refresh(s);
        logPanel.update(s.log);
        updateViewMenu(s);
        preloadTiles(s);
        updateStatus(null);
        // 本局结算页：后端说 awaiting_next（或整场结束）并且给了 hand_result 才弹，不猜
        String round = s.seed + "/" + s.handNo;
        if (s.handResult != null && (s.awaitingNext || s.finished)
                && !round.equals(shownRound)) {
            shownRound = round;
            resultCard.show(s);
            showCard(CARD_RESULT);
        }
        if (s.finished && s.result != null && !resultShown) {
            resultShown = true;
            SwingUtilities.invokeLater(this::showResult);
        }
    }

    private void applyFailure(Exception e) {
        if (shuttingDown) {
            return;
        }
        failCount++;
        nextTryAt = System.currentTimeMillis() + Math.min(4000, 400L * failCount);
        String msg = "连接断开，重试中…（" + shortMessage(e) + "）";
        table.setBanner(msg);
        status.setForeground(Palette.RED);
        status.setText(" " + msg + "  已失败 " + failCount + " 次");
        actions.refresh(null);
    }

    private void updateStatus(Exception lastError) {
        State s = state;
        if (s == null) {
            status.setForeground(Palette.DIM);
            status.setText(" " + client.base() + " · 还没拿到局面");
            return;
        }
        State.Seat me = s.seat(viewSeat);
        StringBuilder sb = new StringBuilder();
        sb.append(' ').append(client.base())
                .append(" · ").append(s.players() >= 4 ? "四麻" : "三麻")
                .append(" · 视角 P").append(viewSeat)
                .append(me == null ? "" : "（" + me.name + "）")
                .append(" · 节拍 ").append(String.format("%.1f", s.delay)).append("s")
                .append(" · 种子 ").append(s.seed);
        if (lastLatency >= 0) {
            sb.append(" · ").append(lastLatency).append("ms");
        }
        sb.append(" · 牌图 ").append(images.readyCount()).append("/37");
        sb.append(" · 后端").append(backend != null && backend.owned() ? "本窗口启动" : "外部");
        if (lastError != null) {
            status.setForeground(Palette.RED);
            sb.append(" · ").append(shortMessage(lastError));
        } else {
            status.setForeground(Palette.DIM);
        }
        status.setText(sb.toString());
    }

    private static String shortMessage(Throwable e) {
        String m = e.getMessage();
        if (m == null || m.isBlank()) {
            m = e.getClass().getSimpleName();
        }
        m = m.replace('\n', ' ').trim();
        return m.length() > 90 ? m.substring(0, 90) + "…" : m;
    }

    private void preloadTiles(State s) {
        Set<String> ids = new LinkedHashSet<>();
        if (!preloadedAll) {
            ids.addAll(Arrays.asList(Tile.IMAGE_IDS));
            preloadedAll = true;
        }
        ids.addAll(s.dora);
        for (State.Player p : s.details) {
            ids.addAll(p.pond);
            if (p.hand != null) {
                ids.addAll(p.hand);
            }
            for (State.Meld m : p.melds) {
                ids.addAll(m.tiles());
            }
        }
        if (s.pending != null) {
            ids.addAll(s.pending.yourHand);
            if (s.pending.drawn != null) {
                ids.add(s.pending.drawn);
            }
            if (s.pending.tile != null) {
                ids.add(s.pending.tile);
            }
        }
        if (s.handResult != null) {          // 结算页要用到的牌
            ids.addAll(s.handResult.hand);
            if (!s.handResult.winTile.isEmpty()) {
                ids.add(s.handResult.winTile);
            }
            for (State.Meld m : s.handResult.melds) {
                ids.addAll(m.tiles());
            }
        }
        images.preload(ids);
    }

    private void fetchModels() {
        net.submit(() -> {
            try {
                MajClient.Models m = client.models();
                SwingUtilities.invokeLater(() -> {
                    models = m;
                    startPanel.setModels(m);
                    startPanel.setBackendState(true, backend != null && backend.owned(),
                            client.base());
                });
            } catch (Exception ignored) {
                SwingUtilities.invokeLater(() -> startPanel.setBackendState(false, false,
                        client.base()));
            }
        });
    }

    // ------------------------------------------------------------ 开局

    @Override
    public void startGame(MajClient.Config cfg) {
        if (cfg == null) {
            return;
        }
        status.setForeground(Palette.BLUE);
        status.setText(" 正在开局：" + cfg + " …");
        net.submit(() -> {
            // 1) 后端没起就自己起（并且拿着句柄，关窗口一起停）
            if (serverAllowed && (backend == null || !client.ping())) {
                if (backend != null) {
                    backend.start(String.join(",", cfg.seats), cfg.delay, cfg.seed, client);
                }
            }
            boolean up = client.ping();
            if (!up) {
                SwingUtilities.invokeLater(() -> {
                    status.setForeground(Palette.RED);
                    status.setText(" 后端起不来：" + (backend == null ? "没有可用的 python"
                            : backend.lastMessage()));
                    startPanel.setBackendState(false, false, client.base());
                });
                return;
            }
            // 后端是刚拉起来的：这时模型列表可能还没到手，座位表会全是教师。
            // 重新拉一次 /api/models，让开局面板把默认值填成模型，再收集配置。
            MajClient.Config use = cfg;
            try {
                MajClient.Models m = client.models();
                boolean noModel = true;
                for (String seat : cfg.seats) {
                    if (m.isModel(seat)) {
                        noModel = false;
                        break;
                    }
                }
                if (noModel && !m.models.isEmpty()) {
                    final MajClient.Models fresh = m;
                    final MajClient.Config[] recollected = new MajClient.Config[1];
                    SwingUtilities.invokeAndWait(() -> {
                        startPanel.setModels(fresh);
                        recollected[0] = startPanel.collect();
                    });
                    if (recollected[0] != null && recollected[0].seats.stream()
                            .anyMatch(fresh::isModel)) {
                        use = recollected[0];
                        SwingUtilities.invokeLater(() -> status.setForeground(Palette.BLUE));
                    }
                }
            } catch (Exception ignored) {
                // 拿不到模型就按原来的座位表开，后面校验会提示
            }
            final MajClient.Config chosen = use;
            try {
                Map<String, Object> r = client.setConfig(chosen);
                boolean ok = Json.b(r.get("ok"), false);
                SwingUtilities.invokeLater(() -> {
                    if (!ok) {
                        status.setForeground(Palette.RED);
                        status.setText(" 开局失败：" + Json.str(r.get("error"), "未知原因"));
                        return;
                    }
                    lastConfig = chosen;
                    resultShown = false;
                    shownRound = "";
                    actedSignature = "";
                    lastSeenSeed = Long.MIN_VALUE;
                    state = null;
                    int human = chosen.firstHumanSeat();
                    viewSeat = human >= 0 ? human
                            : (viewSeat >= chosen.players ? 0 : viewSeat);
                    viewMenuSignature = "";
                    showCard(CARD_TABLE);
                    status.setForeground(Palette.GREEN);
                    status.setText(" 开局：" + chosen + " · 后端"
                            + (backend != null && backend.owned() ? "本窗口启动" : "外部"));
                    tick();
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    status.setForeground(Palette.RED);
                    status.setText(" 开局失败：" + shortMessage(e));
                });
            }
        });
    }

    @Override
    public void nextHand() {
        net.submit(() -> {
            try {
                client.nextHand();
                SwingUtilities.invokeLater(() -> {
                    actedSignature = "";
                    shownRound = "";
                    showCard(CARD_TABLE);
                    status.setForeground(Palette.GREEN);
                    status.setText(" 下一局开始…");
                    tick();
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    status.setForeground(Palette.RED);
                    status.setText(" 继续下一局失败：" + shortMessage(e));
                });
            }
        });
    }

    @Override
    public void startBackend() {
        MajClient.Config cfg = startPanel.collect();
        if (client.ping()) {
            status.setForeground(Palette.GREEN);
            status.setText(" 后端已经在跑（" + (backend != null && backend.owned()
                    ? "本窗口启动" : "外部服务") + "）：" + client.base());
            startPanel.setBackendState(true, backend != null && backend.owned(), client.base());
            return;
        }
        status.setForeground(Palette.BLUE);
        status.setText(" 正在启动后端：" + backendLogHint() + " …");
        net.submit(() -> {
            boolean ok = backend != null && backend.start(String.join(",", cfg.seats),
                    cfg.delay, cfg.seed, client);
            SwingUtilities.invokeLater(() -> {
                startPanel.setBackendState(ok, backend != null && backend.owned(), client.base());
                status.setForeground(ok ? Palette.GREEN : Palette.RED);
                status.setText(ok ? " 后端就绪：" + client.base()
                        : " 后端没起来：" + (backend == null ? "无" : backend.lastMessage()));
                if (ok) {
                    fetchModels();
                }
            });
        });
    }

    private String backendLogHint() {
        return backend == null ? "（没有后端配置）" : backend.logFile().toString();
    }

    private void newGame() {
        net.submit(() -> {
            try {
                client.newGame();
                SwingUtilities.invokeLater(() -> {
                    resultShown = false;
                    shownRound = "";
                    actedSignature = "";
                    state = null;
                    status.setForeground(Palette.GREEN);
                    status.setText(" 已开新的一局…");
                    tick();
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    status.setForeground(Palette.RED);
                    status.setText(" 换一批失败：" + shortMessage(e));
                });
            }
        });
    }

    private void showResult() {
        State s = state;
        if (s == null || s.result == null) {
            return;
        }
        ResultDialog d = new ResultDialog(this, s);
        d.setVisible(true);
        if (d.againClicked()) {
            newGame();
        }
    }

    // ------------------------------------------------------------ 动作

    @Override
    public void submitOption(int optionIndex, String why) {
        State s = state;
        if (s == null || s.pending == null || !s.pending.yours) {
            return;
        }
        State.Option choice = null;
        for (State.Option o : s.pending.options) {
            if (o.index == optionIndex) {
                choice = o;
            }
        }
        if (choice == null) {
            status.setForeground(Palette.RED);
            status.setText(" 后端没给下标 " + optionIndex + " 这个选项，忽略");
            return;
        }
        String sig = s.pending.signature();
        if (sig.equals(actedSignature) && System.currentTimeMillis() - actedAt < 5000) {
            return;
        }
        actedSignature = sig;
        actedAt = System.currentTimeMillis();

        final int seat = s.pending.seat;
        final State.Option picked = choice;
        actions.showSubmitted(why);
        status.setForeground(Palette.BLUE);
        status.setText(" 已提交 " + picked.label + "（" + picked.cn + "）给 P" + seat);
        net.submit(() -> {
            try {
                MajClient.ActResult r = client.act(seat, picked.index);
                SwingUtilities.invokeLater(() -> {
                    if (!r.ok()) {
                        status.setForeground(Palette.RED);
                        status.setText(" 后端拒绝了这次动作：" + r.describe());
                        actedSignature = "";
                    }
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    status.setForeground(Palette.RED);
                    status.setText(" 提交失败：" + shortMessage(e));
                    actedSignature = "";
                });
            }
        });
    }

    private void submitByKey(int n) {
        State s = state;
        if (s == null || s.pending == null || !s.pending.yours) {
            return;
        }
        if (n < 1 || n > s.pending.options.size()) {
            return;
        }
        submitOption(s.pending.options.get(n - 1).index, "键盘 " + n);
    }

    private void bindKeys() {
        JComponent pane = getRootPane();
        for (int i = 1; i <= 9; i++) {
            final int n = i;
            bind(pane, KeyStroke.getKeyStroke(KeyEvent.VK_0 + i, 0), "opt" + i, () -> {
                if (replayCard.isShowing()) {
                    if (n <= 4) {
                        replayCard.setViewSeat(n - 1);   // 回放里数字直接换主视角
                    }
                } else {
                    submitByKey(n);
                }
            });
            bind(pane, KeyStroke.getKeyStroke(KeyEvent.VK_0 + i, shortcutMask()),
                    "seat" + i, () -> switchSeat(n - 1));
        }
        bind(pane, KeyStroke.getKeyStroke(KeyEvent.VK_TAB, 0), "nextSeat", this::cycleViewSeat);
        bind(pane, KeyStroke.getKeyStroke(KeyEvent.VK_LEFT, 0), "rprev",
                () -> replayCard.step(-1));
        bind(pane, KeyStroke.getKeyStroke(KeyEvent.VK_RIGHT, 0), "rnext",
                () -> replayCard.step(1));
        bind(pane, KeyStroke.getKeyStroke(KeyEvent.VK_SPACE, 0), "rplay",
                replayCard::togglePlay);
        bind(pane, KeyStroke.getKeyStroke(KeyEvent.VK_HOME, 0), "rfirst",
                () -> replayCard.seek(0));
        bind(pane, KeyStroke.getKeyStroke(KeyEvent.VK_END, 0), "rlast",
                () -> replayCard.seek(replayCard.frames() - 1));
        bind(pane, KeyStroke.getKeyStroke(KeyEvent.VK_ESCAPE, 0), "rexit", () -> {
            if (replayCard.isShowing()) {
                replayCard.close();
                showCard(CARD_TABLE);
            }
        });
    }

    private void bind(JComponent pane, KeyStroke stroke, String key, Runnable action) {
        pane.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(stroke, key);
        pane.getActionMap().put(key, new javax.swing.AbstractAction() {
            private static final long serialVersionUID = 1L;

            @Override
            public void actionPerformed(java.awt.event.ActionEvent e) {
                action.run();
            }
        });
    }

    // ------------------------------------------------------------ 菜单

    private JMenuBar buildMenu() {
        JMenuBar bar = new JMenuBar();
        JMenu game = new JMenu("对局");
        JMenuItem back = new JMenuItem("回到开局选择页");
        back.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_R, shortcutMask()));
        back.addActionListener(e -> showCard(CARD_START));
        JMenuItem again = new JMenuItem("换一批（同配置）");
        again.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_N, shortcutMask()));
        again.addActionListener(e -> newGame());
        JMenuItem replay = new JMenuItem("回放这一场（一手一手看）");
        replay.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_L, shortcutMask()));
        replay.addActionListener(e -> openReplay());
        JMenuItem showResultCard = new JMenuItem("本局结算");
        showResultCard.addActionListener(e -> {
            if (state != null) {
                resultCard.show(state);
            }
            showCard(CARD_RESULT);
        });
        JMenuItem stopBackend = new JMenuItem("停止后端服务");
        stopBackend.addActionListener(e -> stopBackendDialog());
        JMenuItem quit = new JMenuItem("退出");
        quit.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_Q, shortcutMask()));
        quit.addActionListener(e -> shutdown());
        game.add(back);
        game.add(replay);
        game.add(showResultCard);
        game.add(again);
        game.addSeparator();
        game.add(stopBackend);
        game.addSeparator();
        game.add(quit);

        JMenuItem about = new JMenuItem("快捷键 / 后端状态");
        about.addActionListener(e -> JOptionPane.showMessageDialog(this, """
                1~9          选当前决策的第 N 项（打牌就是牌上角标那个数字）
                点手牌        直接打出
                Cmd/Ctrl+R   回到开局选择页
                Cmd/Ctrl+N   同配置换一批
                视角菜单      切到任一座位看牌
                对局菜单      可以随时「停止后端服务」

                后端：%s
                """.formatted(backend == null ? "无" : backend.describe()
                        + "\n日志：" + backend.logFile()), "说明", JOptionPane.INFORMATION_MESSAGE));
        JMenu help = new JMenu("帮助");
        help.add(about);

        JCheckBoxMenuItem showLog = new JCheckBoxMenuItem("显示牌谱", true);
        showLog.addActionListener(e -> {
            boolean v = showLog.isSelected();
            logPanel.setVisible(v);
            split.setDividerLocation(v ? 0.76 : 1.0);
            split.revalidate();
            split.repaint();
        });
        JCheckBoxMenuItem animItem = new JCheckBoxMenuItem("动画：开（发牌 / 出牌）", true);
        animItem.addActionListener(e -> {
            table.setAnimations(animItem.isSelected());
            animItem.setText(animItem.isSelected() ? "动画：开（发牌 / 出牌）" : "动画：关");
        });
        JCheckBoxMenuItem stopOnExit = new JCheckBoxMenuItem("关窗口时停掉后端", true);
        stopOnExit.addActionListener(e -> stopBackendOnExit = stopOnExit.isSelected());
        JMenu view = new JMenu("视图");
        view.add(showLog);
        view.add(animItem);
        view.add(stopOnExit);

        bar.add(game);
        bar.add(viewMenu);
        bar.add(view);
        bar.add(help);
        return bar;
    }

    private void stopBackendDialog() {
        if (backend == null) {
            return;
        }
        if (!backend.owned()) {
            JOptionPane.showMessageDialog(this,
                    "这个后端不是本窗口启动的（" + backend.describe() + "），我不去动它。\n"
                            + "要停的话请在你自己起它的终端里 Ctrl-C。",
                    "停止后端服务", JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        int r = JOptionPane.showConfirmDialog(this,
                "确定停掉本窗口启动的后端吗？（" + client.base() + "）\n牌桌会变成「连接断开」。",
                "停止后端服务", JOptionPane.OK_CANCEL_OPTION);
        if (r != JOptionPane.OK_OPTION) {
            return;
        }
        net.submit(() -> {
            boolean stopped = backend.stop();
            boolean free = backend.waitPortFree(client, java.time.Duration.ofSeconds(5));
            SwingUtilities.invokeLater(() -> {
                status.setForeground(stopped ? Palette.GREEN : Palette.DIM);
                status.setText(stopped
                        ? " 已停掉后端，端口" + (free ? "已释放" : "还没释放")
                        : " 没有本窗口启动的后端需要停");
                startPanel.setBackendState(client.ping(), backend.owned(), client.base());
                table.setBanner("后端已停止 —— 想继续就回选择页重新开始");
            });
        });
    }

    private static int shortcutMask() {
        return java.awt.Toolkit.getDefaultToolkit().getMenuShortcutKeyMaskEx();
    }

    private void updateViewMenu(State s) {
        StringBuilder sig = new StringBuilder();
        for (State.Seat st : s.seats) {
            sig.append(st.name).append('|');
        }
        if (sig.toString().equals(viewMenuSignature)) {
            return;
        }
        viewMenuSignature = sig.toString();
        viewMenu.removeAll();
        ButtonGroup bg = new ButtonGroup();
        for (int i = 0; i < s.players(); i++) {
            State.Seat st = s.seat(i);
            final int seat = i;
            JRadioButtonMenuItem item = new JRadioButtonMenuItem(
                    "P" + i + (st == null ? "" : " · " + st.name), i == viewSeat);
            item.addActionListener(e -> switchSeat(seat));
            bg.add(item);
            viewMenu.add(item);
        }
    }

    private void switchSeat(int seat) {
        if (seat == viewSeat) {
            return;
        }
        viewSeat = seat;
        state = null;
        table.setState(null);
        nextTryAt = 0;
        viewMenuSignature = "";
        tick();
    }

    // ------------------------------------------------------------ 退出

    /** 自检 / 渲染工具的调试入口。 */
    String debugStatusText() {
        return status.getText();
    }

    State debugState() {
        return state;
    }

    void debugShowTable() {
        showCard(CARD_TABLE);
    }

    void debugShowStart() {
        showCard(CARD_START);
    }

    /** 自检用：拿到开局选择页，好模拟「选完点开始」。 */
    StartPanel debugStartPanel() {
        return startPanel;
    }

    boolean debugOnStartPage() {
        return onStartPage;
    }

    void debugQuit() {
        shutdown();
    }

    private void shutdown() {
        debugShutdown(true);
    }

    /** 关窗口那套清理；exit=false 只是不真的退出（自检要接着验证端口释放）。 */
    void debugShutdown(boolean exit) {
        if (shuttingDown) {
            return;
        }
        shuttingDown = true;
        timer.stop();
        images.shutdown();
        net.shutdownNow();
        if (backend != null && backend.owned() && stopBackendOnExit) {
            backend.stop();
        }
        if (exit) {
            dispose();
            System.exit(0);
        }
    }

    /** 给启动参数用：外面起的后端就不管了。 */
    public boolean serverAllowed() {
        return serverAllowed;
    }

    /** 未使用的占位，保持接口整洁。 */
    static List<String> emptySeats() {
        return new ArrayList<>();
    }
}
