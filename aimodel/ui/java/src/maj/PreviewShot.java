package maj;

import java.awt.BorderLayout;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

/**
 * 渲染 / 冒烟自检工具（开发用）。
 *
 * <pre>
 *   # 牌桌出图（会先按 --players 开一局，再自动打 N 手，让牌河有内容）
 *   java -cp out maj.PreviewShot http://127.0.0.1:8844 a.png --players 3 --autoplay 20
 *   java -cp out maj.PreviewShot http://127.0.0.1:8844 b.png --players 4 --autoplay 20
 *
 *   # 开局选择页出图
 *   java -cp out maj.PreviewShot http://127.0.0.1:8844 start.png --start
 *
 *   # 结算面板出图（那一局要已经打完）
 *   java -cp out maj.PreviewShot http://127.0.0.1:8844 r.png --result
 *
 *   # 真开一次窗口跑几秒（检查 Swing 装配 / 轮询 / 图片加载）
 *   java -cp out maj.PreviewShot http://127.0.0.1:8844 --frame
 *
 *   # 关窗停服务验证：UI 自己起后端 → 走关窗清理 → 检查端口是否释放
 *   java -cp out maj.PreviewShot http://127.0.0.1:8844 --closetest 8844
 * </pre>
 */
public final class PreviewShot {

    private PreviewShot() {
    }

    public static void main(String[] args) throws Exception {
        String base = "http://127.0.0.1:8844";
        String out = null;
        int seat = 0;
        int width = 1400;
        int height = 900;
        int players = 0;
        int autoplay = 0;
        int closePort = 0;
        boolean frame = false;
        boolean startShot = false;
        boolean resultShot = false;
        boolean stress = false;
        boolean replayShot = false;
        boolean animShot = false;
        boolean resultCardShot = false;
        boolean drawCardShot = false;
        boolean winCardShot = false;
        boolean dealShot = false;
        int viewSeatShot = -1;
        int replayAt = -1;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--seat" -> seat = Integer.parseInt(args[++i]);
                case "--w" -> width = Integer.parseInt(args[++i]);
                case "--h" -> height = Integer.parseInt(args[++i]);
                case "--players" -> players = Integer.parseInt(args[++i]);
                case "--autoplay" -> autoplay = Integer.parseInt(args[++i]);
                case "--frame" -> frame = true;
                case "--start" -> startShot = true;
                case "--result" -> resultShot = true;
                case "--stress" -> stress = true;
                case "--replay" -> replayShot = true;
                case "--anim" -> animShot = true;
                case "--at" -> replayAt = Integer.parseInt(args[++i]);
                case "--resultcard" -> resultCardShot = true;
                case "--drawcard" -> drawCardShot = true;
                case "--wincard" -> winCardShot = true;
                case "--anim-deal" -> dealShot = true;
                case "--viewseat" -> viewSeatShot = Integer.parseInt(args[++i]);
                case "--closetest" -> closePort = Integer.parseInt(args[++i]);
                default -> {
                    if (a.startsWith("http")) {
                        base = a;
                    } else if (!a.startsWith("--")) {
                        out = a;
                    }
                }
            }
        }
        MajClient client = new MajClient(base);

        if (closePort > 0) {
            flowTest(client, closePort);
            return;
        }
        if (frame) {
            runFrame(client, seat);
            return;
        }
        if (animShot || dealShot) {
            animShot(client, out == null ? "anim.png" : out, width, height, seat, dealShot);
            return;
        }
        if (replayShot) {
            replayShot(client, out == null ? "replay.png" : out, width, height,
                    replayAt < 0 ? -1 : replayAt, seat);
            return;
        }
        if (resultCardShot) {
            resultCardShot(client, out == null ? "resultcard.png" : out, width, height, seat,
                    false);
            return;
        }
        if (drawCardShot) {
            resultCardShot(client, out == null ? "drawcard.png" : out, width, height, seat, true);
            return;
        }
        if (winCardShot) {
            resultCardShot(client, out == null ? "wincard.png" : out, width, height, seat, false);
            return;
        }
        if (viewSeatShot >= 0) {
            viewSeatShot(client, out == null ? "view.png" : out, width, height, viewSeatShot);
            return;
        }
        if (stress) {
            stressShot(client, players == 0 ? 3 : players, out == null ? "stress.png" : out,
                    width, height);
            return;
        }
        if (resultShot) {
            shotResult(client, seat, out == null ? "result.png" : out);
            return;
        }
        if (startShot) {
            shotStart(client, out == null ? "start.png" : out, width, height, players);
            return;
        }

        if (players == 3 || players == 4) {
            List<String> seats = new ArrayList<>();
            seats.add("human");
            for (int i = 1; i < players; i++) {
                seats.add("teacher");
            }
            MajClient.Config cfg = client.config();
            client.setConfig(new MajClient.Config(players, seats, Math.min(cfg.delay, 0.3),
                    cfg.seed + 1));
            System.out.println("▸ 已切到 " + players + " 人：" + seats);
            Thread.sleep(800);
        }

        // 自动打几手：牌河 / 副露 / 立直都有东西可看
        for (int n = 0; n < autoplay; n++) {
            State s = client.state(seat);
            if (s.pending != null && s.pending.yours && !s.pending.options.isEmpty()) {
                int idx = s.pending.options.get(s.pending.options.size() / 2).index;
                client.act(s.pending.seat, idx);
            }
            Thread.sleep(Math.max(120, (long) (s.delay * 1000) + 60));
        }

        State state = client.state(seat);
        TileImages images = new TileImages(client, Path.of(".tilecache"));
        preload(images, state);

        JPanel root = new JPanel(new BorderLayout());
        root.setBackground(Palette.PANEL);
        TablePanel table = new TablePanel(images, null);
        table.setState(state);
        ActionBar bar = new ActionBar(null);
        bar.refresh(state);
        root.add(table, BorderLayout.CENTER);
        root.add(bar, BorderLayout.SOUTH);
        shoot(root, out == null ? "table.png" : out, width, height);
        System.out.println("▸ 牌桌：" + state.players() + " 人，视角 P" + state.you
                + "，座位块 " + table.seatCount() + " 个，牌 " + layoutSpots(table) + " 张"
                + table.tileSizeText());
        images.shutdown();
        System.exit(0);
    }

    /**
     * 抓动画中间态：连续拉两次 /api/state，把第二份喂给牌桌后立刻出图，
     * 这时「刚打出的那张」还在从手牌飞向牌河的路上。
     */
    /**
     * 动画中间态出图。
     *
     * @param deal true = 抓「开新局发牌」那一下（自己会 POST /api/next）；
     *             false = 抓一次普通的出牌 / 摸牌
     */
    private static void animShot(MajClient client, String out, int w, int h, int seat, boolean deal)
            throws Exception {
        State s1 = client.state(seat);
        State s2 = s1;
        if (deal) {
            // 先把它推到局间（该我们决策就随便出一张），再开下一局，然后盯着 hand_no 变
            for (int i = 0; i < 400 && !s1.awaitingNext && !s1.finished; i++) {
                if (s1.pending != null && s1.pending.yours && !s1.pending.options.isEmpty()) {
                    client.act(s1.pending.seat, s1.pending.options.get(0).index);
                }
                Thread.sleep(120);
                s1 = client.state(seat);
            }
            if (s1.awaitingNext || s1.confirmBetween) {
                client.nextHand();
            }
            int hand0 = s1.handNo;
            for (int i = 0; i < 250; i++) {
                Thread.sleep(60);
                State t = client.state(seat);
                if (t.handNo != hand0) {
                    s2 = t;
                    break;
                }
            }
        } else {
            for (int i = 0; i < 30; i++) {
                Thread.sleep(220);
                State t = client.state(seat);
                boolean grew = false;
                for (int k = 0; k < Math.min(t.players(), s1.players()); k++) {
                    State.Player a = s1.player(k);
                    State.Player b = t.player(k);
                    if (a != null && b != null && b.pond.size() > a.pond.size()) {
                        grew = true;
                    }
                }
                if (grew || t.handNo != s1.handNo) {
                    s2 = t;
                    break;
                }
                s1 = t;
            }
        }
        TileImages images = new TileImages(client, Path.of(".tilecache"));
        preload(images, s2);
        JPanel root = new JPanel(new BorderLayout());
        root.setBackground(Palette.PANEL);
        TablePanel table = new TablePanel(images, null);
        root.add(table, BorderLayout.CENTER);
        root.add(new ActionBar(null), BorderLayout.SOUTH);
        root.setSize(w, h);
        layoutDeep(root);                // 先摆好尺寸，再喂状态 —— 否则布局算出来是空的
        table.setState(s1);
        table.setState(s2);              // 触发动画
        System.out.println("  （触发后正在飞 " + table.flyingCount() + " 张，桌子 "
                + table.getWidth() + "×" + table.getHeight() + "）");
        Thread.sleep(deal ? 300 : 90);   // 等动画飞到半路（发牌是一张张来的，多等一会儿）
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        try {
            root.paint(g);
        } finally {
            g.dispose();
        }
        ImageIO.write(img, "png", Path.of(out).toFile());
        int grewBy = 0;
        for (int k = 0; k < Math.min(s2.players(), s1.players()); k++) {
            State.Player a = s1.player(k);
            State.Player b = s2.player(k);
            if (a != null && b != null && b.pond.size() > a.pond.size()) {
                grewBy += b.pond.size() - a.pond.size();
            }
        }
        System.out.println("▸ 动画中间态：" + out + (deal
                ? "（发牌：第 " + s2.handNo + " 局刚开，牌正从桌心一张张飞到位）"
                : "（出牌：这次牌河多了 " + grewBy + " 张，正在飞的牌画在半路上）")
                + " · 手上正在飞 " + table.flyingCount() + " 张 · 动画开关=" + table.animationsOn());
        images.shutdown();
        System.exit(0);
    }

    /** 回放模式出图：默认自动找一帧「荣和」的（这样顺便能看到铳牌标红）。 */
    private static void replayShot(MajClient client, String out, int w, int h, int at, int seat)
            throws Exception {
        MajClient.ReplayMeta meta = client.replayMeta();
        List<Map<String, Object>> frames = client.replayFrames(0, Math.min(meta.frames() - 1, 199));
        int pick = at;
        if (pick < 0) {
            pick = frames.size() / 2;
            for (Map<String, Object> f : frames) {
                Object ag = f.get("agari");
                if (ag instanceof Map) {
                    Map<String, Object> a = Json.obj(ag);
                    if (!Json.b(a.get("tsumo"), false)) {
                        pick = Json.i(f.get("i"), 0);   // 荣和帧：能同时看到铳牌标红
                        break;
                    }
                }
            }
        }
        TileImages images = new TileImages(client, Path.of(".tilecache"));
        preload(images, client.state(seat));
        java.util.concurrent.ExecutorService pool =
                java.util.concurrent.Executors.newFixedThreadPool(2, r -> {
                    Thread t = new Thread(r, "shot-net");
                    t.setDaemon(true);
                    return t;
                });
        final ReplayCard[] holder = new ReplayCard[1];
        final int at2 = pick;
        SwingUtilities.invokeAndWait(() -> {
            ReplayCard card = new ReplayCard(client, images, pool, new ReplayCard.Sink() {
                @Override
                public void exitReplay() {
                    // 出图用
                }

                @Override
                public void replayViewChanged(int s2) {
                    // 出图用
                }
            });
            card.debugLoad(meta, frames, at2);
            holder[0] = card;
        });
        shoot(holder[0], out, w, h);
        Map<String, Object> fr = holder[0].currentFrame();
        System.out.println("▸ 回放图：第 " + (pick + 1) + "/" + meta.frames() + " 手"
                + " · text=" + (fr == null ? "?" : Json.str(fr.get("text"), "")));
        System.out.println("  全亮？" + (fr != null && Json.arr(fr.get("players")).stream()
                .allMatch(o -> Json.obj(o).get("hand") instanceof List)));
        images.shutdown();
        System.exit(0);
    }

    /**
     * 本局结算页出图。
     *
     * @param draw true = 用合成的流局数据；false = 用<b>后端真实</b>的 hand_result
     *             （要 awaiting_next / finished 才有）
     */
    private static void resultCardShot(MajClient client, String out, int w, int h, int seat,
            boolean draw) throws Exception {
        State st;
        if (draw) {
            st = Fixtures.draw();
        } else {
            st = client.state(seat);
            if (st.handResult == null) {
                System.out.println("▸ 这一局还没结算，用合成数据出图（--wincard）");
                st = Fixtures.win();
            }
        }
        TileImages images = new TileImages(client, Path.of(".tilecache"));
        preload(images, st);
        final State shot = st;
        final ResultCard[] holder = new ResultCard[1];
        SwingUtilities.invokeAndWait(() -> {
            ResultCard card = new ResultCard(images, new ResultCard.Sink() {
                @Override
                public void nextHand() {
                    // 出图用
                }

                @Override
                public void replayThisHand() {
                    // 出图用
                }

                @Override
                public void backToTable() {
                    // 出图用
                }
            });
            card.show(shot);
            holder[0] = card;
        });
        shoot(holder[0], out, w, h);
        State.RoundResult r = st.handResult;
        System.out.println("▸ 结算图：" + (r.isDraw() ? "流局" : r.label + " " + r.totalHan + "番"
                + r.fu + "符") + " · 役 " + r.yaku.size() + " 个 · deltas=" + r.deltas
                + " · awaiting_next=" + st.awaitingNext + " · 「下一局」按钮="
                + holder[0].nextButton().isEnabled()
                + " · 牌型条 " + holder[0].strip().getWidth() + "px");
        images.shutdown();
        System.exit(0);
    }

    /** 换主视角出图：/api/state?seat=N（AI 桌会带 reveal=1 全亮）。 */
    private static void viewSeatShot(MajClient client, String out, int w, int h, int seat)
            throws Exception {
        State probe = client.state(seat);
        State st = probe.spectatorOk ? client.state(seat, true) : probe;
        TileImages images = new TileImages(client, Path.of(".tilecache"));
        preload(images, st);
        JPanel root = new JPanel(new BorderLayout());
        root.setBackground(Palette.PANEL);
        TablePanel table = new TablePanel(images, null);
        table.setState(st);
        ActionBar bar = new ActionBar(null);
        bar.refresh(st);
        root.add(table, BorderLayout.CENTER);
        root.add(bar, BorderLayout.SOUTH);
        shoot(root, out, w, h);
        System.out.println("▸ 换主视角图：you=" + st.you + " · " + st.players() + " 人 · reveal="
                + st.spectatorOk + " · 别家手牌="
                + (st.details.size() > 1 && st.details.get(1).hand != null ? "全亮" : "盖着"));
        images.shutdown();
        System.exit(0);
    }

    private static void stressShot(MajClient client, int players, String out, int w, int h)
            throws Exception {
        State st = Fixtures.stress(players);
        TileImages images = new TileImages(client, Path.of(".tilecache"));
        preload(images, st);
        JPanel root = new JPanel(new BorderLayout());
        root.setBackground(Palette.PANEL);
        TablePanel table = new TablePanel(images, null);
        table.setState(st);
        ActionBar bar = new ActionBar(null);
        bar.refresh(st);
        root.add(table, BorderLayout.CENTER);
        root.add(bar, BorderLayout.SOUTH);
        shoot(root, out, w, h);
        System.out.println("▸ 压测图：" + players + " 人 · " + w + "×" + h
                + " · 每家 4 组副露（吃/碰/暗杠/加杠）+ 20 张牌河 + 14 张手牌");
        images.shutdown();
        System.exit(0);
    }

    private static int layoutSpots(TablePanel table) throws Exception {
        final int[] n = {0};
        SwingUtilities.invokeAndWait(() -> n[0] = table.clickable().size());
        return n[0];
    }

    /** 开局选择页出图。 */
    private static void shotStart(MajClient client, String out, int w, int h, int players)
            throws Exception {
        MajClient.Models models;
        try {
            models = client.models();
        } catch (MajException e) {
            models = new MajClient.Models();
            System.out.println("⚠️ 拿不到模型列表（" + e.getMessage() + "），用默认座位名出图");
        }
        final MajClient.Models m = models;
        final StartPanel[] holder = new StartPanel[1];
        SwingUtilities.invokeAndWait(() -> {
            StartPanel p = new StartPanel(new StartPanel.Sink() {
                @Override
                public void startGame(MajClient.Config cfg) {
                    // 出图用，不真的开局
                }

                @Override
                public void startBackend() {
                    // 出图用
                }
            });
            p.setModels(m);
            p.setBackendState(true, true, client.base());
            if (players == 4 || players == 3) {
                p.debugSelectMode(players);      // 切到这个人数并重新填默认值
            }
            holder[0] = p;
        });
        shoot(holder[0], out, w, h);
        final List<List<String>> seats = new ArrayList<>();
        SwingUtilities.invokeAndWait(() -> seats.add(holder[0].currentSeats()));
        System.out.println("▸ 选择页已写出：人数=" + (players == 4 ? 4 : 3)
                + " 默认座位=" + seats.get(0)
                + " 其中模型座位=" + holder[0].modelSeats());
        System.out.println("  三麻候选：" + m.modelsFor(3));
        System.out.println("  四麻候选：" + m.modelsFor(4));
        System.exit(0);
    }

    /** 结算面板出图。 */
    private static void shotResult(MajClient client, int seat, String out) throws Exception {
        State state = client.state(seat);
        if (state.result == null) {
            System.out.println("✗ 这一局还没结束（result=null），没法画结算面板");
            System.exit(1);
        }
        final ResultDialog[] holder = new ResultDialog[1];
        SwingUtilities.invokeAndWait(() -> holder[0] = new ResultDialog(null, state));
        ResultDialog dialog = holder[0];
        dialog.pack();
        java.awt.Container pane = dialog.getContentPane();
        shoot(pane, out, pane.getPreferredSize().width, pane.getPreferredSize().height);
        System.out.println("▸ 结算面板：" + state.players() + " 人，"
                + state.result.hands.size() + " 局");
        System.exit(0);
    }

    /** 把任意组件离屏画成 PNG。 */
    private static void shoot(java.awt.Component c, String out, int w, int h) throws Exception {
        if (c instanceof java.awt.Container container) {
            container.setSize(w, h);
            layoutDeep(container);
        } else {
            c.setSize(w, h);
        }
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        try {
            c.paint(g);
        } finally {
            g.dispose();
        }
        Path path = Path.of(out);
        ImageIO.write(img, "png", path.toFile());
        System.out.println("▸ 已写出 " + path.toAbsolutePath() + "（" + w + "×" + h + "）");
    }

    private static void layoutDeep(java.awt.Container c) {
        c.doLayout();
        for (java.awt.Component child : c.getComponents()) {
            if (child instanceof java.awt.Container cc) {
                layoutDeep(cc);
            }
        }
    }

    private static void preload(TileImages images, State state) throws InterruptedException {
        List<String> ids = new ArrayList<>(List.of(Tile.IMAGE_IDS));
        ids.addAll(state.dora);
        for (State.Player p : state.details) {
            ids.addAll(p.pond);
            if (p.hand != null) {
                ids.addAll(p.hand);
            }
            for (State.Meld m : p.melds) {
                ids.addAll(m.tiles());
            }
        }
        if (state.pending != null) {
            ids.addAll(state.pending.yourHand);
        }
        if (state.handResult != null) {
            ids.addAll(state.handResult.hand);
            if (!state.handResult.winTile.isEmpty()) {
                ids.add(state.handResult.winTile);
            }
        }
        images.preload(ids);
        for (int i = 0; i < 100 && images.readyCount() < Tile.IMAGE_IDS.length; i++) {
            Thread.sleep(100);
        }
        System.out.println("▸ 牌图就绪 " + images.readyCount() + "/" + Tile.IMAGE_IDS.length);
    }

    /** 真开一次窗口，跑几秒看看会不会炸。 */
    private static void runFrame(MajClient client, int seat) throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            MainFrame f = new MainFrame(client, null, false, seat, true);
            f.setVisible(true);
            f.debugShowTable();
            javax.swing.Timer t = new javax.swing.Timer(3000, e -> {
                System.out.println("▸ 检查：" + f.debugStatusText());
                State s = f.debugState();
                System.out.println("   局面：" + (s == null ? "null"
                        : s.players() + " 人 · " + s.wallLeft + " 张 · " + s.turn + " 巡"));
            });
            t.start();
            javax.swing.Timer stop = new javax.swing.Timer(7000, e -> {
                System.out.println("▸ 窗口装配 / 轮询正常，关闭退出");
                f.debugQuit();
            });
            stop.setRepeats(false);
            stop.start();
        });
        Thread.sleep(9000);
        System.out.println("（窗口模式结束）");
        System.exit(0);
    }

    /**
     * 关窗停服务验证：由 UI 自己起后端 → 跑几秒 → 走「关窗口」那套清理 →
     * 检查端口是否真的释放。
     */
    /** 把 /api/state 里的 seats 原样打出来（用户要看的证据）。 */
    private static boolean printSeats(MajClient client, String title) throws Exception {
        String raw = Http.getText(client.base() + "/api/state?seat=0",
                java.time.Duration.ofSeconds(6));
        Map<String, Object> m = Json.obj(Json.parse(raw));
        System.out.println("  " + title + " /api/state 实际输出：");
        System.out.println("    player_count = " + m.get("player_count"));
        System.out.println("    seats        = " + Json.write(m.get("seats")));
        List<Object> seats = Json.arr(m.get("seats"));
        List<String> types = new ArrayList<>();
        for (Object o : seats) {
            types.add(Json.str(Json.obj(o).get("type"), ""));
        }
        boolean anyModel = types.stream().anyMatch(t -> !t.equals("human") && !t.equals("teacher")
                && !t.equals("random") && !t.equals("external"));
        boolean anyTeacher = types.contains("teacher");
        System.out.println("    -> type 列表 " + types + (anyModel ? " ✓ 有模型"
                : " ✗ 一个模型都没有") + (anyTeacher ? "（含 teacher）" : ""));
        return anyModel && !anyTeacher;
    }

    private static void flowTest(MajClient client, int port) throws Exception {
        Path root = Backend.findProjectRoot();
        if (root == null) {
            System.out.println("✗ 找不到项目根，没法起后端");
            System.exit(1);
        }
        Backend backend = new Backend(root, port);
        System.out.println("▸ 项目根：" + root);
        System.out.println("▸ 故意先不起后端，直接开窗口（= 用户 bash run.sh 之后的第一次）");

        final MainFrame[] holder = new MainFrame[1];
        SwingUtilities.invokeAndWait(() -> {
            MainFrame f = new MainFrame(client, backend, true, 0, true);
            f.setVisible(true);
            holder[0] = f;
        });
        Thread.sleep(2500);
        final List<List<String>> cold = new ArrayList<>();
        SwingUtilities.invokeAndWait(() -> cold.add(holder[0].debugStartPanel().currentSeats()));
        System.out.println("▸ 第一眼是选择页？" + holder[0].debugOnStartPage()
                + " · 状态：" + holder[0].debugStatusText());
        System.out.println("  这时后端还没起、模型列表为空，座位默认是 " + cold.get(0)
                + "（预期是 teacher，因为还不知道有哪些模型）");

        // 用户这时候就点了「开始对局」：界面应该自己起后端、拿到模型、把座位改成模型再开
        final List<MajClient.Config> coldCfg = new ArrayList<>();
        SwingUtilities.invokeAndWait(() -> coldCfg.add(holder[0].debugStartPanel().collect()));
        System.out.println("▸ 用户直接点「开始对局」（带出去的座位还是 " + coldCfg.get(0).seats + "）");
        SwingUtilities.invokeAndWait(() -> holder[0].startGame(coldCfg.get(0)));
        Thread.sleep(9000);
        System.out.println("  UI 自己起后端：" + backend.describe() + " · owned=" + backend.owned()
                + " alive=" + backend.alive() + " · lastMessage=" + backend.lastMessage());
        System.out.println("  状态：" + holder[0].debugStatusText());
        final List<List<String>> afterStart = new ArrayList<>();
        SwingUtilities.invokeAndWait(() -> afterStart.add(holder[0].debugStartPanel().currentSeats()));
        System.out.println("  开完这一局，选择页座位已经变成 " + afterStart.get(0));
        boolean coldOk = printSeats(client, "第一次点开始对局后（冷启动）");

        // ① 三麻：选择页默认值应该是三麻模型
        List<String> three = new ArrayList<>();
        SwingUtilities.invokeAndWait(() -> {
            holder[0].debugStartPanel().debugSelectMode(3);
            three.addAll(holder[0].debugStartPanel().currentSeats());
        });
        System.out.println("▸ 切到三麻，选择页默认座位 = " + three
                + "（模型：" + holder[0].debugStartPanel().modelSeats() + "）");
        MajClient.Config cfg3 = new MajClient.Config(3, three, 0.2, 20261013);
        SwingUtilities.invokeAndWait(() -> holder[0].startGame(cfg3));
        Thread.sleep(4000);
        boolean threeOk = printSeats(client, "三麻（点开始对局后）");
        State st3 = holder[0].debugState();
        System.out.println("    牌桌：还在选择页？" + holder[0].debugOnStartPage() + " · "
                + (st3 == null ? "没拿到局面 ✗" : st3.players() + " 人 · player_count="
                  + st3.playerCount + " · 明细 " + st3.details.size() + " 条"));

        // ② 四麻：切过去应该重新填四麻模型
        List<String> four = new ArrayList<>();
        SwingUtilities.invokeAndWait(() -> {
            holder[0].debugStartPanel().debugSelectMode(4);
            four.addAll(holder[0].debugStartPanel().currentSeats());
        });
        System.out.println("▸ 切到四麻，选择页默认座位 = " + four
                + "（模型：" + holder[0].debugStartPanel().modelSeats() + "）");
        MajClient.Config cfg4 = new MajClient.Config(4, four, 0.2, 20261014);
        SwingUtilities.invokeAndWait(() -> holder[0].startGame(cfg4));
        Thread.sleep(4000);
        boolean fourOk = printSeats(client, "四麻（点开始对局后）");
        State st4 = holder[0].debugState();
        System.out.println("    牌桌：" + (st4 == null ? "没拿到局面 ✗" : st4.players()
                + " 人 · player_count=" + st4.playerCount + " · 明细 " + st4.details.size() + " 条"));

        // 后端日志里不该有「退化成教师」的警告
        String log = backend.tailLog(80);
        boolean degraded = log.contains("找不到模型");
        System.out.println("▸ 后端日志退化警告：" + (degraded ? "✗ 有「找不到模型」：" + log
                : "✓ 没有（座位都真的用上了模型）"));

        // 座位校验：故意选一个四麻模型放到三麻桌上，应该被 checkSeats 抓到
        final List<List<String>> probs = new ArrayList<>();
        SwingUtilities.invokeAndWait(() -> {
            holder[0].debugStartPanel().debugSelectMode(3);
            holder[0].debugStartPanel().currentSeats();
            probs.add(holder[0].debugStartPanel().checkSeats());
        });
        System.out.println("▸ 三麻默认值的校验结果："
                + (probs.get(0).isEmpty() ? "通过（无问题）" : probs.get(0)));

        // 故意选错：三麻桌上放四麻模型 + 一个不存在的模型，应该被 checkSeats 抓住
        MajClient.Config bad = new MajClient.Config(3,
                new ArrayList<>(List.of("human", "maj-flash4", "no-such-model")), 0.2, 1);
        SwingUtilities.invokeAndWait(() -> holder[0].debugStartPanel().apply(bad));
        final List<List<String>> badProbs = new ArrayList<>();
        SwingUtilities.invokeAndWait(() -> badProbs.add(holder[0].debugStartPanel().checkSeats()));
        System.out.println("▸ 故意选错（三麻里塞 maj-flash4 + no-such-model）的校验结果：");
        for (String p2 : badProbs.get(0)) {
            System.out.println("    ⚠ " + p2);
        }
        boolean caught = badProbs.get(0).size() >= 2;
        System.out.println(caught ? "    ✓ 两条都抓到了（开始前会弹窗拦一下）"
                : "    ✗ 没抓到，校验有问题");
        boolean flowOk = coldOk && threeOk && fourOk && !degraded && probs.get(0).isEmpty()
                && caught;
        // 走的就是关窗口那一套：停 timer / 停线程池 / 停后端（只是不真的 exit，好接着验证）
        SwingUtilities.invokeAndWait(() -> holder[0].debugShutdown(false));
        boolean free = backend.waitPortFree(client, java.time.Duration.ofSeconds(6));
        System.out.println("▸ 关窗清理后：后端进程活着？" + backend.alive() + " · 端口 " + port
                + (free ? " 已释放 ✓" : " 还在占着 ✗"));
        boolean allOk = free && !backend.alive() && flowOk;
        System.out.println(allOk
                ? "FLOWTEST PASS（默认=模型 · 三/四麻各自正确 · 无退化 · 关窗停服务）"
                : "FLOWTEST FAIL（端口释放=" + free + " 进程已停=" + !backend.alive()
                  + " 流程=" + flowOk + "）");
        System.exit(allOk ? 0 : 1);
    }
}
