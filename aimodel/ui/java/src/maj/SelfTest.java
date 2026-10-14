package maj;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 命令行自检（不碰 Swing，headless 可跑）：
 *
 * <pre>
 *   java -cp out maj.SelfTest http://127.0.0.1:8844
 * </pre>
 *
 * <p>检查：/api/models；/api/state 里 {@code player_count} / {@code players} /
 * {@code players_detail} 三个字段都能读且一致；pending.tile；以人类座位提交一次
 * 合法 discard（{@code options[0].index}）；**越界下标必须回 {@code ok:false}**；
 * POST /api/config 切四麻再切回来；下载 /tiles/1m.png。
 * 全过退出码 0，有失败退出码 1。
 */
public final class SelfTest {

    private static int passed;
    private static int failed;

    private SelfTest() {
    }

    public static void main(String[] args) {
        String base = "http://127.0.0.1:8844";
        for (String a : args) {
            if (a.startsWith("http")) {
                base = a;
            }
        }
        MajClient client = new MajClient(base);

        System.out.println("═══ 土豆麻将 Java 客户端自检 ═══");
        System.out.println("后端      : " + client.base());
        System.out.println("Java      : " + System.getProperty("java.version")
                + "（" + System.getProperty("java.vendor") + "）");

        MajClient.Models models = step1(client);
        State s = step2(client);
        step3Discard(client, s);
        step4Config(client);
        step5Tile(client);
        step6OutOfRange(client);
        step7Replay(client);
        step8NextHand(client);
        step9Reveal(client);

        System.out.println();
        System.out.println("──── 结果 ────");
        System.out.println("通过 " + passed + " 项，失败 " + failed + " 项");
        System.out.println(failed == 0 ? "全部通过 ✓" : "有失败 ✗");
        System.out.println("（配置面板可选座位：" + models.seatChoices() + "）");
        System.exit(failed == 0 ? 0 : 1);
    }

    // ------------------------------------------------------------ 1 模型

    private static MajClient.Models step1(MajClient client) {
        section("1/6 GET /api/models");
        try {
            MajClient.Models m = client.models();
            ok("models=" + m.models + "  builtin=" + m.builtin + "  seats=" + m.seats);
            ok("观战模式会用的座位值：" + m.spectateSeat());
            return m;
        } catch (MajException e) {
            bad("拿不到模型列表：" + e.getMessage());
            return new MajClient.Models();
        }
    }

    // ------------------------------------------------------------ 2 局面

    /** 顺便把三个字段都验一遍。 */
    private static State step2(MajClient client) {
        section("2/6 GET /api/state?seat=0：player_count / players / players_detail 三字段都要能读");
        State s;
        try {
            s = client.state(0);
        } catch (MajException e) {
            bad("拉不到局面：" + e.getMessage());
            return null;
        }
        System.out.println("      原始 JSON 前 300 字：");
        System.out.println("        " + s.raw.substring(0, Math.min(300, s.raw.length())));
        int rawCount = rawInt(s.raw, "player_count");
        int rawPlayers = rawListLen(s.raw, "players");
        int rawDetail = rawListLen(s.raw, "players_detail");
        ok("player_count=" + rawCount + "（int）· players 是数组，"
                + rawPlayers + " 项 · players_detail 是数组，" + rawDetail + " 项");
        if (rawCount == rawPlayers && rawCount == rawDetail && rawCount == s.players()) {
            ok("三个字段一致，解析出来的人数=" + s.players() + "，座位明细 " + s.details.size() + " 条");
        } else {
            bad("三字段对不上：player_count=" + rawCount + " players=" + rawPlayers
                    + " players_detail=" + rawDetail + " 解析人数=" + s.players());
        }
        ok("parse 出来的 seats=" + s.seats.size() + " details=" + s.details.size()
                + "  剩余=" + s.wallLeft + " 巡=" + s.turn + " 场风=" + s.roundWind
                + " 宝牌指示=" + s.dora + " 种子=" + s.seed);
        ok("吃=" + s.allowChi + " 拔北=" + s.allowKita + " 节拍=" + s.delay + "s"
                + " 结束=" + s.finished);

        System.out.println("      座位一览（三字段渲染用到的全部信息）：");
        for (int i = 0; i < s.players(); i++) {
            State.Seat info = s.seat(i);
            State.Player p = s.player(i);
            String hand;
            if (p == null) {
                hand = "（没有明细！）";
            } else if (p.hand != null) {
                StringBuilder sb = new StringBuilder();
                for (String t : p.hand) {
                    sb.append(Tile.cn(t)).append(' ');
                }
                hand = p.hand.size() + " 张 " + sb.toString().trim();
            } else {
                hand = "牌背 " + p.handCount + " 张";
            }
            System.out.printf("        P%d %-20s %6d 点 %s家%s 副露%d 牌河%d %s %s%n",
                    i, info == null ? "?" : info.name, p == null ? 0 : p.score,
                    p == null ? "?" : p.seatWind, p != null && p.isDealer ? " 亲" : "  ",
                    p == null ? 0 : p.melds.size(), p == null ? 0 : p.pond.size(),
                    p != null && p.kita > 0 ? "拔北" + p.kita : "", hand);
        }
        boolean leak = false;
        for (int i = 0; i < s.players(); i++) {
            State.Player p = s.player(i);
            if (i != s.you && p != null && p.hand != null) {
                leak = true;
            }
        }
        if (leak) {
            bad("别人（非 you）的手牌不是 null —— 视角过滤有问题");
        } else {
            ok("视角过滤正确：只有 seat=" + s.you + " 能看到手牌");
        }
        if (s.pending != null) {
            System.out.println("      当前决策：seat=" + s.pending.seat + " kind=" + s.pending.kind
                    + "（" + s.pending.kindCn + "）yours=" + s.pending.yours
                    + " 选项 " + s.pending.options.size() + " 个");
            if (s.pending.tile != null) {
                ok("pending.tile=" + s.pending.tile + "（" + s.pending.tileCn
                        + "）—— 被鸣/被荣的那张，UI 会高亮");
            } else {
                System.out.println("      · 这次 pending 没有 tile（后端只在鸣/荣时给）");
            }
        } else {
            ok("当前没有待决策（pending=null）"
                    + (s.waitingFor.isEmpty() ? "" : "，waiting_for=" + s.waitingFor));
        }
        return s;
    }

    // ------------------------------------------------------------ 3 出牌

    private static void step3Discard(MajClient client, State first) {
        section("3/6 以人类座位提交一次合法 discard（pending.options[0].index）");
        if (first == null) {
            bad("没有第一步的局面，跳过");
            return;
        }
        int seat = 0;
        for (int i = 0; i < first.seats.size(); i++) {
            if (first.isHumanSeat(i)) {
                seat = i;
                break;
            }
        }
        if (!first.isHumanSeat(seat)) {
            bad("座位表里没有 human / external 座位");
            return;
        }
        System.out.println("      人 / 外部接口座位 = P" + seat);
        try {
            State s = client.state(seat);
            for (int guard = 0; guard < 90; guard++) {
                if (s.awaitingNext) {
                    // 局间等确认：先让它开下一局，不然永远等不到 discard
                    System.out.println("      这局结束了（" + s.lastHandNote + "），先 POST /api/next 继续");
                    client.nextHand();
                    Thread.sleep(500);
                    s = client.state(seat);
                    continue;
                }
                if (s.pending == null || !s.pending.yours) {
                    Thread.sleep(300);
                    s = client.state(seat);
                    continue;
                }
                State.Option pick = s.pending.options.isEmpty() ? null
                        : s.pending.options.get(0);
                if (pick == null) {
                    bad("决策没有任何选项");
                    return;
                }
                if (!s.pending.isDiscard()) {
                    System.out.println("      当前是 " + s.pending.kind + "（" + s.pending.kindCn
                            + "），先提交 options[0]=" + pick.label + " 让流程往下走");
                    client.act(s.pending.seat, pick.index);
                    Thread.sleep(400);
                    s = client.state(seat);
                    continue;
                }
                System.out.println("      拿到 discard：可选 " + s.pending.options.size()
                        + " 种，提交 options[0] → index=" + pick.index + " label=" + pick.label
                        + "（" + pick.cn + "）"
                        + (s.pending.drawn == null ? "" : "，刚摸 " + Tile.cn(s.pending.drawn)));
                MajClient.ActResult r = client.act(s.pending.seat, pick.index);
                if (r.ok()) {
                    ok("POST /api/act → " + r.raw());
                } else {
                    bad("POST /api/act 被拒：" + r.describe());
                    return;
                }
                Thread.sleep(600);
                State after = client.state(seat);
                ok("提交后：pending=" + (after.pending == null ? "null"
                        : after.pending.kind + "/seat" + after.pending.seat)
                        + "，牌谱最后一行："
                        + (after.log.isEmpty() ? "（空）" : after.log.get(after.log.size() - 1)));
                return;
            }
            bad("等不到 discard 决策");
        } catch (MajException e) {
            bad("出牌过程中出错：" + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            bad("被中断");
        }
    }

    // ------------------------------------------------------------ 4 配置

    private static void step4Config(MajClient client) {
        section("4/6 POST /api/config 切四麻再切回三麻（返回体要有 human_wait）");
        try {
            MajClient.Config now = client.config();
            ok("当前配置：" + now);
            long seed = now.seed + 1;

            List<String> four = new ArrayList<>(List.of("human", "teacher", "teacher", "teacher"));
            var r4 = client.setConfig(new MajClient.Config(4, four, 0.2, seed));
            ok("切四麻返回：" + r4);
            if (r4.get("human_wait") != null) {
                ok("返回体带 human_wait=" + r4.get("human_wait"));
            } else {
                bad("返回体没有 human_wait");
            }
            Thread.sleep(700);
            State s4 = client.state(0);
            if (s4.players() == 4 && s4.details.size() == 4 && s4.allowChi && !s4.allowKita) {
                ok("复查四麻：player_count=" + s4.playerCount + " 明细=" + s4.details.size()
                        + " allow_chi=true allow_kita=false");
            } else {
                bad("四麻复查不对：players=" + s4.players() + " 明细=" + s4.details.size()
                        + " allow_chi=" + s4.allowChi + " allow_kita=" + s4.allowKita);
            }

            List<String> three = new ArrayList<>(List.of("human", "teacher", "teacher"));
            var r3 = client.setConfig(new MajClient.Config(3, three, 0.2, seed + 1));
            ok("切回三麻返回：" + r3);
            Thread.sleep(700);
            State s3 = client.state(0);
            if (s3.players() == 3 && s3.details.size() == 3 && s3.allowKita && !s3.allowChi) {
                ok("复查三麻：player_count=" + s3.playerCount + " 明细=" + s3.details.size()
                        + " allow_kita=true allow_chi=false");
            } else {
                bad("三麻复查不对：players=" + s3.players() + " 明细=" + s3.details.size());
            }
        } catch (MajException e) {
            bad("配置接口出错：" + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            bad("被中断");
        }
    }

    // ------------------------------------------------------------ 5 牌图

    private static void step5Tile(MajClient client) {
        section("5/6 GET /tiles/1m.png 存到临时文件");
        try {
            byte[] png = client.tilePng("1m");
            Path out = Files.createTempFile("maj-1m-", ".png");
            Files.write(out, png);
            if (png.length > 1000) {
                ok("已保存 " + out + "（" + png.length + " 字节 > 1000，PNG 头 "
                        + (png.length > 8 && (png[0] & 0xFF) == 0x89 ? "正常" : "可疑") + "）");
                Files.deleteIfExists(out);
            } else {
                bad("牌图只有 " + png.length + " 字节");
            }
            StringBuilder sb = new StringBuilder();
            for (String id : new String[] {"0m", "0p", "0s", "1z", "7z"}) {
                sb.append(id).append('=').append(client.tilePng(id).length).append("B ");
            }
            ok("其它牌图：" + sb.toString().trim());
        } catch (MajException e) {
            bad("拉牌图失败：" + e.getMessage());
        } catch (java.io.IOException e) {
            bad("写临时文件失败：" + e.getMessage());
        }
    }

    // ------------------------------------------------------------ 6 越界

    private static void step6OutOfRange(MajClient client) {
        section("6/6 越界下标必须被后端拒绝（ok:false）");
        try {
            State s = client.state(0);
            int seat = -1;
            for (int i = 0; i < s.seats.size(); i++) {
                if (s.isHumanSeat(i)) {
                    seat = i;
                    break;
                }
            }
            if (seat < 0) {
                bad("没有人类座位，跳过越界测试");
                return;
            }
            if (s.pending != null) {
                System.out.println("      这个座位现在还有待决策（" + s.pending.kind
                        + "），越界测试可能会被「先到先得」吃掉，先等它过去");
            }
            MajClient.ActResult r = client.act(seat, 999);
            System.out.println("      提交 seat=" + seat + " index=999 → HTTP " + r.status()
                    + " " + r.raw());
            if (!r.ok() && r.status() >= 400) {
                ok("越界被拒：ok=false，HTTP " + r.status() + "，error=" + r.error());
            } else {
                bad("越界居然被接受了（ok=" + r.ok() + "）—— 后端校验没生效？");
            }
            MajClient.ActResult r2 = client.act(seat, -1);
            if (!r2.ok()) {
                ok("负数下标也被拒：HTTP " + r2.status() + " " + r2.error());
            } else {
                bad("负数下标被接受了");
            }
        } catch (MajException e) {
            bad("越界测试出错：" + e.getMessage());
        }
    }

    // ------------------------------------------------------------ 7 回放

    private static void step7Replay(MajClient client) {
        section("7/9 回放：/api/replay 元信息 · 单帧 · 批量 · 越界 400");
        try {
            MajClient.ReplayMeta meta = client.replayMeta();
            ok("frames=" + meta.frames() + " players=" + meta.players()
                    + " spectator_ok=" + meta.spectatorOk() + " seats=" + meta.seats());
            if (meta.frames() <= 0) {
                bad("回放帧数是 0（是不是还没打完一局？）");
                return;
            }
            Map<String, Object> f0 = client.replayFrame(0);
            ok("单帧 i=0：text=「" + Json.str(f0.get("text"), "") + "」 hand_no="
                    + Json.i(f0.get("hand_no"), 0) + " turn=" + Json.i(f0.get("turn"), 0)
                    + " wall=" + Json.i(f0.get("wall"), 0));
            List<Object> players = Json.arr(f0.get("players"));
            boolean allHands = !players.isEmpty();
            for (Object o : players) {
                if (!(Json.obj(o).get("hand") instanceof List)) {
                    allHands = false;
                }
            }
            if (allHands) {
                ok("回放帧里 " + players.size() + " 家 hand 都是全的（没有视角过滤）—— 全亮画得出来");
            } else {
                bad("回放帧里有 hand=null，全亮画不出来");
            }
            int to = Math.min(meta.frames() - 1, 99);
            List<Map<String, Object>> batch = client.replayFrames(0, to);
            if (batch.size() == to + 1) {
                ok("批量取帧 0~" + to + "：拿到 " + batch.size() + " 帧（一次预取一批，不是一帧一请求）");
            } else {
                bad("批量取帧数量不对：拿到 " + batch.size() + "，期望 " + (to + 1));
            }
            Http.Response resp = Http.get(client.base() + "/api/replay?i=99999",
                    java.time.Duration.ofSeconds(5));
            if (resp.status() == 400 && resp.body().contains("\"ok\": false")
                    || resp.status() == 400) {
                ok("越界帧号 i=99999 → HTTP " + resp.status() + " " + resp.body().trim());
            } else {
                bad("越界帧号没有返回 400：" + resp.status() + " " + resp.body());
            }
            // 和牌帧（有 agari 的）顺便验一下铳牌信息
            Map<String, Object> goal = null;
            for (Map<String, Object> fr : batch) {
                if (fr.get("agari") instanceof Map) {
                    goal = fr;
                    break;
                }
            }
            if (goal != null) {
                Map<String, Object> ag = Json.obj(goal.get("agari"));
                ok("和牌帧 i=" + Json.i(goal.get("i"), 0) + "：winner=" + Json.i(ag.get("winner"), -1)
                        + " loser=" + (ag.get("loser") == null ? "null"
                        : String.valueOf(Json.i(ag.get("loser"), -1)))
                        + " tile=" + Json.str(ag.get("tile"), "") + " tsumo="
                        + Json.b(ag.get("tsumo"), false)
                        + (Json.b(ag.get("tsumo"), false) ? "" : "（荣和 → 铳牌要标红）"));
            } else {
                System.out.println("      · 这 100 帧里没有和牌帧，跳过铳牌信息检查");
            }
        } catch (Exception e) {
            bad("回放接口出错：" + e.getMessage());
        }
    }

    // ------------------------------------------------------------ 8 局间确认

    private static void step8NextHand(MajClient client) {
        section("8/9 局间确认：hand_result 字段齐全 + POST /api/next");
        try {
            State s = client.state(0);
            int seat = 0;
            for (int i = 0; i < s.seats.size(); i++) {
                if (s.isHumanSeat(i)) {
                    seat = i;
                    break;
                }
            }
            // 打到这一局结束（有 hand_result 为止）
            for (int guard = 0; guard < 80 && s.handResult == null && !s.finished; guard++) {
                if (s.pending != null && s.pending.yours && !s.pending.options.isEmpty()) {
                    client.act(s.pending.seat, s.pending.options.get(0).index);
                }
                Thread.sleep(350);
                s = client.state(seat);
            }
            if (s.handResult == null) {
                bad("打了一局也没有 hand_result（awaiting_next=" + s.awaitingNext
                        + " finished=" + s.finished + "）");
                return;
            }
            State.RoundResult r = s.handResult;
            ok("hand_result：第 " + r.handNo + " 局 " + r.label + " kind=" + r.kind
                    + " 赢家=P" + r.winner + " 放铳=" + (r.loser < 0 ? "无（自摸/流局）"
                    : "P" + r.loser));
            boolean fieldsOk = r.isDraw()
                    ? !r.tenpai.isEmpty() && !r.deltas.isEmpty() && !r.scores.isEmpty()
                    : !r.yaku.isEmpty() && r.totalHan > 0 && r.fu > 0 && !r.deltas.isEmpty()
                      && !r.scores.isEmpty() && !r.winTile.isEmpty();
            if (fieldsOk) {
                ok(r.isDraw()
                        ? "流局字段齐全：tenpai=" + r.tenpai + " deltas=" + r.deltas
                        : "字段齐全：yaku " + r.yaku.size() + " 个 · " + r.totalHan + " 番 "
                          + r.fu + " 符（宝牌 " + r.doraHan + " 番）· 和牌张 " + r.winTile
                          + " · deltas=" + r.deltas);
            } else {
                bad("hand_result 字段缺：yaku=" + r.yaku.size() + " total_han=" + r.totalHan
                        + " fu=" + r.fu + " win_tile=" + r.winTile + " deltas=" + r.deltas);
            }
            System.out.println("      awaiting_next=" + s.awaitingNext + " confirm_between="
                    + s.confirmBetween + " last_hand_note=" + s.lastHandNote);
            if (s.awaitingNext) {
                Map<String, Object> nr = client.nextHand();
                Thread.sleep(600);
                State after = client.state(seat);
                if (!after.awaitingNext) {
                    ok("POST /api/next → " + nr + "，之后 awaiting_next=false（牌局继续）");
                } else {
                    bad("POST /api/next 之后 awaiting_next 还是 true");
                }
            } else {
                System.out.println("      · 这张桌子是全 AI（不用等确认），next 用不上，跳过");
            }
        } catch (Exception e) {
            bad("局间确认检查出错：" + e.getMessage());
        }
    }

    // ------------------------------------------------------------ 9 reveal

    private static void step9Reveal(MajClient client) {
        section("9/9 观战全亮：reveal=1 只能在 spectator_ok=true 时拿到别家手牌");
        try {
            State plain = client.state(0);
            State rev = client.state(0, true);
            boolean leak = false;
            for (int i = 0; i < rev.details.size(); i++) {
                if (i != rev.you && rev.details.get(i).hand != null) {
                    leak = true;
                }
            }
            System.out.println("      spectator_ok=" + plain.spectatorOk + "（桌上有没有人）");
            if (plain.spectatorOk) {
                if (!leak) {
                    ok("这张桌子是全 AI：reveal=1 → " + rev.details.size()
                            + " 家手牌全亮（观战看得出来）");
                } else {
                    bad("spectator_ok=true 但 reveal=1 没给别家手牌");
                }
            } else {
                if (!leak) {
                    ok("桌上有真人：reveal=1 也拿不到别家手牌（后端防作弊生效，没被绕过）");
                } else {
                    bad("有人类在座时 reveal=1 泄露了别家手牌！");
                }
                ok("自己的手牌照常可见（" + (rev.myHand().size()) + " 张）");
            }
        } catch (Exception e) {
            bad("reveal 检查出错：" + e.getMessage());
        }
    }

    // ------------------------------------------------------------ 小工具

    /** 从原始 JSON 里抠一个整数（只为了证明字段类型，不依赖解析器）。 */
    private static int rawInt(String json, String key) {
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\"" + key + "\"\\s*:\\s*(-?\\d+)").matcher(json);
        return m.find() ? Integer.parseInt(m.group(1)) : -1;
    }

    /** 数组字段有几项（-1 表示不是数组）。 */
    private static int rawListLen(String json, String key) {
        int i = json.indexOf("\"" + key + "\"");
        if (i < 0) {
            return -1;
        }
        int colon = json.indexOf(':', i);
        int p = colon + 1;
        while (p < json.length() && Character.isWhitespace(json.charAt(p))) {
            p++;
        }
        if (p >= json.length() || json.charAt(p) != '[') {
            return -1;
        }
        int depth = 0;
        int items = 0;
        boolean any = false;
        for (int k = p; k < json.length(); k++) {
            char c = json.charAt(k);
            if (c == '[' || c == '{') {
                depth++;
                if (depth == 1 || (depth == 2 && c == '{')) {
                    if (c == '{') {
                        items++;
                    }
                }
            } else if (c == ']' || c == '}') {
                depth--;
                if (depth == 0) {
                    return items;
                }
            } else if (depth == 1 && !Character.isWhitespace(c) && c != ',') {
                any = true;
            }
        }
        return any ? items : 0;
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("── " + title);
    }

    private static void ok(String msg) {
        passed++;
        System.out.println("      ✓ " + msg);
    }

    private static void bad(String msg) {
        failed++;
        System.out.println("      ✗ " + msg);
    }
}
