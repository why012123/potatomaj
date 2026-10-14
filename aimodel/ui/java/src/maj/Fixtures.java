package maj;

/**
 * 出图 / 自检用的合成局面（字段跟 {@code /api/state} 一模一样，不联网也能画）。
 *
 * <p>这些数据是「布局铁律」的测试夹具：满副露 + 长牌河 + 满手牌，
 * 也就是最容易把四块矩形挤重叠的那种局面。
 */
public final class Fixtures {

    private Fixtures() {
    }

    /** 牌河素材（循环取用，够长就行）。 */
    private static final String[] POND_POOL = {
        "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z",
        "2z", "3z", "4z", "5z", "6z", "7z", "9m", "1p", "2s", "3p",
        "2m", "5m", "8m", "3s",
    };

    /** 手牌素材（14 张，带对子 / 搭子）。 */
    private static final String[] HAND_POOL = {
        "1m", "1m", "9m", "1p", "2p", "3p", "5s", "6s", "7s", "2z",
        "2z", "7z", "7z", "5z",
    };

    /**
     * 三 / 四麻通用的极端局面：每家 {@code melds} 组副露 + {@code pond} 张牌河 + 14 张手牌。
     *
     * <p>跟后端一样：<b>只有主视角那家给 {@code hand}</b>，别家只给 {@code hand_count}
     * （所以画出来是牌背）；{@code reveal=true} 才是观战 / 回放那种全亮。
     */
    public static State stress(int players, int pond, int melds) throws Json.JsonException {
        return stress(players, pond, melds, false);
    }

    /** @param reveal true = 全亮（观战 / 回放），每家的手牌都给 */
    public static State stress(int players, int pond, int melds, boolean reveal)
            throws Json.JsonException {
        State s = State.fromJson(stressJson(players, pond, melds, reveal));
        s.revealAll = reveal;
        return s;
    }

    /** 默认压测局面：每家 4 组副露（吃 / 碰 / 暗杠 / 加杠）+ 20 张牌河 + 14 张手牌。 */
    public static State stress(int players) throws Json.JsonException {
        return stress(players, 20, 4);
    }

    /** 空桌：刚发完牌，没有副露也没有牌河。 */
    public static State fresh(int players) throws Json.JsonException {
        return stress(players, 0, 0);
    }

    /** 合成的流局 {@code hand_result}（截图用）。 */
    public static State draw() throws Json.JsonException {
        return State.fromJson(drawJson());
    }

    /** 合成的荣和 {@code hand_result}（截图 / 结算页自检用）。 */
    public static State win() throws Json.JsonException {
        return State.fromJson(winJson());
    }

    // ------------------------------------------------------------ JSON

    private static String stressJson(int players, int pond, int melds, boolean reveal) {
        int handCount = HAND_POOL.length;
        StringBuilder sb = new StringBuilder();
        sb.append("{\"player_count\":").append(players)
                .append(",\"you\":0,\"wall_left\":12,\"turn\":16,\"dealer\":0,")
                .append("\"hand_no\":2,\"round_wind\":\"东\",\"dora\":[\"5s\",\"1m\"],")
                .append("\"seats\":[");
        for (int i = 0; i < players; i++) {
            sb.append(i > 0 ? "," : "")
                    .append("{\"index\":").append(i)
                    .append(",\"type\":\"teacher\",\"name\":\"P").append(i)
                    .append("（压测）\"}");
        }
        sb.append("],\"players_detail\":[");
        String[] winds = {"东", "南", "西", "北"};
        String[][] meldPool = {
            {"chi", "1p", "吃一筒二筒三筒"},
            {"pon", "9p", "碰九筒"},
            {"ankan", "3s", "暗杠三索"},
            {"kakan", "5z", "加杠白"},
        };
        for (int i = 0; i < players; i++) {
            sb.append(i > 0 ? "," : "").append("{\"score\":").append(25000 + i * 1000)
                    .append(",\"pond\":[");
            for (int k = 0; k < pond; k++) {
                sb.append(k > 0 ? "," : "").append('"')
                        .append(POND_POOL[k % POND_POOL.length]).append('"');
            }
            sb.append("],\"melds\":[");
            for (int k = 0; k < melds && k < meldPool.length; k++) {
                sb.append(k > 0 ? "," : "").append("{\"kind\":\"").append(meldPool[k][0])
                        .append("\",\"tile\":\"").append(meldPool[k][1])
                        .append("\",\"label\":\"").append(meldPool[k][2]).append("\"}");
            }
            sb.append("],\"kita\":1,\"riichi\":true,\"ippatsu\":false,\"is_dealer\":")
                    .append(i == 0).append(",\"seat_wind\":\"")
                    .append(winds[i % 4]).append("\",\"hand_count\":").append(handCount);
            // 后端只在「你能看这家牌」的时候才给 hand（主视角 / reveal=1）
            if (i == 0 || reveal) {
                sb.append(",\"hand\":[");
                for (int k = 0; k < handCount; k++) {
                    sb.append(k > 0 ? "," : "").append('"').append(HAND_POOL[k]).append('"');
                }
                sb.append("],\"red\":[\"5p\"]}");
            } else {
                sb.append(",\"hand\":null,\"red\":null}");
            }
        }
        sb.append("],\"log\":[],\"pending\":null,\"hint\":{\"shanten\":2,\"ukeire\":31,")
                .append("\"menzen\":false}}");
        return sb.toString();
    }

    private static String drawJson() {
        return "{\"player_count\":3,\"you\":0,\"wall_left\":0,\"turn\":18,\"dealer\":0,"
                + "\"hand_no\":2,\"round_wind\":\"东\",\"dora\":[\"3s\"],"
                + "\"awaiting_next\":true,\"confirm_between\":true,"
                + "\"last_hand_note\":\"第 2 局结束（流局）\",\"spectator_ok\":false,"
                + "\"seats\":[{\"index\":0,\"type\":\"human\",\"name\":\"你（P0）\"},"
                + "{\"index\":1,\"type\":\"maj-flash\",\"name\":\"脚本·maj-flash（P1）\"},"
                + "{\"index\":2,\"type\":\"maj-huge\",\"name\":\"脚本·maj-huge（P2）\"}],"
                + "\"players_detail\":[{\"score\":28000,\"pond\":[],\"melds\":[],\"kita\":0,"
                + "\"riichi\":false,\"ippatsu\":false,\"is_dealer\":true,\"seat_wind\":\"东\","
                + "\"hand_count\":13,\"hand\":null,\"red\":null},"
                + "{\"score\":42000,\"pond\":[],\"melds\":[],\"kita\":0,\"riichi\":false,"
                + "\"ippatsu\":false,\"is_dealer\":false,\"seat_wind\":\"南\","
                + "\"hand_count\":13,\"hand\":null,\"red\":null},"
                + "{\"score\":35000,\"pond\":[],\"melds\":[],\"kita\":0,\"riichi\":false,"
                + "\"ippatsu\":false,\"is_dealer\":false,\"seat_wind\":\"西\","
                + "\"hand_count\":13,\"hand\":null,\"red\":null}],"
                + "\"log\":[],\"pending\":null,"
                + "\"hand_result\":{\"hand_no\":2,\"kind\":\"draw\",\"label\":\"流局\","
                + "\"turns\":18,\"players\":3,\"tenpai\":[true,false,true],"
                + "\"deltas\":[1500,-3000,1500],\"scores\":[28000,42000,35000],"
                + "\"sticks\":0,\"notes\":\"荒牌流局\",\"describe\":\"流局\"}}";
    }

    private static String winJson() {
        return "{\"player_count\":3,\"you\":0,\"wall_left\":0,\"turn\":11,\"dealer\":0,"
                + "\"hand_no\":3,\"round_wind\":\"东\",\"dora\":[\"3s\",\"1z\"],"
                + "\"awaiting_next\":true,\"confirm_between\":true,"
                + "\"last_hand_note\":\"第 3 局结束（荣和）\",\"spectator_ok\":false,"
                + "\"seats\":[{\"index\":0,\"type\":\"human\",\"name\":\"你（P0）\"},"
                + "{\"index\":1,\"type\":\"maj-normal\",\"name\":\"脚本·maj-normal（P1）\"},"
                + "{\"index\":2,\"type\":\"maj-flash\",\"name\":\"脚本·maj-flash（P2）\"}],"
                + "\"players_detail\":[{\"score\":35000,\"pond\":[],\"melds\":[],\"kita\":0,"
                + "\"riichi\":false,\"ippatsu\":false,\"is_dealer\":true,\"seat_wind\":\"东\","
                + "\"hand_count\":13,\"hand\":null,\"red\":null},"
                + "{\"score\":27800,\"pond\":[],\"melds\":[],\"kita\":0,\"riichi\":false,"
                + "\"ippatsu\":false,\"is_dealer\":false,\"seat_wind\":\"南\","
                + "\"hand_count\":13,\"hand\":null,\"red\":null},"
                + "{\"score\":30600,\"pond\":[],\"melds\":[],\"kita\":0,\"riichi\":false,"
                + "\"ippatsu\":false,\"is_dealer\":false,\"seat_wind\":\"西\","
                + "\"hand_count\":13,\"hand\":null,\"red\":null}],"
                + "\"log\":[],\"pending\":null,"
                + "\"hand_result\":{\"hand_no\":3,\"kind\":\"ron\",\"label\":\"荣和\","
                + "\"tsumo\":false,\"winner\":0,\"loser\":1,\"players\":3,\"turns\":11,"
                + "\"sticks\":0,\"han\":2,\"dora_han\":1,\"total_han\":3,\"fu\":40,"
                + "\"yakuman\":0,\"points\":5200,\"win_tile\":\"3p\","
                + "\"winners\":[0],\"deltas\":[5200,-5200,0],"
                + "\"scores\":[40200,27800,30600],"
                + "\"hand\":[\"1m\",\"1m\",\"9m\",\"1p\",\"2p\",\"5s\",\"6s\",\"7s\","
                + "\"2z\",\"2z\"],"
                + "\"melds\":[{\"kind\":\"chi\",\"tile\":\"7p\",\"label\":\"吃七筒八筒九筒\"}],"
                + "\"yaku\":[{\"name\":\"立直\",\"key\":\"riichi\",\"han\":1},"
                + "{\"name\":\"役牌 白\",\"key\":\"yakuhai\",\"han\":1},"
                + "{\"name\":\"宝牌\",\"key\":\"dora\",\"han\":1}],"
                + "\"tenpai\":[true,false,false],"
                + "\"notes\":\"荣和\",\"describe\":\"荣和\"}}";
    }
}
