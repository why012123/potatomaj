package maj;

/**
 * 牌 id 的小工具：{@code "1m"} / {@code "0p"}（赤 5） / {@code "7z"}（中）。
 *
 * <p>牌型下标沿用后端：0..8 = 1m..9m，9..17 = 1p..9p，18..26 = 1s..9s，
 * 27..33 = 东 南 西 北 白 发 中。赤 5 在后端是「5 的另一种颜色」，
 * 牌型下标还是 5，只是图片用 {@code 0m/0p/0s}。
 */
public final class Tile {

    /** 后端 TILE_IDS 的顺序，用来在 id 和下标之间换算。 */
    public static final String[] IDS = {
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
        "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
        "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
        "1z", "2z", "3z", "4z", "5z", "6z", "7z",
    };

    /** 所有可能出现过的图片 id（含赤 5），启动时全部预取。 */
    public static final String[] IMAGE_IDS = {
        "0m", "0p", "0s",
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
        "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
        "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
        "1z", "2z", "3z", "4z", "5z", "6z", "7z",
    };

    private static final String[] NUM_CN = {"一", "二", "三", "四", "五", "六", "七", "八", "九"};
    private static final String[] HONOR_CN = {"东", "南", "西", "北", "白", "发", "中"};
    private static final String[] HONOR_CN_FULL = {"东风", "南风", "西风", "北风", "白板", "发财", "红中"};

    private Tile() {
    }

    /** 牌 id 是否合法（形如 0m..9m / 0p..9p / 0s..9s / 1z..7z）。 */
    public static boolean isValid(String id) {
        if (id == null || id.length() != 2) {
            return false;
        }
        char n = id.charAt(0);
        char s = id.charAt(1);
        if (n < '0' || n > '9') {
            return false;
        }
        return switch (s) {
            case 'm', 'p', 's' -> n <= '9';
            case 'z' -> n >= '1' && n <= '7';
            default -> false;
        };
    }

    /** 是不是赤 5 的图片 id（0m / 0p / 0s）。 */
    public static boolean isRedId(String id) {
        return id != null && id.length() == 2 && id.charAt(0) == '0';
    }

    /** 把赤 5 图片 id 折回普通牌型 id：{@code 0p -> 5p}。 */
    public static String normalize(String id) {
        return isRedId(id) ? "5" + id.charAt(1) : id;
    }

    /** 5m / 5p / 5s 对应的赤 5 图片 id，其它牌返回 null。 */
    public static String redIdOf(String id) {
        if (id != null && id.length() == 2 && id.charAt(0) == '5'
                && (id.charAt(1) == 'm' || id.charAt(1) == 'p' || id.charAt(1) == 's')) {
            return "0" + id.charAt(1);
        }
        return null;
    }

    /** 花色序号：m=0 p=1 s=2 z=3，未知返回 9。 */
    public static int suit(String id) {
        if (id == null || id.length() != 2) {
            return 9;
        }
        return switch (id.charAt(1)) {
            case 'm' -> 0;
            case 'p' -> 1;
            case 's' -> 2;
            case 'z' -> 3;
            default -> 9;
        };
    }

    /** 点数 1..9（赤 5 算 5）；字牌 1..7。 */
    public static int number(String id) {
        if (id == null || id.length() != 2) {
            return 0;
        }
        char c = id.charAt(0);
        if (c < '0' || c > '9') {
            return 0;
        }
        return c == '0' ? 5 : c - '0';
    }

    /** 牌型下标 0..33；非法返回 -1。 */
    public static int index(String id) {
        String n = normalize(id);
        if (n.length() != 2) {
            return -1;
        }
        for (int i = 0; i < IDS.length; i++) {
            if (IDS[i].equals(n)) {
                return i;
            }
        }
        return -1;
    }

    /** 牌型下标 -> id；越界返回 "?"。 */
    public static String id(int index) {
        return index >= 0 && index < IDS.length ? IDS[index] : "?";
    }

    /** 中文牌名，例如 {@code 1m -> 一万}、{@code 0p -> 赤五筒}、{@code 7z -> 中}。 */
    public static String cn(String id) {
        if (id == null || id.length() != 2) {
            return "?";
        }
        int n = number(id);
        if (n == 0) {
            return "?";
        }
        boolean red = isRedId(id);
        char s = id.charAt(1);
        if (s == 'z') {
            String hon = n <= HONOR_CN.length ? HONOR_CN[n - 1] : "?";
            return red ? "赤" + hon : hon;
        }
        String num = NUM_CN[n - 1];
        String suit = switch (s) {
            case 'm' -> "万";
            case 'p' -> "筒";
            case 's' -> "索";
            default -> "?";
        };
        return (red ? "赤" : "") + num + suit;
    }

    /** 完整中文牌名（含「风」字），给宝牌 / 场风用。 */
    public static String cnFull(String id) {
        if (id != null && id.length() == 2 && id.charAt(1) == 'z') {
            int n = number(id);
            if (n >= 1 && n <= HONOR_CN_FULL.length) {
                return HONOR_CN_FULL[n - 1];
            }
        }
        return cn(id);
    }

    /** 排序键：先花色（万筒索字），再点数，赤 5 排在普通 5 后面。 */
    public static int sortKey(String id) {
        return suit(id) * 100 + number(id) * 2 + (isRedId(id) ? 1 : 0);
    }
}
