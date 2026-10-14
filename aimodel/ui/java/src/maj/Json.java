package maj;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 极简 JSON 解析 / 生成器。
 *
 * <p>后端只吐 JSON，而项目禁止第三方依赖，所以这里手写一个够用的小解析器：
 * 支持对象 / 数组 / 字符串 / 数字 / true / false / null。整数解析成
 * {@link Long}，带小数点或指数的解析成 {@link Double}。解析失败抛
 * {@link JsonException}，调用方负责兜住 —— UI 不能因为一段坏 JSON 就崩。
 */
public final class Json {

    private final String src;
    private int pos;

    private Json(String src) {
        this.src = src;
    }

    /** 解析一段 JSON 文本。 */
    public static Object parse(String text) throws JsonException {
        if (text == null) {
            throw new JsonException("响应为空");
        }
        Json p = new Json(text);
        p.skipWs();
        Object v = p.value();
        p.skipWs();
        if (p.pos != p.src.length()) {
            throw new JsonException("JSON 尾部有多余内容（位置 " + p.pos + "）");
        }
        return v;
    }

    /** 生成一段 JSON 文本（请求体用）。 */
    public static String write(Object o) {
        StringBuilder sb = new StringBuilder();
        writeTo(sb, o);
        return sb.toString();
    }

    // ---------------------------------------------------------------- 写

    private static void writeTo(StringBuilder sb, Object o) {
        if (o == null) {
            sb.append("null");
        } else if (o instanceof String s) {
            escape(sb, s);
        } else if (o instanceof Boolean || o instanceof Integer || o instanceof Long) {
            sb.append(o);
        } else if (o instanceof Double || o instanceof Float) {
            double d = ((Number) o).doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                sb.append("0");
            } else {
                sb.append(trim(d));
            }
        } else if (o instanceof Number n) {
            sb.append(n);
        } else if (o instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) {
                    sb.append(',');
                }
                first = false;
                escape(sb, String.valueOf(e.getKey()));
                sb.append(':');
                writeTo(sb, e.getValue());
            }
            sb.append('}');
        } else if (o instanceof Iterable<?> it) {
            sb.append('[');
            boolean first = true;
            for (Object e : it) {
                if (!first) {
                    sb.append(',');
                }
                first = false;
                writeTo(sb, e);
            }
            sb.append(']');
        } else if (o.getClass().isArray()) {
            sb.append('[');
            int n = java.lang.reflect.Array.getLength(o);
            for (int i = 0; i < n; i++) {
                if (i > 0) {
                    sb.append(',');
                }
                writeTo(sb, java.lang.reflect.Array.get(o, i));
            }
            sb.append(']');
        } else {
            escape(sb, o.toString());
        }
    }

    private static String trim(double d) {
        if (d == Math.rint(d) && Math.abs(d) < 1e15) {
            return String.valueOf((long) d);
        }
        return String.valueOf(d);
    }

    private static void escape(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        sb.append('"');
    }

    // ---------------------------------------------------------------- 读

    private void skipWs() {
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos++;
            } else {
                break;
            }
        }
    }

    private char peek() throws JsonException {
        if (pos >= src.length()) {
            throw new JsonException("JSON 意外结束");
        }
        return src.charAt(pos);
    }

    private Object value() throws JsonException {
        char c = peek();
        return switch (c) {
            case '{' -> object();
            case '[' -> array();
            case '"' -> string();
            case 't' -> literal("true", Boolean.TRUE);
            case 'f' -> literal("false", Boolean.FALSE);
            case 'n' -> literal("null", null);
            default -> number();
        };
    }

    private Object literal(String word, Object result) throws JsonException {
        if (!src.startsWith(word, pos)) {
            throw new JsonException("非法字面量（位置 " + pos + "）");
        }
        pos += word.length();
        return result;
    }

    private Map<String, Object> object() throws JsonException {
        expect('{');
        Map<String, Object> map = new LinkedHashMap<>();
        skipWs();
        if (peek() == '}') {
            pos++;
            return map;
        }
        while (true) {
            skipWs();
            String key = string();
            skipWs();
            expect(':');
            skipWs();
            map.put(key, value());
            skipWs();
            char c = peek();
            if (c == ',') {
                pos++;
            } else if (c == '}') {
                pos++;
                return map;
            } else {
                throw new JsonException("对象里期望 , 或 }（位置 " + pos + "）");
            }
        }
    }

    private List<Object> array() throws JsonException {
        expect('[');
        List<Object> list = new ArrayList<>();
        skipWs();
        if (peek() == ']') {
            pos++;
            return list;
        }
        while (true) {
            skipWs();
            list.add(value());
            skipWs();
            char c = peek();
            if (c == ',') {
                pos++;
            } else if (c == ']') {
                pos++;
                return list;
            } else {
                throw new JsonException("数组里期望 , 或 ]（位置 " + pos + "）");
            }
        }
    }

    private String string() throws JsonException {
        expect('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            if (pos >= src.length()) {
                throw new JsonException("字符串没有收尾引号");
            }
            char c = src.charAt(pos++);
            if (c == '"') {
                return sb.toString();
            }
            if (c != '\\') {
                sb.append(c);
                continue;
            }
            if (pos >= src.length()) {
                throw new JsonException("转义符后意外结束");
            }
            char e = src.charAt(pos++);
            switch (e) {
                case '"' -> sb.append('"');
                case '\\' -> sb.append('\\');
                case '/' -> sb.append('/');
                case 'b' -> sb.append('\b');
                case 'f' -> sb.append('\f');
                case 'n' -> sb.append('\n');
                case 'r' -> sb.append('\r');
                case 't' -> sb.append('\t');
                case 'u' -> {
                    if (pos + 4 > src.length()) {
                        throw new JsonException("\\u 转义不完整");
                    }
                    sb.append((char) Integer.parseInt(src.substring(pos, pos + 4), 16));
                    pos += 4;
                }
                default -> throw new JsonException("未知转义 \\" + e);
            }
        }
    }

    private Object number() throws JsonException {
        int start = pos;
        if (peek() == '-' || peek() == '+') {
            pos++;
        }
        boolean fp = false;
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c >= '0' && c <= '9') {
                pos++;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                fp = fp || c == '.' || c == 'e' || c == 'E';
                pos++;
            } else {
                break;
            }
        }
        String tok = src.substring(start, pos);
        if (tok.isEmpty()) {
            throw new JsonException("这里不是合法 JSON 值（位置 " + start + "）");
        }
        try {
            return fp ? (Object) Double.valueOf(tok) : (Object) Long.valueOf(tok);
        } catch (NumberFormatException ex) {
            throw new JsonException("非法数字: " + tok);
        }
    }

    private void expect(char c) throws JsonException {
        if (pos >= src.length() || src.charAt(pos) != c) {
            throw new JsonException("期望 '" + c + "'（位置 " + pos + "）");
        }
        pos++;
    }

    // ------------------------------------------------------------ 取值助手

    /** 取一个对象；不是对象就返回空 Map（UI 宁愿画空白也不要崩）。 */
    public static Map<String, Object> obj(Object o) {
        if (o instanceof Map<?, ?> m) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                out.put(String.valueOf(e.getKey()), e.getValue());
            }
            return out;
        }
        return new LinkedHashMap<>();
    }

    /** 取一个数组；不是数组就返回空 List。 */
    public static List<Object> arr(Object o) {
        if (o instanceof List<?> l) {
            return new ArrayList<>(l);
        }
        return new ArrayList<>();
    }

    public static String str(Object o, String def) {
        if (o == null) {
            return def;
        }
        if (o instanceof String s) {
            return s;
        }
        return String.valueOf(o);
    }

    public static String str(Object o) {
        return str(o, "");
    }

    public static int i(Object o, int def) {
        if (o instanceof Number n) {
            return n.intValue();
        }
        if (o instanceof String s) {
            try {
                return (int) Double.parseDouble(s.trim());
            } catch (NumberFormatException ignored) {
                return def;
            }
        }
        return def;
    }

    public static double d(Object o, double def) {
        if (o instanceof Number n) {
            return n.doubleValue();
        }
        if (o instanceof String s) {
            try {
                return Double.parseDouble(s.trim());
            } catch (NumberFormatException ignored) {
                return def;
            }
        }
        return def;
    }

    public static boolean b(Object o, boolean def) {
        if (o instanceof Boolean bo) {
            return bo;
        }
        if (o instanceof String s) {
            return "true".equalsIgnoreCase(s.trim());
        }
        return def;
    }

    /** 从对象里取一个字符串数组。 */
    public static List<String> strList(Object o) {
        List<String> out = new ArrayList<>();
        for (Object e : arr(o)) {
            if (e != null) {
                out.add(String.valueOf(e));
            }
        }
        return out;
    }

    /** JSON 解析失败。 */
    public static final class JsonException extends Exception {
        private static final long serialVersionUID = 1L;

        public JsonException(String msg) {
            super(msg);
        }
    }
}
