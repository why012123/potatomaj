package maj;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * 薄薄一层 HTTP：只用 JDK 自带的 {@link HttpClient}，没有第三方依赖。
 *
 * <p>这个类不碰 Swing，后台线程随便调。**非 2xx 也算正常返回**
 * （{@link Response#status} 里带着状态码），因为后端用 400 + JSON
 * {@code {"ok":false,"error":...}} 表达「下标越界」这类业务错误，
 * 那时我们要读 body 而不是抛异常。连不上 / 超时才抛 {@link MajException}。
 */
public final class Http {

    /** 一次 HTTP 返回：状态码 + body。 */
    public record Response(int status, String body) {
        public boolean ok2xx() {
            return status >= 200 && status < 300;
        }
    }

    private static final HttpClient CLIENT = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            .version(HttpClient.Version.HTTP_1_1)
            .followRedirects(HttpClient.Redirect.NEVER)
            .proxy(HttpClient.Builder.NO_PROXY)
            .build();

    private Http() {
    }

    /** GET（不关心状态码）。 */
    public static Response get(String url, Duration timeout) throws MajException {
        return send(request(url, timeout).GET().build());
    }

    /** GET，而且要求 2xx。 */
    public static Response getChecked(String url, Duration timeout) throws MajException {
        Response r = get(url, timeout);
        if (!r.ok2xx()) {
            throw new MajException("HTTP " + r.status() + " " + path(url)
                    + (r.body().isEmpty() ? "" : "：" + shorten(r.body())));
        }
        return r;
    }

    /** GET 一段二进制（牌图用）。 */
    public static byte[] getBytes(String url, Duration timeout) throws MajException {
        HttpRequest req = request(url, timeout).GET().build();
        try {
            HttpResponse<byte[]> resp = CLIENT.send(req, HttpResponse.BodyHandlers.ofByteArray());
            if (resp.statusCode() < 200 || resp.statusCode() >= 300) {
                throw new MajException("HTTP " + resp.statusCode() + " " + path(url));
            }
            return resp.body();
        } catch (IOException e) {
            throw new MajException(transportMessage(e), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new MajException("请求被中断", e);
        }
    }

    /** GET 一段文本（UTF-8，要求 2xx）。 */
    public static String getText(String url, Duration timeout) throws MajException {
        return getChecked(url, timeout).body();
    }

    /** POST 一段 JSON（不关心状态码，body 一律返回）。 */
    public static Response postJson(String url, String json, Duration timeout)
            throws MajException {
        HttpRequest req = request(url, timeout)
                .header("Content-Type", "application/json; charset=utf-8")
                .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                .build();
        return send(req);
    }

    // ------------------------------------------------------------ 内部

    private static HttpRequest.Builder request(String url, Duration timeout) {
        return HttpRequest.newBuilder(URI.create(url)).timeout(timeout);
    }

    private static Response send(HttpRequest req) throws MajException {
        try {
            HttpResponse<byte[]> resp = CLIENT.send(req, HttpResponse.BodyHandlers.ofByteArray());
            return new Response(resp.statusCode(), new String(resp.body(), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new MajException(transportMessage(e), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new MajException("请求被中断", e);
        }
    }

    private static String transportMessage(IOException e) {
        String m = e.getMessage() == null ? "" : ": " + e.getMessage();
        return "连不上后端（" + e.getClass().getSimpleName() + m + "）";
    }

    private static String path(String url) {
        try {
            String p = URI.create(url).getPath();
            return p == null ? url : p;
        } catch (RuntimeException e) {
            return url;
        }
    }

    private static String shorten(String s) {
        String one = s.replace('\n', ' ').trim();
        return one.length() > 200 ? one.substring(0, 200) + "…" : one;
    }
}
