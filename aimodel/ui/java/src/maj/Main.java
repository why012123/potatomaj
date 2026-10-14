package maj;

import java.awt.GraphicsEnvironment;
import java.net.URI;
import java.nio.file.Path;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;

/**
 * 程序入口：{@code java -cp out maj.Main [--url URL] [--port N] [--seat N] [--no-log]
 * [--no-server] [--start-page]}。
 *
 * <p>默认会由本进程托管后端（{@link Backend} 用 ProcessBuilder 起、拿着句柄、
 * 关窗口一起停）；{@code --no-server} 表示后端是外面起的，UI 只连不管。
 */
public final class Main {

    private Main() {
    }

    public static void main(String[] args) {
        String url = null;
        int port = 8765;
        int seat = 0;
        boolean showLog = true;
        boolean serverAllowed = true;
        boolean forceStartPage = false;
        String rootArg = null;

        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--url", "-u" -> {
                    if (i + 1 < args.length) {
                        url = args[++i];
                    }
                }
                case "--port", "-p" -> {
                    if (i + 1 < args.length) {
                        port = parseInt(args[++i], port);
                    }
                }
                case "--seat", "-s" -> {
                    if (i + 1 < args.length) {
                        seat = parseInt(args[++i], seat);
                    }
                }
                case "--root" -> {
                    if (i + 1 < args.length) {
                        rootArg = args[++i];
                    }
                }
                case "--no-log" -> showLog = false;
                case "--no-server" -> serverAllowed = false;
                case "--start-page" -> forceStartPage = true;
                case "--help", "-h" -> {
                    System.out.println("""
                            土豆麻将 Java 牌桌
                              --url URL     后端地址（默认 http://127.0.0.1:8765）
                              --port N      等价于 --url http://127.0.0.1:N
                              --seat N      以哪个座位为视角（默认 0）
                              --root DIR    项目根（默认自动往上找 aimodel/ui/server.py）
                              --no-server   后端是外面起的，UI 不去启动/停止它
                              --no-log      不显示右侧牌谱
                              --start-page  直接停在开局选择页
                            """);
                    return;
                }
                default -> {
                    if (a.startsWith("http")) {
                        url = a;
                    }
                }
            }
        }
        if (url == null) {
            url = "http://127.0.0.1:" + port;
        }
        if (isLocal(url)) {
            port = portOf(url, port);
        } else {
            serverAllowed = false;   // 远端后端不归我管
        }

        Thread.setDefaultUncaughtExceptionHandler((t, e) -> {
            System.err.println("[maj] " + t.getName() + " 出异常（界面继续跑）：" + e);
            e.printStackTrace();
        });

        if (GraphicsEnvironment.isHeadless()) {
            System.err.println("没有图形环境（headless），起不了界面。");
            System.err.println("想验证接口能不能用：java -cp out maj.SelfTest " + url);
            System.exit(2);
        }

        System.setProperty("apple.awt.application.name", "土豆麻将");
        System.setProperty("apple.laf.useScreenMenuBar", "true");
        System.setProperty("awt.useSystemAAFontSettings", "on");
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {
            // 系统外观用不了就用默认的
        }

        Path root = rootArg == null ? Backend.findProjectRoot() : Path.of(rootArg);
        Backend backend = null;
        if (serverAllowed && root != null) {
            backend = new Backend(root, port);
        }
        final MajClient client = new MajClient(url);
        final Backend be = backend;
        final int seatNo = seat;
        final boolean log = showLog;
        final boolean allowed = serverAllowed;
        final boolean startPage = forceStartPage;
        SwingUtilities.invokeLater(() -> {
            MainFrame frame = new MainFrame(client, be, allowed, seatNo, log);
            frame.setVisible(true);
            if (startPage) {
                frame.debugShowStart();
            }
        });
    }

    private static boolean isLocal(String url) {
        try {
            String host = URI.create(url).getHost();
            return host == null || host.equals("127.0.0.1") || host.equals("localhost")
                    || host.equals("::1");
        } catch (RuntimeException e) {
            return false;
        }
    }

    private static int portOf(String url, int def) {
        try {
            int p = URI.create(url).getPort();
            return p > 0 ? p : def;
        } catch (RuntimeException e) {
            return def;
        }
    }

    private static int parseInt(String s, int def) {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }
}
