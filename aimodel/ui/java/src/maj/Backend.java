package maj;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

/**
 * 后端进程托管：**UI 自己用 {@link ProcessBuilder} 起 Python 服务，并且拿着句柄**，
 * 窗口一关就把它带走（{@link #stop()}），另有菜单项可以随时手动停。
 *
 * <p>如果这个端口的服务是别人（用户自己）起的，我们只连不管 ——
 * {@link #owned()} 会是 false，界面上也会写清楚。
 */
public final class Backend {

    private final Path projectRoot;
    private final Path python;
    private final int port;
    private final Path logFile;
    private Process process;
    private boolean owned;
    private String lastMessage = "";

    public Backend(Path projectRoot, int port) {
        this.projectRoot = projectRoot;
        this.port = port;
        this.python = projectRoot.resolve(".venv/bin/python");
        this.logFile = projectRoot.resolve("build/java-ui-server-" + port + ".log");
    }

    /** 从当前目录往上找项目根（认得 aimodel/ui/server.py 的那一层）。 */
    public static Path findProjectRoot() {
        Path cur = Path.of("").toAbsolutePath();
        for (int i = 0; i < 6 && cur != null; i++) {
            if (Files.isReadable(cur.resolve("aimodel/ui/server.py"))) {
                return cur;
            }
            cur = cur.getParent();
        }
        // 退一步：aimodel/ui/java -> 项目根
        Path guess = Path.of("").toAbsolutePath().resolve("../../..").normalize();
        return Files.isReadable(guess.resolve("aimodel/ui/server.py")) ? guess : null;
    }

    public boolean owned() {
        return owned;
    }

    public boolean alive() {
        return process != null && process.isAlive();
    }

    public String describe() {
        if (owned && alive()) {
            return "本窗口启动的后端（PID " + process.pid() + "，端口 " + port + "）";
        }
        if (owned) {
            return "本窗口启动的后端已退出（端口 " + port + "）";
        }
        return "后端不是本窗口启动的（端口 " + port + "）";
    }

    public String lastMessage() {
        return lastMessage;
    }

    public Path logFile() {
        return logFile;
    }

    /** close 窗口时要不要带走它（外面起的就不动）。 */
    public boolean shouldStopOnExit(boolean userWants) {
        return owned && userWants;
    }

    /**
     * 起后端并等它就绪。
     *
     * @return 就绪返回 true
     */
    public synchronized boolean start(String seats, double delay, long seed, MajClient client) {
        if (!python.toFile().canExecute()) {
            lastMessage = "找不到 " + python + "，没法自动起后端";
            return false;
        }
        if (client != null && client.ping()) {
            owned = false;   // 已经有人在跑了，那就不动它
            lastMessage = "端口 " + port + " 上已经有后端在跑，直接连它";
            return true;
        }
        try {
            Files.createDirectories(logFile.getParent());
            ProcessBuilder pb = new ProcessBuilder(
                    python.toString(), "-m", "aimodel.ui.server",
                    "--seats", seats,
                    "--port", String.valueOf(port),
                    "--delay", String.valueOf(delay),
                    "--seed", String.valueOf(seed));
            pb.directory(projectRoot.toFile());
            pb.redirectErrorStream(true);
            pb.redirectOutput(ProcessBuilder.Redirect.appendTo(logFile.toFile()));
            process = pb.start();
            owned = true;
        } catch (IOException e) {
            lastMessage = "起后端失败：" + e.getMessage();
            owned = false;
            return false;
        }
        for (int i = 0; i < 60; i++) {
            if (client != null && client.ping()) {
                lastMessage = "后端已就绪（PID " + process.pid() + "）";
                return true;
            }
            if (!process.isAlive()) {
                lastMessage = "后端进程起来了又退了，看日志 " + logFile;
                owned = false;
                return false;
            }
            try {
                Thread.sleep(250);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        lastMessage = "后端 15 秒内没就绪，看日志 " + logFile;
        return false;
    }

    /** 停掉本窗口起的后端；外面起的不动。 */
    public synchronized boolean stop() {
        Process p = process;
        process = null;
        if (p == null) {
            lastMessage = "没有本窗口启动的后端需要停";
            return false;
        }
        boolean wasOwned = owned;
        owned = false;
        try {
            // 先收子孙进程（Python 起过子进程的话）
            p.descendants().forEach(ProcessHandle::destroy);
            p.destroy();
            if (!p.waitFor(3, TimeUnit.SECONDS)) {
                p.descendants().forEach(ProcessHandle::destroyForcibly);
                p.destroyForcibly();
                p.waitFor(3, TimeUnit.SECONDS);
            }
            lastMessage = "已停掉后端（PID " + p.pid() + "）";
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            p.destroyForcibly();
        }
        return wasOwned;
    }

    /** 等端口真的空出来（自检 / 关窗时用）。 */
    public boolean waitPortFree(MajClient client, Duration timeout) {
        long end = System.nanoTime() + timeout.toNanos();
        while (System.nanoTime() < end) {
            if (client == null || !client.ping()) {
                return true;
            }
            try {
                Thread.sleep(150);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return client == null || !client.ping();
    }

    /** 日志文件存在的话读最后几行（出问题给用户看）。 */
    public String tailLog(int lines) {
        try {
            if (!Files.isReadable(logFile)) {
                return "";
            }
            java.util.List<String> all = Files.readAllLines(logFile);
            int from = Math.max(0, all.size() - lines);
            return String.join("\n", all.subList(from, all.size()));
        } catch (IOException e) {
            return "";
        }
    }

    /** 项目根（自检里打印用）。 */
    public File root() {
        return projectRoot.toFile();
    }
}
