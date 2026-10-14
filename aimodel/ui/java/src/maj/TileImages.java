package maj;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.SwingUtilities;

/**
 * 牌图缓存：把 {@code /tiles/<id>.png} 拉下来，按几档尺寸 × 正放/横放
 * 全部缩放好缓存起来。
 *
 * <p>下载和解码都在后台线程做，EDT 上只取现成的 {@link ImageIcon}；
 * 没加载好的牌返回 null，绘制方画占位牌 —— 服务没起也不会白屏。
 * 磁盘缓存放在 {@code java/.tilecache/}。
 */
public final class TileImages {

    /**
     * 牌图尺寸（竖放时的宽高）。三档：L 大窗口 / M 中等 / S 小窗口，
     * 由 {@link TableLayout} 按窗口高度挑一档，保证「手牌 + 副露 + 牌河」
     * 在任何尺寸下都排得下、不重叠。
     */
    public enum Size {
        /** 手牌（四家一样大，俯视看过去比例统一）。 */
        HAND(52, 84),
        /** 副露 / 宝牌。 */
        MED(40, 65),
        /** 牌河。 */
        POND(28, 45),
        /** 徽标位。 */
        MINI(22, 36),
        /** 中窗口：手牌 / 副露 / 牌河 / 徽标。 */
        HAND_M(44, 71),
        MED_M(34, 55),
        POND_M(24, 39),
        MINI_M(19, 30),
        /** 小窗口。 */
        HAND_S(36, 58),
        MED_S(28, 45),
        POND_S(20, 32),
        MINI_S(16, 26),
        /** 很窄的窗口（比如 1000×700 的三麻，两家并排要放得下 14 张）。 */
        HAND_XS(31, 50),
        MED_XS(24, 38),
        POND_XS(17, 27),
        MINI_XS(13, 21);

        public final int w;
        public final int h;

        Size(int w, int h) {
            this.w = w;
            this.h = h;
        }

        /** 横放（左右家）时的宽高。 */
        public int w(boolean rotated) {
            return rotated ? h : w;
        }

        public int h(boolean rotated) {
            return rotated ? w : h;
        }

        /** 小一档（L → M → S → XS，退到底就返回自己）。 */
        public Size smaller() {
            return switch (this) {
                case HAND -> HAND_M;
                case HAND_M -> HAND_S;
                case HAND_S -> HAND_XS;
                case HAND_XS -> HAND_XS;
                case MED -> MED_M;
                case MED_M -> MED_S;
                case MED_S -> MED_XS;
                case MED_XS -> MED_XS;
                case POND -> POND_M;
                case POND_M -> POND_S;
                case POND_S -> POND_XS;
                case POND_XS -> POND_XS;
                case MINI -> MINI_M;
                case MINI_M -> MINI_S;
                case MINI_S -> MINI_XS;
                case MINI_XS -> MINI_XS;
            };
        }
    }

    private final MajClient client;
    private final Path cacheDir;
    private final ExecutorService pool;
    /** key = id，value[8]：4 档尺寸 × 正/横。 */
    private final Map<String, ImageIcon[]> ready = new ConcurrentHashMap<>();
    /** 原图（按需缩放用）。 */
    private final Map<String, BufferedImage> source = new ConcurrentHashMap<>();
    private final Set<String> inflight = ConcurrentHashMap.newKeySet();
    private final AtomicBoolean repaintQueued = new AtomicBoolean();
    private volatile Runnable onLoaded = () -> { };

    public TileImages(MajClient client, Path cacheDir) {
        this.client = client;
        this.cacheDir = cacheDir;
        this.pool = Executors.newFixedThreadPool(4, r -> {
            Thread t = new Thread(r, "tile-loader");
            t.setDaemon(true);
            return t;
        });
    }

    private static int key(Size size, boolean rotated) {
        return size.ordinal() * 2 + (rotated ? 1 : 0);
    }

    /** 加载完一批牌图后回调（在 EDT 上跑，用来重绘）。 */
    public void setOnLoaded(Runnable r) {
        this.onLoaded = r == null ? () -> { } : r;
    }

    /** 拿一张牌图；还没加载好返回 null。某个尺寸第一次用到时现缩一张（有缓存）。 */
    public ImageIcon get(String id, Size size, boolean rotated) {
        if (id == null) {
            return null;
        }
        ImageIcon[] arr = ready.get(id);
        if (arr == null) {
            return null;
        }
        int k = key(size, rotated);
        ImageIcon icon = arr[k];
        if (icon != null) {
            return icon;
        }
        BufferedImage src = source.get(id);
        if (src == null) {
            return null;
        }
        synchronized (arr) {
            if (arr[k] == null) {
                BufferedImage scaled = scale(src, size.w, size.h);
                arr[k] = new ImageIcon(rotated ? rotate(scaled) : scaled);
            }
            return arr[k];
        }
    }

    public ImageIcon get(String id, Size size) {
        return get(id, size, false);
    }

    public boolean isReady(String id) {
        return ready.containsKey(id);
    }

    public int readyCount() {
        return ready.size();
    }

    /** 请求一批牌图（已缓存的跳过，失败的下次还能再试）。 */
    public void preload(Collection<String> ids) {
        for (String id : ids) {
            if (id == null || !Tile.isValid(id) || ready.containsKey(id)) {
                continue;
            }
            if (!inflight.add(id)) {
                continue;
            }
            pool.submit(() -> load(id));
        }
    }

    /** 关窗口时收工。 */
    public void shutdown() {
        pool.shutdownNow();
    }

    // ------------------------------------------------------------ 内部

    private void load(String id) {
        try {
            byte[] data = fromDisk(id);
            if (data == null) {
                data = client.tilePng(id);
                toDisk(id, data);
            }
            BufferedImage src = ImageIO.read(new ByteArrayInputStream(data));
            if (src == null) {
                delete(id);   // 缓存坏了，删掉下次重来
                return;
            }
            ImageIcon[] arr = new ImageIcon[Size.values().length * 2];
            arr[0] = new ImageIcon(scale(src, Size.HAND.w, Size.HAND.h));  // 占位，标记已加载
            ready.put(id, arr);
            source.put(id, src);
            fireLoaded();
        } catch (MajException | IOException | RuntimeException e) {
            delete(id);   // 服务没起 / 图坏了都不该崩，画占位牌就行
        } finally {
            inflight.remove(id);
        }
    }

    private void fireLoaded() {
        if (repaintQueued.compareAndSet(false, true)) {
            SwingUtilities.invokeLater(() -> {
                repaintQueued.set(false);
                try {
                    onLoaded.run();
                } catch (RuntimeException ignored) {
                    // 回调炸了不影响加载
                }
            });
        }
    }

    private static BufferedImage scale(BufferedImage src, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = out.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.drawImage(src, 0, 0, w, h, null);
        } finally {
            g.dispose();
        }
        return out;
    }

    /** 顺时针 90°：左右两家的牌是横着躺的。 */
    private static BufferedImage rotate(BufferedImage src) {
        int w = src.getWidth();
        int h = src.getHeight();
        BufferedImage out = new BufferedImage(h, w, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = out.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.translate(h, 0);
            g.rotate(Math.PI / 2);
            g.drawImage(src, 0, 0, null);
        } finally {
            g.dispose();
        }
        return out;
    }

    private Path fileOf(String id) {
        return cacheDir.resolve(id + ".png");
    }

    private byte[] fromDisk(String id) {
        try {
            Path p = fileOf(id);
            if (Files.isReadable(p)) {
                byte[] b = Files.readAllBytes(p);
                if (b.length > 200) {
                    return b;
                }
            }
        } catch (IOException ignored) {
            // 缓存读不出来就当没有
        }
        return null;
    }

    private void toDisk(String id, byte[] data) {
        try {
            Files.createDirectories(cacheDir);
            Files.write(fileOf(id), data);
        } catch (IOException ignored) {
            // 写不进去也无所谓
        }
    }

    private void delete(String id) {
        try {
            Files.deleteIfExists(fileOf(id));
        } catch (IOException ignored) {
            // 删不掉就算了
        }
    }
}
