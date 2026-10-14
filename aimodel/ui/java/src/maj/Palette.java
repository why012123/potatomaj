package maj;

import java.awt.Color;
import java.awt.Font;
import java.awt.GraphicsEnvironment;
import java.awt.image.BufferedImage;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.Set;

/** 配色、字体、桌布纹理：深木框 + 墨绿绒布 + 米白牌面。 */
public final class Palette {

    // 桌框（木头）
    public static final Color WOOD_HI = new Color(0x5A, 0x40, 0x2C);
    public static final Color WOOD = new Color(0x3A, 0x2A, 0x1E);
    public static final Color WOOD_LO = new Color(0x1E, 0x14, 0x0E);

    // 桌布
    public static final Color FELT_HI = new Color(0x1F, 0x6E, 0x51);
    public static final Color FELT = new Color(0x17, 0x55, 0x3E);
    public static final Color FELT_LO = new Color(0x0C, 0x35, 0x27);

    // 面板
    public static final Color PANEL = new Color(0x16, 0x1C, 0x1A);
    public static final Color PANEL_SOFT = new Color(0x1E, 0x28, 0x24);
    public static final Color LINE = new Color(0x33, 0x51, 0x45);

    // 文字
    public static final Color INK = new Color(0xF2, 0xF0, 0xE8);
    public static final Color DIM = new Color(0xA3, 0xB6, 0xAC);
    public static final Color GOLD = new Color(0xE8, 0xC4, 0x6A);
    public static final Color RED = new Color(0xE0, 0x57, 0x4A);
    public static final Color BLUE = new Color(0x86, 0xBC, 0xEC);
    public static final Color GREEN = new Color(0x6A, 0xCE, 0x96);

    // 牌
    public static final Color TILE_FACE = new Color(0xE6, 0xE2, 0xD8);
    public static final Color TILE_EDGE = new Color(0xBF, 0xB9, 0xAB);
    public static final Color TILE_BACK = new Color(0x2C, 0x62, 0x96);
    public static final Color TILE_BACK_LO = new Color(0x18, 0x3C, 0x60);
    public static final Color TILE_BACK_EDGE = new Color(0x0E, 0x24, 0x3A);
    public static final Color SHADOW = new Color(0, 0, 0, 105);
    public static final Color PLATE = new Color(0x0B, 0x22, 0x1A, 210);

    private static final String FAMILY = pickFamily();
    private static BufferedImage noise;

    private Palette() {
    }

    public static Font font(int size) {
        return new Font(FAMILY, Font.PLAIN, size);
    }

    public static Font bold(int size) {
        return new Font(FAMILY, Font.BOLD, size);
    }

    /** 一小块可平铺的噪点，用来给桌布加质感（只生成一次）。 */
    public static BufferedImage noiseTile() {
        if (noise == null) {
            int n = 64;
            BufferedImage img = new BufferedImage(n, n, BufferedImage.TYPE_INT_ARGB);
            Random rnd = new Random(20261013L);
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    int v = rnd.nextInt(255);
                    int a = 3 + rnd.nextInt(6);          // 很淡的一层绒布颗粒
                    img.setRGB(x, y, (v << 24) | (a << 16) | (a << 8) | a);
                }
            }
            noise = img;
        }
        return noise;
    }

    private static String pickFamily() {
        Set<String> have = new LinkedHashSet<>();
        try {
            for (String f : GraphicsEnvironment.getLocalGraphicsEnvironment()
                    .getAvailableFontFamilyNames()) {
                have.add(f);
            }
        } catch (RuntimeException | LinkageError ignored) {
            // 无头 / 拿不到字体列表：用逻辑字体
        }
        for (String want : new String[] {"PingFang SC", "Hiragino Sans GB", "Heiti SC",
                "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei"}) {
            if (have.contains(want)) {
                return want;
            }
        }
        return Font.SANS_SERIF;
    }
}
