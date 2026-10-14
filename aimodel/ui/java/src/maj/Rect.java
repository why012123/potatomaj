package maj;

/** 一个整数矩形（自己写一个，省得为了画个牌就把 java.awt 拖进来）。 */
public final class Rect {

    public final int x;
    public final int y;
    public final int w;
    public final int h;

    public Rect(int x, int y, int w, int h) {
        this.x = x;
        this.y = y;
        this.w = Math.max(0, w);
        this.h = Math.max(0, h);
    }

    public int right() {
        return x + w;
    }

    public int bottom() {
        return y + h;
    }

    public int centerX() {
        return x + w / 2;
    }

    public int centerY() {
        return y + h / 2;
    }

    public boolean contains(int px, int py) {
        return px >= x && px < right() && py >= y && py < bottom();
    }

    /** 判定时把范围放大几个像素，点牌更宽容。 */
    public boolean hits(int px, int py, int pad) {
        return px >= x - pad && px < right() + pad && py >= y - pad && py < bottom() + pad;
    }

    public Rect inset(int d) {
        return new Rect(x + d, y + d, w - 2 * d, h - 2 * d);
    }

    @Override
    public String toString() {
        return "Rect(" + x + "," + y + "," + w + "x" + h + ")";
    }
}
