import { defineConfig, type Plugin } from "vite";

/**
 * 给 `index.html` 里引用的 js / css 追加 `?v=<构建时间戳>`。
 *
 * 产物是固定名（`app.js` / `asset.css`，不带内容哈希），牌桌又是长期开着的
 * 页面，所以必须有办法让浏览器丢掉旧的。加时间戳而不是开哈希命名的理由：
 * `index.html` 本身已经带 `Cache-Control: no-store`（见服务端 `index_handler`），
 * 它每次都会被重新取，所以只要它引用的 URL 变了，新产物就一定生效；
 * 而固定文件名对人肉排障、无头浏览器自检都省事得多（不用去翻新文件名）。
 */
function cacheBust(): Plugin {
  const stamp = Date.now().toString(36);
  return {
    name: "maj-cache-bust",
    enforce: "post",
    transformIndexHtml(html: string): string {
      return html
        .replace(/(src|href)="(\/app\/[^"?]+\.(?:js|css))"/g, `$1="$2?v=${stamp}"`);
    },
  };
}

/**
 * 前端构建配置。
 *
 * 三个关键点，都是被服务端的路由**钉死**的：
 *
 * 1. `base: "/app/"` —— 服务端 `add_static("/app/", STATIC/"app")`，页面本身是
 *    服务端渲染出来的 `index.html`，所以资源必须挂在 `/app/` 下；用相对路径
 *    在 `/room/XXXX` 这种多级路由上会解析成 `/room/app/xxx` 直接 404。
 * 2. `outDir: "../static/app"` —— 服务端只从这里读 `index.html` 和静态资源。
 * 3. `emptyOutDir: true` —— 不清理的话上一版带 hash 的旧产物会一直躺着，
 *    排查「为什么改了代码没生效」时会白掉很多头发。
 */
export default defineConfig({
  base: "/app/",
  plugins: [cacheBust()],
  build: {
    outDir: "../static/app",
    emptyOutDir: true,
    target: "es2020",
    sourcemap: false,
    cssCodeSplit: false,
    assetsDir: ".",
    rollupOptions: {
      output: {
        entryFileNames: "app.js",
        chunkFileNames: "[name].js",
        assetFileNames: "asset.[ext]",
      },
    },
  },
  server: {
    // 本地 `vite dev` 时把 API / WebSocket 透给后端，省得开两个端口调跨域
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8790", ws: true },
      "/api": { target: "http://127.0.0.1:8790", changeOrigin: true },
      "/tiles": { target: "http://127.0.0.1:8790", changeOrigin: true },
    },
  },
});
