import { defineConfig } from "vite";

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
 *
 * 产物**不带 hash**（`app.js` / `asset.css`）是刻意的：`index.html` 由 Vite
 * 生成、自己引用正确的文件名，而人肉排障 / CDP 自检时固定路径省事得多。
 * 静态资源本来就带 `Cache-Control: no-store`（`index_handler`），不怕缓存。
 */
export default defineConfig({
  base: "/app/",
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
