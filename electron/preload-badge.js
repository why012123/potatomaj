// 角标窗口的 preload。
//
// 关键：**页面脚本里没有 require**（nodeIntegration 默认关闭），所以
// IPC 必须在 preload 里注册，再用 window 上的接口 / 自定义事件交给页面。
// 之前把 require('electron') 写在 badge.html 的 <script> 里，直接抛异常，
// 监听器一个都没注册上 —— 表现就是角标永远显示 0/0、手牌一直是「—」。
const { ipcRenderer } = require('electron');

const fire = (name, detail) => window.dispatchEvent(new CustomEvent(name, { detail }));

ipcRenderer.on('stats', (_e, s) => fire('pm-stats', s));
ipcRenderer.on('gamestate', (_e, g) => fire('pm-gamestate', g));

window.addEventListener('DOMContentLoaded', () => {
  // 页面能用的接口都挂在这里
  window.potatomaj = {
    openDumpDir: () => ipcRenderer.invoke('open-dump-dir'),
    setSeat: (s) => ipcRenderer.invoke('set-seat', s),
    refresh: async () => {
      try {
        fire('pm-stats', await ipcRenderer.invoke('get-stats'));
        fire('pm-gamestate', await ipcRenderer.invoke('get-gamestate'));
      } catch (e) { /* 忽略 */ }
    },
  };
  window.potatomaj.refresh();
  // 注意：**不要**加定时兜底刷新。
  // 这个窗口是「透明 + 置顶」的，每秒重写 DOM 会让系统合成器不断重绘那一块，
  // 压在 WebGL 游戏上就表现为「一闪一闪」。改成纯事件驱动。
});
