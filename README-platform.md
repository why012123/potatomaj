# 土豆麻将 · 网页对战平台

在线地址：**<https://maj.xjtuwhy.com>**（香港 VPS 只做 nginx + frp 中转，
真正的服务和 AI 推理在本机 i9 上跑）

一套从零搭的日式立直麻将（Riichi）网页对战平台：房间制多人对战、
AI 对手 / 纯 AI 观战、天凤式牌桌、决策级数据记录、玩家画像与 ELO。

**规则内核原样复用仓库里的 `aimodel/`**（`sanma.py` 算役算符算点、
`engine.py` 跑对局），平台侧只做「服务层 + 前端 + 数据」——
见文末「对 `aimodel/` 的改动」。

---

## 一、一键命令

### 本机（服务 + 隧道一起管）

```powershell
pwsh deploy\maj.ps1 start        # 启动（run.py + frpc），带看门狗
pwsh deploy\maj.ps1 stop         # 停止
pwsh deploy\maj.ps1 restart      # 重启
pwsh deploy\maj.ps1 status       # 进程 / 端口 / 健康接口 / 公网可达性
pwsh deploy\maj.ps1 logs         # 跟着看日志
pwsh deploy\maj.ps1 watch        # 只跑看门狗（崩了自动重启）
```

不想用脚本也可以直接来：

```powershell
.\.venv\Scripts\python.exe run.py --host 127.0.0.1 --port 8790 --ai-delay 0.6
```

### 前端构建

```powershell
cd webapp\client
npm install
npx tsc --noEmit          # 类型检查（strict）
npx vite build            # 产物 -> webapp/static/app/
```

### 牌面素材重新生成（素材本身不入库，可重跑）

```powershell
npm pack riichi-mahjong-tiles --registry https://registry.npmmirror.com
.\.venv\Scripts\python.exe tools\extract_tiles.py --tgz riichi-mahjong-tiles-0.1.0.tgz
```

### 对局数据导出

```powershell
.\.venv\Scripts\python.exe tools\dump_games.py --list                # 有哪些分片
.\.venv\Scripts\python.exe tools\dump_games.py --stats               # 全局统计
.\.venv\Scripts\python.exe tools\dump_games.py --since 2026-10-01 --out games.jsonl
.\.venv\Scripts\python.exe tools\dump_games.py --show-profiles       # 玩家画像 / ELO 榜
```

### VPS 侧（nginx + TLS）

```powershell
psql -c ""            # （没有 psql，看下面）
pwsh deploy\deploy-vps.ps1          # 上传配置 + 申请/续期证书 + 公网验证
```

### 验收

```powershell
.\.venv\Scripts\python.exe aimodel\tests\run_all.py                   # 1 规则测试
.\.venv\Scripts\python.exe -m webapp.tests.isolation                  # 2 房间隔离
.\.venv\Scripts\python.exe -m webapp.tests.cheat                      # 3 防作弊
.\.venv\Scripts\python.exe -m webapp.tests.selfplay --games 200       # 4 AI 自对弈
.\.venv\Scripts\python.exe -m webapp.tests.loadtest --rooms 10        # 5 并发 + 6 重连
.\.venv\Scripts\python.exe -m webapp.tests.recording_check --games 10 # 7/8 数据记录
.\.venv\Scripts\python.exe -m webapp.tests.shots                      # 9 截图
```

---

## 二、目录结构

```
potatomaj/
├── run.py                        启动入口（就一行：调 webapp.server.app.main）
├── aimodel/                      ★ 规则内核 + AI（大部分是复用，见文末）
│   ├── sanma.py                  役 / 符 / 点数（原样复用）
│   ├── engine.py                 对局引擎（原样复用）
│   ├── agents.py                 HeuristicAgent / RandomAgent / MLAgent（复用）
│   ├── variants.py               模型注册表 + 权重自检（复用）
│   ├── net.py  encode.py         纯 numpy 网络与特征（复用）
│   └── weights/*.npz             25 个模型权重
├── webapp/
│   ├── server/                   服务层（本平台新写）
│   │   ├── app.py                aiohttp：HTTP + WebSocket + 静态资源
│   │   ├── table.py              ★ 一个房间 = 一个引擎线程 + 事件队列
│   │   ├── rooms.py              房间注册表 + 回收
│   │   ├── protocol.py           ★ 座位视角的 JSON 协议（服务端权威）
│   │   ├── agents.py             真人座位 / AI 座位适配（推理走线程池）
│   │   ├── recording.py          ★ 决策级记录 + 玩家画像 + ELO
│   │   ├── tiles.py              牌张换算
│   │   ├── config.py             运行参数
│   │   └── logging_setup.py      文件日志 + 轮转
│   ├── client/                   Vite + TypeScript 前端源码
│   │   └── src/{main,net,protocol}.ts + ui/{table,lobby,result,tiles}.ts
│   ├── static/
│   │   ├── app/                  前端构建产物
│   │   └── tiles/                37 张牌面 SVG（脚本生成）
│   └── tests/                    验收脚本
│       ├── bot.py                会自己打牌的测试客户端（后台读协程）
│       ├── cdp.py                无头浏览器驱动（原生 CDP，零额外依赖）
│       ├── smoke.py              最小闭环
│       ├── isolation.py          验收 2
│       ├── cheat.py              验收 3
│       ├── selfplay.py           验收 4
│       ├── loadtest.py           验收 5/6
│       ├── recording_check.py    验收 7/8
│       ├── shots.py              验收 9（截图）
│       └── tilepreview.py        牌面素材自检
├── tools/
│   ├── extract_tiles.py          从 CC0 素材生成牌面 SVG
│   ├── dump_games.py             对局数据导出 / 统计 / 画像
│   ├── jsonlgz.py                读 gzip JSONL（容忍截断）
│   └── ai_comm.py                协作频道客户端（读/写/长轮询）
├── deploy/
│   ├── maj.ps1                   ★ 一键启停 + 看门狗（Windows 本机）
│   ├── deploy-vps.ps1            VPS 部署（nginx + TLS + 证书）
│   ├── nginx-maj.conf            nginx 反代（含 WebSocket 升级）
│   ├── frpc-maj.toml             frp 隧道
│   ├── maj.service               systemd（开机自启 + 崩溃重启）
│   └── maj.logrotate             日志轮转
├── data/                         运行时：对局记录 + 玩家画像（gitignore）
└── logs/                         运行时：日志（gitignore）
```

---

## 三、调用链

```
浏览器
  │  HTTPS/WSS  maj.xjtuwhy.com
  ▼
VPS nginx ──反代──▶ frps :8080 ──frp 隧道──▶ 本机 frpc ──▶ 127.0.0.1:8790
                                                              │
                                                    ┌─────────┴──────────┐
                                                    ▼                    ▼
                                            aiohttp 事件循环        AI 推理线程池
                                            （WS 收发 / 房间 / 静态）   （默认 4 worker）
                                                    │                    ▲
                                        submit_action│                    │ 推理
                                        broadcast_later                  │
                                                    ▼                    │
                                        ┌───────────────────────┐        │
                                        │ 房间（一个引擎线程）    │────────┘
                                        │  Game(agents, ...)     │
                                        │   ├─ HumanSeat.decide  │◀── 挂住等人答
                                        │   └─ AISeat.decide     │
                                        └───────────────────────┘
                                                    │ on_action（结构化动作流）
                                                    ▼
                                        状态快照（**按座位裁剪**）→ 推送
                                                    │
                                                    ├─▶ 决策级记录（异步写 gzip JSONL）
                                                    └─▶ 玩家画像 + ELO
```

**关键设计**

| 主题 | 做法 | 为什么 |
|---|---|---|
| 服务端权威 | 客户端只发「我要打这张」，服务端拿**引擎给的合法选项**校验 | 前端藏牌等于没藏 |
| 隔离 | 一个房间一个 `Game` + 一条引擎线程 + 自己的事件序号；连接换房间时**先清空**上一条快照 | 房间 A 的数据不可能出现在 B 的消息里 |
| 座位视角 | 别家的 `hand` **在协议层就是 `null`**，只给张数 | 拿不到才算真的看不到 |
| 观战全亮 | 只有**纯 AI 房**允许；有真人在座服务端直接拒绝并说明原因 | 否则观众能念牌 |
| 不阻塞 | AI 推理丢线程池；节拍睡在引擎线程；每连接独立有界发送队列 | 慢客户端拖不住别人，模型卡住不冻结整桌 |
| 超时 | 真人 30s 不出牌 → 内置教师托管；AI 异常 → 降级并**在界面标注** | 桌子永远不卡死 |
| 广播合并 | 30fps 合并 + **自驱动重排**（窗口内又来请求就自己续一轮） | 引擎跑满速时不会刷爆客户端，链条也不会断 |
| 记录 | 异步写、失败只计数；gzip 按天分片；容量上限 + 轮转 | 记录绝不能影响、更不能搞崩对局 |
| ELO | 人和 AI **同一套公式**（成对比较、零和、同一个 K） | 人和模型分数在同一量纲上 |

---

## 四、对局数据（决策级）

每个决策点一条记录，**这才是能拿去训练的东西**：

```json
{"kind":"decision","hand_no":3,"seat":1,"player":"p_ab12cd34","kind_player":"human",
 "decision":"discard","target_tile":null,
 "options":[{"index":0,"tile":"2m","raw":1}, ...],   ← 引擎给的**完整**合法动作，原样
 "choice_index":7,"choice":{"index":7,"tile":"9p","raw":26},
 "elapsed_ms":1342,                                   ← 从轮到他到决定
 "profile":{"uid":"p_ab12cd34","games":37,"avg_rank":2.14,"first_rate":0.243,
            "deal_in_rate":0.121,"agari_rate":0.198,"elo":1623.4},
 "view":{"hand":[...],"ponds":[[...]],"melds_all":[...],"scores":[...],
         "dora_indicators":[...],"wall_left":42,"turn":9,"riichi":[...], ...}}
```

* 只有 `player` 是匿名随机串（`p_xxxxxxxx`）——**不含昵称 / 邮箱 / 手机号**
* 一局结束额外落一条 `hand`：**四家手牌全亮** + 和牌详情（役种、番符、收支、
  谁放铳、铳的是哪张）→ 危险度模型的正样本
* 一场结束落 `match`：顺位、点数、ELO 变化、画像快照
* **收集时不筛选**，弱玩家的数据一条不丢（筛选不可逆）
* 玩家可以在建房面板勾「本局不记录我的数据」，勾了就**真的一条不写**

存储：`data/games/games-YYYY-MM-DD.jsonl.gz`（256MB 换分片，目录上限 8GB
删最旧），画像在 `data/profiles.json`。

> **读这些分片请用 `tools/dump_games.py` 或 `webapp.server.recording.iter_jsonl_gz`**。
> 它们是**追加**写的，进程被强杀时最后一个 gzip 成员会残缺，而
> `gzip.open` 会在读第一行之前就抛 `EOFError`，把整个分片判死
> （这个坑花了一轮才定位到）。

---

## 五、部署

```
玩家 ──443/TLS──▶ VPS(47.82.118.193) nginx ──8080──▶ frps
                     └─ 证书 /etc/letsencrypt/live/maj.xjtuwhy.com/
     ──frp 隧道──▶ 本机 frpc ──▶ 127.0.0.1:8790  python run.py
```

* **服务在本机**：VPS 只有 2 核 / ~890MB，而 AI 推理吃 CPU；
  本机 i9-14900HX / 32GB 更合适。VPS 只做 TLS 终结 + 反代。
* nginx 三条要点（少一条就坏）：**WebSocket 要能升级**（`Upgrade`/`Connection`
  透传 + `proxy_read_timeout 3600s`）、**Host 头要原样透传**（frps 靠
  `customDomains` 路由）、**不能开代理缓冲**（牌桌是持续小包推送）。
* `/api/profiles` 和 `/api/record` 在 nginx 里加了 basic auth
  （带匿名 ID 的画像表不该裸奔）。
* 日志：应用自己写 `logs/*.log`（8MB × 5 份轮转），VPS 侧 `logrotate` 管
  nginx / frps 的日志。
* 身份：昵称 + 匿名签名 cookie（HMAC），房间可选口令。没有账号密码体系 ——
  这是「轻量身份识别」，够用但不适合真运营。

---

## 六、对 `aimodel/` 的改动

**最终状态：零改动。** `git diff origin/main -- aimodel/` 是空的（除了
`weights/train_log.json`，那是另一台机器的训练进程在写，不是平台侧动的）。

过程中我确实加过一套只读观测钩子（`on_ui` / `Game.pump()` /
`emit_pending_hand_end()` / `Player.pond_red`），但它让另一台机器的
`engine.py` patch 产生了 8 个 hunk 冲突。收到协作频道里的说明后**已全部回退**，
补丁留在 `E:\WHYidea\_verify\my-engine-on_ui.patch` 备查。现在平台只用
现成的 `on_action`（训练侧已经提供的那条结构化动作流），一个钩子够用：

* 牌河赤 5 —— 在 `discard` 事件里从 `pl.red[tile] > pl.hand[tile]` 现读
* 一局结束 —— 在 `win` / `ryukyoku` 事件里抓盘面（此刻四家手牌还完整）

---

## 七、已知问题（如实写）

1. **`aimodel/tests/run_all.py` 在 `origin/main` 上是红的（2/18）**，
   **都不是平台侧造成的**，`tests/` 是共享只读所以我没动：
   * `test_chi_4p.py`：「缺 chi 头的旧权重应该抛异常」—— 但 `a88f417` 把
     `Net.load` 默认改成 `strict=False` 了，断言没跟着改。
   * `test_models.py`：「应该有 8 个模型，得到 10」—— 有人加了
     `maj-normal4v1.3`，测试写死数量。
   已在 `ai-communication` 频道里报告（含建议改法）。
2. **存量权重缺 `chi` 头**：25 个模型全部可用，但其中 24 个没训过「吃」。
   平台的做法是**这一手直接交给内置教师**（绝不让零初始化头参与决策），
   并在模型列表里如实标注。四麻的吃牌因此不是模型决策。带 chi 头的
   `maj-normal4v1.3` 是唯一一个完整的。
3. **`policy.npz` / `policy_bc4.npz` 没放进模型下拉** —— 它们是内部对照产物
   （`variants` 里也没注册），玩家不该选到。
4. **没有账号体系**：身份 = 昵称 + 匿名 cookie，清 cookie 就换人。
   真要运营需要补登录。
5. **断线 5 分钟后座位交回给 AI**：回来会变成观战（可以接着看，坐不回去）。
6. **单进程**：10 房 × 4 人实测没问题，但没有做多进程 / 多机扩展；
   `max_rooms=200` 是软上限，真到那个量级要重新压测。
7. **`data/profiles.json` 是整体重写的**：进程被强杀可能丢掉最后几局画像
   （对局记录本身是追加写的，不受影响）。
8. **Windows 上没有 `chmod` 那种语义**：验收 7 的「落盘目录只读」是用
   「父级是个文件、目录创建必然失败」等价替代的，效果相同但不是同一件事。
9. **前端 UI 是重写的**：工作区被误清过一次（源码丢失、构建产物还在），
   现在的 `webapp/client/src/**` 是重写版本；牌面素材也从 CC0 源重新生成。
10. **名牌位置**：别家的名牌有时候会压在自家手牌上（视觉上挤，但牌本身
    看得见）。不影响可玩性，属于待打磨项。
11. **两台机器在同一个工作树上**：另一台机器（Mac）的 AI 用
    `ai-communication` 仓库当信箱、本机（训练侧）在同一个 `potatomaj`
    工作树里改 `aimodel/`。平台侧只用 `origin/webapp` 分支，不碰 `main`。
