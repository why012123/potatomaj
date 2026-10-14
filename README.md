# 土豆麻将 🀄

雀魂（Mahjong Soul）**三麻**实时辅助。自带浏览器开游戏，把「打这张的收益」直接画在牌面上。

```bash
./run-electron.sh          # 或者在 Finder 里双击「土豆麻将.command」
```

---

## 一、怎么用

双击 **`土豆麻将.command`**（macOS 只有 `.command` 能双击运行）。启动后：

* 自动开窗口打开雀魂，正常登录、正常打牌
* 轮到你别人的时候，每张候选牌上方挂一个小牌子，写打它的收益
* 左下角信息框：手牌 / 宝牌 / 牌河 / 向听 / 立直提醒 / 我坐哪家

### 命令行参数

```bash
./run-electron.sh                  # 默认
./run-electron.sh --badge          # 额外开一个独立状态窗口
./run-electron.sh --no-dump        # 不记录牌局帧
./run-electron.sh --verbose        # 逐帧打印（排查用）
./run-electron.sh --setup-only     # 只配好环境就退出
```

日志同时写到 `build/app.log`。

---

## 二、数据全部来自**协议**

程序直接解析雀魂的 WebSocket 消息，**不做任何截图或识图**。拿到的东西：

| 数据 | 来源消息（字段） |
|---|---|
| **我的手牌** | `ActionNewRound` 字段 4 |
| **宝牌指示牌** | `ActionNewRound` 字段 14 |
| **摸到的牌** | `ActionDealTile` 字段 2 |
| **副露的具体牌** | `ActionChiPengGang` 字段 3（含类型和来源）|
| **四家牌河** | `ActionDiscardTile` 字段 2 |
| **立直 / 摸切 / 振听** | `ActionDiscardTile` 字段 3 / 5 / 6 |
| 拔北、暗杠、加杠 | `ActionBaBei`、`ActionAnGangAddGang` |

字段号照着 [`.refs/mahjongsoul.proto`](.refs/mahjongsoul.proto) 抄的，解码在
[`electron/lib/actions.js`](electron/lib/actions.js)。

### 那层「加密」

`ActionPrototype.data` 上盖了一层位置相关的异或。**它不是密码学加密**，
算法一共 8 行：

```rust
const KEYS: [u8; 9] = [0x84, 0x5E, 0x4E, 0x42, 0x39, 0xA2, 0x1F, 0x60, 0x1C];
let base = 23 ^ data.len();
data[i] ^= (base + 5 * i + KEYS[i % 9]) as u8;
```

**这段算法来自 [shinkuan/Akagi](https://github.com/shinkuan/Akagi) 的
`src/bridge/majsoul/parser.rs`（Apache-2.0）**，不是我自己推的。
Rust 版的解码器实现得很完整，本项目的 JS 移植见
[`electron/lib/liqi.js`](electron/lib/liqi.js)。

> 这一条值得写清楚：我一开始不知道它是公开的，花了很多轮去逆向，
> 还得出过「密钥流按长度分套、每种长度要攒 10~15 局样本」的结论，
> 并为此写了一大套识图兜底。**那些现在全删了。**
> 如果你要改这一层，先去看 Akagi，不要自己 bit-twiddle。

---

## 三、判断逻辑

### 打哪张的收益

1. **神经网络**（`aimodel/weights/policy.npz`，同事训的三麻模型，35 万参数）
2. **纯牌效**（进张枚数）—— 网络用不上时兜底
3. **危险度** —— 给气泡加红圈警示

网络的适用性闸门在 [`potatomaj/ai.py`](potatomaj/ai.py)：它是三麻专用的，
所以手里出现 2m~8m（只有四麻才有）或不是刚摸完的 14 张时，会如实说明原因
并退回纯牌效。

### 危险度

先用**协议解出来的立直**做确定判断：立直那家一定听牌，对他的现物绝对安全、
筋明显更安全、无筋直接高危。没有立直信息时退回「按巡目估算」的相对排序。

红圈警示取这一手牌里**最危险的那几张**（不看绝对值）—— 因为危险度的量纲
（`0.25 + 0.075×巡目`）让绝对值没有意义，按绝对阈值划会让每张都亮。

---

## 四、结构

```
electron/                  外壳（Node）
├── main.js                主进程：抓帧、状态机、气泡、建议
├── preload-game.js        WebSocket 钩子 + 页内 HUD + 牌面气泡
├── lib/
│   ├── liqi.js            那层异或的解码（算法来自 Akagi）
│   ├── actions.js         按 proto 把每条动作读成结构化数据
│   ├── gamestate.js       牌局状态（全协议）
│   ├── rowgeom.js         手牌顺序 + 位置推算
│   ├── pb.js              liqi 封装解析
│   └── bridge.js          Python 桥
└── test/                  离线测试（不需要雀魂、不需要网络）

potatomaj/                 算牌（Python）
├── tiles.py               牌名与宝牌推导
├── shanten.py             向听 / 进张 / 每张牌的收益
├── danger.py              危险度
├── ai.py                  接 aimodel 神经网络
└── bridge.py              给 Electron 用的 stdio JSON 服务

aimodel/                   三麻 AI（同事的模块）
tools/decode.js            把录下来的帧解码成牌谱（排查用）
.refs/                     参考：mahjongsoul.proto 等
```

---

## 五、测试

全部**离线**，不依赖雀魂、不依赖网络：

```bash
for t in tests/*.py; do .venv/bin/python $t; done   # Python 侧（牌效 / 危险度 / AI）
node electron/test/protocol.js                      # 协议解码（拿真实帧跑）
node electron/test/riichi.js                        # 立直/摸切语义
node electron/test/rowgeom.js                       # 手牌顺序与位置推算
node electron/test/ipc-wiring.js                     # IPC 接线
```

需要窗口的（加 `--no-sandbox`）：

```bash
ELECTRON=electron/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron
HOME=$PWD/build/fakehome env -u ELECTRON_RUN_AS_NODE "$ELECTRON" \
  electron/test/hud-test.js --no-sandbox --enable-unsafe-swiftshader \
  --user-data-dir="$PWD/build/fakehome/ud"
```

长时间挂机压测（重放真实帧，量内存）：

```bash
bash build/soak-test.sh 120
```

---

## 六、已知限制

* **只打三麻**。四麻能认出来（牌河里出现 2m~8m 就判定四麻）并如实说明，
  但三麻模型用不了，会退回纯牌效
* **读不到他家的手牌** —— 服务器不下发
* 危险度只用「现物 / 筋 / 可见张数 / 立直」这几样，**读不到听牌形状**，
  所以是相对安全度排序，不是绝对保证
* 辅助工具。**别拿去打段位场** —— WebSocket 钩子会增加客户端指纹暴露面

---

## 七、隐私

抓包只保留**牌局消息**。登录、账号、聊天、大厅那类走黑名单（`NEVER_DUMP`），
**不落盘**。抓到的帧在 `build/ws-frames/`，随时可以删；不想记录就加 `--no-dump`。

---

## 八、致谢

* 协议解码算法：[shinkuan/Akagi](https://github.com/shinkuan/Akagi)（Apache-2.0）
* protobuf 定义：[Cryolite/mahjongsoul_sniffer](https://github.com/Cryolite/mahjongsoul_sniffer)、
  [jimsavage/majsoul-api](https://github.com/jimsavage/majsoul-api)
* 同类项目（值得一看，都比这个成熟）：
  [EndlessCheng/mahjong-helper](https://github.com/EndlessCheng/mahjong-helper)、
  [latorc/MahjongCopilot](https://github.com/latorc/MahjongCopilot)
