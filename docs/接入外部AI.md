# 接入外部 AI（mjai 协议）

本文回答三件事：

1. 市面上哪些开源日本麻将 AI **真的能当对手**，怎么调用；
2. 本项目的协议桥 `aimodel/external.py` 怎么用、装在哪；
3. 平台（`webapp/`）要接的话改哪几行。

> 代码里的协议桥只有 `aimodel/external.py` 一个文件。**不把任何第三方
> 源码/权重放进 git**；默认安装目录是 `_data/external/`，已被
> `.gitignore` 的 `_data/` 忽略。

---

## 1. 调研表

star 数是 2026-10-21 用 GitHub API 查的；许可证以仓库 `LICENSE` / README
为准（GitHub 的自动识别有时不准，见 Akagi 那行）。

| 项目 | ★ | 许可证 | 语言 | 能当 self-play 对手吗 | 怎么调用 |
|---|---:|---|---|---|---|
| [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) | 1591 | **AGPL-3.0** | Rust + Python | ✅ 能，**mjai stdio** | `python mortal.py <player_id>`；stdin 一行一个 mjai 事件，stdout 一行一个动作。需要：Rust 编译 `libriichi`、PyTorch、`mortal.pth` 权重（仓库不带权重） |
| [Apricot-S/mjai-manue-go](https://github.com/Apricot-S/mjai-manue-go) | 5 | **BSD-3-Clause** | Go | ✅ 能，**mjai stdio**（也有 TCP） | `mjai-manue --name X --id <seat> --seed <n>`；纯 stdio，不需要权重。**本次实际跑通的就是它** |
| [gimite/mjai-manue](https://github.com/gimite/mjai-manue) | 37 | New BSD（README/gemspec） | CoffeeScript + Ruby | ⚠️ 只能当 **TCP mjai client** | `bin/mjai-manue <mjai-server-url>`；要配 `gimite/mjai` server，不能直接做 stdio 子进程 |
| [critter-mj/akochan](https://github.com/critter-mj/akochan) | 299 | 自定义（非 OSI；对 AI 权重的再发布有限制） | C++ | ✅ 自对弈；mjai 只支持 **TCP client** | 自对弈 `./system.exe test <begin> <end>`；mjai `./system.exe mjai_client <port> setup_mjai.json`，需要 mjai server。编译要 Boost |
| [Cryolite/kanachan](https://github.com/Cryolite/kanachan) | 346 | **仓库没有 LICENSE 文件**（API 也识别为 null） | C++ + Python | ⚠️ 有 `mjai.app/bot.py` stdio 包装，但**不给训练好的模型** | 需要自己按 `mjai.app/build.sh` 打包（Docker + C++ 依赖），并提供 `model.kanachan`；仓库不提供权重 |
| [shinkuan/Akagi-MjaiBot-Mortal](https://github.com/shinkuan/Akagi-MjaiBot-Mortal) | 10 | **AGPL-3.0** | Python + 预编译 libriichi | ✅ 能，**mjai stdio（批量数组）** | `python bot.py`；一行收一个 JSON **数组**，回一行动作。自带 `mortal.pth` + macOS/Python 3.12 aarch64 的 libriichi；需要 Python 3.12 + torch |
| [Nitasurin/Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy) | 55 | AGPL-3.0 | Rust | ✅ Mortal 的 policy 分支，mjai 接口同 Mortal | 同 Mortal；仓库不带权重 |
| [Equim-chan/mjai-reviewer](https://github.com/Equim-chan/mjai-reviewer) | 1259 | Apache-2.0 | Rust | ❌ 不是 agent | 批量**复盘**工具，内部调 Mortal/akochan 引擎；没有「持续对局」接口 |
| [shinkuan/Akagi](https://github.com/shinkuan/Akagi) | 1107 | README 写 **AGPL-3.0 + Commons Clause**（GitHub API 识别成 Apache-2.0，以 LICENSE 文件为准） | Rust/TS + Python | ❌ 不是 agent | 实时**分析助手**（雀魂/天凤客户端）；它的 `mjai/bot/bot.py` 是 Mortal 的 bot 包装，本身不产生 self-play |
| [EndlessCheng/mahjong-helper](https://github.com/EndlessCheng/mahjong-helper) | 2347 | MIT | Go | ❌ 不是 agent | 牌效/防守**助手**，读雀魂/天凤日志；没有 mjai self-play 接口 |
| [gimite/mjai](https://github.com/gimite/mjai) | 67 | 未声明 | Ruby | ❌ 不是 AI | mjai **对局服务器**；给 Manue/akochan 这类 TCP client 用 |

**一句话结论**：能直接接进本项目的，只有 **mjai stdio** 这一类。
Mortal 最强但依赖最重；manue-go 最轻，本次用它做了端到端验证。
Akagi / mahjong-helper 是助手不是对手；mjai-reviewer 是复盘器。

---

## 2. 协议桥：`aimodel/external.py`

### 2.1 已注册的名字

```
mortal           Mortal（上游，stdio）
mortal-akagi     Mortal（Akagi 打包版，stdio 批量数组）
manue            mjai-manue-go（stdio，本次验证用它）
kanachan         Kanachan 的 mjai.app 包装
akochan          TCP-only，构造时明确拒绝（不是静默失败）
manue-original   原版 Manue，TCP-only，明确拒绝
```

别名：`mjai-manue` / `manue-go` → `manue`；`akagi-mortal` / `mortal-bot` → `mortal-akagi`。
`--third external:mortal` 也可以。

### 2.2 命令行

```bash
# 外部 AI 当第三家
.venv/bin/python -m aimodel.ab \
    --a aimodel/weights/maj-pro4v1.1_mac_rl.npz \
    --b aimodel/weights/maj-normalv1.1.npz \
    --third manue --matches 10

# 外部 AI 当被测方 A / B
.venv/bin/python -m aimodel.arena_test \
    --a manue --b aimodel/weights/maj-pro4v1.1_mac_rl.npz \
    --games 20 --jobs 4 --no-sprt

# 看安装状态（不启动进程）
.venv/bin/python -c "from aimodel import external; print(external.check_spec('manue'))"
```

**没装时**：构造 `ExternalMJaiAgent` 抛 `ExternalAINotInstalled`，消息里有
`docs/接入外部AI.md`；`ab` / `arena_test` 在开跑前用 `external.require()`
直接 `SystemExit`，不会跑一半才崩。

### 2.3 代码里

```python
from aimodel import external
from aimodel.engine import Game
from aimodel import sanma as S

ag = external.ExternalMJaiAgent("manue", seed=1, player_id=2, players=4)
game = Game([..., ag, ...], seed=1, rule=S.rule_for(4))
external.attach(game)      # 把 Game.on_action 转发给外部 AI
result = game.play()
ag.close()
```

`external.attach(game)` 只包一层 `game.on_action`，**不修改 `engine.py`**；
原来的钩子（平台的广播）会先被调用。

### 2.4 配置：环境变量 / JSON 文件

优先级（后者覆盖前者）：`SPECS` 默认值 → JSON 配置 → 环境变量 → 代码参数。

环境变量（`<KEY>` 是大写、`-` 换 `_`）：

```bash
export POTATOMAJ_EXTERNAL_MANUE_BIN=/abs/path/mjai-manue
export POTATOMAJ_EXTERNAL_MORTAL_CMD="/abs/venv/bin/python mortal.py {player_id}"
export POTATOMAJ_EXTERNAL_MORTAL_DIR=/abs/path/Mortal/mortal
export POTATOMAJ_EXTERNAL_MORTAL_MODEL=/abs/path/Mortal/mortal/mortal.pth
export POTATOMAJ_EXTERNAL_MORTAL_AKAGI_CMD="/abs/python3.12 /abs/Akagi-MjaiBot-Mortal/bot.py"
```

占位符：`{player_id}`、`{seed}`、`{python}`（当前解释器）、`{manue_bin}`、
`{root}`、`{external_dir}`。

JSON 配置（`POTATOMAJ_EXTERNAL_CONFIG=/abs/external.json`）：

```json
{
  "mortal": {
    "cmd": ["/abs/venv/bin/python", "mortal.py", "{player_id}"],
    "cwd": "/abs/Mortal/mortal",
    "env": {"MORTAL_REVIEW_MODE": "0"}
  },
  "manue": {
    "cmd": ["/abs/bin/mjai-manue", "--name", "Manue", "--id", "{player_id}", "--seed", "{seed}"],
    "restart_each_hand": true
  }
}
```

### 2.5 超时 / 崩溃 / 确定性

- **超时**：读 stdout 走独立线程 + `queue.Queue.get(timeout=...)`，
  默认 20s（`timeout=` 可调）。卡死抛 `ExternalAITimeout`，不会拖死整局。
- **崩溃**：`ExternalAICrashed`。适配层会先**重启并重放本局已发生的事件**
  （默认最多 2 次，`max_restarts=` 可调）—— 因为外部 AI 是确定性的，
  同一串事件应得到同一个动作。
- **仍然失败**：
  - 构造时给了 `fallback=` → 降级到教师，`degraded=True`，`summary()` 里记
    `errors / restarts / timeouts / crashes / bad_actions`；
  - 没给 → 直接抛。**评测默认走这条**，宁可中止也不拿脏数据。
- **非法动作**：外部 AI 返回的牌不在 `decision.options` 里 → 抛
  `ExternalAIProtocolError`，消息里列出「它想打什么 / 引擎允许什么」。
  **绝不 `return 0`** —— 静默 0 在麻将里等于随便打一张，会污染整个天梯。

### 2.6 事件翻译里必须知道的几个坑

1. **赤 5 写法**：`aimodel/mjai.py` 用 `0m`，Mortal/manue 用 `5mr`。桥内部
   统一发 `5mr`；读回来两种都认。
2. **赤 5 归一化（当前 MVP 限制）**：桥把**所有 5 都按普通 5**发给外部 AI
   （`start_kyoku` 手牌、`tsumo`）。原因：本项目的 `chi` 不维护 `red` 计数
   （`_do_chi` 只减手牌、不减赤），如果保留赤，外部 AI 的手牌模型会和引擎
   对不上、直接判非法打牌。代价是外部 AI 看不到赤 dora，强度略低。
   要恢复赤 dora，需要先修引擎的 chi 赤计数，或在桥里自己维护完整手牌模型。
3. **别家摸切**：引擎的 `discard` 事件里 `tsumogiri` 在 `_finish_discard`
   里先清 `pl.drawn` 再 `_act`，所以**永远是 True**。桥对自己按「摸到的牌
   和打出的牌是否同型」现算；对别家一律发 `True`（本来就不该知道别家摸牌，
   而且 manue-go 会对非摸切做喰い替え校验，规则一有差异就崩）。
4. **别家手牌**：`start_kyoku` 的别家 `tehais` 全是 `"?"`；别家 `tsumo`
   也是 `"?"`。**不能把 View 里的别家手牌泄漏给外部 AI**，否则它作弊，
   评测数字就废了。
5. **明杠**：引擎 `call=="kan"`，Mortal 的事件名是 `daiminkan`；桥发明杠用
   `daiminkan`，读动作时 `kan`/`daiminkan` 都认。
6. **立直**：先发 `reach`，宣言牌 `dahai` 之后再发 `reach_accepted`。
7. **每局重启**：manue 不认 `can_act`，上一局迟到的动作会污染下一局第一手
   （实测它想打上一局摸到的 8s）。所以 `manue` spec 开了
   `restart_each_hand=True`；Mortal 认 `can_act`，不重启。
8. **四人限定**：mjai 的 `start_kyoku` 是 4 人格式；外部 AI 只支持四麻。
   三麻/三麻权重用 `--third manue` 会得到明确报错，不会静默降级。

---

## 3. 安装外部 AI

### 3.1 mjai-manue-go（推荐先试，本次已验证）

需要 Go **1.27+**（仓库 README 明确要求）。

```bash
# 方式 A：go install
go install github.com/Apricot-S/mjai-manue-go/cmd/mjai-manue@latest

# 方式 B：从源码构建（二进制放 _data/ 下，不会进 git）
git clone https://github.com/Apricot-S/mjai-manue-go.git _data/external/mjai-manue-go
cd _data/external/mjai-manue-go
go build -o ../bin/mjai-manue ./cmd/mjai-manue

# 让桥找到它（或放到 PATH 里）
export POTATOMAJ_EXTERNAL_MANUE_BIN=$PWD/_data/external/bin/mjai-manue
.venv/bin/python -c "from aimodel import external; print(external.check_spec('manue'))"
```

不需要权重、不需要 torch。**本次实测**：四麻一局 2 手、26 次决策、
0 错误；`arena_test --a manue --b maj-pro4v1.1_mac_rl` 能正常出 Δ/SE。

### 3.2 Mortal（上游，最强但最重）

前置：Rust（rustup）、Python 3.9–3.12、PyTorch、`mortal.pth` 权重。
**权重不在仓库里**（Mortal README 指向作者 gist），需要自己准备。

```bash
git clone https://github.com/Equim-chan/Mortal.git _data/external/Mortal
cd _data/external/Mortal
cargo build -p libriichi --lib --release

# Linux：
cp target/release/libriichi.so mortal/libriichi.so
# macOS：产物名可能是 libriichi.dylib，需要确认/改名为 libriichi.so
# （Akagi 的预编译包用的是 libriichi-3.12-aarch64-apple-darwin.so）

# 用 Python 3.12 建 venv 并装 torch（不要用本仓库的 3.14，Mortal 不支持）
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch
# 把 mortal.pth 放到 mortal/ 下（上游 config 的 state_file 也可以指别处）

export POTATOMAJ_EXTERNAL_MORTAL_CMD="$PWD/.venv/bin/python mortal.py {player_id}"
export POTATOMAJ_EXTERNAL_MORTAL_DIR="$PWD/mortal"
export POTATOMAJ_EXTERNAL_MORTAL_MODEL="$PWD/mortal/mortal.pth"
```

**本次状态**：这台机器**没有 Rust 工具链**（`rustc`/`cargo` 都没有），
上游 Mortal 未能在本次会话内编译/跑通。`check_spec("mortal")` 会明确报
「工作目录不存在 / 缺 Mortal 权重」，不是静默失败。需要确认的地方：
macOS 下 `libriichi` 的最终文件名、以及 `mortal.pth` 的获取方式（上游
不随仓库分发）。

### 3.3 Mortal（Akagi 打包版，自带权重 + 预编译 libriichi）

仓库：`shinkuan/Akagi-MjaiBot-Mortal`（AGPL-3.0）。解压后约 **93 MB**
（`mortal.pth` 4.9 MB + 各平台 libriichi ~70 MB），**不要提交进 git**。
它要求 Python **3.12** + `torch<2.9`，并且 stdin 一行是 JSON **数组**。

```bash
# 解压/克隆到（已在 .gitignore 的 _data/ 下）
_data/external/Akagi-MjaiBot-Mortal/
  bot.py  model.py  mortal.pth  libriichi/...

uv venv --python 3.12 _data/external/Akagi-MjaiBot-Mortal/.venv
uv pip install --python _data/external/Akagi-MjaiBot-Mortal/.venv/bin/python \
    "torch<2.9" "numpy>=1.24" "requests>=2.28"

export POTATOMAJ_EXTERNAL_MORTAL_AKAGI_CMD="$PWD/_data/external/Akagi-MjaiBot-Mortal/.venv/bin/python bot.py"
.venv/bin/python -m aimodel.arena_test --a mortal-akagi --b <权重> --games 4 --jobs 1 --no-sprt
```

适配层给 `mortal-akagi` 开了 `batch_events=True`：一批事件发一行 JSON
数组，Akagi bot 每批只回一行。**本次状态**：包已下载、Python 3.12 环境
已建，torch 安装/端到端对局属于「第二优先」，见最终报告。

### 3.4 akochan（只能 TCP）

```bash
cd _data/external/akochan
make -f Makefile_MacOS          # 需要 Boost；本机未装
# 自对弈：./system.exe test 100 102
# mjai：./system.exe mjai_client <port> setup_mjai.json
```

它不是 stdio bot，本桥会明确拒绝（`ExternalAIUnsupported`）。要接的话，
需要先跑一个 mjai server（`gimite/mjai`），把 akochan 当 TCP client，
桥这边再写一个 TCP↔Agent 的适配 —— 目前**没做**。

### 3.5 Kanachan（没有权重，跑不了）

`mjai.app/` 下有 `bot.py`（stdio），但：
- 仓库**不提供**训练好的 `model.kanachan`；
- 打包要 Docker + C++/Boost/Marisa 等依赖。

所以只登记 spec，`check_spec("kanachan")` 会报缺 `model.kanachan`；
没有实际跑。**不要**把它当成「装一下就能用的对手」。

---

## 4. 平台接入（`webapp/`）

本任务按约定**没有改 webapp/**（改动只在 `aimodel/`、`tools/`、`docs/`）。
下面给出精确到文件+行的接法；每一处都很小。

### 4.1 模型列表：`webapp/server/agents.py`

1. 文件顶部 import 区（约 **L20–L24**，`from aimodel import variants` 附近）加：

   ```python
   from aimodel import external
   ```

2. `available_models()` 里，在 `rows = list(out.values())`（约 **L237**）
   之前加一行：

   ```python
   for row in external.external_models(4):
       out[row["name"]] = row
   ```

   `external_models()` **只检查文件是否存在，不启动进程**；每行带
   `name / external / spec / title / license / repo / players / usable / note`，
   前端现有的下拉（`webapp/static/app/app.js` 的 `models.filter(...)`）
   会直接把它当普通模型显示；`usable=False` 的置灰/显示 note。

3. `AISeat._build()`（约 **L473–L503**）在
   `name = (model or "").strip().lower()` 之后、`try: spec = variants.get(name)`
   之前加分支：

   ```python
   ext_key = external.normalize_key(name)
   if ext_key is not None:
       if self.players != 4:
           self.degraded = True
           self.degrade_reason = "外部 AI 只支持四麻"
           return self.teacher
       try:
           return external.ExternalMJaiAgent(
               ext_key, seed=self.seed, player_id=self.seat,
               players=4, timeout=APPROX_20S,
               fallback=self.teacher,          # 平台要的是「桌子别崩」
               log_path=self.cfg.log_dir / f"external-{self.seat}.log",
           )
       except Exception as exc:
           self.degraded = True
           self.degrade_reason = f"{type(exc).__name__}: {exc}"
           return self.teacher
   ```

   `AISeat.__init__` 里已经有 `self.teacher`；`AISeat.decide` 的线程池
   超时是 20s（`fut.result(timeout=20.0)`），桥默认 `timeout=20`，对得上。

### 4.2 事件转发：`webapp/server/table.py`

`Room._on_action()`（**L633** 起）在 `kind = ev.get("kind")`（**L648**）
之后加：

```python
# 外部 mjai AI 需要看到完整事件流（桥自己缓冲，feed 很快、不推理）
for seat_agent in getattr(self.game, "agents", ()) or ():
    inner = getattr(seat_agent, "agent", seat_agent)
    if isinstance(inner, external.ExternalMJaiAgent):
        inner.feed(ev)
```

文件顶部同样加 `from aimodel import external`。

> 为什么要在这里：`ExternalMJaiAgent` 在收到 `start_hand` / `draw` /
> `dahai` / `call` 后只在内存里排队，真正发给子进程是在下一次 `decide()`
> 时。`feed` 本身不做 IO 等待，放在引擎线程里不会拖慢推帧。

### 4.3 房间创建：`webapp/server/table.py` L363–L370

`AISeat(i, s.ai or "teacher", ...)` 不用改：`s.ai` 直接传
`"external:mortal"` 就行。**但**要在创建前做一次可用性检查，避免选了一个
没装的模型整桌降级：

```python
if external.is_external_spec(s.ai):
    ok, why = external.check_spec(s.ai)
    if not ok:
        raise RoomError(f"外部 AI 不可用：{why}")
```

### 4.4 前端

`webapp/client/src/ui/lobby.ts` / `protocol.ts` 的模型类型如果有
「必填字段」校验，给 `external` 行补 `external: true`、`title`、`repo`、
`license` 即可；`webapp/static/app/app.js` 是打包产物，改完客户端要
重新 `pnpm build`。**本任务没有改这些文件。**

---

## 5. 验证

```bash
# 协议桥单测（不需要网络/不需要装外部 AI）
.venv/bin/python aimodel/tests/test_external.py

# 全量测试
.venv/bin/python -m aimodel.tests.run_all

# manue 当被测方（本次实际跑）
export POTATOMAJ_EXTERNAL_MANUE_BIN=/abs/path/mjai-manue
.venv/bin/python -m aimodel.arena_test \
    --a manue --b aimodel/weights/maj-pro4v1.1_mac_rl.npz \
    --games <N> --jobs <J> --no-sprt
```

最终报告里的 Δ / SE / n、以及哪些外部 AI 实际跑通、哪些卡在哪，都以
那次运行的输出为准。

---

## 6. 还没做 / 需要确认

- [ ] 上游 **Mortal** 的编译与权重获取：本机没有 Rust；权重不随仓库分发。
      需要确认 macOS 下 `libriichi` 的文件名，以及 `mortal.pth` 的合法来源。
- [ ] **Akagi 打包版**的 torch 端到端：包已下载、Python 3.12 环境已建，
      但 torch 安装/跑通未在本轮完成。
- [ ] **akochan** 的 TCP ↔ stdio 桥：没写。需要先跑 mjai server。
- [ ] **Kanachan**：没有权重，未跑。
- [ ] **赤 5**：当前按普通 5 归一化；恢复赤 dora 需要先处理引擎 chi 的
      `red` 计数（或桥里维护完整手牌模型）。
- [ ] **平台**：`webapp/` 的四处接法只写在本文，没有实际改代码/跑前端。
- [ ] **多人外部 AI 同桌**：`attach` 支持多个 `ExternalMJaiAgent`，但
      每个都是独立子进程；4 个 Mortal 同桌的内存/延迟没测。
