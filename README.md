# 土豆麻将 🀄

日本麻将 AI。**三麻（三人麻将）和四麻（四人麻将）都支持** —— 规则引擎、训练管线、
对战平台全部自己写，**只依赖 numpy**（没有 PyTorch、不需要显卡，一台普通机器就能训）。

```bash
# 看 AI 自己打一局
.venv/bin/python -m aimodel.play --agents maj-flashv1.1,teacher,teacher

# 开一桌：你打一个位子，两个 AI 坐对面
./aimodel/ui/run-arena.sh            # 或者双击「土豆麻将竞技场.command」
```

---

## 一、这是什么

一个从零写的日麻 AI 项目，分四块：

| 模块 | 是什么 |
|---|---|
| **规则引擎** | 三麻 108 张 / 四麻 136 张。役种、符数、点数、吃碰杠拔北立直振听流局，全部实现并测试 |
| **八个模型** | 四种「得分条件」（标准 / 防守 / 速攻 / 大牌）× 三麻 / 四麻，另有 v1/v1.1/v1.2 三个版本 |
| **训练管线** | 行为克隆（跟启发式教师学）→ 自对弈 PPO（对手池混入其它风格的模型） |
| **对战与评测** | 命令行对局、Java 牌桌、网页对战台、ELO 混战天梯 |

---

## 二、规则引擎

`aimodel/sanma.py` + `aimodel/engine.py`。对齐雀魂三麻/四麻段位战：

* **三麻**：108 张（万子只有 1m/9m）、无吃、拔北、ツモ損、不聴罰符 2000、
  1m↔9m 宝牌循环、双响有效
* **四麻**：136 张、有吃（只能吃上家，碰杠优先于吃）、北家北为自风、
  三色同顺、王牌 14 张、不聴罰符 3000、25000 起 / 30000 返

点数表**逐格核对过雀魂官方点数表**（`tests/test_sanma.py`），
役种有**逐个对照表**（`tests/test_yaku.py`，34 个役种 + 易混组合）。

---

## 三、八个模型

规则一样，**目标不同**：

| 模型 | 人数 | 得分条件 |
|---|---|---|
| `maj-normalv1.2` | 三麻 | 雀魂段位分 |
| `maj-mountainv1.2` | 三麻 | 段位分 **− 3.0 × 放铳点数/1000** |
| `maj-flashv1.2` | 三麻 | 段位分 **+ 4.0 × 和了局数** |
| `maj-hugev1.2` | 三麻 | 段位分 **+ 1.5 × 累计番数** |
| `maj-*4v1.2` | 四麻 | 同上四条 |

不带版本号的写法（`maj-flash`、`flash4`）一律指向最新版。

### 训练方法

1. **行为克隆**：让每个模型跟**口味配套的启发式教师**打 5000~6000 局，学它的出牌
   （防守型教师更早弃和、速攻型教师疯狂副露、大牌型教师压进张保打点）
2. **自对弈 PPO**：16~20 轮 × 每轮 300~500 场，**对手池里混入另外三个风格的模型** ——
   只跟自己下学不到「面对速攻该怎么防」

网络 512×512，纯 numpy 手写反向传播 + Adam。特征 486 维（四麻 560），
含**动作特征**（打掉每张牌之后的向听、进张、危险度、是否现物/宝牌/赤牌）。

### 实测

八个模型的「强化学习 vs 纯克隆」配对检验（各 300 场、座位轮换）：

| | 结果 |
|---|---|
| 平均提升 | **+3.65 段位分** |
| 正向 | 7 / 8 个模型 |
| 合并显著性 | Stouffer z = 2.45，**p = 0.014** |

单个模型全是「打平」（三麻方差极大），合起来才看得出效果。详见
[aimodel/RESULTS.md](aimodel/RESULTS.md)。

---

## 四、怎么玩

```bash
# ① 命令行看 AI 对局
.venv/bin/python -m aimodel.play --agents maj-normalv1.2,teacher,teacher
.venv/bin/python -m aimodel.play --agents maj-flash4v1.2,teacher,teacher,teacher   # 四麻

# ② Java 牌桌：人类点牌 / 选模型 / 逐手回放 / 换主视角 / 结算页
bash aimodel/ui/java/build.sh && bash aimodel/ui/java/run.sh

# ③ ELO 混战天梯：不同模型不同版本同池竞技，实时图表
./aimodel/ui/run-arena.sh
./aimodel/ui/run-arena.sh --players 4        # 四麻池

# ④ 网页对战台（多座位、每座位独立视角、逐手回放）
.venv/bin/python -m aimodel.ui.server --seats human,maj-normalv1.2,maj-flashv1.2
```

---

## 五、训练

```bash
.venv/bin/python -m aimodel.train --list-models              # 看八个模型和训练状态
.venv/bin/python -m aimodel.train all --model maj-flash4v1.2 # 训单个

# 服务器上跑一整轮（自动按核数/内存定并行）
./aimodel/train-server.sh --version v1.3 --hands 15000 --iters 30

# 巡检（每小时看一眼，走歪会主动告警）
.venv/bin/python -m aimodel.progress --version v1.3
```

只依赖 numpy。在别的机器上训练见 [aimodel/README.md](aimodel/README.md) 的
「在别的机器（服务器）上训练」一节。

### 真实牌谱：对拍与数据抽取

有一套现成的管线，把 **MJAI 格式**（麻将 AI 的通用牌谱格式，Mortal、akochan
都用它）的牌谱喂回引擎：

```bash
# ① 对拍：把整局喂回引擎，比和牌者 / 和牌张 / 番数 / 符数 / 役种 / 点数收支
python -m aimodel.logcheck game.json
python -m aimodel.logcheck logs/ --limit 500
python -m aimodel.logcheck --selftest          # 自检：自己导出再自己对拍

# ② 抽训练数据：(可见局面 → 动作) 用于行为克隆，(局面 + 打出的牌 → 是否放铳) 用于危险度
python -m aimodel.logdata logs/ --out human.npz
```

**对拍是抓规则 bug 最狠的手段** —— 牌谱写的是 5 番 8000 点、引擎算出来 8 番
12000 点，这种事写用例想不到，拿真实对局一跑就露馅。

自检（导出 → 还原牌山 → 重放 → 比对）目前在**三麻四麻各 20 场、共 108 局**
上逐项一致，`test_logs.py` 把它钉住了。等拿到真实牌谱（天凤 / 雀魂 / 自己录的），
这条链路直接能用。

---

## 六、目录结构

```
aimodel/                  AI 主体
├── sanma.py              规则：役种 / 符 / 点数
├── engine.py             对局引擎
├── fastshanten.py        快速向听（比朴素实现快 550 倍）
├── encode.py             特征编码（含动作特征）
├── net.py                网络 + 手写反向传播 + Adam
├── agents.py             教师 / 神经网络 / 随机 agent
├── train.py              训练（行为克隆 + 自对弈 PPO）
├── train_all.py          批量训练八个模型
├── rewards.py            四种得分条件
├── variants.py           模型注册表（多版本）
├── arena.py              ELO 混战池
├── mjai.py               牌谱格式读写（MJAI）
├── logcheck.py           牌谱对拍：重放真实牌谱、比对规则
├── logdata.py            从牌谱抽训练数据
├── evaluate.py           对战评测
├── progress.py           训练巡检
├── ab.py                 两个模型配对对战
├── weights/              24 个模型权重（v1 / v1.1 / v1.2）
├── ui/                   对战平台（网页 + Java 牌桌 + ELO 竞技场）
└── tests/                8 套测试

potatomaj/                牌效与危险度（aimodel 依赖它）
├── tiles.py              牌名与宝牌推导
├── shanten.py            向听 / 进张
└── danger.py             危险度

tests/                    牌效 / 危险度 / AI 接口的测试
```

---

## 七、测试

```bash
.venv/bin/python aimodel/tests/run_all.py     # 8 套，全绿才算改对了
for t in tests/*.py; do .venv/bin/python "$t"; done
```

面向 AI 的 8 套里，最值得一提的是：

* `test_yaku.py` —— **34 个役种逐个对照表**。写它的时候抓出过「混一色被判成清一色、
  副露白多算 3 番」这种靠人眼永远看不出来的错
* `test_4p.py` —— 四麻的 136 张守恒、吃、北家自风、三色同顺、点数分摊
* `test_engine.py` —— 引擎状态机、牌数守恒、流局/振听/多家和牌
* `test_net.py` —— 数值梯度校验、掩码、存档、PPO 截断
* `test_logs.py` —— 牌谱管线：导出 → 还原牌山 → 重放 → 逐项比对（三麻四麻都测）

---

## 八、依赖

```
numpy>=1.24        # AI 部分只要这个
```

牌面图用的是公开的麻将牌素材，仅用于本项目的对战界面。

---

## 九、已知限制

* **不叠加役满**（只报最高的那个）
* 危险度模型的校准基于自对弈统计，**没有用真实人类对局拟合过**
* 八个模型的风格差异是**统计上成立**（副露 0.67 vs 0.36、放铳率 15% vs 18%）而非一眼能看出
* 四麻模型的水平目前与内置教师持平，三麻标准型略强于教师

---

## 十、许可

MIT（见 [LICENSE](LICENSE)）。牌面图资源版权归原作者所有。
