# 接入 `strong`（纯规则强化 agent）

`strong` 是 `aimodel/strong.py` 里的 `StrongAgent`：**不加载任何权重**，
四麻 / 三麻都能打，决策完全确定。支持 `strong:<变体>`：

| 写法 | 含义 |
|---|---|
| `strong` / `strong:standard` | 默认：牌效优先 + 敢鸣 + 有危险度/押し引き修正 |
| `strong:attack` | 更爱鸣、更少弃和、更爱立直 |
| `strong:defense` | 更早弃和、少鸣 |
| `strong:value` | 打点权重更高，愿意掉 1 向听做染手/全带 |

## 1. 命令行 / 对拍

```bash
# 第三方座位（aimodel/ab.py）
.venv/bin/python -m aimodel.ab --a A.npz --b B.npz --third strong

# 强度评测（aimodel/arena_test.py 现在支持 agent spec）
.venv/bin/python -m aimodel.arena_test --a strong --b teacher \
    --max-games 1500 --no-sprt --jobs 4

# 天梯（aimodel/arena.py）
.venv/bin/python -m aimodel.arena --players 4 --models strong --matches 100
```

`arena_test.py` 的 `--a` / `--b` 现在可以是：权重路径、`variants` 里的模型名、
`teacher`、`random`、`strong[:变体]`、`style:<风格>`。纯规则对照默认四麻；
要三麻就加 `--players 3`。

## 2. 平台（webapp）模型下拉

平台有**两条**路径，只改一条会出现「文件在但选不到」或「选得到但一选就降级」：

1. **列出可选**：`webapp/server/agents.py` 的 `available_models()`
   - 在 `rows = list(out.values())` **之前**加了两行合成项：
     `strong`（4 人）与 `strong3`（3 人），`usable=True`、`missing_heads=[]`。
   - 它们没有 `.npz`，不要往 `aimodel/variants.py` 注册（`ModelSpec.path`
     会指向不存在的权重，平台会把它标成「没有权重文件」）。
2. **构造座位**：同文件的 `AISeat._build()`
   - 在 `variants.get(name)` **之前**判断 `name.startswith("strong")`，
     直接走 `make_builtin_agent()`。
   - `make_builtin_agent()` 里也加了 `strong` 分支（懒导入 `aimodel.strong`）。
   - 否则会掉进 `except KeyError` 的「降级为教师」路径，**静默**变成教师。
3. **显示名**：`webapp/server/table.py` 的 `_ai_label()` 加了
   `if model.startswith("strong"): return "AI·强规则"`。

前端不需要改：`webapp/static/app/app.js` 会从 `/api/models` 拿列表，并按
`players` 过滤；`strong`/`strong3` 会自动出现在四麻/三麻的模型下拉里。

## 3. 新增一个变体

在 `aimodel/strong.py` 的 `STRONG_VARIANTS` 里加一个 `StrongConfig` 即可：

```python
"myvariant": StrongConfig(
    name="myvariant",
    call_aggression=1.5,   # 鸣牌倾向
    fold_bias=0.6,         # 弃和倾向
    value_bias=0.2,        # 打点权重（牌效是主项）
    riichi_bias=1.0,       # 立直倾向
),
```

不需要改任何注册代码：`make("strong:myvariant")` 会自动找到它。

## 4. 硬规则：无役不鸣

`_eval_call_action()` 在给吃/碰/杠打分前先做两件事：

1. 对**鸣完之后**的手牌做结构化役种分析（`_analyze_yaku`）：
   断幺、役牌、混一色/清一色、混全/纯全带幺、对对和、七对子、一气、三色、
   混老头、小三元、平和/一杯口（门清价值）。**宝牌/赤/拔北不算役**。
2. 如果鸣完已经听牌，则用 `sanma.evaluate_hand` 对每一个和了张**真算**
   —— 结构分析只能看出「役牌对子/一气苗子」这类潜力，听牌后可能根本不成立。

没有真役路线 -> 直接 `pass`。测试见 `aimodel/tests/test_strong.py` 的 ②：
同一手牌教师会吃、`strong` 会过，且吃完后 `has_yaku=False`。
