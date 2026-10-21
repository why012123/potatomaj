# pro4v1.2 作战计划（2 天：2026-10-21 13:00 → 2026-10-23 13:00）

> 这是一页**只讲"现在到明天要做什么"**的作战板。长期（30 天 dot V1）见 `docs/project-maj-dot-V1.md`。

## 一、目标与判据（预注册，不许事后改）

```
交付物：**pro4v1.2** —— 在配对赛里稳定胜过生产模型 `maj-pro4v1.1_mac_rl.npz` 的权重
判据（照旧）：Welch 单侧 t，df≈6，α=0.05 -> **Δ > 3.24** 才算"证实"
            不足 3.24 但 Δ>0 且方向一致 -> 记为"方向性证据"，**不写进验收**
参考基线：mac_rl 的 sha256 = e2af5d31…，hidden [512,512]
```

## 二、今天已被证伪的四条（别再试 ✗）

| 实验 | 结果 |
|---|---|
| 从 mac_rl 微调（干净数据，lr 1e-4，2 轮） | SPRT 打平 ×2 |
| 干净数据 6 轮 vs 2 轮 BC | Δ=+0.050（SE 1.584） |
| 干净数据 vs 旧数据（4 种子 BC） | Δ=−0.992（SE 1.968）|
| ret 加权 BC | Δ=−0.992 |
| RL 20 轮（Mac，从 mac_rl） | Δ=+0.325（SE 1.169）|
| StrongAgent（手写规则）vs mac_rl | **Δ≈−7**（640 组，SE 2.31）-> 规则打不过网络 |

**唯一正结果**：server 的 RL（从 mac_2ep 起，4 种子）**合并 +1.870 ± 0.706，3/3 全正，p≈0.004**。
→ 结论：**增益来自 RL**，不来自数据清洗/更长 BC/手写规则。

⚠️ **但这个 +1.870 是 vs 起点 mac_2ep，不是 vs 生产 mac_rl** ✗（后来 server 补测：seed1 vs mac_rl 只有 **+0.648 ± 1.179，p=0.291**）
```
合起来读：RL 从 mac_2ep 起能涨 +2.285 ✓ 但 **mac_rl 本来就比 mac_2ep 强** ✓
          -> RL 20 轮只是**追平** mac_rl，不是造出新天花板 ✗
          -> Mac 这边「从 mac_rl 起再 RL 20 轮」= +0.325 ± 1.169（同样的结论）
=> **同配置换起点 = 无效** ✗ 要突破必须换**信息量**（特征 / 数据质量 / 搜索）✓
```

## 二·五、今天被关掉的两条外部路（别再来回试 ✗）

| 路 | 结论 |
|---|---|
| **Mortal 当老师** | 官方**不公开权重**（作者 2022 明确表态）✗ 代码 AGPL ✗ 判死 |
| **找「公开权重的强 NN」** | 日麻生态里**不存在** ✓（Mortal 无权重 / kanachan 要自训 / akochan 的 80 KB 是查表不是 NN / riichienv 也要 mortal_v4.pth）✗ |
| **akochan** | 还能用 ✓ 但只当**评测对手/oracle**（规则+搜索，非 NN）；需 MSYS2+boost 编译，明天 2 小时盒 |
| **manue** | 已接入 ✓ BSD-3，Go 单二进制，今晚 20:00 天梯的外部对手就是它 |

→ 所以 pro4v1.2 只能来自**我们自己的三件事**：① 645 维新特征 ② 鳳凰卓高质量数据 ③ RL（已知只能追平）✗

## 三、今晚的路线（Mac 侧，不依赖网络）

```
① 特征改造（进行中）
   match_state 17 + called_tile 34 + last_discard 34  -> DIM 560 -> 645
   + 修 `visible_counts` 重复计数 bug（自己副露被数两遍 / 自己拔北被数两遍）
   注：`called_tile` 是今天 new 的发现 —— 老特征里**根本没有"被叫的是哪张牌"** ✗
② 重抽 40,000 局（27 批，4 进程并行，约 25~35 分钟）
   产物：_data/pro4/tbc_v12/ -> _data/pro4/bc_all_v12.npz
   脚本：_tools/reextract_parallel.sh / _tools/merge_shards.py
③ BC 训练 4 种子 × 2 轮（约 40 分钟，jobs=4）
④ 配对赛 vs mac_rl（4 种子 × 1500 组）
   -> 得到 BC_new 相对 mac_rl 的 Δ。预期仍为负（BC 天花板），但**看它比 BC_old 好多少**
⑤ 把 Δ 最好的种子交给 server 做 RL（RL 是唯一被证实的增益，+1.87）
   -> pro4v1.2 = BC_new + RL，明天中午前结算
```

## 四、并行线（server 侧，网络/大机器）

```
P0  mjlog XML -> MJAI 转换器打通（8 千~8 万局新数据；**必须保留 dan/rate**）
P1  akochan 编译 + 能否吐 MJAI 动作（四麻外部锚点；只内部用）
P2  继续找有公开权重的强对手（60 分钟盒）
P3  **RL 最好种子 vs mac_rl 配对赛**（今天 20:00 前，2100 组）—— 决定今天能不能就换
P4  PyTorch 训练器（4060 上跑通一步）
```

## 五、外部 AI 进天梯/平台（用户今天问的）

```
现状：aimodel/external.py 已有 MJAI 子进程桥，arena_test 已能跑 external:manue ✓
      但 arena.py（天梯）/ webapp（平台）**还没接** ✗
今天：manue（BSD-3-Clause）上天梯+平台，必须带 超时/崩溃重启/房间清理 三件
明后天：akochan 需要 **TCP mjai server 壳**（它只当客户端，不能直连 stdio）
不能：Mortal（官方不公开权重 ✗ 已把 M6 口径改成 akochan）
```

## 六、进度板（改这里，别靠记）

- [ ] D：encode/engine 改完 + run_all 全绿
- [ ] 重抽 27 批 -> bc_all_v12.npz
- [ ] BC 4 种子 2 轮
- [ ] 配对赛 vs mac_rl（4 种子）
- [ ] 交 server 做 RL
- [ ] server：转换器 / akochan / RL vs mac_rl 数字
- [ ] manue 上天梯+平台
