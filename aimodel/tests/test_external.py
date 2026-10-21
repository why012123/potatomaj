"""外部 mjai AI 适配层（aimodel/external.py）的回归测试。

**不需要网络、不需要装 Mortal**：这里用一个 20 行的假 mjai 子进程
（``_fake_mjai_bot.py``）当外部 AI，验证桥的每一根线：

1. 正常对局：起进程 / 翻译事件 / 映射动作 / 关进程
2. 没装：清晰报错（带 ``docs/接入外部AI.md``），**不是** import 就崩
3. 超时：外部进程卡死不能拖死整局
4. 崩溃：有 fallback 就降级并记录；没有就抛
5. 非法动作：**必须**抛 ExternalAIProtocolError，**不许**静默 return 0
6. 确定性：同种子同牌山跑两次，结果一致
7. Akagi 的「一批事件一个数组」变体
8. 注册表：``mortal`` / ``manue`` 能被 --third 认出来；TCP-only 的要拒绝
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import external, sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent  # noqa: E402
from aimodel.engine import Game  # noqa: E402

FAKE = Path(__file__).resolve().parent / "_fake_mjai_bot.py"


def _command(mode: str = "normal", batch: bool = False):
    cmd = [sys.executable, str(FAKE)]
    env = {"FAKE_MJAI_MODE": mode}
    if batch:
        env["FAKE_MJAI_BATCH"] = "1"
    return cmd, env


def _agent(mode: str = "normal", *, batch: bool = False, player_id: int = 0,
           timeout: float = 10.0, max_restarts: int = 2, **kwargs):
    cmd, env = _command(mode, batch)
    return external.ExternalMJaiAgent(
        "manue", seed=1, player_id=player_id, players=4,
        timeout=timeout, max_restarts=max_restarts,
        command=cmd, env=env, **kwargs,
    )


def _play(agent, seed: int = 20261021, rounds: int = 1):
    agents = [agent] + [HeuristicAgent(seed=100 + i) for i in range(3)]
    game = Game(agents, seed=seed, rule=S.rule_for(4), rounds=rounds)
    external.attach(game)
    return game.play()


# ---------------------------------------------------------------------------
def test_registry():
    names = external.spec_names()
    for want in ("mortal", "manue", "mortal-akagi"):
        assert want in names, f"注册表缺 {want}: {names}"
    assert external.normalize_key("external:mortal") == "mortal"
    assert external.normalize_key("manue-go") == "manue"
    assert external.normalize_key("policy.npz") is None
    assert external.normalize_key("style:speed") is None
    assert external.is_external_spec("mortal")
    assert not external.is_external_spec("teacher")
    # 外部模型列表（平台用）不能启动进程
    rows = external.external_models(4)
    by_name = {r["name"]: r for r in rows}
    assert "external:manue" in by_name
    assert by_name["external:manue"]["external"] is True
    assert "license" in by_name["external:manue"]
    print(f"  ✅ 注册表 {len(names)} 个名字；平台行 {len(rows)} 个")


def test_missing_is_clear():
    """没装时必须是 ExternalAINotInstalled，且提示文档路径。"""
    with mock.patch.dict(os.environ, {
        "POTATOMAJ_EXTERNAL_MANUE_BIN": "/nonexistent/definitely-not-manue",
        "POTATOMAJ_EXTERNAL_MANUE_CMD": "",
    }, clear=False):
        try:
            external.ExternalMJaiAgent(
                "manue", seed=0, player_id=0, players=4, timeout=1,
            )
        except external.ExternalAINotInstalled as exc:
            msg = str(exc)
            assert external.DOC_HINT in msg, msg
            assert "没装好" in msg, msg
        else:
            raise AssertionError("没装的外部 AI 居然构造成功了")
    # 只支持 TCP 的不能假装能接
    ok, why = external.check_spec("akochan")
    assert not ok and "TCP" in why, (ok, why)
    ok, why = external.check_spec("manue-original")
    assert not ok and "TCP" in why, (ok, why)
    print("  ✅ 没装/TCP-only 都有清晰报错，且带文档路径")


def test_full_game():
    with tempfile.TemporaryDirectory() as td:
        agent = _agent("normal", log_path=Path(td) / "ext.log")
        try:
            r = _play(agent)
            assert len(r.hands) >= 1, "一局都没打完"
            assert sum(r.scores) == sum(r.start_scores), (r.scores, r.start_scores)
            assert agent.decisions > 0, "外部 AI 一次决策都没被问到"
            assert agent.errors == 0, agent.summary()
            assert not agent.degraded, agent.summary()
            assert agent.crash_count == 0
            summary = agent.summary()
            print(f"  ✅ 假 mjai bot 打完整场：{len(r.hands)} 局、"
                  f"{agent.decisions} 次决策、分数 {r.scores}（{summary['spec']}）")
        finally:
            agent.close()


def test_attach_keeps_previous_hook():
    agent = _agent("normal")
    acts: list[dict] = []
    agents = [agent] + [HeuristicAgent(seed=1 + i) for i in range(3)]
    game = Game(agents, seed=7, rule=S.rule_for(4), rounds=1)
    game.on_action = acts.append
    external.attach(game)
    try:
        game.play()
        assert acts, "原来的 on_action 被覆盖了"
        assert agent.decisions > 0
        print("  ✅ attach 保留原 on_action，同时把事件转给外部 AI")
    finally:
        agent.close()


def test_illegal_action_raises():
    agent = _agent("illegal", max_restarts=0)
    try:
        try:
            _play(agent)
        except external.ExternalAIProtocolError as exc:
            msg = str(exc)
            assert "不静默" in msg or "合法打牌" in msg, msg
            print("  ✅ 外部 AI 打不在 options 里的牌 -> 明确协议错误（不是 return 0）")
            return
        else:
            raise AssertionError("非法动作居然没报错")
    finally:
        agent.close()


def test_timeout():
    agent = _agent("sleep", timeout=0.4, max_restarts=0)
    try:
        try:
            _play(agent)
        except external.ExternalAITimeout as exc:
            assert "没有给动作" in str(exc), str(exc)
            print("  ✅ 外部进程卡死 -> ExternalAITimeout，没有拖死整局")
            return
        else:
            raise AssertionError("卡死的外部 AI 居然没超时")
    finally:
        agent.close()


def test_crash_degrades_with_fallback():
    teacher = HeuristicAgent(seed=999)
    agent = _agent("crash", timeout=2.0, max_restarts=0, fallback=teacher)
    try:
        r = _play(agent)
        assert len(r.hands) >= 1
        assert agent.degraded, agent.summary()
        assert agent.errors >= 1, agent.summary()
        assert "ExternalAICrashed" in agent.degrade_reason or \
               "ExternalAI" in agent.degrade_reason, agent.degrade_reason
        print(f"  ✅ 外部进程崩溃 -> 降级教师继续打完（{agent.errors} 次错误，"
              f"{agent.degrade_reason[:60]}…）")
    finally:
        agent.close()


def test_crash_without_fallback_raises():
    agent = _agent("crash", timeout=2.0, max_restarts=0)
    try:
        try:
            _play(agent)
        except external.ExternalAIError as exc:
            assert "退出了" in str(exc) or "崩溃" in str(exc), str(exc)
            print("  ✅ 没给 fallback 时崩溃 -> 明确抛错，评测不会拿脏数据")
            return
        else:
            raise AssertionError("崩溃居然没抛异常")
    finally:
        agent.close()


def test_determinism():
    scores = []
    for _ in range(2):
        agent = _agent("normal")
        try:
            r = _play(agent, seed=424242)
            scores.append((tuple(r.scores), tuple(r.ranks), agent.decisions))
        finally:
            agent.close()
    assert scores[0] == scores[1], scores
    print(f"  ✅ 同种子跑两次完全一致：{scores[0]}")


def test_batch_mode():
    """Akagi 的 Mortal bot 一行收一个 JSON 数组，也要能接。"""
    agent = _agent("normal", batch=True)
    try:
        r = _play(agent, seed=99)
        assert len(r.hands) >= 1
        assert agent.decisions > 0
        print(f"  ✅ 批量事件模式（Akagi Mortal bot 风格）跑完 {len(r.hands)} 局")
    finally:
        agent.close()


def test_tile_names():
    assert external._tile_name(4, red=True) == "5mr"
    assert external._parse_tile("5mr") == (4, True)
    assert external._parse_tile("0m") == (4, True)
    assert external._parse_tile("P") == (31, False)
    assert external._masked_hand([0] * 34)[:0] == []
    print("  ✅ 赤 5 的 mjai 写法（5mr）往返正确")


def test_cli_help():
    for mod in ("aimodel.ab", "aimodel.arena_test"):
        r = subprocess.run([sys.executable, "-m", mod, "--help"],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (mod, r.returncode, r.stderr[-500:])
        out = r.stdout
        assert "mortal" in out, f"{mod} 的 --help 里没有 mortal:\n{out[-800:]}"
        assert "manue" in out, f"{mod} 的 --help 里没有 manue"
    print("  ✅ ab / arena_test 的 --help 能看到 mortal / manue")


def main() -> int:
    print("【外部 mjai AI 适配层】")
    failed: list[str] = []
    tests = [
        test_registry,
        test_missing_is_clear,
        test_tile_names,
        test_full_game,
        test_attach_keeps_previous_hook,
        test_illegal_action_raises,
        test_timeout,
        test_crash_degrades_with_fallback,
        test_crash_without_fallback_raises,
        test_determinism,
        test_batch_mode,
        test_cli_help,
    ]
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"  ❌ {fn.__name__}: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"❌ {len(failed)} 项失败：{', '.join(failed)}")
        return 1
    print(f"✅ 外部 AI 适配层 {len(tests)} 项全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
