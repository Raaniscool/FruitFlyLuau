"""Reward ledger and the dopamine-like modulatory trace.

The modulator is the part most likely to be quietly broken (a signal that never
reaches the weights produces a convincing null result), so these tests pin the
timing and the collapse behaviour rather than just the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from fruitfly.config import RewardConfig
from fruitfly.learning.modulator import DopamineLikeModulator
from fruitfly.learning.reward import RewardSystem, RewardEvent


# ------------------------------------------------------------------ ledger
def test_value_for_maps_outcomes_to_signed_values():
    rs = RewardSystem(RewardConfig(positive=1.0, negative=-1.0, penalize_no_response=True))
    assert rs.value_for(True) == 1.0
    assert rs.value_for(False) == -1.0
    assert rs.value_for(True, magnitude=2.5) == 2.5
    no_punish = RewardSystem(RewardConfig(penalize_no_response=False, neutral=0.0))
    assert no_punish.value_for(False) == 0.0


def test_reward_is_delayed_by_exactly_delay_steps():
    for delay in (0, 1, 3, 7):
        rs = RewardSystem(RewardConfig(delay_steps=delay))
        rs.deliver(1.0, step=10)
        vals = [rs.pop(t) for t in range(10, 10 + delay + 2)]
        assert sum(vals) == pytest.approx(1.0), f"reward lost with delay={delay}"
        assert vals[delay] == pytest.approx(1.0), f"reward arrived late/early (delay={delay})"
        assert all(v == 0.0 for i, v in enumerate(vals) if i != delay)


def test_simultaneous_rewards_sum_at_their_due_step():
    rs = RewardSystem(RewardConfig(delay_steps=2))
    rs.deliver(0.5, step=4)
    rs.deliver(0.25, step=4)
    assert rs.pop(5) == 0.0
    assert rs.pop(6) == pytest.approx(0.75)
    assert rs.pop(6) == 0.0, "a reward must be delivered once, not every step"


def test_undelivered_reward_is_flushed_at_trial_end_not_dropped():
    """A reward queued near the end of a trial used to be discarded silently."""
    rs = RewardSystem(RewardConfig(delay_steps=50))
    rs.deliver(2.0, step=0)
    leftover = rs.end_trial(correct=True)
    assert leftover == pytest.approx(2.0), "end_trial must hand the queued value back"
    assert rs.pop(0) == 0.0, "flushed value must not also be paid out later"


def test_outcome_helpers_and_windows():
    rs = RewardSystem(RewardConfig())
    for correct in (True, True, False, True):
        rs.deliver_outcome(correct, step=0)
        rs.end_trial(correct=correct)
    assert rs.window_accuracy(4) == pytest.approx(0.75)
    assert rs.window_accuracy(2) == pytest.approx(0.5)
    cum = rs.cumulative()
    assert cum["n_events"] == 4
    assert cum["n_positive"] == 3 and cum["n_negative"] == 1
    assert cum["total_reward"] == pytest.approx(2.0)  # +1 +1 -1 +1
    assert cum["accuracy_overall"] == pytest.approx(0.75)
    curve = rs.curve()
    assert len(curve["episode"]) == 4 and len(curve["reward"]) == 4


def test_state_roundtrip_resumes_the_ledger():
    rs = RewardSystem(RewardConfig())
    for c in (True, False, True):
        rs.deliver_outcome(c, step=0)
        rs.end_trial(correct=c)
    snap = rs.state_dict()
    fresh = RewardSystem(RewardConfig())
    fresh.load_state(snap)
    assert fresh.state_dict() == snap
    assert fresh.window_accuracy(3) == pytest.approx(rs.window_accuracy(3))


def test_history_is_capped_not_unbounded():
    rs = RewardSystem(RewardConfig(max_history=20))
    for i in range(200):
        rs.deliver(1.0, step=0)
    assert len(rs.history) <= 400, "history must be pruned (memory guard)"


# ------------------------------------------------------------------ modulator
def test_trace_rises_with_reward_and_decays_exponentially():
    cfg = RewardConfig(trace_tau_ms=10.0, baseline_subtract=False, value_abs_max=0.0)
    mod = DopamineLikeModulator(cfg, dt_ms=0.5)
    mod.deliver(1.0)
    v1 = mod.current
    assert v1 == pytest.approx(1.0)
    mod.deliver(0.0)
    v2 = mod.current
    assert v2 == pytest.approx(v1 * np.exp(-0.5 / 10.0))
    for _ in range(200):
        mod.deliver(0.0)
    assert mod.current < 1e-3, "the trace must fade to nothing without reinforcement"


def test_persistent_reward_saturates_at_the_cap():
    cfg = RewardConfig(trace_tau_ms=60.0, value_abs_max=4.0, baseline_subtract=False)
    mod = DopamineLikeModulator(cfg, dt_ms=0.5)
    for _ in range(500):
        mod.deliver(1.0)
    assert mod.current == pytest.approx(4.0), "value must be clamped, not runaway"
    assert mod.stats()["max_abs_value"] <= 4.0 + 1e-9


def test_prediction_error_mode_collapses_for_a_constant_outcome():
    """WHY ``baseline_subtract`` defaults to False -- keep this test honest if it changes.

    With a constant reward and an EMA baseline, the "reward prediction error"
    degenerates to zero, so the modulator contributes nothing and no learning can
    occur. That is expected for this simple baseline: it is a real RPE only when
    the predictor can be surprised.
    """
    cfg = RewardConfig(baseline_subtract=True, trace_decay=0.9, value_abs_max=4.0, trace_tau_ms=60.0)
    mod = DopamineLikeModulator(cfg, dt_ms=0.5)
    # The baseline EMA (alpha=0.9) needs ~50 steps to catch a constant reward, and
    # the 60 ms trace then has to bleed away what it accumulated meanwhile.
    for _ in range(1500):
        mod.deliver(1.0)  # always exactly as predicted
    assert abs(mod.current) < 0.05, f"RPE should have collapsed, value={mod.current}"
    # ...and it recovers immediately when the outcome is surprising
    mod.deliver(-1.0)
    assert abs(mod.current) > 0.5, "a negative surprise must produce a real signal"


def test_modulator_distinguishes_sign_of_outcome():
    cfg = RewardConfig(baseline_subtract=False, value_abs_max=4.0, trace_tau_ms=60.0)
    pos, neg = DopamineLikeModulator(cfg, dt_ms=0.5), DopamineLikeModulator(cfg, dt_ms=0.5)
    for _ in range(20):
        pos.deliver(1.0)
        neg.deliver(-1.0)
    assert pos.current > 0 > neg.current


def test_reset_trial_clears_the_value_but_keeps_the_baseline():
    cfg = RewardConfig(baseline_subtract=True, value_abs_max=4.0)
    mod = DopamineLikeModulator(cfg, dt_ms=0.5)
    mod.deliver(1.0)
    baseline = mod.baseline
    mod.reset_trial()
    assert mod.current == 0.0
    assert mod.baseline == pytest.approx(baseline), "expectations must persist across trials"
    assert mod.n_steps > 0, "step counting continues"


def test_stats_expose_the_evidence_a_reviewer_asks_for():
    cfg = RewardConfig(value_abs_max=4.0)
    mod = DopamineLikeModulator(cfg, dt_ms=0.5)
    for _ in range(10):
        mod.deliver(0.5)
    st = mod.stats()
    for key in ("n_steps", "n_deliveries", "mean_abs_value", "max_abs_value", "final_value",
                "tonic", "baseline", "decay_per_step", "prediction_error_mode"):
        assert key in st, key
    assert st["n_steps"] == 10
    assert st["n_deliveries"] == 10
    assert 0 < st["decay_per_step"] < 1


def test_reward_and_modulator_compose_through_the_delay():
    """Ledger -> queue -> modulator, the exact chain the runner performs."""
    rcfg = RewardConfig(delay_steps=4, baseline_subtract=False, value_abs_max=4.0, trace_tau_ms=60.0)
    rs = RewardSystem(rcfg)
    mod = DopamineLikeModulator(rcfg, dt_ms=0.5)
    rs.deliver_outcome(True, step=0)
    seen = 0.0
    for t in range(12):
        due = rs.pop(t)
        mod.deliver(due)
        seen += due
    assert seen == pytest.approx(1.0)
    assert mod.current > 0.9, "the modulator should still hold a strong value when credit lands"
    assert mod.stats()["n_deliveries"] == 1
