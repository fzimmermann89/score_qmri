"""Tests for schedule classes."""


from datasets.lightning import (
    ConstantSchedule,
    PhaseInSchedule,
    PhaseOutSchedule,
)


def test_constant_schedule():
    """Test ConstantSchedule returns constant weight."""
    schedule = ConstantSchedule(weight=1.0)
    assert schedule(0) == 1.0
    assert schedule(100) == 1.0
    assert schedule(1000000) == 1.0


def test_constant_schedule_different_weights():
    """Test ConstantSchedule with different weight values."""
    schedule_zero = ConstantSchedule(weight=0.0)
    schedule_one = ConstantSchedule(weight=1.0)
    schedule_two = ConstantSchedule(weight=2.0)

    assert schedule_zero(10) == 0.0
    assert schedule_one(10) == 1.0
    assert schedule_two(10) == 2.0


def test_phase_in_schedule_zero_steps():
    """Test PhaseInSchedule with zero steps (instant ramp)."""
    schedule = PhaseInSchedule(steps=0, weight=1.0)
    assert schedule(0) == 1.0
    assert schedule(100) == 1.0


def test_phase_in_schedule_linear_ramp():
    """Test PhaseInSchedule ramps linearly over steps."""
    schedule = PhaseInSchedule(steps=100, weight=2.0)

    # At step 0, should be 0
    assert schedule(0) == 0.0

    # At step 50 (midpoint), should be 1.0
    assert schedule(50) == 1.0

    # At step 100 (end), should be 2.0
    assert schedule(100) == 2.0

    # At step 200 (beyond), should be max weight
    assert schedule(200) == 2.0


def test_phase_in_schedule_negative_weight():
    """Test PhaseInSchedule with negative weight."""
    schedule = PhaseInSchedule(steps=10, weight=-1.0)

    # Should ramp from 0 to -1.0
    assert schedule(0) == 0.0
    assert schedule(10) == -1.0
    assert schedule(20) == -1.0


def test_phase_out_schedule_zero_steps():
    """Test PhaseOutSchedule with zero steps (instant drop to 0)."""
    schedule = PhaseOutSchedule(steps=0, weight=2.0)
    assert schedule(0) == 0.0
    assert schedule(100) == 0.0


def test_phase_out_schedule_linear_decay():
    """Test PhaseOutSchedule decays linearly over steps."""
    schedule = PhaseOutSchedule(steps=100, weight=3.0)

    # At step 0, should be 3.0
    assert schedule(0) == 3.0

    # At step 50 (midpoint), should be 1.5
    assert schedule(50) == 1.5

    # At step 100 (end), should be 0
    assert schedule(100) == 0.0

    # At step 200 (beyond), should be 0
    assert schedule(200) == 0.0


def test_phase_out_schedule_negative_weight():
    """Test PhaseOutSchedule with negative weight."""
    schedule = PhaseOutSchedule(steps=10, weight=-2.0)

    # Should decay from -2.0 to 0
    assert schedule(0) == -2.0
    assert schedule(10) == 0.0
    assert schedule(20) == 0.0


def test_schedule_large_steps():
    """Test schedules with very large step values."""
    const_schedule = ConstantSchedule(weight=5.0)
    phase_in_schedule = PhaseInSchedule(steps=1000000, weight=5.0)
    phase_out_schedule = PhaseOutSchedule(steps=1000000, weight=5.0)

    # All should handle large step values correctly
    assert const_schedule(1000000) == 5.0
    assert phase_in_schedule(1000000) == 5.0
    assert phase_out_schedule(1000000) == 0.0
