"""Impulses: transitions that move their rate at once, in both engines."""

import math

import pytest
from pydantic import ValidationError

from commol import ModelBuilder, Simulation, AxisPattern
from commol.context.model import Model
from tests.support import TIGHT, assert_close, build, build_ode


def _impulse_model(
    *,
    ode: bool,
    p: float = 0.25,
    impulse_at: list[float] | None = None,
    impulse_every: tuple[float, float] | None = None,
) -> Model:
    """A → B moves only by an impulse of fraction ``p``; ``jumped`` counts it."""
    builder = ModelBuilder(name="impulse", version="1.0")
    builder.add_bin("A", "A").add_bin("B", "B").add_accumulator("jumped", "jumped")
    builder.add_parameter("p", p)
    builder.add_transition(
        "jump",
        ["A"],
        ["B"],
        rate="p",
        accumulators=["jumped"],
        impulse_at=impulse_at,
        impulse_every=impulse_every,
    )
    builder.set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
    )
    return build_ode(builder) if ode else build(builder)


class TestJump:
    @pytest.mark.parametrize("ode", [False, True])
    def test_an_impulse_moves_its_fraction_at_once(self, ode: bool) -> None:
        results = Simulation(_impulse_model(ode=ode, impulse_at=[1, 3])).run(4)

        assert results["A"] == pytest.approx([1000, 750, 750, 562.5, 562.5])

    @pytest.mark.parametrize("ode", [False, True])
    def test_an_accumulator_counts_the_jumped_amount(self, ode: bool) -> None:
        results = Simulation(_impulse_model(ode=ode, impulse_every=(2, 0))).run(4)

        assert results["jumped"] == pytest.approx([250, 250, 437.5, 437.5, 578.125])

    def test_the_jump_comes_before_the_flows_of_its_step(self) -> None:
        # The flow `k A` of step 1 sees A after the jump: 750, not 1000.
        builder = ModelBuilder(name="both", version="1.0")
        builder.add_bin("A", "A").add_bin("B", "B")
        builder.add_parameter("p", 0.25).add_parameter("k", 0.1)
        builder.add_transition("jump", ["A"], ["B"], rate="p", impulse_at=[1])
        builder.add_transition("flow", ["A"], ["B"], rate="k")
        builder.set_initial_conditions(
            population_size=1000,
            bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0}],
        )

        results = Simulation(build(builder)).run(2)

        assert results["A"] == pytest.approx([1000, 675, 607.5])

    def test_an_ode_impulse_may_act_between_integers(self) -> None:
        results = Simulation(_impulse_model(ode=True, impulse_at=[0.5])).run_at(
            [0.25, 0.5, 0.75]
        )

        assert results["A"] == pytest.approx([1000, 750, 750])

    def test_pulse_vaccination_reaches_the_periodic_orbit(self) -> None:
        # S' = mu K - mu S between jumps, S -> (1 - p) S at t = k T.
        mu, k, p, period = 0.02, 1000.0, 0.3, 5.0
        builder = ModelBuilder(name="pulse vaccination", version="1.0")
        builder.add_bin("S", "S").add_bin("V", "V")
        for name, value in {"mu": mu, "K": k, "p": p}.items():
            builder.add_parameter(name, value)
        builder.add_transition("birth", [], ["S"], rate="mu * K")
        builder.add_transition("death", ["S"], [], rate="mu")
        builder.add_transition(
            "vaccination", ["S"], ["V"], rate="p", impulse_every=(period, period)
        )
        builder.set_initial_conditions(
            population_size=1000,
            bin_fractions=[{"bin": "S", "fraction": 1.0}, {"bin": "V", "fraction": 0}],
        )
        steps = [j * period for j in range(1, 201)]

        results = Simulation(build_ode(builder, TIGHT)).run_at(steps)

        decay = math.exp(-mu * period)
        before = k * (1 - decay) / (1 - (1 - p) * decay)
        assert_close(results["S"][-1], (1 - p) * before, rel=1e-7)


class TestDefinition:
    def test_an_impulse_is_saved_and_loaded_back(self, tmp_path) -> None:
        model = _impulse_model(ode=True, impulse_every=(7.5, 0.5))
        path = tmp_path / "impulse.json"
        model.to_json(path)

        transition = Model.from_json(path).dynamics.transitions[0]

        assert transition.impulse_every == (7.5, 0.5)
        assert transition.is_impulse

    def test_a_difference_engine_names_an_impulse_between_steps(self) -> None:
        model = _impulse_model(ode=False, impulse_at=[0.5])

        with pytest.raises(RuntimeError, match="'jump' acts at 0.5"):
            Simulation(model)

    def test_impulse_at_and_impulse_every_exclude_each_other(self) -> None:
        with pytest.raises(ValidationError, match="not both"):
            _impulse_model(ode=True, impulse_at=[1], impulse_every=(2, 0))

    @pytest.mark.parametrize("schedule", [(0, 1), (-1, 0), (1, -1)])
    def test_the_period_is_positive_and_the_offset_not_negative(
        self, schedule: tuple[float, float]
    ) -> None:
        with pytest.raises(ValidationError, match="impulse"):
            _impulse_model(ode=True, impulse_every=schedule)

    def test_a_time_pattern_cannot_be_an_impulse(self) -> None:
        builder = ModelBuilder(name="m")
        builder.add_bin("A", "A").add_bin("B", "B")

        with pytest.raises(ValueError, match="not a AxisPattern"):
            builder.add_transition(
                "t",
                ["A"],
                ["B"],
                rate=AxisPattern.pulse(at=1, amount=0.1),
                impulse_at=[1],
            )

    def test_print_equations_lists_an_impulse_apart(self, capsys) -> None:
        _impulse_model(ode=True, impulse_at=[1, 3]).print_equations()

        printed = capsys.readouterr().out
        assert "dA/dt = 0" in printed
        assert "jump, impulse at 1, 3:\n  A -> B: p * A" in printed
