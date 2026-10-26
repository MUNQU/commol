"""Quantities of interest: a series and its reduction, the rules every
quantity follows, the ratio of two quantities, the form they cross to Rust in,
and a propagation filtered by a condition."""

import json
import math

import numpy as np
import pytest
from pydantic import TypeAdapter

from commol import (
    PRCC,
    Aggregate,
    AxisPattern,
    Gain,
    Increments,
    InputSpace,
    Level,
    ModelBuilder,
    Morris,
    ParameterSample,
    Peak,
    PeakPosition,
    PolynomialChaos,
    Propagation,
    Ratio,
    SampleMethod,
    SensitivityAnalysis,
    Simulation,
    Sobol,
    Trajectory,
    ValueAt,
    member_statistics,
    weighted_quantiles,
)
from commol.context.inference.global_sensitivity import (
    PropagationResult,
)
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.engine_json import engine_json
from commol.context.inference.quantities import _Quantity, quantity_labels
from tests.support import bounded, build
from tests.support.models import a_to_b_builder

ENDS = [7.0, 14.0, 21.0]
"""The ends of three windows of 7."""


def _inflow(gains: list[float]) -> SensitivityAnalysis:
    """
    An analysis of one run of a model whose accumulator `acc` gains each of
    `gains` in one window of 7, from a pulse in the middle of the window.
    """
    model = (
        ModelBuilder(name="Inflow", version="1.0")
        .add_bin(id="A", name="A")
        .add_accumulator(id="acc", name="acc")
        .add_parameter(id="k", value=1.0)
        .add_transition(
            id="in",
            source=[],
            target=["A"],
            rate=AxisPattern.pulses([3, 10, 17], [f"{gain} * k" for gain in gains]),
            accumulators=["acc"],
        )
        .build(Aggregate(bin_amounts={"A": 0}), engine="DifferenceEquations")
    )
    sample = ParameterSample(
        names=["k"],
        bounds=[(0.5, 2.0)],
        values=[[1.0]],
        log_weights=[0.0],
        method=SampleMethod(name="one run"),
    )
    inputs = InputSpace.from_sample(sample, [bounded("k", 0.5, 2.0)])
    return SensitivityAnalysis(Simulation(model), inputs, seed=1)


def _values(analysis: SensitivityAnalysis, quantities: list[_Quantity]) -> list[float]:
    """The value of each scalar of `quantities` in the one run."""
    return analysis.run(Propagation(quantities=quantities)).values[0]


class TestSeries:
    def test_the_peak_of_increments_is_the_largest_gain_of_a_window(self) -> None:
        analysis = _inflow([1.0, 4.0, 2.0])
        weekly = Increments("acc", window=7)

        values = _values(
            analysis,
            [
                Peak(series=weekly, positions=ENDS),
                PeakPosition(series=weekly, positions=ENDS),
                Trajectory(series=weekly, positions=ENDS),
                ValueAt(series=Increments("acc", 14), at=21),
            ],
        )

        assert values == [4.0, 14.0, 1.0, 4.0, 2.0, 6.0]

    def test_increments_cannot_start_before_zero(self) -> None:
        with pytest.raises(ValueError, match=r"window 7 that ends at 5 starts before"):
            ValueAt(series=Increments("acc", 7), at=5)
        with pytest.raises(ValueError, match=r"window 7 that ends at 0 starts before"):
            Peak(series=Increments("acc", 7), positions=[0, 7])
        with pytest.raises(ValueError, match=r"greater than 0"):
            Increments("acc", 0)
        # A window that starts at 0 reads the first row.
        assert ValueAt(series=Increments("acc", 7), at=7).at == 7

    def test_the_peak_of_a_level_reads_the_level(self) -> None:
        analysis = _inflow([1.0, 4.0, 2.0])

        values = _values(
            analysis,
            [
                Peak(series="acc", positions=ENDS),
                PeakPosition(series=["acc"], positions=ENDS),
                ValueAt(series=Level("acc"), at=21),
            ],
        )

        # An accumulator never falls: its peak is its value at the last position.
        assert values == [7.0, 21.0, 7.0]

    def test_a_name_is_the_level_of_that_output(self) -> None:
        assert ValueAt(series="A", at=1).series == Level(["A"])
        assert Peak(series=["A", "B"], positions=[1]).series == Level(["A", "B"])
        assert Gain(outputs="A", start=0, end=1).outputs == ["A"]
        with pytest.raises(ValueError, match=r"at least 1 item"):
            ValueAt(series=[], at=1)


class TestRules:
    def test_a_tie_gives_the_first_position(self) -> None:
        analysis = _inflow([2.0, 2.0, 1.0])

        values = _values(
            analysis,
            [PeakPosition(series=Increments("acc", 7), positions=ENDS)],
        )

        assert values == [7.0]

    def test_every_reduction_across_runs_rejects_a_nan(self) -> None:
        # The sum of A, B and acc is above the largest float from position 1,
        # so its increments are inf − inf, NaN, at every run.
        simulation = Simulation(
            build(
                a_to_b_builder(amounts={"A": 0.9e308, "B": 0.85e308}, accumulator="acc")
            )
        )
        inputs = InputSpace.uniform_on_bounds([bounded("k", 0.1, 0.5)])
        analysis = SensitivityAnalysis(simulation, inputs, seed=1)
        peak = Peak(series=Increments(["A", "B", "acc"], 1), positions=[1, 2])

        propagation = analysis.run(Propagation(draws=3, quantities=[peak]))

        assert all(math.isnan(row[0]) for row in propagation.values)
        for method in [
            PRCC(draws=5, quantities=[peak]),
            Morris(trajectories=2, levels=4, quantities=[peak]),
            Sobol(draws=4, replicates=2, confidence_level=0.9, quantities=[peak]),
            PolynomialChaos(degree=1, draws=4, quantities=[peak]),
        ]:
            with pytest.raises(
                ValueError, match=r"quantity 0 .* NaN .* first at the point 0"
            ):
                analysis.run(method)
        with pytest.raises(ValueError, match=r"'peak of A \+ B \+ acc gain over"):
            propagation.summary(0.9)
        with pytest.raises(ValueError, match=r"values at \[1\] are NaN"):
            member_statistics([1.0, math.nan], 0.9)
        with pytest.raises(ValueError, match=r"values at \[0\] are NaN"):
            weighted_quantiles([math.nan], [0.5])


def _two_inflows() -> SensitivityAnalysis:
    """
    An analysis of a sample of two draws, `k` = 1 and 3, of a model whose
    accumulator `n` gains ``2 k`` and `d` gains ``4.5 - 0.5 k`` over (0, 7].
    """
    model = (
        ModelBuilder(name="Two inflows", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_accumulator(id="n", name="n")
        .add_accumulator(id="d", name="d")
        .add_parameter(id="k", value=1.0)
        .add_transition(
            id="to_a",
            source=[],
            target=["A"],
            rate=AxisPattern.pulses([3], ["2 * k"]),
            accumulators=["n"],
        )
        .add_transition(
            id="to_b",
            source=[],
            target=["B"],
            rate=AxisPattern.pulses([3], ["4.5 - 0.5 * k"]),
            accumulators=["d"],
        )
        .build(Aggregate(bin_amounts={"A": 0, "B": 0}), engine="DifferenceEquations")
    )
    sample = ParameterSample(
        names=["k"],
        bounds=[(0.5, 4.0)],
        values=[[1.0], [3.0]],
        log_weights=[0.0, 0.0],
        method=SampleMethod(name="two runs"),
    )
    inputs = InputSpace.from_sample(sample, [bounded("k", 0.5, 4.0)])
    return SensitivityAnalysis(Simulation(model), inputs, seed=1)


class TestRatio:
    def test_a_ratio_is_taken_within_each_run(self) -> None:
        analysis = _two_inflows()
        ratio = Ratio(
            numerator=Gain(outputs="n", start=0, end=7),
            denominator=Gain(outputs="d", start=0, end=7),
            replicate_rule="within",
        )

        result = analysis.run(Propagation(quantities=[ratio]))

        assert result.values == [[0.5], [2.0]]
        assert result.labels == ["(n gain over (0, 7]) / (d gain over (0, 7])"]

    def test_a_ratio_takes_values_only(self) -> None:
        value = ValueAt(series="B", at=1)

        with pytest.raises(ValueError, match=r"got the PeakPosition 'when'"):
            Ratio(
                numerator=PeakPosition(series="B", positions=[1], label="when"),
                denominator=value,
                replicate_rule="pooled",
            )
        with pytest.raises(ValueError, match=r"got the Trajectory 'B at 1'"):
            Ratio(
                numerator=value,
                denominator=Trajectory(series="B", positions=[1]),
                replicate_rule="within",
            )
        with pytest.raises(ValueError, match=r"replicate_rule"):
            Ratio(numerator=value, denominator=value)  # ty: ignore[missing-argument]


class TestLabels:
    def test_each_quantity_has_a_label(self) -> None:
        labels = quantity_labels(
            [
                ValueAt(series="B", at=10),
                Gain(outputs=["B", "C"], start=5, end=10),
                Peak(series="B", positions=[1, 2, 3], label="peak"),
                PeakPosition(series="B", positions=[1, 2]),
                Trajectory(series="C", positions=[1, 2]),
                Peak(series=Increments("B", 0.5), positions=[1]),
            ]
        )
        assert labels == [
            "B at 10",
            "B + C gain over (5, 10]",
            "peak",
            "position of the peak of B",
            "C at 1",
            "C at 2",
            "peak of B gain over windows of 0.5",
        ]

    def test_the_labels_of_close_positions_differ(self) -> None:
        labels = quantity_labels(
            [
                Trajectory(series="B", positions=[1000000, 1000001]),
                ValueAt(series="B", at=0.1234567),
                ValueAt(series="B", at=0.1234568),
                Gain(outputs="C", start=1e-7, end=2.5e16),
            ]
        )
        assert labels == [
            "B at 1000000",
            "B at 1000001",
            "B at 0.1234567",
            "B at 0.1234568",
            "C gain over (1e-07, 2.5e+16]",
        ]

    def test_a_quantity_reads_a_valid_window_of_positions(self) -> None:
        with pytest.raises(ValueError, match=r"ends after it starts"):
            Gain(outputs="C", start=10, end=10)
        with pytest.raises(ValueError, match=r"ascend"):
            Peak(series="B", positions=[2, 1])
        with pytest.raises(ValueError):
            ValueAt(series="B", at=-1)

    def test_quantities_have_labels_of_their_own(self) -> None:
        with pytest.raises(ValueError, match=r"\['B at 1'\]"):
            quantity_labels(
                [ValueAt(series="B", at=1), Trajectory(series="B", positions=[1])]
            )
        with pytest.raises(ValueError, match=r"at least one quantity"):
            quantity_labels([])


class TestBoundary:
    def test_each_quantity_crosses_to_rust_as_it_is(self) -> None:
        quantities: list[_Quantity] = [
            ValueAt(series=["A", "B"], at=1.5),
            ValueAt(series=Increments("A", 2), at=3),
            Gain(outputs="A", start=0.1, end=0.4),
            Peak(series=Increments("A", 7), positions=ENDS),
            PeakPosition(series="A", positions=[1, 2]),
            Trajectory(series=Increments(["A", "B"], 1), positions=[1, 2]),
            Ratio(
                numerator=Peak(series="A", positions=[1, 2]),
                denominator=Gain(outputs="B", start=0, end=1),
                replicate_rule="within_positive",
            ),
        ]
        read = [
            commol_rs.json.as_read(type(quantity).__name__, engine_json(quantity))
            for quantity in quantities
        ]
        labelled = Ratio(
            numerator=ValueAt(series="A", at=1, label="a"),
            denominator=ValueAt(series="B", at=1, label="b"),
            replicate_rule="pooled",
            label="a over b",
        )

        adapter: TypeAdapter[_Quantity] = TypeAdapter(_Quantity)
        assert [adapter.validate_json(text) for text in read] == quantities
        assert (
            json.loads(commol_rs.json.as_read("Ratio", engine_json(labelled)))["label"]
            == "a over b"
        )
        assert labelled.labels() == ["a over b"]
        with pytest.raises(ValueError, match=r"kind"):
            commol_rs.json.as_read("ValueAt", '{"kind": "mean"}')


def _propagation() -> PropagationResult:
    """Four runs of two quantities; the second is NaN in the second run."""
    return PropagationResult(
        sample=ParameterSample(
            names=["k"],
            bounds=[(0.0, 1.0)],
            values=[[0.1], [0.2], [0.3], [0.4]],
            log_weights=[0.0, 0.0, 0.0, 0.0],
            method=SampleMethod(name="test"),
        ),
        labels=["x", "y"],
        values=[[1.0, 10.0], [2.0, math.nan], [3.0, 30.0], [4.0, 40.0]],
        weights=[0.1, 0.2, 0.3, 0.4],
        draw_seeds=[1, 2, 3, 4],
    )


class TestFilter:
    def test_a_filtered_propagation_renormalises_its_weights(self) -> None:
        result = _propagation()
        values = np.asarray(result.values)

        kept = result.where(np.isfinite(values[:, 1]))

        assert kept.values == [[1.0, 10.0], [3.0, 30.0], [4.0, 40.0]]
        assert kept.weights == pytest.approx([0.125, 0.375, 0.5])
        assert kept.sample == result.sample
        for summary, index in zip(kept.summary(0.9), [0, 1]):
            column = [row[index] for row in kept.values]
            statistics = member_statistics(column, 0.9, kept.weights)
            assert summary.mean == pytest.approx(statistics["mean"])
            assert summary.median == pytest.approx(statistics["median"])
            assert summary.lower == pytest.approx(statistics["ci_lower"])
            assert summary.upper == pytest.approx(statistics["ci_upper"])
        weights = [0.1, 0.3, 0.4]
        assert kept.effective_size == pytest.approx(
            sum(weights) ** 2 / sum(w * w for w in weights)
        )
        assert result.effective_size == pytest.approx(1 / 0.3)

    def test_a_mask_must_keep_a_row_of_each_run(self) -> None:
        result = _propagation()

        with pytest.raises(ValueError, match=r"keeps no run"):
            result.where([False] * 4)
        with pytest.raises(ValueError, match=r"one boolean per run: 4 runs"):
            result.where([True] * 3)
        with pytest.raises(ValueError, match=r"one boolean per run"):
            result.where([1, 0, 1, 1])
        zero = result.model_copy(update={"weights": [0.0, 0.5, 0.5, 0.0]})
        with pytest.raises(ValueError, match=r"positive weight"):
            zero.where([True, False, False, True])

    def test_the_effective_size_of_a_propagation_comes_from_the_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = commol_rs.inference.effective_size
        read: list[list[float]] = []

        def recorded(weights: list[float]) -> float:
            read.append(weights)
            return engine(weights)

        monkeypatch.setattr(commol_rs.inference, "effective_size", recorded)
        # Two runs of weights 3 and 1: (3 + 1)² / (3² + 1²).
        weights = [0.75, 0.25, 0.0, 0.0]
        result = _propagation().model_copy(update={"weights": weights})

        assert result.effective_size == pytest.approx(16 / 10)
        assert read == [weights]


@pytest.mark.parametrize("rule", ["within", "within_positive", "pooled"])
def test_a_ratio_takes_a_replicate_rule(rule: str) -> None:
    value = ValueAt(series="A", at=1)
    ratio = Ratio(numerator=value, denominator=value, replicate_rule=rule)

    read = json.loads(commol_rs.json.as_read("Ratio", engine_json(ratio)))
    assert read["replicate_rule"] == rule
