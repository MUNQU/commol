"""Identifiability from a sample: what it learned against its reference, the
reasons of its verdicts and the warnings of its section."""

import json
import math
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from commol import Calibrator
from commol.distributions import Beta, Gamma, Priors
from commol.identifiability import (
    IdentifiabilityReport,
    SampleIdentifiability,
    SampleSection,
)
from commol.uncertainty import ParameterSample, SampleMethod

GIVEN = SampleMethod(name="given")
THRESHOLDS = {
    "correlation_threshold": 0.9,
    "contraction_threshold": 0.5,
    "bound_share": 0.05,
}
BOUNDS = [(0.0, 0.5), (0.0, 0.3)]
CONSTANT_K2 = (
    "Constant in every draw: k2. The sample did not explore these parameters, so "
    "it cannot say that the data fix them."
)


def _section(
    calibrator: Calibrator, sample: ParameterSample, **settings: list[str]
) -> SampleSection:
    """The section of `sample` against its bounds."""
    report = calibrator.identifiability(
        SampleIdentifiability(
            sample=sample, reference="bounds", **THRESHOLDS, **settings
        )
    )
    return report.sample[0]


def _reasons(
    calibrator: Calibrator, sample: ParameterSample
) -> dict[str, tuple[str, list[str]]]:
    """The practical verdict and reasons of each parameter of `sample`."""
    return _verdicts(
        calibrator.identifiability(
            SampleIdentifiability(sample=sample, reference="bounds", **THRESHOLDS)
        )
    )


def _verdicts(report: IdentifiabilityReport) -> dict[str, tuple[str, list[str]]]:
    """The practical verdict and reasons of each parameter of `report`."""
    return {
        name: (verdict.practical.verdict, verdict.practical.reasons)
        for name, verdict in report.verdicts.items()
        if verdict.practical is not None
    }


def test_a_sample_reports_what_it_learned(calibrator: Calibrator) -> None:
    # k1 spreads over its bounds; k2 follows it on a line.
    k1 = [0.5 * (i + 0.5) / 100 for i in range(100)]
    sample = ParameterSample.equally_weighted(
        ["k1", "k2"], BOUNDS, [[k, 0.3 - 0.6 * k] for k in k1], GIVEN
    )

    section = _section(calibrator, sample)

    assert section.contraction["k1"] == pytest.approx(0.0, abs=0.05)
    assert section.correlation["k1"]["k2"] == pytest.approx(-1.0)
    assert [(p.first, p.second) for p in section.correlated_pairs] == [("k1", "k2")]
    # 5 of 100 draws in each outer 5 % of the bounds; one bin per draw share.
    assert section.mass_below["k1"] == pytest.approx(0.05)
    assert section.mass_above["k1"] == pytest.approx(0.05)
    assert section.bins == math.ceil(2 * 100 ** (1 / 3))
    assert section.overlap["k1"] == pytest.approx(1.0)
    assert section.at_bound == {"k1": [], "k2": []}
    # Along the line the sample spreads as its reference; across it, not at all.
    least, most = section.directions
    assert least.contraction < 0.0
    assert most.contraction == pytest.approx(1.0)
    assert most.scales == {"k1": "natural", "k2": "natural"}
    assert section.hidden_ridges == 0
    assert section.warnings == []


def test_a_sample_gives_the_reason_of_each_verdict(calibrator: Calibrator) -> None:
    # k1 and k2 lie on a line over their bounds; then k2 has two clusters.
    k1 = [0.5 * (i + 0.5) / 100 for i in range(100)]
    k2 = [0.05 + 0.2 * (i % 2) + 0.01 * ((i * 37) % 11) / 11 for i in range(100)]
    line = ParameterSample.equally_weighted(
        ["k1", "k2"], BOUNDS, [[k, 0.3 - 0.6 * k] for k in k1], GIVEN
    )
    clusters = ParameterSample.equally_weighted(
        ["k1", "k2"],
        BOUNDS,
        [[0.25 + 0.001 * ((i * 13) % 7), c] for i, c in enumerate(k2)],
        GIVEN,
    )

    assert _reasons(calibrator, line) == {
        "k1": (
            "poorly_determined",
            [
                "Sample: contraction 0.00, below 0.5.",
                "Sample: correlation -1.000 with k2.",
            ],
        ),
        "k2": (
            "poorly_determined",
            [
                "Sample: contraction 0.00, below 0.5.",
                "Sample: correlation -1.000 with k1.",
            ],
        ),
    }
    reasons = _reasons(calibrator, clusters)
    assert reasons["k1"] == (
        "determined",
        ["Sample: contraction 1.00, no strong pair."],
    )
    # Two clusters near the ends of the bounds spread wider than the bounds.
    assert reasons["k2"] == (
        "poorly_determined",
        ["Sample: contraction -0.33, below 0.5.", "Sample: its marginal has 2 modes."],
    )


def test_contraction_against_the_prior_of_a_prior_sample_is_zero(
    calibrator: Calibrator,
) -> None:
    priors = Priors(
        distributions={"k1": Beta(a=2.0, b=5.0), "k2": Gamma(shape=4.0, rate=40.0)}
    )
    columns = [
        priors.distributions[name].sample(4000, bound, seed=index)
        for index, (name, bound) in enumerate(zip(["k1", "k2"], BOUNDS, strict=True))
    ]
    sample = ParameterSample.equally_weighted(
        ["k1", "k2"],
        BOUNDS,
        [list(row) for row in zip(*columns, strict=True)],
        SampleMethod(name="prior"),
    )

    against_prior = calibrator.identifiability(
        SampleIdentifiability(
            sample=sample, reference="prior", priors=priors, **THRESHOLDS
        )
    ).sample[0]
    against_bounds = _section(calibrator, sample)

    assert against_prior.reference == "prior"
    for name in ["k1", "k2"]:
        assert against_prior.contraction[name] == pytest.approx(0.0, abs=0.1)
        # A sample of its prior fills the bins of equal prior mass.
        assert against_prior.overlap[name] > 0.9
        # The priors are narrower than their bounds.
        assert against_bounds.contraction[name] > 0.3


def test_equal_rows_are_merged(calibrator: Calibrator) -> None:
    # Weighted draws: k1 has two modes, at 0.15 and 0.35; k2 has a mode at
    # 0.1 and 30 % of the draws at its upper bound, each a distinct row.
    rng = random.Random(4)
    rows, log_weights = [], []
    for i in range(2000):
        k1 = rng.uniform(0.0, 0.5)
        k2 = 0.3 if i % 10 < 3 else min(max(rng.gauss(0.1, 0.02), 0.0), 0.3)
        rows.append([k1, k2])
        log_weights.append(
            math.log(
                math.exp(-0.5 * ((k1 - 0.15) / 0.03) ** 2)
                + math.exp(-0.5 * ((k1 - 0.35) / 0.03) ** 2)
            )
        )
    weighted = ParameterSample(
        names=["k1", "k2"],
        bounds=BOUNDS,
        values=rows,
        log_weights=log_weights,
        method=GIVEN,
    )
    resampled = weighted.resample(20_000, seed=1)

    by_weight = _section(calibrator, weighted)
    by_copies = _section(calibrator, resampled)

    # The copies of a draw count once, with their summed weight.
    assert by_copies.effective_size < 2000
    for name in ["k1", "k2"]:
        assert len(by_weight.modes[name]) == 2
        assert by_copies.modes[name] == pytest.approx(by_weight.modes[name], abs=0.02)
    assert by_weight.modes["k2"][1] == pytest.approx(0.3)


def test_a_value_at_a_natural_bound_is_determined_and_at_bound(
    calibrator: Calibrator,
) -> None:
    # k1 is a half-normal of sd 0.02 at its lower bound 0.
    rng = random.Random(7)
    sample = ParameterSample.equally_weighted(
        ["k1", "k2"],
        BOUNDS,
        [[abs(rng.gauss(0.0, 0.02)), rng.gauss(0.15, 0.01)] for _ in range(2000)],
        GIVEN,
    )

    section = _section(calibrator, sample)
    verdict, reasons = _reasons(calibrator, sample)["k1"]

    # P(|Z| < 0.025 / 0.02) = 0.79 of the mass in the bottom 5 % of [0, 0.5].
    assert section.mass_below["k1"] == pytest.approx(0.79, abs=0.03)
    assert section.at_bound == {"k1": ["lower"], "k2": []}
    assert verdict == "determined"
    assert reasons == [
        "Sample: contraction 0.99, no strong pair.",
        "Sample: the lower bound, not the data, sets one side; "
        f"{100 * section.mass_below['k1']:.0f} % of its mass is in the bottom "
        "5 % of its reference.",
    ]


def test_log_parameters_need_positive_draws(calibrator: Calibrator) -> None:
    def analyze(bounds: list[tuple[float, float]], k1: float, logs: list[str]) -> None:
        sample = ParameterSample.equally_weighted(
            ["k1", "k2"], bounds, [[k1, 0.1], [0.2, 0.2]], GIVEN
        )
        _section(calibrator, sample, log_parameters=logs)

    errors = [
        (
            ([(-0.1, 0.5), (0.0, 0.3)], 0.1, ["k1"]),
            "log_parameters reads ln θ, so each of its parameters has a lower "
            "bound of at least 0: 'k1' has the bounds [-0.1, 0.5]",
        ),
        (
            (BOUNDS, 0.0, ["k1"]),
            "log_parameters reads ln θ, so each of its parameters has every draw "
            "above 0: 'k1' has a draw at 0",
        ),
        (
            (BOUNDS, 0.1, ["k3"]),
            "log_parameters names 'k3', which is not a parameter of the sample: "
            '["k1", "k2"]',
        ),
    ]
    for arguments, message in errors:
        with pytest.raises(ValueError) as error:
            analyze(*arguments)
        assert str(error.value) == message


def test_a_section_warns_of_a_hidden_ridge_and_of_no_directions(
    calibrator: Calibrator,
) -> None:
    # Two parameters of contraction 0.6 and correlation 0.85: each is
    # determined, and the direction they share contracts by 0.26.
    rng = random.Random(3)
    rows: list[list[float]] = []
    while len(rows) < 2000:
        a, b = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
        row = [
            0.25 + 0.4**0.5 * 0.5 / 12**0.5 * a,
            0.15 + 0.4**0.5 * 0.3 / 12**0.5 * (0.85 * a + (1 - 0.85**2) ** 0.5 * b),
        ]
        if 0.0 <= row[0] <= 0.5 and 0.0 <= row[1] <= 0.3:
            rows.append(row)
    ridge = ParameterSample.equally_weighted(["k1", "k2"], BOUNDS, rows, GIVEN)
    two = ParameterSample.equally_weighted(
        ["k1", "k2"], BOUNDS, [[0.1, 0.1], [0.2, 0.2], [0.1, 0.1]], GIVEN
    )

    section = _section(calibrator, ridge)
    few = _section(calibrator, two)

    assert {v for v, _ in _reasons(calibrator, ridge).values()} == {"determined"}
    assert section.hidden_ridges == 1
    assert section.warnings == [
        "1 direction among the determined parameters contracts less than 0.5: "
        f"a ridge that no pair shows (n_eff / p = {section.draws_per_parameter:.1f})."
    ]
    assert few.directions == []
    assert few.hidden_ridges is None
    # Two distinct draws, of weights 2/3 and 1/3: an effective size of 1.8.
    assert few.warnings == [
        "The effective size 1.8 is below p + 1 = 3 for 2 parameters: the "
        "weights are worth fewer draws than a covariance of the parameters "
        "needs, so no parameter is determined.",
        "The sample has 2 distinct draws for 2 parameters, and no directions: a "
        "direction needs more distinct draws than parameters.",
    ]


def test_a_sample_of_one_effective_draw_determines_nothing(
    calibrator: Calibrator,
) -> None:
    # 500 draws whose weights put 0.999 on one draw: each contraction is near
    # 1, and the effective size, about 1, is below p + 1 = 3.
    rng = random.Random(5)
    rows = [[rng.uniform(0.0, 0.5), rng.uniform(0.0, 0.3)] for _ in range(500)]
    log_weights = [math.log(0.999)] + [math.log(0.001 / 499)] * 499
    sample = ParameterSample(
        names=["k1", "k2"],
        bounds=BOUNDS,
        values=rows,
        log_weights=log_weights,
        method=GIVEN,
    )

    report = calibrator.identifiability(
        SampleIdentifiability(sample=sample, reference="bounds", **THRESHOLDS)
    )
    section = report.sample[0]

    size = f"{section.effective_size:.3g}"
    assert section.effective_size < 1.01
    assert min(section.contraction.values()) > 0.99
    reason = f"Sample: effective size {size}, below p + 1 = 3 for 2 parameters."
    for name in ["k1", "k2"]:
        practical = report.verdicts[name].practical
        assert practical is not None
        assert practical.verdict == "poorly_determined"
        assert practical.reasons == [reason]
    warning = (
        f"The effective size {size} is below p + 1 = 3 for 2 parameters: the "
        "weights are worth fewer draws than a covariance of the parameters "
        "needs, so no parameter is determined."
    )
    assert section.warnings == [warning]
    data = report.report()
    assert data["sections"]["sample"][0]["warnings"] == [warning]
    assert data["parameters"]["k1"]["practical"] == {
        "verdict": "poorly_determined",
        "reasons": [reason],
    }


def test_a_parameter_constant_in_every_draw_is_not_determined(
    calibrator: Calibrator, tmp_path: Path
) -> None:
    # k1 spreads over its bounds in 100 equal draws; k2 never moves.
    k1 = [0.5 * (i + 0.5) / 100 for i in range(100)]
    sample = ParameterSample.equally_weighted(
        ["k1", "k2"], BOUNDS, [[k, 0.05] for k in k1], GIVEN
    )

    report = calibrator.identifiability(
        SampleIdentifiability(sample=sample, reference="bounds", **THRESHOLDS)
    )
    section = report.sample[0]

    reason = "Sample: constant in every draw."
    assert section.effective_size == pytest.approx(100.0)
    assert section.contraction["k2"] == 1.0
    assert section.correlation == {
        "k1": {"k1": 1.0, "k2": None},
        "k2": {"k1": None, "k2": None},
    }
    assert section.correlated_pairs == []
    assert section.modes["k2"] == [0.05]
    assert section.warnings == [CONSTANT_K2]
    assert _verdicts(report) == {
        "k1": ("poorly_determined", ["Sample: contraction 0.00, below 0.5."]),
        "k2": ("poorly_determined", [reason]),
    }
    # The report as plain data, and the file of the report.
    data = report.report()
    assert data["sections"]["sample"][0]["correlation"] == section.correlation
    assert data["parameters"]["k2"]["practical"] == {
        "verdict": "poorly_determined",
        "reasons": [reason],
    }
    path = tmp_path / "report.json"
    report.to_json(path)
    assert IdentifiabilityReport.from_json(path) == report
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["sample"][0]["correlation"]["k2"] == {"k1": None, "k2": None}


def test_a_constant_parameter_of_a_weighted_sample_has_its_reason_and_warning(
    calibrator: Calibrator,
) -> None:
    # Draws of unequal weights: 0.1 is a value whose weighted mean rounds. The
    # draw of another k2 has no weight. Against the priors, 60 draws of k1 near
    # 0.1; then one draw holds nearly all the weight.
    rows = [[0.1 + 0.001 * (i % 7), 0.1] for i in range(60)] + [[0.4, 0.2]]
    spread = [-0.01 * i for i in range(60)] + [-math.inf]
    heavy = [0.0] + [-20.0] * 59 + [-math.inf]
    reports = {
        name: calibrator.identifiability(
            SampleIdentifiability(
                sample=ParameterSample(
                    names=["k1", "k2"],
                    bounds=BOUNDS,
                    values=rows,
                    log_weights=log_weights,
                    method=GIVEN,
                ),
                reference="prior",
                priors=Priors.uniform_on_bounds(),
                **THRESHOLDS,
            )
        )
        for name, log_weights in (("spread", spread), ("heavy", heavy))
    }

    section = reports["spread"].sample[0]
    assert section.contraction["k2"] == 1.0
    assert section.correlation["k2"] == {"k1": None, "k2": None}
    assert section.at_bound["k2"] == []
    assert section.warnings[-1] == CONSTANT_K2
    assert _verdicts(reports["spread"])["k2"] == (
        "poorly_determined",
        ["Sample: constant in every draw."],
    )
    # Too few effective draws and a constant column: both reasons and warnings.
    few = reports["heavy"].sample[0]
    assert few.warnings[0].startswith("The effective size 1 is below p + 1 = 3")
    assert few.warnings[1] == CONSTANT_K2
    verdicts = _verdicts(reports["heavy"])
    size = "Sample: effective size 1, below p + 1 = 3 for 2 parameters."
    assert verdicts["k1"] == ("poorly_determined", [size])
    assert verdicts["k2"] == (
        "poorly_determined",
        [size, "Sample: constant in every draw."],
    )


def test_a_sample_of_one_distinct_draw_warns_of_its_size_alone(
    calibrator: Calibrator,
) -> None:
    # Every parameter is constant because the sample holds one distinct draw.
    sample = ParameterSample.equally_weighted(
        ["k1", "k2"], BOUNDS, [[0.1, 0.1]] * 50, GIVEN
    )

    section = _section(calibrator, sample)

    undefined = {"k1": None, "k2": None}
    assert section.correlation == {"k1": undefined, "k2": undefined}
    size, directions = section.warnings
    assert size.startswith("The effective size 1 is below p + 1 = 3")
    assert "no directions" in directions
    reasons = ["Sample: effective size 1, below p + 1 = 3 for 2 parameters."]
    assert _reasons(calibrator, sample) == {
        "k1": ("poorly_determined", reasons),
        "k2": ("poorly_determined", reasons),
    }


def test_a_section_names_an_effective_size_below_thirty_per_parameter(
    calibrator: Calibrator,
) -> None:
    # 40 draws near the middle of the bounds: above p + 1, below 30 · p.
    rng = random.Random(6)
    rows = [[0.25 + 0.01 * rng.random(), 0.15 + 0.01 * rng.random()] for _ in range(40)]
    sample = ParameterSample.equally_weighted(["k1", "k2"], BOUNDS, rows, GIVEN)

    section = _section(calibrator, sample)

    assert {v for v, _ in _reasons(calibrator, sample).values()} == {"determined"}
    assert section.hidden_ridges == 0
    assert section.effective_size == pytest.approx(40.0)
    assert section.warnings == [
        "The effective size 40 is below 30 · p = 60 for 2 parameters: the count "
        "of hidden ridges over-counts."
    ]


def test_a_sample_of_other_parameters_is_rejected(calibrator: Calibrator) -> None:
    sample = ParameterSample.equally_weighted(["k1"], [(0.0, 0.5)], [[0.3]], GIVEN)

    with pytest.raises(ValueError) as error:
        _section(calibrator, sample)

    assert str(error.value) == (
        "The sample's parameters ['k1'] are not the problem's ['k1', 'k2']."
    )


def test_the_reference_matches_the_priors() -> None:
    # The engine reads the JSON of the method: no priors with "prior", and
    # priors only with "prior".
    sample = ParameterSample.equally_weighted(["k1"], [(0, 1)], [[0.5]], GIVEN)
    method = {"sample": sample, **THRESHOLDS}
    with pytest.raises(ValidationError, match="reference='prior' needs priors"):
        SampleIdentifiability(**method, reference="prior")
    with pytest.raises(ValidationError, match="priors are read only with"):
        SampleIdentifiability(
            **method, reference="bounds", priors=Priors.uniform_on_bounds()
        )
