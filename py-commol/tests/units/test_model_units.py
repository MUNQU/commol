"""``Model.check_unit_consistency``: every flow in ``<bin unit>/<step unit>``."""

import pytest

from commol import ModelBuilder
from commol.context.model import Model
from commol.utils.equations import UnitConsistencyError
from tests.support import a_to_b_builder, build, seir_builder, sir_builder

DAILY = {"beta": "1/day", "gamma": "1/day"}
FAILED = "Unit consistency check failed"


def _sir(**options) -> Model:
    """SIR in persons with rates per day; an option replaces a default."""
    options.setdefault("units", DAILY)
    return build(sir_builder(bin_unit="person", **options))


def _per_age_infection(*rates: str, **options) -> Model:
    """SIR whose infection has one rate per age category, in order."""
    categories = ("child", "adult", "senior")[: len(rates)]
    builder = sir_builder(
        bin_unit="person",
        units=DAILY,
        infection=None,
        stratifications={"age": dict.fromkeys(categories, 1 / len(rates))},
        **options,
    )
    return build(
        builder.add_transition(
            id="infection",
            source=["S"],
            target=["I"],
            stratified_rates=[
                {
                    "conditions": [{"stratification": "age", "category": category}],
                    "rate": rate,
                }
                for category, rate in zip(categories, rates, strict=True)
            ],
        )
    )


def _vaccination(rate: str, threshold_unit: str = "person") -> Model:
    """S, V, I, R: S is vaccinated at ``rate`` besides the SIR flows."""
    return build(
        ModelBuilder(name="SVIR", bin_unit="person")
        .add_bin(id="S", name="Susceptible")
        .add_bin(id="V", name="Vaccinated")
        .add_bin(id="I", name="Infected")
        .add_bin(id="R", name="Recovered")
        .add_parameter(id="beta", value=0.5, unit="1/day")
        .add_parameter(id="gamma", value=0.1, unit="1/day")
        .add_parameter(id="vax_rate", value=0.01, unit="1/day")
        .add_parameter(id="threshold", value=100, unit=threshold_unit)
        .add_transition(id="vaccination", source=["S"], target=["V"], rate=rate)
        .add_transition(
            id="infection", source=["S"], target=["I"], rate="beta * S * I / N"
        )
        .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "S", "fraction": 0.98},
                {"bin": "V", "fraction": 0.0},
                {"bin": "I", "fraction": 0.02},
                {"bin": "R", "fraction": 0.0},
            ],
        )
    )


class TestConsistentModels:
    def test_sir(self) -> None:
        _sir().check_unit_consistency()

    def test_rates_divided_by_population_totals(self) -> None:
        _per_age_infection(
            "beta * S * I / N_child", "beta * S * I / N_adult"
        ).check_unit_consistency()

    def test_a_dimensionless_parameter(self) -> None:
        _sir(
            parameters={"contact_reduction": 0.5, "beta": 0.5, "gamma": 0.1},
            units={**DAILY, "contact_reduction": "dimensionless"},
            infection="beta * contact_reduction * S * I / N",
        ).check_unit_consistency()

    def test_a_custom_bin_unit_and_a_time_alias_are_registered(self) -> None:
        build(
            sir_builder(
                bin_unit="cow",
                parameters={"beta": 0.1},
                units={"beta": "1/semester"},
                recovery=None,
            )
        ).check_unit_consistency()

    def test_a_vaccination_rate(self) -> None:
        _vaccination(
            "vax_rate * S", threshold_unit="dimensionless"
        ).check_unit_consistency()

    def test_exponential_decay_of_a_rate(self) -> None:
        _sir(
            parameters={"beta_0": 0.5, "decay_rate": 0.01, "gamma": 0.1},
            units={"beta_0": "1/day", "decay_rate": "dimensionless", "gamma": "1/day"},
            infection="beta_0 * exp(-decay_rate * step) * S * I / N",
        ).check_unit_consistency()

    def test_seasonal_forcing(self) -> None:
        _sir(
            parameters={"beta_avg": 0.5, "seasonal_amp": 0.1, "gamma": 0.1},
            units={
                "beta_avg": "1/day",
                "seasonal_amp": "dimensionless",
                "gamma": "1/day",
            },
            infection=(
                "beta_avg * (1 + seasonal_amp * sin(2 * pi * step / 365)) * S * I / N"
            ),
        ).check_unit_consistency()

    def test_the_unit_of_a_formula_parameter_is_inferred(self) -> None:
        _sir(
            parameters={
                "beta_base": 0.3,
                "multiplier": 2.0,
                "beta": "beta_base * multiplier",
                "gamma": 0.1,
            },
            units={
                "beta_base": "1/day",
                "multiplier": "dimensionless",
                "gamma": "1/day",
            },
        ).check_unit_consistency()

    def test_a_formula_of_population_totals_is_dimensionless(self) -> None:
        _sir(
            parameters={"young_fraction": "N_young / N", "beta": 0.3, "gamma": 0.1},
            stratifications={"age": {"young": 0.3, "old": 0.7}},
        ).check_unit_consistency()

    def test_chained_formula_parameters(self) -> None:
        _sir(
            parameters={
                "base_rate": 0.1,
                "adjusted_rate": "base_rate * 2",
                "final_rate": "adjusted_rate * 3",
                "gamma": 0.1,
            },
            units={"base_rate": "1/day", "gamma": "1/day"},
            infection="final_rate * S * I / N",
        ).check_unit_consistency()


class TestMissingUnits:
    def test_parameters_without_units_are_named(self) -> None:
        with pytest.raises(UnitConsistencyError, match="missing units"):
            _sir(units={}).check_unit_consistency()

    def test_one_parameter_without_a_unit_is_named(self) -> None:
        with pytest.raises(UnitConsistencyError, match="missing units") as error:
            _sir(units={"beta": "1/day"}).check_unit_consistency()

        assert "gamma" in str(error.value)

    def test_a_formula_of_an_unknown_variable_has_no_unit(self) -> None:
        model = _sir(
            parameters={
                "beta_base": 0.3,
                "multiplier": 2.0,
                "invalid_param": "nonexistent_var * 2",
                "gamma": 0.1,
            },
            units={
                "beta_base": "1/day",
                "multiplier": "dimensionless",
                "gamma": "1/day",
            },
            infection="invalid_param * S * I / N",
        )

        with pytest.raises(UnitConsistencyError, match="no defined unit") as error:
            model.check_unit_consistency()

        assert "invalid_param" in str(error.value)


class TestInconsistentModels:
    def test_a_rate_in_the_wrong_unit_names_its_transition(self) -> None:
        model = _sir(units={"beta": "day", "gamma": "1/day"})

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "infection" in str(error.value)

    @pytest.mark.parametrize(
        ("infection", "extra"),
        [
            pytest.param("beta * log(I) * S / N", {}, id="log-of-a-population"),
            pytest.param(
                "beta * (1 + 0.1 * sin(time_scale)) * S * I / N",
                {"time_scale": (365, "day")},
                id="sin-of-days",
            ),
            pytest.param(
                "beta * (1 + 0.1 * atan2(I, threshold_time)) * S * I / N",
                {"threshold_time": (30, "day")},
                id="atan2-of-persons-and-days",
            ),
            pytest.param(
                "min(beta, capacity) * S * I / N",
                {"capacity": (50, "person")},
                id="min-of-a-rate-and-persons",
            ),
        ],
    )
    def test_a_function_of_a_wrong_unit_fails(
        self, infection: str, extra: dict[str, tuple[float, str]]
    ) -> None:
        model = _sir(
            parameters={"beta": 0.5, "gamma": 0.1}
            | {name: value for name, (value, _) in extra.items()},
            units=DAILY | {name: unit for name, (_, unit) in extra.items()},
            infection=infection,
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert infection in str(error.value)

    def test_exp_of_days_fails(self) -> None:
        model = build(
            seir_builder(
                bin_unit="person",
                parameters={"beta": 0.5, "sigma": 0.2, "gamma": 0.1, "decay_time": 10},
                units={**DAILY, "sigma": "1/day", "decay_time": "day"},
                exposure="beta * exp(decay_time) * S * I / N",
            )
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "exp(decay_time)" in str(error.value)

    def test_sin_of_steps_over_days_fails(self) -> None:
        model = build(
            seir_builder(
                bin_unit="person",
                parameters={
                    "beta_avg": 0.5,
                    "seasonal_amp": 0.2,
                    "period": 365,
                    "sigma": 0.2,
                    "gamma": 0.1,
                },
                units={
                    "beta_avg": "1/day",
                    "seasonal_amp": "dimensionless",
                    "period": "day",
                    "sigma": "1/day",
                    "gamma": "1/day",
                },
                exposure=(
                    "beta_avg * (1 + seasonal_amp * sin(2 * pi * step / period)) "
                    "* S * I / N"
                ),
            )
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "sin(2 * pi * step / period)" in str(error.value)

    def test_sqrt_of_a_population_in_a_stratified_rate_fails(self) -> None:
        model = _per_age_infection(
            "beta * sqrt(I) * S / N_child", "beta * 0.5 * sqrt(I) * S / N_adult"
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "sqrt(I)" in str(error.value)

    def test_floor_of_a_population_in_a_stratified_rate_fails(self) -> None:
        model = _per_age_infection(
            "beta * floor(S) * I / N_child",
            "beta * S * I / N_adult",
            "beta * 0.6 * S * I / N_senior",
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "floor(S)" in str(error.value)

    def test_log_of_persons_in_one_of_four_combinations_fails(self) -> None:
        builder = sir_builder(
            bin_unit="person",
            parameters={"beta": 0.5, "gamma": 0.1, "contact_scale": 50},
            units={**DAILY, "contact_scale": "person"},
            infection=None,
            stratifications={
                "age": {"young": 0.6, "old": 0.4},
                "location": {"urban": 0.7, "rural": 0.3},
            },
        )
        rates = {
            (
                "young",
                "urban",
            ): "beta * exp(log(contact_scale)) * S * I / N_young_urban",
            ("young", "rural"): "beta * 0.5 * S * I / N_young_rural",
            ("old", "urban"): "beta * 0.7 * S * I / N_old_urban",
            ("old", "rural"): "beta * 0.3 * S * I / N_old_rural",
        }
        model = build(
            builder.add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                stratified_rates=[
                    {
                        "conditions": [
                            {"stratification": "age", "category": age},
                            {"stratification": "location", "category": location},
                        ],
                        "rate": rate,
                    }
                    for (age, location), rate in rates.items()
                ],
            )
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "exp(log(contact_scale))" in str(error.value)

    def test_pow_with_an_exponent_in_persons_fails(self) -> None:
        model = _vaccination("vax_rate * pow(I / N, threshold) * S")

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "pow(I / N, threshold)" in str(error.value)


def _two_rates(k1_unit: str, k2_unit: str, bin_unit: str | None = "person") -> Model:
    """A → B → C, each transition with its own rate parameter."""
    return build(
        ModelBuilder(name="Step unit", bin_unit=bin_unit)
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_bin(id="C", name="C")
        .add_parameter(id="k1", value=0.1, unit=k1_unit)
        .add_parameter(id="k2", value=0.1, unit=k2_unit)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 1.0},
                {"bin": "B", "fraction": 0.0},
                {"bin": "C", "fraction": 0.0},
            ],
        )
    )


class TestStepUnit:
    """One step unit, inferred from the parameters, for every rate."""

    @pytest.mark.parametrize("unit", ["1/step", "1/day", "1/week", "1/d"])
    def test_a_model_in_one_unit_of_time_passes(self, unit: str) -> None:
        _two_rates(unit, unit).check_unit_consistency()

    def test_aliases_of_one_unit_are_one_unit(self) -> None:
        _two_rates("1/day", "1/d").check_unit_consistency()

    @pytest.mark.parametrize(
        ("k1_unit", "k2_unit"), [("1/day", "1/week"), ("1/day", "1/step")]
    )
    def test_two_units_of_time_are_an_error(self, k1_unit: str, k2_unit: str) -> None:
        with pytest.raises(UnitConsistencyError) as error:
            _two_rates(k1_unit, k2_unit).check_unit_consistency()

        message = str(error.value)
        assert f"'k1' in '{k1_unit}'" in message
        assert f"'k2' in '{k2_unit}'" in message

    def test_a_model_without_units_of_time_uses_step(self) -> None:
        model = build(
            a_to_b_builder(
                parameters={"share": 0.1},
                rate="share",
                bin_unit="person",
                units={"share": "dimensionless"},
            )
        )

        with pytest.raises(UnitConsistencyError, match="person/step"):
            model.check_unit_consistency()

    def test_a_bin_without_a_unit_is_named(self) -> None:
        with pytest.raises(UnitConsistencyError, match="no unit: A, B, C"):
            _two_rates("1/day", "1/day", bin_unit=None).check_unit_consistency()
