import pytest

from commol.api.model_builder import ModelBuilder
from commol.constants import ModelTypes
from commol.utils.equations import (
    UnitConsistencyError,
    check_all_parameters_have_units,
    check_equation_units,
    get_predefined_variable_units,
    ureg,
)


def register_custom_unit(unit_name: str) -> None:
    """
    Helper function to register a custom unit if not already defined.

    This is used in standalone tests that don't go through the Model object.
    In production code, units are auto-registered by Model._register_custom_units().
    """
    import pint

    try:
        ureg(unit_name)
    except pint.UndefinedUnitError:
        dimension_name = f"{unit_name}_dimension"
        ureg.define(f"{unit_name} = [{dimension_name}]")


def register_common_units() -> None:
    """
    Register commonly used units for epidemiological modeling.

    This includes population units (person, individual, people) and
    abstract time unit, plus commonly used time aliases.
    """
    import pint

    # Register person and aliases
    try:
        ureg("person")
    except pint.UndefinedUnitError:
        ureg.define("person = [population]")
        ureg.define("individual = person")
        ureg.define("people = person")

    # Register abstract time unit
    try:
        ureg("time")
    except pint.UndefinedUnitError:
        ureg.define("time = [time_abstract]")

    # Register time unit aliases
    time_aliases = {
        "semester": "6 * month",
        "wk": "week",
        "mon": "month",
    }

    for unit_name, unit_definition in time_aliases.items():
        try:
            ureg(unit_name)
        except pint.UndefinedUnitError:
            ureg.define(f"{unit_name} = {unit_definition}")


class TestUnitUtilities:
    """Test utility functions for unit handling."""

    @classmethod
    def setup_class(cls):
        """Register common units once for all tests in this class."""
        register_common_units()

    def test_check_equation_units_simple(self):
        """Test simple unit consistency check."""
        variable_units = {
            "beta": "1/day",
            "S": "person",
            "I": "person",
            "N": "person",
        }
        is_consistent, error = check_equation_units(
            "beta * S * I / N", variable_units, "person/day"
        )
        assert is_consistent
        assert error is None

    def test_check_equation_units_mismatch(self):
        """Test unit mismatch detection."""
        variable_units = {"beta": "1/day", "S": "person"}
        is_consistent, error = check_equation_units(
            "beta * S",
            variable_units,
            "person",  # Wrong expected unit
        )
        assert not is_consistent
        assert error is not None
        assert "Unit mismatch" in error

    def test_check_all_parameters_have_units(self):
        """Test checking if all parameters have units."""
        from commol.context.parameter import Parameter

        params_with_units = [
            Parameter(id="beta", value=0.5, unit="1/day"),
            Parameter(id="gamma", value=0.1, unit="1/day"),
        ]
        assert check_all_parameters_have_units(params_with_units)

        params_without_units = [
            Parameter(id="beta", value=0.5, unit="1/day"),
            Parameter(id="gamma", value=0.1, unit=None),
        ]
        assert not check_all_parameters_have_units(params_without_units)

    def test_get_predefined_variable_units(self):
        """Test getting units for predefined variables."""
        from commol.context.stratification import Stratification

        strats = [
            Stratification(id="age", categories=["young", "old"]),
        ]
        units = get_predefined_variable_units(strats, bin_unit="person")

        assert units["N"] == "person"
        assert units["N_young"] == "person"
        assert units["N_old"] == "person"

    def test_get_predefined_variable_units_multiple_stratifications(self):
        """Test getting units for predefined variables with multiple stratifications."""
        from commol.context.stratification import Stratification

        strats = [
            Stratification(id="age", categories=["young", "old"]),
            Stratification(id="location", categories=["urban", "rural"]),
        ]
        units = get_predefined_variable_units(strats, bin_unit="person")

        assert units["N"] == "person"
        assert units["N_young"] == "person"
        assert units["N_urban"] == "person"
        assert units["N_young_urban"] == "person"
        assert units["N_old_rural"] == "person"


class TestModelUnitConsistency:
    """Test unit consistency checking in Model class."""

    def test_sir_model_with_consistent_units(self):
        """Test SIR model with consistent units."""
        builder = (
            ModelBuilder(name="SIR", version="1.0", bin_unit="person")
            # Add disease states
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Add parameters WITH units
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            # Add transitions
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            # Set initial conditions
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should not raise an exception
        model.check_unit_consistency()

    def test_sir_model_without_units_raises_error(self):
        """Test that unit check raises error when parameters lack units."""
        builder = (
            ModelBuilder(name="SIR", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Add parameters WITHOUT units
            .add_parameter("beta", 0.5, "Transmission rate")
            .add_parameter("gamma", 0.1, "Recovery rate")
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should raise error because parameters lack units
        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "missing units" in str(exc_info.value)

    def test_sir_model_with_inconsistent_units(self):
        """Test SIR model with inconsistent units raises error."""
        builder = (
            ModelBuilder(name="SIR", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Add parameters with WRONG units (beta should be 1/day, not day)
            .add_parameter("beta", 0.5, "Transmission rate", unit="day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should raise UnitConsistencyError
        with pytest.raises(UnitConsistencyError) as error:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(error.value)

    def test_stratified_model_with_consistent_units(self):
        """Test stratified model with consistent units."""
        builder = (
            ModelBuilder(name="Stratified SIR", version="1.0", bin_unit="person")
            # Add disease states
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Add stratification
            .add_stratification("age", ["young", "old"])
            # Add parameters with units
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            # Add transitions with predefined variables (N_young, N_old)
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "young"}],
                        "rate": "beta * S * I / N_young",
                    },
                    {
                        "conditions": [{"stratification": "age", "category": "old"}],
                        "rate": "beta * S * I / N_old",
                    },
                ],
            )
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            # Set initial conditions
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.6},
                            {"category": "old", "fraction": 0.4},
                        ],
                    }
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should not raise an exception
        model.check_unit_consistency()

    def test_mixed_units_some_parameters_without_units(self):
        """Test that check raises error when some parameters lack units."""
        builder = (
            ModelBuilder(name="SIR", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Mix: one parameter with unit, one without
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate")  # No unit
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should raise error because gamma lacks a unit
        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "gamma" in str(exc_info.value)
        assert "missing units" in str(exc_info.value)

    def test_dimensionless_parameters(self):
        """Test handling of dimensionless parameters."""
        builder = (
            ModelBuilder(name="Test", version="1.0", bin_unit="person")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            # Dimensionless parameter
            .add_parameter(
                "contact_reduction",
                0.5,
                "Contact reduction factor",
                unit="dimensionless",
            )
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                rate="beta * contact_reduction * S * I / N",
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should not raise an exception
        model.check_unit_consistency()

    def test_custom_bin_unit_auto_registration(self):
        """Test that custom bin units are automatically registered."""
        builder = (
            ModelBuilder(name="Custom Unit Test", version="1.0", bin_unit="cow")
            .add_bin("healthy", "Healthy cows")
            .add_bin("sick", "Sick cows")
            .add_parameter("infection_rate", 0.1, "Infection rate", unit="1/semester")
            .add_transition(
                id="infection",
                source=["healthy"],
                target=["sick"],
                rate="infection_rate * healthy * sick / N",
            )
            .set_initial_conditions(
                population_size=100,
                bin_fractions=[
                    {"bin": "healthy", "fraction": 0.95},
                    {"bin": "sick", "fraction": 0.05},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should auto-register "cow" unit and not raise an exception
        model.check_unit_consistency()


class TestComplexUnits:
    """Test complex unit scenarios."""

    def test_vaccination_rate_with_time_dependent_coverage(self):
        """Test vaccination model with time-dependent coverage."""
        builder = (
            ModelBuilder(name="SIR with Vaccination", version="1.0", bin_unit="person")
            .add_bin("S", "Susceptible")
            .add_bin("V", "Vaccinated")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Parameters
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            .add_parameter("vax_rate", 0.01, "Vaccination rate", unit="1/day")
            # Transitions
            .add_transition("vaccination", ["S"], ["V"], rate="vax_rate * S")
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "V", "fraction": 0.0},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should not raise
        model.check_unit_consistency()


class TestMathFunctionsInUnitChecking:
    """Test that mathematical functions work correctly in unit checking."""

    @classmethod
    def setup_class(cls):
        """Register common units once for all tests in this class."""
        register_common_units()

    def test_exponential_decay(self):
        """Test exponential decay with exp function."""
        variable_units = {
            "beta": "1/day",
            "decay": "dimensionless",
            "step": "dimensionless",
            "S": "person",
            "I": "person",
            "N": "person",
        }
        is_consistent, error = check_equation_units(
            "beta * exp(-decay * step) * S * I / N",
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_abstract_time_unit(self):
        """Test that abstract 'time' unit works."""
        variable_units = {"rate": "1/time", "I": "person"}
        is_consistent, error = check_equation_units(
            "rate * I", variable_units, "person/time"
        )
        assert is_consistent, f"Error: {error}"

    def test_various_time_units(self):
        """Test various time units (week, month, year, hour, etc.)."""
        time_units = [
            "second",
            "minute",
            "hour",
            "day",
            "week",
            "month",
            "fortnight",
            "year",
        ]

        for time_unit in time_units:
            variable_units = {"rate": f"1/{time_unit}", "I": "person"}
            is_consistent, error = check_equation_units(
                "rate * I", variable_units, f"person/{time_unit}"
            )
            assert is_consistent, f"Error for unit '{time_unit}': {error}"

    def test_time_unit_conversions(self):
        """Test that different time units are compatible."""
        variable_units = {"rate": "1/week", "I": "person"}
        # Week and day are compatible time units
        is_consistent, error = check_equation_units(
            "rate * I", variable_units, "person/day"
        )
        assert is_consistent, f"Error: {error}"


class TestFullModelUnitFailures:
    """Test that full models with unit errors properly raise exceptions."""

    @classmethod
    def setup_class(cls):
        """Register common units once for all tests in this class."""
        register_common_units()

    def test_simple_sir_wrong_beta_units(self):
        """Test SIR model fails with wrong beta units."""
        builder = (
            ModelBuilder(name="SIR Wrong Beta", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Wrong units: beta should be 1/day, not day
            .add_parameter("beta", 0.5, "Transmission rate", unit="day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)
        assert "infection" in str(exc_info.value)

    def test_sir_with_log_of_population_fails(self):
        """Test SIR model fails when log receives population (dimensional)."""
        builder = (
            ModelBuilder(name="SIR Bad Log", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            # This should fail: log(I) where I has units of person
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                rate="beta * log(I) * S / N",  # log(I) is invalid!
            )
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_seir_with_exponential_wrong_units(self):
        """Test SEIR model fails when exp receives dimensional parameter."""
        builder = (
            ModelBuilder(name="SEIR Bad Exp", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("E", "Exposed")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("sigma", 0.2, "Incubation rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            .add_parameter("decay_time", 10, "Decay time", unit="day")  # Has units!
            # This should fail: exp(decay_time) where decay_time has units of day
            .add_transition(
                "exposure",
                ["S"],
                ["E"],
                rate="beta * exp(decay_time) * S * I / N",  # exp(decay_time) invalid!
            )
            .add_transition("infection", ["E"], ["I"], rate="sigma * E")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.98},
                    {"bin": "E", "fraction": 0.01},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_stratified_model_with_sqrt_wrong_units(self):
        """Test stratified model fails when sqrt receives dimensional quantity."""
        builder = (
            ModelBuilder(name="Stratified Bad Sqrt", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            .add_stratification("age", ["young", "old"])
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            # This should fail: sqrt(I) where I has units of person
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "young"}],
                        "rate": "beta * sqrt(I) * S / N_young",  # sqrt(I) invalid!
                    },
                    {
                        "conditions": [{"stratification": "age", "category": "old"}],
                        "rate": "beta * 0.5 * sqrt(I) * S / N_old",  # sqrt(I) invalid!
                    },
                ],
            )
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.6},
                            {"category": "old", "fraction": 0.4},
                        ],
                    }
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_vaccination_model_with_pow_wrong_exponent(self):
        """Test vaccination model fails when pow has dimensional exponent."""
        builder = (
            ModelBuilder(name="Vaccination Bad Pow", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("V", "Vaccinated")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            .add_parameter("vax_rate", 0.01, "Vaccination rate", unit="1/day")
            .add_parameter("threshold", 100, "Threshold", unit="person")  # Has units!
            # This should fail: pow(I/N, threshold) where threshold has units
            .add_transition(
                "vaccination",
                ["S"],
                ["V"],
                # threshold must be dimensionless!
                rate="vax_rate * pow(I / N, threshold) * S",
            )
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
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

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_complex_model_seasonal_wrong_sin_argument(self):
        """Test complex seasonal model fails with dimensional sin argument."""
        builder = (
            ModelBuilder(name="Complex Seasonal Bad", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("E", "Exposed")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            .add_parameter("beta_avg", 0.5, "Average transmission", unit="1/day")
            .add_parameter(
                "seasonal_amp", 0.2, "Seasonal amplitude", unit="dimensionless"
            )
            .add_parameter("period", 365, "Period", unit="day")  # Has units!
            .add_parameter("sigma", 0.2, "Incubation rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            # This should fail: sin(2 * pi * step / period) where period has units
            # Results in sin(dimensionless / day) which is wrong
            .add_transition(
                "exposure",
                ["S"],
                ["E"],
                rate=(
                    "beta_avg * (1 + seasonal_amp * sin(2 * pi * step / period)) "
                    "* S * I / N"
                ),
            )
            .add_transition("infection", ["E"], ["I"], rate="sigma * E")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.97},
                    {"bin": "E", "fraction": 0.01},
                    {"bin": "I", "fraction": 0.02},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_multi_stratified_model_with_nested_math_errors(self):
        """Test multi-stratified model fails with nested math function errors."""
        builder = (
            ModelBuilder(name="Multi-Strat Bad Nested", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            .add_stratification("age", ["young", "old"])
            .add_stratification("location", ["urban", "rural"])
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            .add_parameter(
                "contact_scale", 50, "Contact scale", unit="person"
            )  # Has units!
            # This should fail: exp(log(contact_scale)) where contact_scale has units
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                stratified_rates=[
                    {
                        "conditions": [
                            {"stratification": "age", "category": "young"},
                            {"stratification": "location", "category": "urban"},
                        ],
                        "rate": (
                            "beta * exp(log(contact_scale)) * S * I / N_young_urban"
                        ),
                    },
                    {
                        "conditions": [
                            {"stratification": "age", "category": "young"},
                            {"stratification": "location", "category": "rural"},
                        ],
                        "rate": "beta * 0.5 * S * I / N_young_rural",
                    },
                    {
                        "conditions": [
                            {"stratification": "age", "category": "old"},
                            {"stratification": "location", "category": "urban"},
                        ],
                        "rate": "beta * 0.7 * S * I / N_old_urban",
                    },
                    {
                        "conditions": [
                            {"stratification": "age", "category": "old"},
                            {"stratification": "location", "category": "rural"},
                        ],
                        "rate": "beta * 0.3 * S * I / N_old_rural",
                    },
                ],
            )
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.6},
                            {"category": "old", "fraction": 0.4},
                        ],
                    },
                    {
                        "stratification": "location",
                        "fractions": [
                            {"category": "urban", "fraction": 0.7},
                            {"category": "rural", "fraction": 0.3},
                        ],
                    },
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_sirs_model_with_atan2_wrong_units(self):
        """Test SIRS model fails when atan2 receives incompatible units."""
        builder = (
            ModelBuilder(name="SIRS Bad Atan2", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            .add_parameter("waning_rate", 0.01, "Waning immunity rate", unit="1/day")
            .add_parameter(
                "threshold_time", 30, "Threshold time", unit="day"
            )  # Has units!
            # This should fail: atan2(I, threshold_time)
            # - incompatible units (person, day)
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                rate="beta * (1 + 0.1 * atan2(I, threshold_time)) * S * I / N",
            )
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .add_transition("waning", ["R"], ["S"], rate="waning_rate * R")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.95},
                    {"bin": "I", "fraction": 0.04},
                    {"bin": "R", "fraction": 0.01},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_complex_intervention_model_with_min_max_errors(self):
        """Test complex intervention model fails with min/max unit errors."""
        builder = (
            ModelBuilder(name="Intervention Bad Min Max", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("H", "Hospitalized")
            .add_bin("R", "Recovered")
            .add_parameter("beta_max", 0.8, "Max transmission", unit="1/day")
            .add_parameter("beta_min", 0.2, "Min transmission", unit="1/day")
            .add_parameter("hosp_rate", 0.05, "Hospitalization rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            .add_parameter(
                "capacity", 50, "Hospital capacity", unit="person"
            )  # Has units!
            # This should fail: min(beta_max, capacity)
            # - incompatible units (1/day, person)
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                rate="min(beta_max, capacity) * S * I / N",  # Incompatible units!
            )
            .add_transition("hospitalization", ["I"], ["H"], rate="hosp_rate * I")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .add_transition("hospital_recovery", ["H"], ["R"], rate="gamma * 0.5 * H")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.97},
                    {"bin": "I", "fraction": 0.02},
                    {"bin": "H", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_age_structured_with_floor_dimensional_error(self):
        """Test age-structured model fails when floor receives dimensional quantity."""
        builder = (
            ModelBuilder(name="Age Structured Bad Floor", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            .add_stratification("age", ["child", "adult", "senior"])
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            # This should fail: floor(S) where S has units of person
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "child"}],
                        "rate": "beta * floor(S) * I / N_child",  # floor(S) invalid!
                    },
                    {
                        "conditions": [{"stratification": "age", "category": "adult"}],
                        "rate": "beta * S * I / N_adult",
                    },
                    {
                        "conditions": [{"stratification": "age", "category": "senior"}],
                        "rate": "beta * 0.6 * S * I / N_senior",
                    },
                ],
            )
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "child", "fraction": 0.2},
                            {"category": "adult", "fraction": 0.5},
                            {"category": "senior", "fraction": 0.3},
                        ],
                    }
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_sin_with_dimensional_argument_fails(self):
        """Test that sin with dimensional argument is caught by pint."""
        variable_units = {
            "beta": "1/day",
            "I": "person",  # Has units, not dimensionless!
            "S": "person",
        }
        # sin(I) should fail because I has units of person
        is_consistent, error = check_equation_units(
            "beta * sin(I) * S",
            variable_units,
            "person/day",
        )
        assert not is_consistent
        assert error is not None
        assert "Cannot convert" in error or "Error" in error

    def test_log_with_dimensional_argument_fails(self):
        """Test that log with dimensional argument is caught by pint."""
        variable_units = {
            "base_rate": "1/day",
            "I": "person",  # Has units, not dimensionless!
            "S": "person",
        }
        # log(I) should fail because I has units of person
        is_consistent, error = check_equation_units(
            "base_rate * log(I) * S",
            variable_units,
            "person/day",
        )
        assert not is_consistent
        assert error is not None

    def test_exp_with_dimensional_argument_fails(self):
        """Test that exp with dimensional argument is caught by pint."""
        variable_units = {
            "beta": "1/day",
            "time": "day",  # Has units, not dimensionless!
            "S": "person",
            "I": "person",
            "N": "person",
        }
        # exp(time) should fail because time has units of day
        is_consistent, error = check_equation_units(
            "beta * exp(time) * S * I / N",
            variable_units,
            "person/day",
        )
        assert not is_consistent
        assert error is not None
        assert "Cannot convert" in error or "Error" in error

    def test_sqrt_with_dimensional_argument_fails(self):
        """Test that sqrt with dimensional argument fails (requires dimensionless)."""
        variable_units = {
            "rate": "1/day",
            "area": "person**2",
        }
        # sqrt(area) should fail because area has units
        is_consistent, error = check_equation_units(
            "rate * sqrt(area)",
            variable_units,
            "person/day",
        )
        assert not is_consistent
        assert error is not None
        assert "Cannot convert" in error or "Error" in error

    def test_math_function_result_unit_mismatch(self):
        """Test that incorrect expected units are caught even with math functions."""
        variable_units = {
            "beta": "1/day",
            "amp": "dimensionless",
            "step": "dimensionless",
            "pi": "dimensionless",
            "S": "person",
            "I": "person",
            "N": "person",
        }
        # Equation produces person/day but we expect person (wrong!)
        is_consistent, error = check_equation_units(
            "beta * (1 + amp * sin(2 * pi * step / 365)) * S * I / N",
            variable_units,
            "person",  # Wrong! Should be person/day
        )
        assert not is_consistent
        assert error is not None
        assert "Unit mismatch" in error

    def test_atan2_with_mismatched_units_fails(self):
        """Test that atan2 with incompatible unit arguments fails."""
        variable_units = {
            "rate": "1/day",
            "y": "person",
            "x": "day",  # Different units than y!
            "S": "person",
        }
        # atan2(y, x) where y and x have incompatible units
        is_consistent, error = check_equation_units(
            "rate * atan2(y, x) * S",
            variable_units,
            "person/day",
        )
        assert not is_consistent
        assert error is not None

    def test_pow_with_dimensional_exponent_fails(self):
        """Test that pow with dimensional exponent is caught."""
        variable_units = {
            "rate": "1/day",
            "base": "dimensionless",
            "exponent": "person",  # Should be dimensionless!
            "S": "person",
        }
        # pow(base, exponent) where exponent has units should fail
        is_consistent, error = check_equation_units(
            "rate * pow(base, exponent) * S",
            variable_units,
            "person/day",
        )
        assert not is_consistent
        assert error is not None

    def test_sir_model_with_dimensional_sin_argument_fails(self):
        """Test SIR model fails when sin receives dimensional argument."""
        builder = (
            ModelBuilder(name="SIR Bad Seasonality", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Add parameters with units
            .add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
            .add_parameter("time_scale", 365, "Time scale", unit="day")  # Has units!
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            # This should fail: sin(time_scale) where time_scale has units of day
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                rate="beta * (1 + 0.1 * sin(time_scale)) * S * I / N",
            )
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should raise UnitConsistencyError
        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "Unit consistency check failed" in str(exc_info.value)

    def test_division_in_math_function_correct_dimensionality(self):
        """Test that division inside math functions correctly handles units."""
        variable_units = {
            "beta": "1/day",
            "I": "person",
            "N": "person",
            "S": "person",
        }
        # I/N is dimensionless, so sin(I/N) should work
        is_consistent, error = check_equation_units(
            "beta * sin(I / N) * S",
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_nested_math_functions_with_wrong_units_fails(self):
        """Test that nested math functions properly validate units."""
        variable_units = {
            "rate": "1/day",
            "x": "person",  # Has units!
            "S": "person",
        }
        # exp(sin(x)) where x has units should fail
        is_consistent, error = check_equation_units(
            "rate * exp(sin(x)) * S",
            variable_units,
            "person/day",
        )
        assert not is_consistent
        assert error is not None

    def test_min_max_with_incompatible_units_fails(self):
        """Test that min/max with incompatible units fails."""
        variable_units = {
            "rate1": "1/day",
            "rate2": "person",  # Incompatible with rate1!
            "S": "person",
        }
        # max(rate1, rate2) where they have different units should fail
        is_consistent, error = check_equation_units(
            "max(rate1, rate2) * S",
            variable_units,
            "person/day",
        )
        assert not is_consistent
        assert error is not None

    def test_floor_with_dimensional_argument_fails(self):
        """Test that floor with dimensional argument fails (requires dimensionless)."""
        variable_units = {
            "rate": "1/day",
            "S": "person",
        }
        # floor(S) should fail because S has units
        is_consistent, error = check_equation_units(
            "rate * floor(S)",
            variable_units,
            "person/day",
        )
        assert not is_consistent
        assert error is not None
        assert "Cannot convert" in error or "Error" in error

    def test_seasonal_forcing_sin(self):
        """Test seasonal forcing with sin function."""
        variable_units = {
            "beta": "1/day",
            "amplitude": "dimensionless",
            "pi": "dimensionless",
            "step": "dimensionless",
            "S": "person",
            "I": "person",
            "N": "person",
        }
        is_consistent, error = check_equation_units(
            "beta * (1 + amplitude * sin(2 * pi * step / 365)) * S * I / N",
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_seasonal_forcing_cos(self):
        """Test seasonal forcing with cos function."""
        variable_units = {
            "beta": "1/day",
            "amplitude": "dimensionless",
            "pi": "dimensionless",
            "step": "dimensionless",
            "S": "person",
            "I": "person",
            "N": "person",
        }
        is_consistent, error = check_equation_units(
            "beta * (1 + amplitude * cos(2 * pi * step / 365)) * S * I / N",
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_logarithmic_rate(self):
        """Test logarithmic function in rate equation."""
        variable_units = {
            "base_rate": "1/day",
            "I": "person",
            "scale": "person",
            "S": "person",
            "N": "person",
        }
        is_consistent, error = check_equation_units(
            "base_rate * log(1 + I / scale) * S * I / N",
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_sqrt_function(self):
        """Test square root function in rate equation."""
        variable_units = {
            "base_rate": "1/day",
            "I": "person",
            "threshold": "person",
            "S": "person",
        }
        is_consistent, error = check_equation_units(
            "base_rate * sqrt(I / threshold) * S",
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_pow_function(self):
        """Test power function in rate equation."""
        variable_units = {
            "base_rate": "1/day",
            "I": "person",
            "threshold": "person",
            "exponent": "dimensionless",
            "S": "person",
        }
        is_consistent, error = check_equation_units(
            "base_rate * pow(I / threshold, exponent) * S",
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_min_max_functions(self):
        """Test min and max functions."""
        variable_units = {
            "beta_min": "1/day",
            "beta_max": "1/day",
            "I": "person",
            "threshold": "person",
            "S": "person",
        }
        is_consistent, error = check_equation_units(
            "max(beta_min, min(beta_max, beta_max * (1 - I / threshold))) * S",
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_abs_function(self):
        """Test absolute value function."""
        variable_units = {
            "rate": "1/day",
            "I": "person",
        }
        is_consistent, error = check_equation_units(
            "abs(rate) * I",
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_sir_model_with_exponential_decay(self):
        """Test SIR model with exponential decay in transmission rate."""
        builder = (
            ModelBuilder(name="SIR with Decay", version="1.0", bin_unit="person")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Add parameters with units
            .add_parameter("beta_0", 0.5, "Initial transmission rate", unit="1/day")
            .add_parameter("decay_rate", 0.01, "Decay rate", unit="dimensionless")
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            # Add transitions with exponential decay
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                rate="beta_0 * exp(-decay_rate * step) * S * I / N",
            )
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should not raise an exception
        model.check_unit_consistency()

    def test_sir_model_with_seasonal_forcing(self):
        """Test SIR model with seasonal forcing using sin."""
        builder = (
            ModelBuilder(name="SIR with Seasonality", version="1.0", bin_unit="person")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Add parameters with units
            .add_parameter("beta_avg", 0.5, "Average transmission rate", unit="1/day")
            .add_parameter(
                "seasonal_amp", 0.1, "Seasonal amplitude", unit="dimensionless"
            )
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            # Add transitions with seasonal forcing
            .add_transition(
                "infection",
                ["S"],
                ["I"],
                rate=(
                    "beta_avg * (1 + seasonal_amp * sin(2 * pi * step / 365)) "
                    "* S * I / N"
                ),
            )
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should not raise an exception
        model.check_unit_consistency()

    def test_combined_math_functions(self):
        """Test equation with multiple math functions combined."""
        variable_units = {
            "base": "1/day",
            "amp1": "dimensionless",
            "amp2": "dimensionless",
            "decay": "dimensionless",
            "step": "dimensionless",
            "pi": "dimensionless",
            "S": "person",
            "I": "person",
            "N": "person",
        }
        # Complex equation with sin, cos, and exp
        is_consistent, error = check_equation_units(
            (
                "base * exp(-decay * step) * (1 + amp1 * sin(2 * pi * step / 365) "
                "+ amp2 * cos(4 * pi * step / 365)) * S * I / N"
            ),
            variable_units,
            "person/day",
        )
        assert is_consistent, f"Error: {error}"
        assert error is None

    def test_formula_parameter_unit_inference(self):
        """Test that units are correctly inferred for formula parameters."""
        builder = (
            ModelBuilder(
                name="Model with Formula Parameters", version="1.0", bin_unit="person"
            )
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Base parameters with units
            .add_parameter("beta_base", 0.3, "Base transmission rate", unit="1/day")
            .add_parameter(
                "multiplier", 2.0, "Contact multiplier", unit="dimensionless"
            )
            # Formula parameter WITHOUT explicit unit - should be inferred
            .add_parameter(
                "beta",
                "beta_base * multiplier",
                "Effective transmission rate",
                # No unit specified - should be inferred as "1/day"
            )
            .add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Unit checking should pass because beta's unit is inferred as "1/day"
        model.check_unit_consistency()

    def test_formula_parameter_with_N_reference(self):
        """Test formula parameter referencing N and N_category."""
        builder = (
            ModelBuilder(
                name="Model with N-based Formula", version="1.0", bin_unit="person"
            )
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            .add_stratification("age", ["young", "old"])
            # Formula parameter using N variables - should infer as dimensionless
            .add_parameter(
                "young_fraction",
                "N_young / N",
                "Fraction of young population",
                # No unit - should be inferred as dimensionless
            )
            .add_parameter("beta", 0.3, unit="1/day")
            .add_parameter("gamma", 0.1, unit="1/day")
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.3},
                            {"category": "old", "fraction": 0.7},
                        ],
                    }
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        model.check_unit_consistency()

    def test_formula_parameter_missing_dependency_units(self):
        """
        Test that error is raised when formula parameter references variables without
        units.
        """
        builder = (
            ModelBuilder(name="Model with Invalid Formula", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # All base parameters have units
            .add_parameter("beta_base", 0.3, unit="1/day")
            .add_parameter("multiplier", 2.0, unit="dimensionless")
            # This parameter has no unit AND references non-existent variable
            .add_parameter("invalid_param", "nonexistent_var * 2")  # Should fail!
            .add_parameter("gamma", 0.1, unit="1/day")
            # Use invalid_param in a transition so it gets checked
            .add_transition("infection", ["S"], ["I"], rate="invalid_param * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should raise error because invalid_param has no defined unit
        # (because it references nonexistent_var which has no unit)
        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        # The error should mention that invalid_param has no defined unit
        assert "invalid_param" in str(exc_info.value)
        assert "no defined unit" in str(exc_info.value)

    def test_formula_parameter_chained_dependencies(self):
        """Test formula parameters with chained dependencies."""
        builder = (
            ModelBuilder(
                name="Model with Chained Formulas", version="1.0", bin_unit="person"
            )
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Base parameter
            .add_parameter("base_rate", 0.1, unit="1/day")
            # First level formula
            .add_parameter("adjusted_rate", "base_rate * 2")  # Should infer 1/day
            # Second level formula (depends on first level)
            .add_parameter("final_rate", "adjusted_rate * 3")  # Should infer 1/day
            .add_parameter("gamma", 0.1, unit="1/day")
            .add_transition("infection", ["S"], ["I"], rate="final_rate * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        # Should successfully infer units through the chain
        model.check_unit_consistency()

    def test_constant_parameter_without_unit_raises_error(self):
        """
        Test that constant parameters without units raise an error during unit checking.
        """
        builder = (
            ModelBuilder(name="Model with Missing Unit", version="1.0")
            .add_bin("S", "Susceptible")
            .add_bin("I", "Infected")
            .add_bin("R", "Recovered")
            # Constant parameter without unit
            .add_parameter("beta", 0.3)  # Missing unit!
            .add_parameter("gamma", 0.1, unit="1/day")
            .add_transition("infection", ["S"], ["I"], rate="beta * S * I / N")
            .add_transition("recovery", ["I"], ["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should raise error because beta (constant) has no unit
        with pytest.raises(UnitConsistencyError) as exc_info:
            model.check_unit_consistency()

        assert "beta" in str(exc_info.value)
        assert "missing units" in str(exc_info.value)
