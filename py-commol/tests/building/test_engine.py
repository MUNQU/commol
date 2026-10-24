"""The engine of a model: the list of engines, `build(structure, engine)`, saving."""

import math
from typing import get_args

import pytest
from pydantic import ValidationError

from commol import (
    Model,
    ODESolver,
    Simulation,
    StochasticEventsSettings,
    StochasticStepsSettings,
)
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.engine import EngineName
from tests.support.kinds import TEST_ENGINE, TEST_STOCHASTIC_ENGINE
from tests.support import a_to_b_builder, assert_close, build, build_ode


class TestEngineList:
    def test_the_python_names_equal_the_engines_of_the_list(self) -> None:
        engines = [
            name
            for name in commol_rs.engine.EngineType.names()
            if name not in {TEST_ENGINE, TEST_STOCHASTIC_ENGINE}
        ]

        assert list(get_args(EngineName)) == engines

    def test_the_aggregate_kind_lists_its_engines(self) -> None:
        kind = commol_rs.engine.ModelKind("aggregate")

        assert kind.engines() == [
            "DifferenceEquations",
            "DifferentialEquations",
            "StochasticEvents",
            "StochasticSteps",
        ]

    def test_each_engine_says_what_a_rate_means(self) -> None:
        meanings = {
            name: commol_rs.engine.EngineType(name).rate_meaning
            for name in get_args(EngineName)
        }

        assert meanings == {
            "DifferenceEquations": "per_step",
            "DifferentialEquations": "continuous",
            "StochasticEvents": "continuous",
            "StochasticSteps": "per_step",
        }

    def test_an_unknown_engine_lists_the_engines(self) -> None:
        with pytest.raises(ValueError, match="DifferentialEquations"):
            commol_rs.engine.EngineType("Stochastic")

    def test_every_engine_gives_the_form_of_its_dynamics(self) -> None:
        forms = {
            "difference_equation",
            "differential_equation",
            "stochastic_events",
            "rules",
        }
        for name in commol_rs.engine.EngineType.names():
            engine = commol_rs.engine.EngineType(name)
            assert engine.dynamics_form in forms
            assert engine.schedule

    def test_an_engine_of_equations_gives_their_left_hand_side(self) -> None:
        difference = commol_rs.engine.EngineType("DifferenceEquations")
        differential = commol_rs.engine.EngineType("DifferentialEquations")

        assert difference.dynamics_form == "difference_equation"
        assert difference.equation_lhs("A", False) == "[A(t+Dt) - A(t)] / Dt"
        assert differential.equation_lhs("A", False) == "dA/dt"
        assert differential.equation_lhs("A", True) == "\\frac{dA}{dt}"

    def test_every_engine_runs_a_kind_of_the_list(self) -> None:
        kinds = commol_rs.engine.ModelKind.names()

        assert "aggregate" in kinds
        for name in commol_rs.engine.EngineType.names():
            engine_kinds = commol_rs.engine.EngineType(name).kinds
            assert engine_kinds
            assert set(engine_kinds) <= set(kinds)

    def test_an_aggregate_model_accepts_initial_conditions(self) -> None:
        kind = commol_rs.engine.ModelKind("aggregate")

        assert kind.value_types() == ["initial_condition"]

    def test_an_unknown_kind_lists_the_kinds(self) -> None:
        engine = commol_rs.engine.EngineType("DifferenceEquations")

        with pytest.raises(ValueError, match="aggregate"):
            engine.check_kind("network")


class TestBuild:
    @pytest.mark.parametrize("name", get_args(EngineName))
    def test_one_process_builds_with_an_aggregate_for_each_engine(
        self, name: EngineName
    ) -> None:
        builder = a_to_b_builder()
        model = builder.build(builder.structure, engine=name)

        simulation = Simulation(model)

        assert model.engine.type == name
        assert simulation.engine.engine_type == name

    def test_each_engine_runs_the_model_with_its_meaning_of_a_rate(self) -> None:
        builder = a_to_b_builder()
        difference = Simulation(
            builder.build(builder.structure, engine="DifferenceEquations")
        ).run(1)
        differential = Simulation(
            builder.build(
                builder.structure,
                engine="DifferentialEquations",
                settings=ODESolver(rtol=1e-10),
            )
        ).run(1)

        assert_close(difference["A"][1], 1000 * (1 - 0.1), rel=1e-12)
        assert_close(differential["A"][1], 1000 * math.exp(-0.1), rel=1e-7)

    def test_the_engine_takes_its_settings(self) -> None:
        settings = ODESolver(method="bdf", rtol=1e-8)

        builder = a_to_b_builder()
        model = builder.build(
            builder.structure, engine="DifferentialEquations", settings=settings
        )

        assert model.engine.settings == settings

    def test_the_engine_settings_take_the_defaults_of_the_engine(self) -> None:
        engine = commol_rs.engine

        assert (ODESolver().rtol, ODESolver().atol) == (
            engine.ODE_DEFAULT_RTOL,
            engine.ODE_DEFAULT_ATOL,
        )
        events = StochasticEventsSettings()
        assert (events.epsilon, events.tolerance) == (
            engine.STOCHASTIC_EVENTS_DEFAULT_EPSILON,
            engine.STOCHASTIC_EVENTS_DEFAULT_TOLERANCE,
        )
        builder = a_to_b_builder()
        unset, default = (
            Simulation(
                builder.build(
                    builder.structure, engine="DifferentialEquations", settings=settings
                )
            ).run(5)
            for settings in (None, ODESolver())
        )
        assert unset == default

    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            (
                lambda: StochasticEventsSettings(epsilon=1.5),
                "epsilon is strictly between 0 and 1, got 1.5",
            ),
            (
                lambda: ODESolver(rtol=0.0),
                "rtol is a finite number greater than 0, got 0",
            ),
        ],
        ids=["epsilon", "rtol"],
    )
    def test_the_engine_checks_the_settings(self, settings, message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            settings()

    def test_settings_of_another_engine_name_both_engines(self) -> None:
        builder = a_to_b_builder()
        with pytest.raises(ValidationError) as error:
            builder.build(
                builder.structure, engine="DifferenceEquations", settings=ODESolver()
            )

        assert "DifferentialEquations" in str(error.value)
        assert "DifferenceEquations" in str(error.value)

    def test_an_unknown_engine_is_rejected(self) -> None:
        builder = a_to_b_builder()
        with pytest.raises(ValidationError, match="DifferentialEquations"):
            builder.build(builder.structure, engine="Stochastic")  # ty: ignore[invalid-argument-type]


class TestSaveAndLoad:
    def test_settings_are_saved_and_loaded_back(self, tmp_path) -> None:
        model = build_ode(a_to_b_builder(), ODESolver(method="bdf", rtol=1e-8))
        path = tmp_path / "model.json"
        model.to_json(path)

        loaded = Model.from_json(path)

        assert loaded.engine == model.engine

    @pytest.mark.parametrize(
        ("engine", "settings"),
        [
            ("StochasticEvents", StochasticEventsSettings(method="tau_leaping")),
            ("StochasticEvents", StochasticEventsSettings(rounding="random")),
            ("StochasticSteps", StochasticStepsSettings(rounding="random")),
        ],
    )
    def test_stochastic_settings_load_back_as_their_class(
        self, tmp_path, engine: EngineName, settings: object
    ) -> None:
        builder = a_to_b_builder()
        model = builder.build(builder.structure, engine=engine, settings=settings)  # ty: ignore[invalid-argument-type]
        path = tmp_path / "model.json"
        model.to_json(path)

        loaded = Model.from_json(path)

        assert loaded == model
        assert type(loaded.engine.settings) is type(settings)

    def test_a_difference_model_saves_no_settings(self) -> None:
        model = build(a_to_b_builder())

        assert '"engine":{"type":"DifferenceEquations","settings":null}' in (
            model.model_dump_json()
        )

    def test_a_saved_model_with_settings_of_another_engine_does_not_load(
        self,
    ) -> None:
        payload = build(a_to_b_builder()).model_dump_json()
        payload = payload.replace(
            '"settings":null', '"settings":{"method":"bdf","rtol":1e-6,"atol":1e-6}'
        )

        with pytest.raises(ValidationError, match="DifferentialEquations"):
            Model.model_validate_json(payload)
