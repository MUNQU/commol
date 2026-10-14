"""The engine of a model: the engine's list, `build(engine, settings)`, saving."""

import math
from typing import get_args

import pytest
from pydantic import ValidationError

from commol import Model, OdeSolver, Simulation
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.engine import CompartmentalEngineType
from tests.support import a_to_b_builder, assert_close, build, build_ode


class TestEngineList:
    def test_the_python_names_equal_the_compartmental_engines(self) -> None:
        compartmental = [
            name
            for name in commol_rs.engine.EngineType.names()
            if commol_rs.engine.EngineType(name).kind == "compartmental"
        ]

        assert list(get_args(CompartmentalEngineType)) == compartmental

    def test_each_engine_says_what_a_rate_means(self) -> None:
        meanings = {
            name: commol_rs.engine.EngineType(name).rate_meaning
            for name in get_args(CompartmentalEngineType)
        }

        assert meanings == {
            "DifferenceEquations": "per_step",
            "DifferentialEquations": "continuous",
        }

    def test_an_unknown_engine_lists_the_engines(self) -> None:
        with pytest.raises(ValueError, match="DifferentialEquations"):
            commol_rs.engine.EngineType("Stochastic")


class TestBuild:
    @pytest.mark.parametrize("name", get_args(CompartmentalEngineType))
    def test_one_builder_builds_the_model_for_each_engine(self, name: str) -> None:
        model = a_to_b_builder().build(engine=name)

        simulation = Simulation(model)

        assert model.engine.type == name
        assert simulation.engine.engine_type == name

    def test_each_engine_runs_the_model_with_its_meaning_of_a_rate(self) -> None:
        builder = a_to_b_builder()
        difference = Simulation(builder.build(engine="DifferenceEquations")).run(1)
        differential = Simulation(
            builder.build(
                engine="DifferentialEquations", settings=OdeSolver(rtol=1e-10)
            )
        ).run(1)

        assert_close(difference["A"][1], 1000 * (1 - 0.1), rel=1e-12)
        assert_close(differential["A"][1], 1000 * math.exp(-0.1), rel=1e-7)

    def test_the_engine_takes_its_settings(self) -> None:
        settings = OdeSolver(method="bdf", rtol=1e-8)

        model = a_to_b_builder().build(
            engine="DifferentialEquations", settings=settings
        )

        assert model.engine.settings == settings

    def test_settings_of_another_engine_name_both_engines(self) -> None:
        with pytest.raises(ValidationError) as error:
            a_to_b_builder().build(engine="DifferenceEquations", settings=OdeSolver())

        assert "DifferentialEquations" in str(error.value)
        assert "DifferenceEquations" in str(error.value)

    def test_an_unknown_engine_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="DifferentialEquations"):
            a_to_b_builder().build(engine="Stochastic")  # ty: ignore[invalid-argument-type]


class TestSaveAndLoad:
    def test_settings_are_saved_and_loaded_back(self, tmp_path) -> None:
        model = build_ode(a_to_b_builder(), OdeSolver(method="bdf", rtol=1e-8))
        path = tmp_path / "model.json"
        model.to_json(path)

        loaded = Model.from_json(path)

        assert loaded.engine == model.engine

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
