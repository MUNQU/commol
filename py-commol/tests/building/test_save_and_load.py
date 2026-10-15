"""A model saved to JSON loads back as an equal model."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from commol.context.model import Model
from commol.context.parameter import SeriesValue
from tests.support import a_to_b_builder, build, initial_state


def _model(k: float | None = 0.1, fraction: float | None = 1.0) -> Model:
    """Its parameter and bin fraction may be left for calibration."""
    return build(
        a_to_b_builder(
            parameters={"k": k},
            total=100,
            fractions={"A": fraction, "B": 0.0},
        )
    )


class TestToJson:
    def test_the_file_holds_the_header_the_engine_and_the_structure(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "model.json"
        _model().to_json(path)

        data = json.loads(path.read_text())

        assert list(data) == [
            "name",
            "description",
            "version",
            "parameters",
            "engine",
            "structure",
            "step_unit",
        ]
        assert data["structure"]["kind"] == "compartmental"
        assert len(data["structure"]["transitions"]) == 1

    def test_a_file_with_population_and_dynamics_names_them(
        self, tmp_path: Path
    ) -> None:
        data = _model().model_dump(mode="json")
        structure = data.pop("structure")
        structure.pop("kind")
        data["dynamics"] = {"transitions": structure["transitions"]}
        data["population"] = structure
        path = tmp_path / "old.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValidationError) as error:
            Model.from_json(path)

        assert "population" in str(error.value)
        assert "dynamics" in str(error.value)

    def test_a_file_with_population_size_names_it(self, tmp_path: Path) -> None:
        data = _model().model_dump(mode="json")
        conditions = data["structure"]["initial_conditions"]
        conditions["population_size"] = conditions.pop("total")
        path = tmp_path / "old.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValidationError, match="population_size"):
            Model.from_json(path)

    def test_a_real_total_reloads_with_the_same_initial_state(
        self, tmp_path: Path
    ) -> None:
        model = build(a_to_b_builder(total=2.5, fractions={"A": 0.4, "B": 0.6}))
        path = tmp_path / "model.json"

        model.to_json(path)

        assert initial_state(Model.from_json(path)) == initial_state(model)
        assert initial_state(model) == pytest.approx({"A": 1.0, "B": 1.5})

    def test_round_trips_to_an_equal_model(self, tmp_path: Path) -> None:
        model = _model()
        path = tmp_path / "model.json"

        model.to_json(path)

        assert Model.from_json(path) == model

    def test_an_uncalibrated_parameter_round_trips(self, tmp_path: Path) -> None:
        model = _model(k=None)
        path = tmp_path / "model.json"

        model.to_json(path)
        restored = Model.from_json(path)

        assert restored == model
        assert restored.get_uncalibrated_parameters() == ["k"]

    def test_an_uncalibrated_bin_fraction_round_trips(self, tmp_path: Path) -> None:
        model = _model(fraction=None)
        path = tmp_path / "model.json"

        model.to_json(path)
        restored = Model.from_json(path)

        assert restored == model
        assert restored.structure.uncalibrated_values() == ["A"]

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        model = _model()
        path = tmp_path / "model.json"

        model.to_json(str(path))

        assert Model.from_json(path) == model

    def test_long_numeric_arrays_are_written_on_one_line(self, tmp_path: Path) -> None:
        model = _model()
        model.parameters[0].value = SeriesValue(
            data=[(step, float(step)) for step in range(20)]
        )
        path = tmp_path / "model.json"

        model.to_json(path)
        text = path.read_text()

        assert Model.from_json(path) == model
        assert max(len(line) for line in text.splitlines()) > 100

    def test_indent_is_configurable(self, tmp_path: Path) -> None:
        model = _model()
        wide = tmp_path / "wide.json"
        narrow = tmp_path / "narrow.json"

        model.to_json(wide, indent=8)
        model.to_json(narrow, indent=2)

        assert len(wide.read_text()) > len(narrow.read_text())
        assert Model.from_json(wide) == Model.from_json(narrow)
