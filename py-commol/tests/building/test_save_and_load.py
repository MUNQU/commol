"""A model saved to JSON loads back as an equal model."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from commol.context._model.serialization import INLINE_ARRAY_MIN_LENGTH, render_json
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
    def test_the_file_holds_the_header_the_process_and_the_structure(
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
            "process",
            "structure",
            "step_unit",
        ]
        assert data["structure"]["kind"] == "aggregate"
        assert data["structure"]["variant"] is None
        assert len(data["process"]["transitions"]) == 1
        assert data["structure"]["bin_fractions"] == {"A": 1.0, "B": 0.0}

    def test_a_file_of_the_compartmental_kind_lists_the_kinds(
        self, tmp_path: Path
    ) -> None:
        data = _model().model_dump(mode="json")
        data["structure"]["kind"] = "compartmental"
        path = tmp_path / "old.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValidationError, match="aggregate"):
            Model.from_json(path)

    def test_a_file_with_population_and_dynamics_names_them(
        self, tmp_path: Path
    ) -> None:
        data = _model().model_dump(mode="json")
        structure = data.pop("structure")
        structure.pop("kind")
        data["dynamics"] = {"transitions": data.pop("process")["transitions"]}
        data["population"] = structure
        path = tmp_path / "old.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValidationError) as error:
            Model.from_json(path)

        assert "population" in str(error.value)
        assert "dynamics" in str(error.value)

    def test_a_file_with_population_size_names_it(self, tmp_path: Path) -> None:
        data = _model().model_dump(mode="json")
        conditions = data["structure"]
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

    def test_a_file_is_ascii_and_a_file_in_utf_8_loads(self, tmp_path: Path) -> None:
        model = _model().model_copy(update={"description": "Población en régimen"})
        written = tmp_path / "written.json"
        by_hand = tmp_path / "by_hand.json"

        model.to_json(written)
        by_hand.write_bytes(
            json.dumps(model.model_dump(mode="json"), ensure_ascii=False).encode()
        )

        assert written.read_bytes().isascii()
        assert not by_hand.read_bytes().isascii()
        assert Model.from_json(written) == model
        assert Model.from_json(by_hand) == model

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

    @pytest.mark.parametrize(
        "array",
        [
            list(range(INLINE_ARRAY_MIN_LENGTH)),
            [1.5, None] * (INLINE_ARRAY_MIN_LENGTH // 2),
            [[0, 1]] * INLINE_ARRAY_MIN_LENGTH,
        ],
    )
    def test_an_array_of_numbers_or_nulls_of_the_minimum_length_is_one_line(
        self, array: list
    ) -> None:
        assert render_json({"a": array}) == f'{{\n  "a": {json.dumps(array)}\n}}'

    @pytest.mark.parametrize(
        "array",
        [
            list(range(INLINE_ARRAY_MIN_LENGTH - 1)),
            ["x"] * INLINE_ARRAY_MIN_LENGTH,
            [True] * INLINE_ARRAY_MIN_LENGTH,
            [["x"]] * INLINE_ARRAY_MIN_LENGTH,
            [1, [2]] * (INLINE_ARRAY_MIN_LENGTH // 2),
        ],
    )
    def test_a_shorter_array_or_one_of_other_values_is_indented(
        self, array: list
    ) -> None:
        assert render_json({"a": array}) == json.dumps({"a": array}, indent=2)

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

        default = tmp_path / "default.json"

        model.to_json(wide, indent=8)
        model.to_json(narrow, indent=2)
        model.to_json(default)

        assert len(wide.read_text()) > len(narrow.read_text())
        assert default.read_text() == narrow.read_text()
        assert Model.from_json(wide) == Model.from_json(narrow)
