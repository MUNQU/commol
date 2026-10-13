"""A model saved to JSON loads back as an equal model."""

from pathlib import Path

from commol.context.model import Model
from commol.context.parameter import SeriesValue
from tests.support import a_to_b_builder, build


def _model(k: float | None = 0.1, fraction: float | None = 1.0) -> Model:
    """Its parameter and bin fraction may be left for calibration."""
    return build(
        a_to_b_builder(
            parameters={"k": k},
            population=100,
            fractions={"A": fraction, "B": 0.0},
        )
    )


class TestToJson:
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
        assert restored.get_uncalibrated_initial_conditions() == ["A"]

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
