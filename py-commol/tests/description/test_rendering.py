"""Renderings of a description: text, Markdown, LaTeX, JSON and DOT.

The Mermaid, DOT and LaTeX outputs are compared with the snapshots of
``tests/description/snapshots/``. The DOT output also parses with ``pydot`` and
the LaTeX output compiles with ``latexmk`` when they are installed. To accept
a new rendering, delete the snapshot and run the test once: it writes the
snapshot and fails, so the change is reviewed in the diff.
"""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from commol import Aggregate, ModelBuilder
from commol.description.blocks import (
    Equation,
    EquationGroup,
    Equations,
    Matrix,
    ModelDescription,
    Section,
    Table,
)
from commol._model.model import Model
from commol.description.render import mermaid
from tests.support import FactoryBuilder, abc_builder, build
from tests.support.described import (
    every_flow,
    stochastic_with_settings,
    weighted_groups,
)
from tests.support.description import matches_snapshot


def _graph_model() -> Model:
    """Several sources, an inflow, an outflow and a category transition."""
    builder = (
        FactoryBuilder(name="Graph", version="1.0")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_parameter(id="k1", value=0.001)
        .add_parameter(id="b", value=1.0)
        .add_transition(
            id="pair", source=["A", "B"], target=["C", "C"], rate="k1 * A * B"
        )
        .add_transition(id="inflow", source=[], target=["A"], rate="b")
        .add_transition(id="outflow", source=["C"], target=[], rate="0.01")
        .add_transition(id="move", source=["g1"], target=["g2"], rate="0.02")
    )
    builder.set_structure(
        Aggregate(
            bin_amounts={"A": 500, "B": 500, "C": 0},
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
        )
    )
    return build(builder)


class TestText:
    def test_the_text_keeps_the_compact_and_expanded_forms(self) -> None:
        text = build(abc_builder()).describe().render("text")

        compact = (
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  A -> B: k1 * A * B / total\n"
            "\n"
            "bc (B -> C):\n"
            "  B -> C: k2 * B\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
        )
        expanded = (
            "[A(t+Dt) - A(t)] / Dt = - (k1 * A * B / total)\n"
            "[B(t+Dt) - B(t)] / Dt = (k1 * A * B / total) - (k2 * B)\n"
            "[C(t+Dt) - C(t)] / Dt = (k2 * B)\n"
        )
        assert compact in text
        assert expanded in text
        assert build(abc_builder()).describe().render() == text

    def test_the_text_draws_no_diagram(self) -> None:
        text = _graph_model().describe().render("text")

        assert "flowchart" not in text
        assert "digraph" not in text

    def test_a_section_with_no_blocks_is_left_out(self) -> None:
        description = build(abc_builder()).describe()
        description.section("outputs").blocks.clear()

        assert "Outputs\n-------" not in description.render("text")
        assert "## Outputs" not in description.render("markdown")


class TestMarkdown:
    def test_markdown_draws_the_graphs_of_the_bins_and_the_strata(self) -> None:
        description = _graph_model().describe()

        markdown = description.render("markdown")

        bins, strata, _ = description.graphs()
        assert markdown.count("```mermaid") == 2
        assert mermaid(bins) in markdown
        assert mermaid(strata) in markdown
        diagrams = [part.split("```")[0] for part in markdown.split("```mermaid")[1:]]
        assert not any("A_g1" in diagram for diagram in diagrams)
        assert "The graph of the expanded compartments has 6 nodes" in markdown

    def test_a_notebook_shows_the_markdown(self) -> None:
        description = build(abc_builder()).describe()

        assert description._repr_markdown_() == description.render("markdown")

    def test_the_mermaid_graphs_match_their_snapshot(self) -> None:
        bins, strata, _ = _graph_model().describe().graphs()

        matches_snapshot("graph.mmd", f"{mermaid(bins)}\n\n{mermaid(strata)}\n")

    def test_the_outflows_of_a_graph_end_at_one_node(self) -> None:
        builder = abc_builder()
        for bin_id in ("A", "B", "C"):
            builder.add_transition(
                id=f"out_{bin_id}", source=[bin_id], target=[], rate="0.01"
            )

        diagram = mermaid(build(builder).describe().graphs()[0])

        assert diagram.count('(("none"))') == 1
        assert diagram.count("| outflow") == 3


class TestDot:
    def test_a_dot_file_holds_every_graph(self, tmp_path: Path) -> None:
        path = _graph_model().describe().save(tmp_path / "model.dot")

        dot = path.read_text(encoding="utf-8")
        assert dot.count("digraph") == 3
        assert 'digraph "states"' in dot
        assert 'digraph "strata"' in dot
        assert 'digraph "compartments"' in dot
        matches_snapshot("graph.dot", dot)

    @pytest.mark.skipif(
        importlib.util.find_spec("pydot") is None, reason="pydot is not installed"
    )
    # pydot 4 calls pyparsing names that pyparsing 3.3 deprecates.
    @pytest.mark.filterwarnings("ignore::DeprecationWarning:pydot.dot_parser")
    def test_the_dot_file_parses(self) -> None:
        import pydot

        graphs = pydot.graph_from_dot_data(_graph_model().describe().render("dot"))

        assert graphs is not None
        assert [graph.get_name() for graph in graphs] == [
            '"states"',
            '"strata"',
            '"compartments"',
        ]


class TestLatex:
    def test_the_latex_document_matches_its_snapshot(self) -> None:
        matches_snapshot("graph.tex", _graph_model().describe().render("latex"))

    @pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk is missing")
    @pytest.mark.parametrize("layout", ["default", "odd"])
    def test_the_latex_document_compiles(self, tmp_path: Path, layout: str) -> None:
        path = _graph_model().describe(layout=layout).save(tmp_path / "model.tex")  # ty: ignore[invalid-argument-type]

        run = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                path.name,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert run.returncode == 0, run.stdout[-2000:]
        assert (tmp_path / "model.pdf").exists()


class TestJson:
    def test_the_json_loads_back_into_an_equal_description(self) -> None:
        description = _graph_model().describe(layout="odd")

        loaded = ModelDescription.model_validate_json(description.render("json"))

        assert loaded == description
        assert json.loads(description.render("json"))["layout"] == "odd"


class TestSave:
    @pytest.mark.parametrize(
        ("suffix", "start"),
        [
            (".txt", "Graph\n====="),
            (".md", "# Graph"),
            (".tex", "\\documentclass"),
            (".json", "{"),
            (".dot", 'digraph "states"'),
        ],
    )
    def test_the_format_comes_from_the_suffix(
        self, tmp_path: Path, suffix: str, start: str
    ) -> None:
        path = _graph_model().describe().save(tmp_path / f"model{suffix}")

        assert path.read_text(encoding="utf-8").startswith(start)

    def test_an_unknown_suffix_lists_the_suffixes(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"\.md"):
            _graph_model().describe().save(tmp_path / "model.html")

    def test_a_format_overrides_the_suffix(self, tmp_path: Path) -> None:
        path = _graph_model().describe().save(tmp_path / "model.out", format="markdown")

        assert path.read_text(encoding="utf-8").startswith("# Graph")


class TestEveryKindRenders:
    @pytest.mark.parametrize("layout", ["default", "odd"])
    @pytest.mark.parametrize("format", ["text", "markdown", "latex", "json", "dot"])
    def test_every_layout_renders_in_every_format(
        self, layout: str, format: str
    ) -> None:
        rendered = _graph_model().describe(layout=layout).render(format)  # ty: ignore[invalid-argument-type]

        assert rendered.strip()

    def test_a_builder_without_a_structure_still_describes_through_its_model(
        self,
    ) -> None:
        builder = ModelBuilder(name="Plain").add_bin(id="A", name="A")
        builder.add_transition(id="out", source=["A"], target=[], rate="0.1")

        model = builder.build(
            Aggregate(bin_amounts={"A": 10}), engine="DifferenceEquations"
        )

        assert "dA/dt" not in model.describe().render("text")
        assert "[A(t+Dt) - A(t)] / Dt = - (0.1 * A)" in model.describe().render("text")


class TestEveryBlock:
    """Every kind of block, with and without a heading and with empty cells."""

    def test_a_matrix_has_one_value_per_row_and_column(self) -> None:
        with pytest.raises(ValueError, match="2 rows and 2 columns has rows of"):
            Matrix(row_labels=["a", "b"], column_labels=["a", "b"], values=[[1.0, 0.0]])

    def test_a_matrix_shows_its_heading(self) -> None:
        block = Matrix(
            heading="Weights",
            row_labels=["a", "b"],
            column_labels=["a", "b"],
            values=[[1.0, 0.5], [0.5, 1.0]],
        )
        description = ModelDescription(
            title="t",
            layout="default",
            sections=[Section(key="s", title="S", blocks=[block])],
        )

        assert "### Weights\n" in description.render("markdown")
        assert "Weights" in description.render("text")

    def test_a_row_has_one_cell_per_column(self) -> None:
        with pytest.raises(ValueError, match="Row 1 of the table has 1 cells"):
            Table(columns=["a", "b"], rows=[["x", "y"], ["z"]])

    def test_a_group_without_a_heading_is_followed_by_the_next(self) -> None:
        def equation(name: str) -> Equation:
            return Equation(left=name, right="0", left_latex=name, right_latex="0")

        block = Equations(
            groups=[
                EquationGroup(equations=[equation("x")]),
                EquationGroup(heading="h", equations=[equation("y")]),
            ]
        )
        description = ModelDescription(
            title="t",
            layout="default",
            sections=[Section(key="s", title="S", blocks=[block])],
        )

        assert "x = 0\n\nh:\n  y = 0" in description.render("text")

    @pytest.mark.parametrize(
        ("model", "layout"),
        [
            (weighted_groups, "default"),
            (stochastic_with_settings, "odd"),
            (every_flow, "default"),
        ],
    )
    @pytest.mark.parametrize(
        ("format", "suffix"),
        [("text", "txt"), ("markdown", "md"), ("latex", "tex"), ("dot", "dot")],
    )
    def test_a_rendering_matches_its_snapshot(
        self, model, layout: str, format: str, suffix: str
    ) -> None:
        rendered = model().describe(layout).render(format)

        matches_snapshot(f"render_{model.__name__}_{layout}.{suffix}", rendered)
