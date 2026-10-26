"""Check ``.github/scripts/affected.py``: the open issues whose body names
what a branch changes. The rules are in ``docs/development/workflow.md``,
"Lifecycle of a proposal"."""

from tests.support.planning import load

affected = load("affected")

RESULTS = "py-commol/commol/_simulation/results.py"
CHANGED = {RESULTS: "@@ -10,0 +11 @@\n+    def quantities(self) -> None:\n"}


def test_a_dotted_symbol_is_not_a_path() -> None:
    body = f"`Model.save` writes the file; see `{RESULTS}`."

    assert not affected.is_path("Model.save")
    assert [match.item for match in affected.matches(body, CHANGED)] == [RESULTS]


def test_an_item_matches_a_changed_file_its_directory_or_its_symbol() -> None:
    body = f"""## Current state

`_simulation/results.py::quantities` reads `_simulation/results.py::plot`;
run `pytest -m docs`.
Every change edits `AGENTS.md` and `tests/`.

```bash
cat py-commol/commol/_simulation/windows.py
```

## Touches

- Changes: `{RESULTS}`
- Adds: `py-commol/commol/_simulation/`
"""

    assert affected.matches(body, CHANGED) == [
        affected.Match("_simulation/results.py::quantities", "body", [RESULTS]),
        affected.Match(RESULTS, "Touches (Changes)", [RESULTS]),
        affected.Match("py-commol/commol/_simulation/", "Touches (Adds)", [RESULTS]),
    ]
