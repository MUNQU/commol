# Quantities API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.quantities`: the quantities of interest that a method reads from each run. Guide: [Sensitivity analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/index.md).

## Level

Bases: `_Outputs`

The level of some outputs: their sum at each position.

A quantity given a name or a list of names reads this series.

Attributes:

| Name      | Type        | Description                                    |
| --------- | ----------- | ---------------------------------------------- |
| `outputs` | `list[str]` | The outputs summed; one name is a list of one. |

Source code in `commol/quantities/quantities.py`

```python
class Level(_Outputs):
    """
    The level of some outputs: their sum at each position.

    A quantity given a name or a list of names reads this series.

    Attributes
    ----------
    outputs : list[str]
        The outputs summed; one name is a list of one.
    """

    kind: Literal["level"] = "level"

    def __init__(
        self, outputs: str | list[str], *, kind: Literal["level"] = "level"
    ) -> None:
        BaseModel.__init__(self, outputs=outputs, kind=kind)
```

## Increments

Bases: `_Outputs`, `RangeChecked`

The increments of some outputs: at each end, the gain of their sum over `(end - window, end]`, as an observation with the window `window` reads it.

The positions of a quantity of increments are window ends, each at least `window` within the rounding of the window (`window_start` of `crates/commol-calibration/src/observation.rs`), so no window starts before 0.

Attributes:

| Name      | Type        | Description                                    |
| --------- | ----------- | ---------------------------------------------- |
| `outputs` | `list[str]` | The outputs summed; one name is a list of one. |
| `window`  | `float`     | The width of each window, above 0.             |

Source code in `commol/quantities/quantities.py`

```python
class Increments(_Outputs, RangeChecked):
    """
    The increments of some outputs: at each end, the gain of their sum over
    ``(end - window, end]``, as an observation with the window `window`
    reads it.

    The positions of a quantity of increments are window ends, each at
    least `window` within the rounding of the window (`window_start` of
    `crates/commol-calibration/src/observation.rs`), so no window starts
    before 0.

    Attributes
    ----------
    outputs : list[str]
        The outputs summed; one name is a list of one.
    window : float
        The width of each window, above 0.
    """

    kind: Literal["increments"] = "increments"
    window: float = Field(description="The width.")

    def __init__(
        self,
        outputs: str | list[str],
        window: float,
        *,
        kind: Literal["increments"] = "increments",
    ) -> None:
        BaseModel.__init__(self, outputs=outputs, window=window, kind=kind)
```

## ValueAt

Bases: `_OfSeries`, `RangeChecked`

The value of a series at a position: the level of the outputs, or their gain over the window that ends there.

Attributes:

| Name     | Type    | Description                                           |
| -------- | ------- | ----------------------------------------------------- |
| `series` | \`Level | Increments\`                                          |
| `at`     | `float` | The position; with Increments, the end of the window. |
| `label`  | \`str   | None\`                                                |

Source code in `commol/quantities/quantities.py`

```python
class ValueAt(_OfSeries, RangeChecked):
    """
    The value of a series at a position: the level of the outputs, or their
    gain over the window that ends there.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    at : float
        The position; with `Increments`, the end of the window.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["value_at"] = "value_at"
    at: float = Field(description="The position.")
```

## Gain

Bases: `_Outputs`, `QuantityBase`, `RangeChecked`

The value at `end` minus the value at `start`: what entered a cumulative output over `(start, end]`, as an observation with the window `end - start` that starts at exactly `start` reads it.

Attributes:

| Name      | Type        | Description                                    |
| --------- | ----------- | ---------------------------------------------- |
| `outputs` | `list[str]` | The outputs summed; one name is a list of one. |
| `start`   | `float`     | The start of the window.                       |
| `end`     | `float`     | The end of the window, after start.            |
| `label`   | \`str       | None\`                                         |

Source code in `commol/quantities/quantities.py`

```python
class Gain(_Outputs, QuantityBase, RangeChecked):
    """
    The value at `end` minus the value at `start`: what entered a cumulative
    output over ``(start, end]``, as an observation with the window
    ``end - start`` that starts at exactly `start` reads it.

    Attributes
    ----------
    outputs : list[str]
        The outputs summed; one name is a list of one.
    start : float
        The start of the window.
    end : float
        The end of the window, after `start`.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["gain"] = "gain"
    start: float = Field(description="The start of the window.")
    end: float = Field(description="The end of the window.")
```

## Peak

Bases: `_AtPositions`, `RangeChecked`

The largest value of a series at the positions.

The series decides what the peak is: the peak of `Increments(acc, window=7)` at the ends 7, 14, ... is the largest gain of a window of 7, while the peak of the level of an accumulator, which never falls, is its value at the last position. A NaN at a position makes the peak NaN.

Attributes:

| Name        | Type          | Description                                                      |
| ----------- | ------------- | ---------------------------------------------------------------- |
| `series`    | \`Level       | Increments\`                                                     |
| `positions` | `list[float]` | The positions compared, ascending; with Increments, window ends. |
| `label`     | \`str         | None\`                                                           |

Source code in `commol/quantities/quantities.py`

```python
class Peak(_AtPositions, RangeChecked):
    """
    The largest value of a series at the positions.

    The series decides what the peak is: the peak of `Increments(acc,
    window=7)` at the ends 7, 14, ... is the largest gain of a window of 7,
    while the peak of the level of an accumulator, which never falls, is
    its value at the last position. A NaN at a position makes the peak NaN.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    positions : list[float]
        The positions compared, ascending; with `Increments`, window ends.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["peak"] = "peak"
```

## PeakPosition

Bases: `_AtPositions`, `RangeChecked`

The first of the positions where the value of a series is largest; NaN when the value at a position is NaN.

Attributes:

| Name        | Type          | Description                                                      |
| ----------- | ------------- | ---------------------------------------------------------------- |
| `series`    | \`Level       | Increments\`                                                     |
| `positions` | `list[float]` | The positions compared, ascending; with Increments, window ends. |
| `label`     | \`str         | None\`                                                           |

Source code in `commol/quantities/quantities.py`

```python
class PeakPosition(_AtPositions, RangeChecked):
    """
    The first of the positions where the value of a series is largest; NaN
    when the value at a position is NaN.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    positions : list[float]
        The positions compared, ascending; with `Increments`, window ends.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["peak_position"] = "peak_position"
```

## Trajectory

Bases: `_AtPositions`, `RangeChecked`

The value of a series at each position: one scalar per position, for an analysis over the axis.

Attributes:

| Name        | Type          | Description                                             |
| ----------- | ------------- | ------------------------------------------------------- |
| `series`    | \`Level       | Increments\`                                            |
| `positions` | `list[float]` | The positions, ascending; with Increments, window ends. |
| `label`     | \`str         | None\`                                                  |

Source code in `commol/quantities/quantities.py`

```python
class Trajectory(_AtPositions, RangeChecked):
    """
    The value of a series at each position: one scalar per position, for an
    analysis over the axis.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    positions : list[float]
        The positions, ascending; with `Increments`, window ends.
    label : str | None
        The name of the series; each scalar is named after it and its
        position.
    """

    kind: Literal["trajectory"] = "trajectory"
```

## Ratio

Bases: `QuantityBase`

The ratio of two quantities of one value each, numerator N over denominator D.

The ratio of a run whose denominator is 0 is NaN, also when the numerator is not 0, and follows the NaN rule of every method. A method that reduces the points of a design (`PRCC`, `Morris`, `Sobol`, `PolynomialChaos`) reads one value per point; with a stochastic engine, `replicate_rule` says how the replicates of a point combine. A `Propagation`, the `Scenarios` and the quantities of one run keep the ratio of each run.

Attributes:

| Name             | Type                                | Description                                                                                                                                                                                     |
| ---------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `numerator`      | \`ValueAt                           | Gain                                                                                                                                                                                            |
| `denominator`    | \`ValueAt                           | Gain                                                                                                                                                                                            |
| `replicate_rule` | `{within, within_positive, pooled}` | "within": the mean of the ratio of each replicate, E[N/D]; a replicate with D = 0 leaves the point with no value. "within_positive": the mean of the ratio of the replicates with D > 0, E\[N/D |
| `label`          | \`str                               | None\`                                                                                                                                                                                          |

Source code in `commol/quantities/quantities.py`

```python
class Ratio(QuantityBase):
    """
    The ratio of two quantities of one value each, numerator N over
    denominator D.

    The ratio of a run whose denominator is 0 is NaN, also when the
    numerator is not 0, and follows the NaN rule of every method. A method
    that reduces the points of a design (`PRCC`, `Morris`, `Sobol`,
    `PolynomialChaos`) reads one value per point; with a stochastic engine,
    `replicate_rule` says how the replicates of a point combine. A `Propagation`,
    the `Scenarios` and the quantities of one run keep the ratio of each run.

    Attributes
    ----------
    numerator : ValueAt | Gain | Peak
        The quantity divided.
    denominator : ValueAt | Gain | Peak
        The quantity it is divided by.
    replicate_rule : {"within", "within_positive", "pooled"}
        ``"within"``: the mean of the ratio of each replicate, E[N/D]; a
        replicate with D = 0 leaves the point with no value.
        ``"within_positive"``: the mean of the ratio of the replicates with
        D > 0, E[N/D | D > 0]; a point with no such replicate has no value.
        ``"pooled"``: the ratio of the means, E[N]/E[D]; a point with
        E[D] = 0 has no value. With a deterministic engine, the three give
        N/D, and a point with D = 0 has no value.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["ratio"] = "ratio"
    numerator: _Operand = Field(description="The quantity divided.")
    denominator: _Operand = Field(description="The quantity it is divided by.")
    replicate_rule: Literal["within", "within_positive", "pooled"] = Field(
        description="How the replicates of a stochastic run combine."
    )
```

## quantity_schedule

```python
quantity_schedule(quantities: Sequence[QuantityBase]) -> list[float]
```

The positions a run must report at for `quantities` to read it.

They are every position a quantity reads and the start of every window, sorted, each once. A `SimulationResults` of `Simulation.run_at(quantity_schedule(quantities))` holds them all, so `SimulationResults.quantities` reads it; a `Propagation` runs at the same positions.

Parameters:

| Name         | Type                 | Description     | Default    |
| ------------ | -------------------- | --------------- | ---------- |
| `quantities` | `Sequence[quantity]` | The quantities. | *required* |

Returns:

| Type          | Description               |
| ------------- | ------------------------- |
| `list[float]` | The positions, ascending. |

Raises:

| Type         | Description              |
| ------------ | ------------------------ |
| `ValueError` | If there is no quantity. |

Source code in `commol/quantities/quantities.py`

```python
def quantity_schedule(quantities: Sequence[QuantityBase]) -> list[float]:
    """
    The positions a run must report at for `quantities` to read it.

    They are every position a quantity reads and the start of every window,
    sorted, each once. A `SimulationResults` of
    ``Simulation.run_at(quantity_schedule(quantities))`` holds them all, so
    `SimulationResults.quantities` reads it; a `Propagation` runs at the same
    positions.

    Parameters
    ----------
    quantities : Sequence[quantity]
        The quantities.

    Returns
    -------
    list[float]
        The positions, ascending.

    Raises
    ------
    ValueError
        If there is no quantity.
    """
    return commol_rs.inference.quantity_schedule(engine_json(list(quantities)))
```
