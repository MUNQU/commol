# Simulation API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Runs a model and reads its outputs. Guide: [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/index.md).

## Simulation

A Facade for running a simulation from a defined Model.

The engine follows the model: `run` and `engine` rebuild it when `model_definition` no longer serializes to the JSON the engine was built from, for example after `Model.update_parameters`.

Source code in `commol/api/simulation.py`

```python
class Simulation:
    """
    A Facade for running a simulation from a defined Model.

    The engine follows the model: ``run`` and ``engine`` rebuild it when
    ``model_definition`` no longer serializes to the JSON the engine was built
    from, for example after ``Model.update_parameters``.
    """

    def __init__(self, model: Model):
        """
        Initializes the simulation engine from a Pydantic Model definition.

        Parameters
        ----------
        model : Model
            A fully constructed and validated model object, of any kind.
            None values for parameters/initial conditions if used for calibration.

        Raises
        ------
        TypeError
            If the structure of the model is not a `ModelStructure`.

        """
        if not isinstance(model.structure, ModelStructure):
            raise TypeError(
                f"The structure of model '{model.name}' is a "
                f"{type(model.structure).__name__}, not a ModelStructure. A model "
                "holds the structure of its kind, such as a CompartmentalStructure."
            )
        logging.info(f"Initializing Simulation with model: '{model.name}'")
        self.model_definition: Model = model
        self._build_engine(model.model_dump_json())

    def _build_engine(self, model_json: str) -> None:
        """Build the engine and its output names from a model JSON."""
        self._model_json: str = model_json
        self._engine: "EngineProtocol" = build_engine(model_json)
        self._simulation_outputs: list[str] = self._engine.output_names
        logging.info(
            f"Simulation engine ready. Outputs: {len(self._simulation_outputs)}"
        )

    def _sync_engine(self) -> None:
        """Rebuild the engine when the model changed since it was built."""
        model_json = self.model_definition.model_dump_json()
        if model_json != self._model_json:
            logging.info("Model changed since the engine was built; rebuilding it.")
            self._build_engine(model_json)

    def _validate_all_parameters_calibrated(self, model: Model) -> None:
        """
        Validates that every parameter, and every value of the model kind, has
        a value (is calibrated).

        Parameters
        ----------
        model : Model
            The model to validate.

        Raises
        ------
        ValueError
            If any parameter, or any value of the model kind such as an
            initial condition, has a None value.
        """
        uncalibrated_params = model.get_uncalibrated_parameters()
        uncalibrated_ics = model.structure.uncalibrated_values()

        errors = []
        if uncalibrated_params:
            errors.append(
                f"Parameters requiring calibration: {', '.join(uncalibrated_params)}"
            )
        if uncalibrated_ics:
            errors.append(
                f"Initial conditions requiring calibration: "
                f"{', '.join(uncalibrated_ics)}"
            )

        if errors:
            raise ValueError(
                f"Cannot run Simulation: {'; '.join(errors)}. "
                f"Please calibrate these values before running a simulation."
            )

    def run(self, num_steps: int) -> SimulationResults:
        """
        Runs the simulation from the initial state on its default grid.

        Every call starts at position 0 with the current ``model_definition``,
        so two calls with the same model give the same result. To continue a
        run, call ``simulation.engine.step()``.

        Parameters
        ----------
        num_steps : int
            The number of steps: the results hold the positions
            ``0, 1, ..., num_steps``.

        Returns
        -------
        SimulationResults
            One value per position for each output, and the positions.

        Raises
        ------
        InfeasibleStateError
            If a step leaves an amount below zero; the message names the
            compartment, the step and the flows out of it.
        ValueError
            If a parameter or initial condition has no value.
        """
        self._validate_all_parameters_calibrated(self.model_definition)
        self._sync_engine()
        self._engine.reset()
        logging.info(f"Running simulation for {num_steps} steps.")
        start = time.time()
        positions, rows = self._engine.run(num_steps)
        logging.info(f"Simulation complete. It took {time.time() - start} seconds.")
        return SimulationResults.from_rows(positions, self._simulation_outputs, rows)

    def run_at(self, at: Sequence[float]) -> SimulationResults:
        """
        Runs the simulation from the initial state and reports at given positions.

        The positions are on the axis, the independent variable of the model.
        The engine decides which positions its model type reports at: a
        DifferentialEquations model at any finite, non-negative position, from
        the dense output of its solver; a DifferenceEquations model only at
        integer steps.

        Parameters
        ----------
        at : Sequence[float]
            Sorted, unique positions to report at.

        Returns
        -------
        SimulationResults
            One value per position for each output, and the positions.

        Raises
        ------
        InfeasibleStateError
            If a step leaves an amount below zero; the message names the
            compartment, the step and the flows out of it.
        ValueError
            If a parameter or initial condition has no value, the positions are
            not sorted and unique, or the engine does not report at one of them;
            the message is the engine's.
        """
        positions = [float(position) for position in at]
        self._validate_all_parameters_calibrated(self.model_definition)
        self._sync_engine()
        self._engine.reset()
        rows = self._engine.run_at(positions)
        return SimulationResults.from_rows(positions, self._simulation_outputs, rows)

    @property
    def engine(self) -> "EngineProtocol":
        """
        The engine for the current ``model_definition``.

        It is rebuilt, at step 0, when the model changed since it was built.
        Otherwise it is the same object, in the state the last ``run`` or
        ``step`` left it.
        """
        self._sync_engine()
        return self._engine

    @property
    def simulation_outputs(self) -> list[str]:
        """Names of the columns returned by simulation runs."""
        self._sync_engine()
        return self._simulation_outputs

    def outputs_for(self, id: str) -> list[str]:
        """
        Names of the outputs a source id expands into.

        The structure of the model groups its outputs by source
        (`ModelStructure.outputs_by_source`): for a compartmental model, by
        bin and accumulator.

        Parameters
        ----------
        id : str
            A source id, such as a bin id or an accumulator id.

        Returns
        -------
        list[str]
            Output names for that id, in the order the engine reports them.

        Raises
        ------
        KeyError
            If `id` is not a source of the model.
        """
        outputs = self.model_definition.structure.outputs_by_source()
        if id not in outputs:
            raise KeyError(
                f"'{id}' is not a bin or accumulator of model "
                f"'{self.model_definition.name}'. Available: {sorted(outputs)}"
            )
        return outputs[id]

    def group_outputs(self, ids: Iterable[str]) -> dict[str, list[str]]:
        """
        Group output names by the bin or accumulator they belong to.

        Parameters
        ----------
        ids : Iterable[str]
            Bin and accumulator ids.

        Returns
        -------
        dict[str, list[str]]
            Each id mapped to its output names, keeping the order given.

        Raises
        ------
        KeyError
            If an id is not a bin or accumulator of the model.
        """
        return {id: self.outputs_for(id) for id in ids}

    def total_series(
        self,
        results: Mapping[str, Sequence[float]],
        ids: Iterable[str],
    ) -> list[float]:
        """
        Sum the outputs of the given bins or accumulators at each position.

        Parameters
        ----------
        results : Mapping[str, Sequence[float]]
            Simulation results, such as those of `run` or `run_at`.
        ids : Iterable[str]
            Bin and accumulator ids whose outputs are summed together.

        Returns
        -------
        list[float]
            One value per position of `results`.

        Raises
        ------
        KeyError
            If an id is not a bin or accumulator of the model, or an expected
            output is missing from `results`.
        """
        names = [name for id in ids for name in self.outputs_for(id)]
        missing = [name for name in names if name not in results]
        if missing:
            raise KeyError(f"Missing output series in results: {sorted(missing)}")
        if not names:
            return []
        return [
            sum(values)
            for values in zip(*(results[name] for name in names), strict=True)
        ]

    def windowed_totals(
        self,
        results: SimulationResults,
        ids: Iterable[str],
        window: float,
        at: Iterable[float] | None = None,
    ) -> list[float]:
        """
        Amount the given accumulators gained over each window.

        Parameters
        ----------
        results : SimulationResults
            Simulation results, with their positions.
        ids : Iterable[str]
            Accumulator ids whose outputs are summed before windowing.
        window : float
            Length of one window on the axis.
        at : Iterable[float] | None, optional
            Positions at which windows close. Defaults to every complete
            window of the results (:func:`commol.window_ends`). Pass the
            observation positions to reproduce exactly the values a
            calibration used.

        Returns
        -------
        list[float]
            One increment per window.

        Raises
        ------
        KeyError
            If an id is not a bin or accumulator of the model, or an expected
            output is missing from `results`.
        ValueError
            If `window` is not positive, or the results have no value at the
            end or the start of a requested window.
        """
        return windowed_totals(
            self.total_series(results, ids), results.positions, window, at
        )
```

### Attributes

#### engine

```python
engine: EngineProtocol
```

The engine for the current `model_definition`.

It is rebuilt, at step 0, when the model changed since it was built. Otherwise it is the same object, in the state the last `run` or `step` left it.

#### simulation_outputs

```python
simulation_outputs: list[str]
```

Names of the columns returned by simulation runs.

### Methods:

#### __init__

```python
__init__(model: Model)
```

Initializes the simulation engine from a Pydantic Model definition.

Parameters:

| Name    | Type    | Description                                                                                                                         | Default    |
| ------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `model` | `Model` | A fully constructed and validated model object, of any kind. None values for parameters/initial conditions if used for calibration. | *required* |

Raises:

| Type        | Description                                            |
| ----------- | ------------------------------------------------------ |
| `TypeError` | If the structure of the model is not a ModelStructure. |

Source code in `commol/api/simulation.py`

```python
def __init__(self, model: Model):
    """
    Initializes the simulation engine from a Pydantic Model definition.

    Parameters
    ----------
    model : Model
        A fully constructed and validated model object, of any kind.
        None values for parameters/initial conditions if used for calibration.

    Raises
    ------
    TypeError
        If the structure of the model is not a `ModelStructure`.

    """
    if not isinstance(model.structure, ModelStructure):
        raise TypeError(
            f"The structure of model '{model.name}' is a "
            f"{type(model.structure).__name__}, not a ModelStructure. A model "
            "holds the structure of its kind, such as a CompartmentalStructure."
        )
    logging.info(f"Initializing Simulation with model: '{model.name}'")
    self.model_definition: Model = model
    self._build_engine(model.model_dump_json())
```

#### run

```python
run(num_steps: int) -> SimulationResults
```

Runs the simulation from the initial state on its default grid.

Every call starts at position 0 with the current `model_definition`, so two calls with the same model give the same result. To continue a run, call `simulation.engine.step()`.

Parameters:

| Name        | Type  | Description                                                               | Default    |
| ----------- | ----- | ------------------------------------------------------------------------- | ---------- |
| `num_steps` | `int` | The number of steps: the results hold the positions 0, 1, ..., num_steps. | *required* |

Returns:

| Type                | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `SimulationResults` | One value per position for each output, and the positions. |

Raises:

| Type                   | Description                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `InfeasibleStateError` | If a step leaves an amount below zero; the message names the compartment, the step and the flows out of it. |
| `ValueError`           | If a parameter or initial condition has no value.                                                           |

Source code in `commol/api/simulation.py`

```python
def run(self, num_steps: int) -> SimulationResults:
    """
    Runs the simulation from the initial state on its default grid.

    Every call starts at position 0 with the current ``model_definition``,
    so two calls with the same model give the same result. To continue a
    run, call ``simulation.engine.step()``.

    Parameters
    ----------
    num_steps : int
        The number of steps: the results hold the positions
        ``0, 1, ..., num_steps``.

    Returns
    -------
    SimulationResults
        One value per position for each output, and the positions.

    Raises
    ------
    InfeasibleStateError
        If a step leaves an amount below zero; the message names the
        compartment, the step and the flows out of it.
    ValueError
        If a parameter or initial condition has no value.
    """
    self._validate_all_parameters_calibrated(self.model_definition)
    self._sync_engine()
    self._engine.reset()
    logging.info(f"Running simulation for {num_steps} steps.")
    start = time.time()
    positions, rows = self._engine.run(num_steps)
    logging.info(f"Simulation complete. It took {time.time() - start} seconds.")
    return SimulationResults.from_rows(positions, self._simulation_outputs, rows)
```

#### run_at

```python
run_at(at: Sequence[float]) -> SimulationResults
```

Runs the simulation from the initial state and reports at given positions.

The positions are on the axis, the independent variable of the model. The engine decides which positions its model type reports at: a DifferentialEquations model at any finite, non-negative position, from the dense output of its solver; a DifferenceEquations model only at integer steps.

Parameters:

| Name | Type              | Description                            | Default    |
| ---- | ----------------- | -------------------------------------- | ---------- |
| `at` | `Sequence[float]` | Sorted, unique positions to report at. | *required* |

Returns:

| Type                | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `SimulationResults` | One value per position for each output, and the positions. |

Raises:

| Type                   | Description                                                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InfeasibleStateError` | If a step leaves an amount below zero; the message names the compartment, the step and the flows out of it.                                                           |
| `ValueError`           | If a parameter or initial condition has no value, the positions are not sorted and unique, or the engine does not report at one of them; the message is the engine's. |

Source code in `commol/api/simulation.py`

```python
def run_at(self, at: Sequence[float]) -> SimulationResults:
    """
    Runs the simulation from the initial state and reports at given positions.

    The positions are on the axis, the independent variable of the model.
    The engine decides which positions its model type reports at: a
    DifferentialEquations model at any finite, non-negative position, from
    the dense output of its solver; a DifferenceEquations model only at
    integer steps.

    Parameters
    ----------
    at : Sequence[float]
        Sorted, unique positions to report at.

    Returns
    -------
    SimulationResults
        One value per position for each output, and the positions.

    Raises
    ------
    InfeasibleStateError
        If a step leaves an amount below zero; the message names the
        compartment, the step and the flows out of it.
    ValueError
        If a parameter or initial condition has no value, the positions are
        not sorted and unique, or the engine does not report at one of them;
        the message is the engine's.
    """
    positions = [float(position) for position in at]
    self._validate_all_parameters_calibrated(self.model_definition)
    self._sync_engine()
    self._engine.reset()
    rows = self._engine.run_at(positions)
    return SimulationResults.from_rows(positions, self._simulation_outputs, rows)
```

#### outputs_for

```python
outputs_for(id: str) -> list[str]
```

Names of the outputs a source id expands into.

The structure of the model groups its outputs by source (`ModelStructure.outputs_by_source`): for a compartmental model, by bin and accumulator.

Parameters:

| Name | Type  | Description                                         | Default    |
| ---- | ----- | --------------------------------------------------- | ---------- |
| `id` | `str` | A source id, such as a bin id or an accumulator id. | *required* |

Returns:

| Type        | Description                                                     |
| ----------- | --------------------------------------------------------------- |
| `list[str]` | Output names for that id, in the order the engine reports them. |

Raises:

| Type       | Description                         |
| ---------- | ----------------------------------- |
| `KeyError` | If id is not a source of the model. |

Source code in `commol/api/simulation.py`

```python
def outputs_for(self, id: str) -> list[str]:
    """
    Names of the outputs a source id expands into.

    The structure of the model groups its outputs by source
    (`ModelStructure.outputs_by_source`): for a compartmental model, by
    bin and accumulator.

    Parameters
    ----------
    id : str
        A source id, such as a bin id or an accumulator id.

    Returns
    -------
    list[str]
        Output names for that id, in the order the engine reports them.

    Raises
    ------
    KeyError
        If `id` is not a source of the model.
    """
    outputs = self.model_definition.structure.outputs_by_source()
    if id not in outputs:
        raise KeyError(
            f"'{id}' is not a bin or accumulator of model "
            f"'{self.model_definition.name}'. Available: {sorted(outputs)}"
        )
    return outputs[id]
```

#### group_outputs

```python
group_outputs(ids: Iterable[str]) -> dict[str, list[str]]
```

Group output names by the bin or accumulator they belong to.

Parameters:

| Name  | Type            | Description              | Default    |
| ----- | --------------- | ------------------------ | ---------- |
| `ids` | `Iterable[str]` | Bin and accumulator ids. | *required* |

Returns:

| Type                   | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `dict[str, list[str]]` | Each id mapped to its output names, keeping the order given. |

Raises:

| Type       | Description                                        |
| ---------- | -------------------------------------------------- |
| `KeyError` | If an id is not a bin or accumulator of the model. |

Source code in `commol/api/simulation.py`

```python
def group_outputs(self, ids: Iterable[str]) -> dict[str, list[str]]:
    """
    Group output names by the bin or accumulator they belong to.

    Parameters
    ----------
    ids : Iterable[str]
        Bin and accumulator ids.

    Returns
    -------
    dict[str, list[str]]
        Each id mapped to its output names, keeping the order given.

    Raises
    ------
    KeyError
        If an id is not a bin or accumulator of the model.
    """
    return {id: self.outputs_for(id) for id in ids}
```

#### total_series

```python
total_series(results: Mapping[str, Sequence[float]], ids: Iterable[str]) -> list[float]
```

Sum the outputs of the given bins or accumulators at each position.

Parameters:

| Name      | Type                            | Description                                                | Default    |
| --------- | ------------------------------- | ---------------------------------------------------------- | ---------- |
| `results` | `Mapping[str, Sequence[float]]` | Simulation results, such as those of run or run_at.        | *required* |
| `ids`     | `Iterable[str]`                 | Bin and accumulator ids whose outputs are summed together. | *required* |

Returns:

| Type          | Description                        |
| ------------- | ---------------------------------- |
| `list[float]` | One value per position of results. |

Raises:

| Type       | Description                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------------- |
| `KeyError` | If an id is not a bin or accumulator of the model, or an expected output is missing from results. |

Source code in `commol/api/simulation.py`

```python
def total_series(
    self,
    results: Mapping[str, Sequence[float]],
    ids: Iterable[str],
) -> list[float]:
    """
    Sum the outputs of the given bins or accumulators at each position.

    Parameters
    ----------
    results : Mapping[str, Sequence[float]]
        Simulation results, such as those of `run` or `run_at`.
    ids : Iterable[str]
        Bin and accumulator ids whose outputs are summed together.

    Returns
    -------
    list[float]
        One value per position of `results`.

    Raises
    ------
    KeyError
        If an id is not a bin or accumulator of the model, or an expected
        output is missing from `results`.
    """
    names = [name for id in ids for name in self.outputs_for(id)]
    missing = [name for name in names if name not in results]
    if missing:
        raise KeyError(f"Missing output series in results: {sorted(missing)}")
    if not names:
        return []
    return [
        sum(values)
        for values in zip(*(results[name] for name in names), strict=True)
    ]
```

#### windowed_totals

```python
windowed_totals(results: SimulationResults, ids: Iterable[str], window: float, at: Iterable[float] | None = None) -> list[float]
```

Amount the given accumulators gained over each window.

Parameters:

| Name      | Type                | Description                                                | Default                                                                                                                                                                                           |
| --------- | ------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results` | `SimulationResults` | Simulation results, with their positions.                  | *required*                                                                                                                                                                                        |
| `ids`     | `Iterable[str]`     | Accumulator ids whose outputs are summed before windowing. | *required*                                                                                                                                                                                        |
| `window`  | `float`             | Length of one window on the axis.                          | *required*                                                                                                                                                                                        |
| `at`      | \`Iterable[float]   | None\`                                                     | Positions at which windows close. Defaults to every complete window of the results (:func:commol.window_ends). Pass the observation positions to reproduce exactly the values a calibration used. |

Returns:

| Type          | Description               |
| ------------- | ------------------------- |
| `list[float]` | One increment per window. |

Raises:

| Type         | Description                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| `KeyError`   | If an id is not a bin or accumulator of the model, or an expected output is missing from results.      |
| `ValueError` | If window is not positive, or the results have no value at the end or the start of a requested window. |

Source code in `commol/api/simulation.py`

```python
def windowed_totals(
    self,
    results: SimulationResults,
    ids: Iterable[str],
    window: float,
    at: Iterable[float] | None = None,
) -> list[float]:
    """
    Amount the given accumulators gained over each window.

    Parameters
    ----------
    results : SimulationResults
        Simulation results, with their positions.
    ids : Iterable[str]
        Accumulator ids whose outputs are summed before windowing.
    window : float
        Length of one window on the axis.
    at : Iterable[float] | None, optional
        Positions at which windows close. Defaults to every complete
        window of the results (:func:`commol.window_ends`). Pass the
        observation positions to reproduce exactly the values a
        calibration used.

    Returns
    -------
    list[float]
        One increment per window.

    Raises
    ------
    KeyError
        If an id is not a bin or accumulator of the model, or an expected
        output is missing from `results`.
    ValueError
        If `window` is not positive, or the results have no value at the
        end or the start of a requested window.
    """
    return windowed_totals(
        self.total_series(results, ids), results.positions, window, at
    )
```

## SimulationResults

Bases: `dict[str, list[float]]`

Values of every output at each position on the axis.

A `dict` from output name to one value per position, so `results["A"]`, `json.dumps(results)` and `pandas.DataFrame(results)` work as for any mapping. The positions the values were read at are in :attr:`positions`.

Attributes:

| Name        | Type          | Description                                                                              |
| ----------- | ------------- | ---------------------------------------------------------------------------------------- |
| `positions` | `list[float]` | Positions on the axis, ascending. results[name][i] is the value of name at positions[i]. |

Source code in `commol/api/results.py`

```python
class SimulationResults(dict[str, list[float]]):
    """
    Values of every output at each position on the axis.

    A ``dict`` from output name to one value per position, so
    ``results["A"]``, ``json.dumps(results)`` and
    ``pandas.DataFrame(results)`` work as for any mapping. The positions the
    values were read at are in :attr:`positions`.

    Attributes
    ----------
    positions : list[float]
        Positions on the axis, ascending. ``results[name][i]`` is the value of
        ``name`` at ``positions[i]``.
    """

    positions: list[float]

    def __init__(
        self,
        positions: Iterable[float],
        values: Mapping[str, Sequence[float]] | None = None,
    ):
        """
        Results at the given positions.

        Parameters
        ----------
        positions : Iterable[float]
            Positions on the axis, ascending.
        values : Mapping[str, Sequence[float]] | None, optional
            One value per position for each output.

        Raises
        ------
        ValueError
            If an output does not have one value per position.
        """
        super().__init__(
            {name: list(series) for name, series in (values or {}).items()}
        )
        self.positions = [float(position) for position in positions]
        for name, series in self.items():
            if len(series) != len(self.positions):
                raise ValueError(
                    f"Output '{name}' has {len(series)} values for "
                    f"{len(self.positions)} positions."
                )

    @classmethod
    def from_rows(
        cls,
        positions: Sequence[float],
        names: Sequence[str],
        rows: Sequence[Sequence[float]],
    ) -> "SimulationResults":
        """
        Results from one row of output values per position.

        Parameters
        ----------
        positions : Sequence[float]
            Positions on the axis, ascending.
        names : Sequence[str]
            Output names, in the order of each row.
        rows : Sequence[Sequence[float]]
            One row per position.

        Returns
        -------
        SimulationResults
            One series per output name.
        """
        columns = list(zip(*rows)) if rows else [() for _ in names]
        return cls(positions, dict(zip(names, columns, strict=True)))

    def rows(self) -> list[list[float]]:
        """
        Every output at each position, one row per position.

        Returns
        -------
        list[list[float]]
            ``rows()[i][j]`` is the value of the ``j``-th output, in key order
            (the order of ``Simulation.simulation_outputs``), at
            ``positions[i]``.
        """
        return [list(row) for row in zip(*self.values())] if self else []

    def __repr__(self) -> str:
        """The positions and the values."""
        return f"SimulationResults(positions={self.positions!r}, {dict.__repr__(self)})"
```

### Methods:

#### __init__

```python
__init__(positions: Iterable[float], values: Mapping[str, Sequence[float]] | None = None)
```

Results at the given positions.

Parameters:

| Name        | Type                              | Description                       | Default                                 |
| ----------- | --------------------------------- | --------------------------------- | --------------------------------------- |
| `positions` | `Iterable[float]`                 | Positions on the axis, ascending. | *required*                              |
| `values`    | \`Mapping\[str, Sequence[float]\] | None\`                            | One value per position for each output. |

Raises:

| Type         | Description                                        |
| ------------ | -------------------------------------------------- |
| `ValueError` | If an output does not have one value per position. |

Source code in `commol/api/results.py`

```python
def __init__(
    self,
    positions: Iterable[float],
    values: Mapping[str, Sequence[float]] | None = None,
):
    """
    Results at the given positions.

    Parameters
    ----------
    positions : Iterable[float]
        Positions on the axis, ascending.
    values : Mapping[str, Sequence[float]] | None, optional
        One value per position for each output.

    Raises
    ------
    ValueError
        If an output does not have one value per position.
    """
    super().__init__(
        {name: list(series) for name, series in (values or {}).items()}
    )
    self.positions = [float(position) for position in positions]
    for name, series in self.items():
        if len(series) != len(self.positions):
            raise ValueError(
                f"Output '{name}' has {len(series)} values for "
                f"{len(self.positions)} positions."
            )
```

#### from_rows

```python
from_rows(positions: Sequence[float], names: Sequence[str], rows: Sequence[Sequence[float]]) -> SimulationResults
```

Results from one row of output values per position.

Parameters:

| Name        | Type                        | Description                             | Default    |
| ----------- | --------------------------- | --------------------------------------- | ---------- |
| `positions` | `Sequence[float]`           | Positions on the axis, ascending.       | *required* |
| `names`     | `Sequence[str]`             | Output names, in the order of each row. | *required* |
| `rows`      | `Sequence[Sequence[float]]` | One row per position.                   | *required* |

Returns:

| Type                | Description                 |
| ------------------- | --------------------------- |
| `SimulationResults` | One series per output name. |

Source code in `commol/api/results.py`

```python
@classmethod
def from_rows(
    cls,
    positions: Sequence[float],
    names: Sequence[str],
    rows: Sequence[Sequence[float]],
) -> "SimulationResults":
    """
    Results from one row of output values per position.

    Parameters
    ----------
    positions : Sequence[float]
        Positions on the axis, ascending.
    names : Sequence[str]
        Output names, in the order of each row.
    rows : Sequence[Sequence[float]]
        One row per position.

    Returns
    -------
    SimulationResults
        One series per output name.
    """
    columns = list(zip(*rows)) if rows else [() for _ in names]
    return cls(positions, dict(zip(names, columns, strict=True)))
```

#### rows

```python
rows() -> list[list[float]]
```

Every output at each position, one row per position.

Returns:

| Type                | Description                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `list[list[float]]` | rows()[i][j] is the value of the j-th output, in key order (the order of Simulation.simulation_outputs), at positions[i]. |

Source code in `commol/api/results.py`

```python
def rows(self) -> list[list[float]]:
    """
    Every output at each position, one row per position.

    Returns
    -------
    list[list[float]]
        ``rows()[i][j]`` is the value of the ``j``-th output, in key order
        (the order of ``Simulation.simulation_outputs``), at
        ``positions[i]``.
    """
    return [list(row) for row in zip(*self.values())] if self else []
```

#### __repr__

```python
__repr__() -> str
```

The positions and the values.

Source code in `commol/api/results.py`

```python
def __repr__(self) -> str:
    """The positions and the values."""
    return f"SimulationResults(positions={self.positions!r}, {dict.__repr__(self)})"
```

## OdeSolver

Bases: `BaseModel`

Settings of the `DifferentialEquations` engine.

Attributes:

| Name     | Type                       | Description                                                                                                                      |
| -------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `method` | `Literal['tsit45', 'bdf']` | "tsit45": explicit Tsitouras 5(4) Runge-Kutta, for non-stiff models. "bdf": backward differentiation formulas, for stiff models. |
| `rtol`   | `float`                    | Relative tolerance of each state component.                                                                                      |
| `atol`   | `float`                    | Absolute tolerance of each state component, in the unit of the bins.                                                             |

Source code in `commol/context/engine.py`

```python
class OdeSolver(BaseModel):
    """
    Settings of the ``DifferentialEquations`` engine.

    Attributes
    ----------
    method : Literal["tsit45", "bdf"]
        ``"tsit45"``: explicit Tsitouras 5(4) Runge-Kutta, for non-stiff
        models. ``"bdf"``: backward differentiation formulas, for stiff models.
    rtol : float
        Relative tolerance of each state component.
    atol : float
        Absolute tolerance of each state component, in the unit of the bins.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["tsit45", "bdf"] = Field(
        default="tsit45", description="Integration method."
    )
    rtol: float = Field(default=1e-6, gt=0.0, description="Relative tolerance.")
    atol: float = Field(default=1e-6, gt=0.0, description="Absolute tolerance.")
```

## InfeasibleStateError

```python
InfeasibleStateError = _engine.InfeasibleStateError
```

A run left the values its model allows, such as an amount below zero.

A difference-equations step that leaves a compartment below zero raises it. It is a `ValueError`; a calibration treats such a run as infeasible.

## Windows

## window_start

```python
window_start(index: int, window: float) -> float
```

Position at which a window opens.

Parameters:

| Name     | Type    | Description                       | Default    |
| -------- | ------- | --------------------------------- | ---------- |
| `index`  | `int`   | Zero-based window index.          | *required* |
| `window` | `float` | Length of one window on the axis. | *required* |

Returns:

| Type    | Description     |
| ------- | --------------- |
| `float` | index * window. |

Raises:

| Type         | Description                                     |
| ------------ | ----------------------------------------------- |
| `ValueError` | If window is not positive or index is negative. |

Source code in `commol/api/windows.py`

```python
def window_start(index: int, window: float) -> float:
    """
    Position at which a window opens.

    Parameters
    ----------
    index : int
        Zero-based window index.
    window : float
        Length of one window on the axis.

    Returns
    -------
    float
        ``index * window``.

    Raises
    ------
    ValueError
        If `window` is not positive or `index` is negative.
    """
    _check_window(window)
    if index < 0:
        raise ValueError(f"index must not be negative, got {index}.")
    return index * window
```

## window_end

```python
window_end(index: int, window: float) -> float
```

Position at which a window closes, equal to the start of the next.

Parameters:

| Name     | Type    | Description                       | Default    |
| -------- | ------- | --------------------------------- | ---------- |
| `index`  | `int`   | Zero-based window index.          | *required* |
| `window` | `float` | Length of one window on the axis. | *required* |

Returns:

| Type    | Description           |
| ------- | --------------------- |
| `float` | (index + 1) * window. |

Raises:

| Type         | Description                                     |
| ------------ | ----------------------------------------------- |
| `ValueError` | If window is not positive or index is negative. |

Source code in `commol/api/windows.py`

```python
def window_end(index: int, window: float) -> float:
    """
    Position at which a window closes, equal to the start of the next.

    Parameters
    ----------
    index : int
        Zero-based window index.
    window : float
        Length of one window on the axis.

    Returns
    -------
    float
        ``(index + 1) * window``.

    Raises
    ------
    ValueError
        If `window` is not positive or `index` is negative.
    """
    return window_start(index + 1, window)
```

## window_index

```python
window_index(at: float, window: float) -> int
```

Window a position belongs to.

The exact inverse of :func:`window_end`, and the containing window for any position in between: `max(ceil(at / window) - 1, 0)`. Position 0 belongs to window 0. :func:`commol.api.reporting.observed_report` places each observation with this rule.

Parameters:

| Name     | Type    | Description                       | Default    |
| -------- | ------- | --------------------------------- | ---------- |
| `at`     | `float` | A position on the axis.           | *required* |
| `window` | `float` | Length of one window on the axis. | *required* |

Returns:

| Type  | Description              |
| ----- | ------------------------ |
| `int` | Zero-based window index. |

Raises:

| Type         | Description                                  |
| ------------ | -------------------------------------------- |
| `ValueError` | If window is not positive or at is negative. |

Source code in `commol/api/windows.py`

```python
def window_index(at: float, window: float) -> int:
    """
    Window a position belongs to.

    The exact inverse of :func:`window_end`, and the containing window for
    any position in between: ``max(ceil(at / window) - 1, 0)``. Position 0
    belongs to window 0. :func:`commol.api.reporting.observed_report` places
    each observation with this rule.

    Parameters
    ----------
    at : float
        A position on the axis.
    window : float
        Length of one window on the axis.

    Returns
    -------
    int
        Zero-based window index.

    Raises
    ------
    ValueError
        If `window` is not positive or `at` is negative.
    """
    _check_window(window)
    if not at >= 0:
        raise ValueError(f"at must not be negative, got {at}.")
    return max(math.ceil(_windows_in(at, window)) - 1, 0)
```

## window_ends

```python
window_ends(window: float, end: float) -> list[float]
```

Positions closing each complete window up to a position.

Parameters:

| Name     | Type    | Description                                                | Default    |
| -------- | ------- | ---------------------------------------------------------- | ---------- |
| `window` | `float` | Length of one window on the axis.                          | *required* |
| `end`    | `float` | Last position covered, such as the last position of a run. | *required* |

Returns:

| Type          | Description                                                                   |
| ------------- | ----------------------------------------------------------------------------- |
| `list[float]` | window, 2 * window, ... up to end. A trailing partial window is not included. |

Raises:

| Type         | Description                |
| ------------ | -------------------------- |
| `ValueError` | If window is not positive. |

Source code in `commol/api/windows.py`

```python
def window_ends(window: float, end: float) -> list[float]:
    """
    Positions closing each complete window up to a position.

    Parameters
    ----------
    window : float
        Length of one window on the axis.
    end : float
        Last position covered, such as the last position of a run.

    Returns
    -------
    list[float]
        ``window``, ``2 * window``, ... up to `end`. A trailing partial window
        is not included.

    Raises
    ------
    ValueError
        If `window` is not positive.
    """
    _check_window(window)
    count = math.floor(_windows_in(end, window)) if end > 0 else 0
    return [window_end(index, window) for index in range(count)]
```

## windowed_totals

```python
windowed_totals(series: Sequence[float], positions: Sequence[float], window: float, at: Iterable[float] | None = None) -> list[float]
```

Amount a cumulative series gained over each window.

Each value is the series at the end of the window minus the series at its start, `at - window`: the same quantity a calibration compares an observation against when that observation sets `window`.

Parameters:

| Name        | Type              | Description                                                              | Default                                                                                                                                                                                   |
| ----------- | ----------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `series`    | `Sequence[float]` | A cumulative series, one value per position.                             | *required*                                                                                                                                                                                |
| `positions` | `Sequence[float]` | Positions of the series, ascending, such as SimulationResults.positions. | *required*                                                                                                                                                                                |
| `window`    | `float`           | Length of one window on the axis.                                        | *required*                                                                                                                                                                                |
| `at`        | \`Iterable[float] | None\`                                                                   | Positions at which windows close. Defaults to every complete window of the series (:func:window_ends). Pass the observation positions to reproduce exactly the values a calibration used. |

Returns:

| Type          | Description                           |
| ------------- | ------------------------------------- |
| `list[float]` | One increment per requested position. |

Raises:

| Type         | Description                                                                                                                             |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If window is not positive, a window starts before position 0, or the series has no value at the end or the start of a requested window. |

Source code in `commol/api/windows.py`

```python
def windowed_totals(
    series: Sequence[float],
    positions: Sequence[float],
    window: float,
    at: Iterable[float] | None = None,
) -> list[float]:
    """
    Amount a cumulative series gained over each window.

    Each value is the series at the end of the window minus the series at its
    start, ``at - window``: the same quantity a calibration compares an
    observation against when that observation sets ``window``.

    Parameters
    ----------
    series : Sequence[float]
        A cumulative series, one value per position.
    positions : Sequence[float]
        Positions of the series, ascending, such as
        ``SimulationResults.positions``.
    window : float
        Length of one window on the axis.
    at : Iterable[float] | None, optional
        Positions at which windows close. Defaults to every complete window of
        the series (:func:`window_ends`). Pass the observation positions to
        reproduce exactly the values a calibration used.

    Returns
    -------
    list[float]
        One increment per requested position.

    Raises
    ------
    ValueError
        If `window` is not positive, a window starts before position 0, or the
        series has no value at the end or the start of a requested window.
    """
    _check_window(window)
    ends = (
        window_ends(window, positions[-1] if positions else 0.0)
        if at is None
        else list(at)
    )

    totals = []
    for end in ends:
        start = end - window
        if start < -POSITION_TOLERANCE * max(abs(end), 1.0):
            raise ValueError(
                f"Position {end} has no complete window of {window} before it."
            )
        totals.append(
            series[index_at(positions, end)]
            - series[index_at(positions, max(start, 0.0))]
        )
    return totals
```
