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
                "holds the structure of its kind, such as an Aggregate."
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

    @property
    def is_stochastic(self) -> bool:
        """Whether the engine of the model draws random numbers."""
        return commol_rs.engine.EngineType(
            self.model_definition.engine.type
        ).is_stochastic

    def _seed_run(self, seed: int | None) -> int | None:
        """Seed the engine of a stochastic model; None for a deterministic one."""
        if not self.is_stochastic:
            return None
        seed = draw_seed() if seed is None else check_seed(seed)
        self._engine.set_seed(seed)
        return seed

    def run(self, num_steps: int, *, seed: int | None = None) -> SimulationResults:
        """
        Runs the simulation from the initial state on its default grid.

        Every call starts at position 0 with the current ``model_definition``,
        so two calls with the same model, and the same seed for a stochastic
        engine, give the same result. To continue a run, call
        ``simulation.engine.step()``.

        Parameters
        ----------
        num_steps : int
            The number of steps: the results hold the positions
            ``0, 1, ..., num_steps``.
        seed : int | None, optional
            The seed of a stochastic engine, a whole number from 0 to
            ``2**64 - 1``. None draws one, which ``results.seed`` keeps, so
            every run can be repeated. A deterministic engine ignores it.

        Returns
        -------
        SimulationResults
            One value per position for each output, the positions, and the
            seed of a stochastic run.

        Raises
        ------
        InfeasibleStateError
            If a step leaves an amount below zero; the message names the
            compartment, the step and the flows out of it.
        ValueError
            If a parameter or initial condition has no value, or the seed is
            not a whole number from 0 to ``2**64 - 1``.
        """
        self._validate_all_parameters_calibrated(self.model_definition)
        self._sync_engine()
        used_seed = self._seed_run(seed)
        self._engine.reset()
        logging.info(f"Running simulation for {num_steps} steps.")
        start = time.time()
        positions, rows = self._engine.run(num_steps)
        logging.info(f"Simulation complete. It took {time.time() - start} seconds.")
        return SimulationResults.from_rows(
            positions, self._simulation_outputs, rows, seed=used_seed
        )

    def run_at(
        self, at: Sequence[float], *, seed: int | None = None
    ) -> SimulationResults:
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
        seed : int | None, optional
            The seed of a stochastic engine; see `run`.

        Returns
        -------
        SimulationResults
            One value per position for each output, the positions, and the
            seed of a stochastic run.

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
        used_seed = self._seed_run(seed)
        self._engine.reset()
        rows = self._engine.run_at(positions)
        return SimulationResults.from_rows(
            positions, self._simulation_outputs, rows, seed=used_seed
        )

    def replicates(self, n: int, *, seed: int | None = None) -> "ReplicateRuns":
        """
        Replicates of the runs of a stochastic engine.

        The returned object has ``run`` and ``run_at``, as `Simulation`, and
        they return `SimulationReplicates`: ``n`` runs, each with its own seed,
        run in parallel in the engine. Replicate ``i`` takes the seed derived
        from ``seed`` and ``i`` (the SplitMix64 of ``seed + i``), so it gives
        the same run however many replicates run.

        Parameters
        ----------
        n : int
            How many replicates, at least 1.
        seed : int | None, optional
            The seed the replicate seeds derive from, a whole number from 0 to
            ``2**64 - 1``. None draws one, which the replicates keep.

        Returns
        -------
        ReplicateRuns
            The runner of the replicates.

        Raises
        ------
        ValueError
            If the engine is deterministic (every replicate would be equal),
            ``n`` is below 1, or the seed is not valid.
        """
        if not self.is_stochastic:
            raise ValueError(
                f"The {self.model_definition.engine.type} engine is deterministic: "
                "every replicate would be equal. Replicates need a stochastic engine."
            )
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"The number of replicates is at least 1, got {n!r}.")
        return ReplicateRuns(self, n, draw_seed() if seed is None else check_seed(seed))

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
        (`ModelStructure.outputs_by_source`): for an aggregate model, by
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
        model = self.model_definition
        outputs = model.structure.outputs_by_source(model)
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

#### is_stochastic

```python
is_stochastic: bool
```

Whether the engine of the model draws random numbers.

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
            "holds the structure of its kind, such as an Aggregate."
        )
    logging.info(f"Initializing Simulation with model: '{model.name}'")
    self.model_definition: Model = model
    self._build_engine(model.model_dump_json())
```

#### run

```python
run(num_steps: int, *, seed: int | None = None) -> SimulationResults
```

Runs the simulation from the initial state on its default grid.

Every call starts at position 0 with the current `model_definition`, so two calls with the same model, and the same seed for a stochastic engine, give the same result. To continue a run, call `simulation.engine.step()`.

Parameters:

| Name        | Type  | Description                                                               | Default                                                                                                                                                                           |
| ----------- | ----- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_steps` | `int` | The number of steps: the results hold the positions 0, 1, ..., num_steps. | *required*                                                                                                                                                                        |
| `seed`      | \`int | None\`                                                                    | The seed of a stochastic engine, a whole number from 0 to 2\*\*64 - 1. None draws one, which results.seed keeps, so every run can be repeated. A deterministic engine ignores it. |

Returns:

| Type                | Description                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `SimulationResults` | One value per position for each output, the positions, and the seed of a stochastic run. |

Raises:

| Type                   | Description                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `InfeasibleStateError` | If a step leaves an amount below zero; the message names the compartment, the step and the flows out of it. |
| `ValueError`           | If a parameter or initial condition has no value, or the seed is not a whole number from 0 to 2\*\*64 - 1.  |

Source code in `commol/api/simulation.py`

```python
def run(self, num_steps: int, *, seed: int | None = None) -> SimulationResults:
    """
    Runs the simulation from the initial state on its default grid.

    Every call starts at position 0 with the current ``model_definition``,
    so two calls with the same model, and the same seed for a stochastic
    engine, give the same result. To continue a run, call
    ``simulation.engine.step()``.

    Parameters
    ----------
    num_steps : int
        The number of steps: the results hold the positions
        ``0, 1, ..., num_steps``.
    seed : int | None, optional
        The seed of a stochastic engine, a whole number from 0 to
        ``2**64 - 1``. None draws one, which ``results.seed`` keeps, so
        every run can be repeated. A deterministic engine ignores it.

    Returns
    -------
    SimulationResults
        One value per position for each output, the positions, and the
        seed of a stochastic run.

    Raises
    ------
    InfeasibleStateError
        If a step leaves an amount below zero; the message names the
        compartment, the step and the flows out of it.
    ValueError
        If a parameter or initial condition has no value, or the seed is
        not a whole number from 0 to ``2**64 - 1``.
    """
    self._validate_all_parameters_calibrated(self.model_definition)
    self._sync_engine()
    used_seed = self._seed_run(seed)
    self._engine.reset()
    logging.info(f"Running simulation for {num_steps} steps.")
    start = time.time()
    positions, rows = self._engine.run(num_steps)
    logging.info(f"Simulation complete. It took {time.time() - start} seconds.")
    return SimulationResults.from_rows(
        positions, self._simulation_outputs, rows, seed=used_seed
    )
```

#### run_at

```python
run_at(at: Sequence[float], *, seed: int | None = None) -> SimulationResults
```

Runs the simulation from the initial state and reports at given positions.

The positions are on the axis, the independent variable of the model. The engine decides which positions its model type reports at: a DifferentialEquations model at any finite, non-negative position, from the dense output of its solver; a DifferenceEquations model only at integer steps.

Parameters:

| Name   | Type              | Description                            | Default                                   |
| ------ | ----------------- | -------------------------------------- | ----------------------------------------- |
| `at`   | `Sequence[float]` | Sorted, unique positions to report at. | *required*                                |
| `seed` | \`int             | None\`                                 | The seed of a stochastic engine; see run. |

Returns:

| Type                | Description                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `SimulationResults` | One value per position for each output, the positions, and the seed of a stochastic run. |

Raises:

| Type                   | Description                                                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InfeasibleStateError` | If a step leaves an amount below zero; the message names the compartment, the step and the flows out of it.                                                           |
| `ValueError`           | If a parameter or initial condition has no value, the positions are not sorted and unique, or the engine does not report at one of them; the message is the engine's. |

Source code in `commol/api/simulation.py`

```python
def run_at(
    self, at: Sequence[float], *, seed: int | None = None
) -> SimulationResults:
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
    seed : int | None, optional
        The seed of a stochastic engine; see `run`.

    Returns
    -------
    SimulationResults
        One value per position for each output, the positions, and the
        seed of a stochastic run.

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
    used_seed = self._seed_run(seed)
    self._engine.reset()
    rows = self._engine.run_at(positions)
    return SimulationResults.from_rows(
        positions, self._simulation_outputs, rows, seed=used_seed
    )
```

#### replicates

```python
replicates(n: int, *, seed: int | None = None) -> ReplicateRuns
```

Replicates of the runs of a stochastic engine.

The returned object has `run` and `run_at`, as `Simulation`, and they return `SimulationReplicates`: `n` runs, each with its own seed, run in parallel in the engine. Replicate `i` takes the seed derived from `seed` and `i` (the SplitMix64 of `seed + i`), so it gives the same run however many replicates run.

Parameters:

| Name   | Type  | Description                      | Default                                                                                                                    |
| ------ | ----- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `n`    | `int` | How many replicates, at least 1. | *required*                                                                                                                 |
| `seed` | \`int | None\`                           | The seed the replicate seeds derive from, a whole number from 0 to 2\*\*64 - 1. None draws one, which the replicates keep. |

Returns:

| Type            | Description                   |
| --------------- | ----------------------------- |
| `ReplicateRuns` | The runner of the replicates. |

Raises:

| Type         | Description                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the engine is deterministic (every replicate would be equal), n is below 1, or the seed is not valid. |

Source code in `commol/api/simulation.py`

```python
def replicates(self, n: int, *, seed: int | None = None) -> "ReplicateRuns":
    """
    Replicates of the runs of a stochastic engine.

    The returned object has ``run`` and ``run_at``, as `Simulation`, and
    they return `SimulationReplicates`: ``n`` runs, each with its own seed,
    run in parallel in the engine. Replicate ``i`` takes the seed derived
    from ``seed`` and ``i`` (the SplitMix64 of ``seed + i``), so it gives
    the same run however many replicates run.

    Parameters
    ----------
    n : int
        How many replicates, at least 1.
    seed : int | None, optional
        The seed the replicate seeds derive from, a whole number from 0 to
        ``2**64 - 1``. None draws one, which the replicates keep.

    Returns
    -------
    ReplicateRuns
        The runner of the replicates.

    Raises
    ------
    ValueError
        If the engine is deterministic (every replicate would be equal),
        ``n`` is below 1, or the seed is not valid.
    """
    if not self.is_stochastic:
        raise ValueError(
            f"The {self.model_definition.engine.type} engine is deterministic: "
            "every replicate would be equal. Replicates need a stochastic engine."
        )
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"The number of replicates is at least 1, got {n!r}.")
    return ReplicateRuns(self, n, draw_seed() if seed is None else check_seed(seed))
```

#### outputs_for

```python
outputs_for(id: str) -> list[str]
```

Names of the outputs a source id expands into.

The structure of the model groups its outputs by source (`ModelStructure.outputs_by_source`): for an aggregate model, by bin and accumulator.

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
    (`ModelStructure.outputs_by_source`): for an aggregate model, by
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
    model = self.model_definition
    outputs = model.structure.outputs_by_source(model)
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
| `seed`      | \`int         | None\`                                                                                   |

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
    seed : int | None
        The seed of the run of a stochastic engine, which repeats it; None for
        a deterministic engine, whose runs depend on no seed.
    """

    positions: list[float]
    seed: int | None

    def __init__(
        self,
        positions: Iterable[float],
        values: Mapping[str, Sequence[float]] | None = None,
        seed: int | None = None,
    ):
        """
        Results at the given positions.

        Parameters
        ----------
        positions : Iterable[float]
            Positions on the axis, ascending.
        values : Mapping[str, Sequence[float]] | None, optional
            One value per position for each output.
        seed : int | None, optional
            The seed of the run, or None when it depends on no seed.

        Raises
        ------
        ValueError
            If an output does not have one value per position.
        """
        super().__init__(
            {name: list(series) for name, series in (values or {}).items()}
        )
        self.positions = [float(position) for position in positions]
        self.seed = seed
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
        seed: int | None = None,
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
        seed : int | None, optional
            The seed of the run, or None when it depends on no seed.

        Returns
        -------
        SimulationResults
            One series per output name.
        """
        columns = list(zip(*rows)) if rows else [() for _ in names]
        return cls(positions, dict(zip(names, columns, strict=True)), seed=seed)

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
        """The positions, the seed and the values."""
        seed = "" if self.seed is None else f"seed={self.seed!r}, "
        return (
            f"SimulationResults(positions={self.positions!r}, {seed}"
            f"{dict.__repr__(self)})"
        )
```

### Methods:

#### __init__

```python
__init__(positions: Iterable[float], values: Mapping[str, Sequence[float]] | None = None, seed: int | None = None)
```

Results at the given positions.

Parameters:

| Name        | Type                              | Description                       | Default                                                  |
| ----------- | --------------------------------- | --------------------------------- | -------------------------------------------------------- |
| `positions` | `Iterable[float]`                 | Positions on the axis, ascending. | *required*                                               |
| `values`    | \`Mapping\[str, Sequence[float]\] | None\`                            | One value per position for each output.                  |
| `seed`      | \`int                             | None\`                            | The seed of the run, or None when it depends on no seed. |

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
    seed: int | None = None,
):
    """
    Results at the given positions.

    Parameters
    ----------
    positions : Iterable[float]
        Positions on the axis, ascending.
    values : Mapping[str, Sequence[float]] | None, optional
        One value per position for each output.
    seed : int | None, optional
        The seed of the run, or None when it depends on no seed.

    Raises
    ------
    ValueError
        If an output does not have one value per position.
    """
    super().__init__(
        {name: list(series) for name, series in (values or {}).items()}
    )
    self.positions = [float(position) for position in positions]
    self.seed = seed
    for name, series in self.items():
        if len(series) != len(self.positions):
            raise ValueError(
                f"Output '{name}' has {len(series)} values for "
                f"{len(self.positions)} positions."
            )
```

#### from_rows

```python
from_rows(positions: Sequence[float], names: Sequence[str], rows: Sequence[Sequence[float]], seed: int | None = None) -> SimulationResults
```

Results from one row of output values per position.

Parameters:

| Name        | Type                        | Description                             | Default                                                  |
| ----------- | --------------------------- | --------------------------------------- | -------------------------------------------------------- |
| `positions` | `Sequence[float]`           | Positions on the axis, ascending.       | *required*                                               |
| `names`     | `Sequence[str]`             | Output names, in the order of each row. | *required*                                               |
| `rows`      | `Sequence[Sequence[float]]` | One row per position.                   | *required*                                               |
| `seed`      | \`int                       | None\`                                  | The seed of the run, or None when it depends on no seed. |

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
    seed: int | None = None,
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
    seed : int | None, optional
        The seed of the run, or None when it depends on no seed.

    Returns
    -------
    SimulationResults
        One series per output name.
    """
    columns = list(zip(*rows)) if rows else [() for _ in names]
    return cls(positions, dict(zip(names, columns, strict=True)), seed=seed)
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

The positions, the seed and the values.

Source code in `commol/api/results.py`

```python
def __repr__(self) -> str:
    """The positions, the seed and the values."""
    seed = "" if self.seed is None else f"seed={self.seed!r}, "
    return (
        f"SimulationResults(positions={self.positions!r}, {seed}"
        f"{dict.__repr__(self)})"
    )
```

## ReplicateRuns

Runs replicates of a stochastic model; `Simulation.replicates` makes it.

Attributes:

| Name         | Type         | Description                                    |
| ------------ | ------------ | ---------------------------------------------- |
| `simulation` | `Simulation` | The simulation whose model the replicates run. |
| `n`          | `int`        | How many replicates.                           |
| `seed`       | `int`        | The seed the replicate seeds derive from.      |

Source code in `commol/api/simulation.py`

```python
class ReplicateRuns:
    """
    Runs replicates of a stochastic model; `Simulation.replicates` makes it.

    Attributes
    ----------
    simulation : Simulation
        The simulation whose model the replicates run.
    n : int
        How many replicates.
    seed : int
        The seed the replicate seeds derive from.
    """

    def __init__(self, simulation: Simulation, n: int, seed: int):
        """
        The replicates of a simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation whose model the replicates run.
        n : int
            How many replicates.
        seed : int
            The seed the replicate seeds derive from.
        """
        self.simulation = simulation
        self.n = n
        self.seed = seed

    def _replicates(
        self, num_steps: int | None, positions: list[float] | None
    ) -> SimulationReplicates:
        simulation = self.simulation
        simulation._validate_all_parameters_calibrated(simulation.model_definition)
        engine = simulation.engine
        grid, seeds, runs = engine.run_replicates(
            self.seed, self.n, num_steps=num_steps, positions=positions
        )
        outputs = simulation.simulation_outputs
        return SimulationReplicates(
            (
                SimulationResults.from_rows(grid, outputs, rows, seed=replicate_seed)
                for replicate_seed, rows in zip(seeds, runs, strict=True)
            ),
            seed=self.seed,
            positions=grid,
        )

    def run(self, num_steps: int) -> SimulationReplicates:
        """
        Every replicate from the initial state, on the default grid.

        Parameters
        ----------
        num_steps : int
            The number of steps: each replicate holds the positions
            ``0, 1, ..., num_steps``.

        Returns
        -------
        SimulationReplicates
            One `SimulationResults` per replicate, each with its seed.

        Raises
        ------
        InfeasibleStateError
            If a replicate leaves the values the model allows.
        ValueError
            If a parameter or a value of the model has no value.
        """
        return self._replicates(num_steps, None)

    def run_at(self, at: Sequence[float]) -> SimulationReplicates:
        """
        Every replicate from the initial state, reported at given positions.

        Parameters
        ----------
        at : Sequence[float]
            Sorted, unique positions to report at.

        Returns
        -------
        SimulationReplicates
            One `SimulationResults` per replicate, each with its seed.

        Raises
        ------
        InfeasibleStateError
            If a replicate leaves the values the model allows.
        ValueError
            If a parameter has no value, the positions are not sorted and
            unique, or the engine does not report at one of them.
        """
        return self._replicates(None, [float(position) for position in at])
```

### Methods:

#### __init__

```python
__init__(simulation: Simulation, n: int, seed: int)
```

The replicates of a simulation.

Parameters:

| Name         | Type         | Description                                    | Default    |
| ------------ | ------------ | ---------------------------------------------- | ---------- |
| `simulation` | `Simulation` | The simulation whose model the replicates run. | *required* |
| `n`          | `int`        | How many replicates.                           | *required* |
| `seed`       | `int`        | The seed the replicate seeds derive from.      | *required* |

Source code in `commol/api/simulation.py`

```python
def __init__(self, simulation: Simulation, n: int, seed: int):
    """
    The replicates of a simulation.

    Parameters
    ----------
    simulation : Simulation
        The simulation whose model the replicates run.
    n : int
        How many replicates.
    seed : int
        The seed the replicate seeds derive from.
    """
    self.simulation = simulation
    self.n = n
    self.seed = seed
```

#### run

```python
run(num_steps: int) -> SimulationReplicates
```

Every replicate from the initial state, on the default grid.

Parameters:

| Name        | Type  | Description                                                                   | Default    |
| ----------- | ----- | ----------------------------------------------------------------------------- | ---------- |
| `num_steps` | `int` | The number of steps: each replicate holds the positions 0, 1, ..., num_steps. | *required* |

Returns:

| Type                   | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `SimulationReplicates` | One SimulationResults per replicate, each with its seed. |

Raises:

| Type                   | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `InfeasibleStateError` | If a replicate leaves the values the model allows.   |
| `ValueError`           | If a parameter or a value of the model has no value. |

Source code in `commol/api/simulation.py`

```python
def run(self, num_steps: int) -> SimulationReplicates:
    """
    Every replicate from the initial state, on the default grid.

    Parameters
    ----------
    num_steps : int
        The number of steps: each replicate holds the positions
        ``0, 1, ..., num_steps``.

    Returns
    -------
    SimulationReplicates
        One `SimulationResults` per replicate, each with its seed.

    Raises
    ------
    InfeasibleStateError
        If a replicate leaves the values the model allows.
    ValueError
        If a parameter or a value of the model has no value.
    """
    return self._replicates(num_steps, None)
```

#### run_at

```python
run_at(at: Sequence[float]) -> SimulationReplicates
```

Every replicate from the initial state, reported at given positions.

Parameters:

| Name | Type              | Description                            | Default    |
| ---- | ----------------- | -------------------------------------- | ---------- |
| `at` | `Sequence[float]` | Sorted, unique positions to report at. | *required* |

Returns:

| Type                   | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `SimulationReplicates` | One SimulationResults per replicate, each with its seed. |

Raises:

| Type                   | Description                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `InfeasibleStateError` | If a replicate leaves the values the model allows.                                                                  |
| `ValueError`           | If a parameter has no value, the positions are not sorted and unique, or the engine does not report at one of them. |

Source code in `commol/api/simulation.py`

```python
def run_at(self, at: Sequence[float]) -> SimulationReplicates:
    """
    Every replicate from the initial state, reported at given positions.

    Parameters
    ----------
    at : Sequence[float]
        Sorted, unique positions to report at.

    Returns
    -------
    SimulationReplicates
        One `SimulationResults` per replicate, each with its seed.

    Raises
    ------
    InfeasibleStateError
        If a replicate leaves the values the model allows.
    ValueError
        If a parameter has no value, the positions are not sorted and
        unique, or the engine does not report at one of them.
    """
    return self._replicates(None, [float(position) for position in at])
```

## SimulationReplicates

Bases: `list[SimulationResults]`

Replicates of a run of a stochastic engine, one `SimulationResults` each.

A `list` of runs at the same positions, each with its own seed: `replicates[i].seed` repeats replicate `i`. The statistics reduce the replicates position by position, and `member_statistics` reduces them as `ci_percentiles` requires.

Attributes:

| Name        | Type          | Description                                       |
| ----------- | ------------- | ------------------------------------------------- |
| `seed`      | `int`         | The seed the replicate seeds derive from.         |
| `positions` | `list[float]` | Positions on the axis, shared by every replicate. |

Source code in `commol/api/results.py`

```python
class SimulationReplicates(list[SimulationResults]):
    """
    Replicates of a run of a stochastic engine, one `SimulationResults` each.

    A ``list`` of runs at the same positions, each with its own seed:
    ``replicates[i].seed`` repeats replicate ``i``. The statistics reduce the
    replicates position by position, and `member_statistics` reduces them as
    ``ci_percentiles`` requires.

    Attributes
    ----------
    seed : int
        The seed the replicate seeds derive from.
    positions : list[float]
        Positions on the axis, shared by every replicate.
    """

    seed: int
    positions: list[float]

    def __init__(
        self, runs: Iterable[SimulationResults], seed: int, positions: Sequence[float]
    ):
        """
        Replicates of one run.

        Parameters
        ----------
        runs : Iterable[SimulationResults]
            One run per replicate, at `positions`.
        seed : int
            The seed the replicate seeds derive from.
        positions : Sequence[float]
            Positions on the axis, ascending.

        Raises
        ------
        ValueError
            If a run is not at `positions`.
        """
        super().__init__(runs)
        self.seed = seed
        self.positions = [float(position) for position in positions]
        for index, run in enumerate(self):
            if run.positions != self.positions:
                raise ValueError(
                    f"Replicate {index} is at other positions than the replicates."
                )

    @property
    def seeds(self) -> list[int]:
        """The seed of each replicate, in order."""
        return [run.seed for run in self if run.seed is not None]

    def _values(self, name: str) -> np.ndarray:
        if not self:
            raise ValueError("There are no replicates.")
        return np.asarray([run[name] for run in self], dtype=float)

    def mean(self) -> SimulationResults:
        """
        The mean of the replicates at each position, for each output.

        Returns
        -------
        SimulationResults
            One series of means per output, with no seed.

        Raises
        ------
        ValueError
            If there are no replicates.
        """
        names = list(self[0]) if self else []
        if not names:
            raise ValueError("There are no replicates.")
        return SimulationResults(
            self.positions,
            {name: self._values(name).mean(axis=0).tolist() for name in names},
        )

    def quantiles(self, levels: Sequence[float]) -> dict[float, SimulationResults]:
        """
        Quantiles of the replicates at each position, for each output.

        Parameters
        ----------
        levels : Sequence[float]
            Quantile levels, each between 0 and 1, such as ``[0.05, 0.5,
            0.95]``.

        Returns
        -------
        dict[float, SimulationResults]
            The results of each level.

        Raises
        ------
        ValueError
            If a level is outside ``[0, 1]`` or there are no replicates.
        """
        for level in levels:
            if not 0.0 <= level <= 1.0:
                raise ValueError(f"A quantile level is between 0 and 1, got {level}.")
        if not self:
            raise ValueError("There are no replicates.")
        names = list(self[0])
        return {
            level: SimulationResults(
                self.positions,
                {
                    name: np.quantile(self._values(name), level, axis=0).tolist()
                    for name in names
                },
            )
            for level in levels
        }

    def member_statistics(
        self, output: str, confidence_level: float
    ) -> dict[str, list[float]]:
        """
        The spread of one output across the replicates, at each position.

        Parameters
        ----------
        output : str
            An output name.
        confidence_level : float
            Confidence level of the interval, strictly between 0 and 1.

        Returns
        -------
        dict[str, list[float]]
            Keys ``mean``, ``median``, ``ci_lower``, ``ci_upper``, ``min`` and
            ``max``, one value per position (`member_statistics`).

        Raises
        ------
        KeyError
            If no replicate has the output.
        ValueError
            If the confidence level is out of range or there are no
            replicates.
        """
        from commol.api.probabilistic.intervals import member_statistics

        if self and output not in self[0]:
            raise KeyError(
                f"'{output}' is not an output of the replicates. "
                f"Outputs: {sorted(self[0])}"
            )
        return member_statistics([run[output] for run in self], confidence_level)

    def __repr__(self) -> str:
        """The number of replicates, their seed and their positions."""
        return (
            f"SimulationReplicates({len(self)} replicates, seed={self.seed!r}, "
            f"positions={self.positions!r})"
        )
```

### Attributes

#### seeds

```python
seeds: list[int]
```

The seed of each replicate, in order.

### Methods:

#### __init__

```python
__init__(runs: Iterable[SimulationResults], seed: int, positions: Sequence[float])
```

Replicates of one run.

Parameters:

| Name        | Type                          | Description                               | Default    |
| ----------- | ----------------------------- | ----------------------------------------- | ---------- |
| `runs`      | `Iterable[SimulationResults]` | One run per replicate, at positions.      | *required* |
| `seed`      | `int`                         | The seed the replicate seeds derive from. | *required* |
| `positions` | `Sequence[float]`             | Positions on the axis, ascending.         | *required* |

Raises:

| Type         | Description                   |
| ------------ | ----------------------------- |
| `ValueError` | If a run is not at positions. |

Source code in `commol/api/results.py`

```python
def __init__(
    self, runs: Iterable[SimulationResults], seed: int, positions: Sequence[float]
):
    """
    Replicates of one run.

    Parameters
    ----------
    runs : Iterable[SimulationResults]
        One run per replicate, at `positions`.
    seed : int
        The seed the replicate seeds derive from.
    positions : Sequence[float]
        Positions on the axis, ascending.

    Raises
    ------
    ValueError
        If a run is not at `positions`.
    """
    super().__init__(runs)
    self.seed = seed
    self.positions = [float(position) for position in positions]
    for index, run in enumerate(self):
        if run.positions != self.positions:
            raise ValueError(
                f"Replicate {index} is at other positions than the replicates."
            )
```

#### mean

```python
mean() -> SimulationResults
```

The mean of the replicates at each position, for each output.

Returns:

| Type                | Description                                   |
| ------------------- | --------------------------------------------- |
| `SimulationResults` | One series of means per output, with no seed. |

Raises:

| Type         | Description                 |
| ------------ | --------------------------- |
| `ValueError` | If there are no replicates. |

Source code in `commol/api/results.py`

```python
def mean(self) -> SimulationResults:
    """
    The mean of the replicates at each position, for each output.

    Returns
    -------
    SimulationResults
        One series of means per output, with no seed.

    Raises
    ------
    ValueError
        If there are no replicates.
    """
    names = list(self[0]) if self else []
    if not names:
        raise ValueError("There are no replicates.")
    return SimulationResults(
        self.positions,
        {name: self._values(name).mean(axis=0).tolist() for name in names},
    )
```

#### quantiles

```python
quantiles(levels: Sequence[float]) -> dict[float, SimulationResults]
```

Quantiles of the replicates at each position, for each output.

Parameters:

| Name     | Type              | Description                                                       | Default    |
| -------- | ----------------- | ----------------------------------------------------------------- | ---------- |
| `levels` | `Sequence[float]` | Quantile levels, each between 0 and 1, such as [0.05, 0.5, 0.95]. | *required* |

Returns:

| Type                             | Description                |
| -------------------------------- | -------------------------- |
| `dict[float, SimulationResults]` | The results of each level. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If a level is outside [0, 1] or there are no replicates. |

Source code in `commol/api/results.py`

```python
def quantiles(self, levels: Sequence[float]) -> dict[float, SimulationResults]:
    """
    Quantiles of the replicates at each position, for each output.

    Parameters
    ----------
    levels : Sequence[float]
        Quantile levels, each between 0 and 1, such as ``[0.05, 0.5,
        0.95]``.

    Returns
    -------
    dict[float, SimulationResults]
        The results of each level.

    Raises
    ------
    ValueError
        If a level is outside ``[0, 1]`` or there are no replicates.
    """
    for level in levels:
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"A quantile level is between 0 and 1, got {level}.")
    if not self:
        raise ValueError("There are no replicates.")
    names = list(self[0])
    return {
        level: SimulationResults(
            self.positions,
            {
                name: np.quantile(self._values(name), level, axis=0).tolist()
                for name in names
            },
        )
        for level in levels
    }
```

#### member_statistics

```python
member_statistics(output: str, confidence_level: float) -> dict[str, list[float]]
```

The spread of one output across the replicates, at each position.

Parameters:

| Name               | Type    | Description                                                 | Default    |
| ------------------ | ------- | ----------------------------------------------------------- | ---------- |
| `output`           | `str`   | An output name.                                             | *required* |
| `confidence_level` | `float` | Confidence level of the interval, strictly between 0 and 1. | *required* |

Returns:

| Type                     | Description                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `dict[str, list[float]]` | Keys mean, median, ci_lower, ci_upper, min and max, one value per position (member_statistics). |

Raises:

| Type         | Description                                                         |
| ------------ | ------------------------------------------------------------------- |
| `KeyError`   | If no replicate has the output.                                     |
| `ValueError` | If the confidence level is out of range or there are no replicates. |

Source code in `commol/api/results.py`

```python
def member_statistics(
    self, output: str, confidence_level: float
) -> dict[str, list[float]]:
    """
    The spread of one output across the replicates, at each position.

    Parameters
    ----------
    output : str
        An output name.
    confidence_level : float
        Confidence level of the interval, strictly between 0 and 1.

    Returns
    -------
    dict[str, list[float]]
        Keys ``mean``, ``median``, ``ci_lower``, ``ci_upper``, ``min`` and
        ``max``, one value per position (`member_statistics`).

    Raises
    ------
    KeyError
        If no replicate has the output.
    ValueError
        If the confidence level is out of range or there are no
        replicates.
    """
    from commol.api.probabilistic.intervals import member_statistics

    if self and output not in self[0]:
        raise KeyError(
            f"'{output}' is not an output of the replicates. "
            f"Outputs: {sorted(self[0])}"
        )
    return member_statistics([run[output] for run in self], confidence_level)
```

#### __repr__

```python
__repr__() -> str
```

The number of replicates, their seed and their positions.

Source code in `commol/api/results.py`

```python
def __repr__(self) -> str:
    """The number of replicates, their seed and their positions."""
    return (
        f"SimulationReplicates({len(self)} replicates, seed={self.seed!r}, "
        f"positions={self.positions!r})"
    )
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
