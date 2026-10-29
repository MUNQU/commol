# Simulation API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Runs a model and reads its outputs. Guide: [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/index.md).

## Simulation

A Facade for running a simulation from a defined Model.

The engine follows the model: `run` and `engine` rebuild it when `model_definition` no longer serializes to the JSON the engine was built from, for example after `Model.update_parameters`.

Source code in `commol/_simulation/simulation.py`

```python
class Simulation:
    """
    A Facade for running a simulation from a defined Model.

    The engine follows the model: ``run`` and ``engine`` rebuild it when
    ``model_definition`` no longer serializes to the JSON the engine was built
    from, for example after ``Model.update_parameters``.
    """

    def __init__(self, model: Model) -> None:
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
        logger.info(f"Initializing Simulation with model: '{model.name}'")
        self.model_definition: Model = model
        self._build_engine(engine_json(model))

    def _build_engine(self, model_json: str) -> None:
        """Build the engine and its output names from a model JSON."""
        self._model_json: str = model_json
        self._engine: "EngineProtocol" = build_engine(model_json)
        self._simulation_outputs: list[str] = self._engine.output_names
        logger.info(
            f"Simulation engine ready. Outputs: {len(self._simulation_outputs)}"
        )

    def _sync_engine(self) -> None:
        """Rebuild the engine when the model changed since it was built."""
        model_json = engine_json(self.model_definition)
        if model_json != self._model_json:
            logger.info("Model changed since the engine was built; rebuilding it.")
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
            engine_json(self.model_definition.engine.type)
        ).is_stochastic

    def _seed_run(self, seed: int | None) -> int | None:
        """
        Seed the engine of a stochastic model, which draws a seed for None;
        the seed it runs with, or None for a deterministic engine.
        """
        if not self.is_stochastic:
            return None
        return self._engine.set_seed(None if seed is None else check_seed(seed))

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
            One value per position for each output, the positions, the seed
            of a stochastic run, and the model.

        Raises
        ------
        InfeasibleStateError
            If a step leaves an amount below zero; the message names the
            compartment, the step and the flows out of it.
        ValueError
            If a parameter or initial condition has no value, or the seed is
            not a whole number from 0 to ``2**64 - 1``.
        RuntimeError
            If a rate cannot be evaluated or is not a number; the message
            names the flow and the position.
        """
        self._validate_all_parameters_calibrated(self.model_definition)
        self._sync_engine()
        used_seed = self._seed_run(seed)
        self._engine.reset()
        logger.info(f"Running simulation for {num_steps} steps.")
        start = time.time()
        positions, rows = self._engine.run(num_steps)
        logger.info(f"Simulation complete. It took {time.time() - start} seconds.")
        return SimulationResults.from_rows(
            positions,
            self._simulation_outputs,
            rows,
            seed=used_seed,
            model=self.model_definition,
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
            One value per position for each output, the positions, the seed
            of a stochastic run, and the model.

        Raises
        ------
        InfeasibleStateError
            If a step leaves an amount below zero; the message names the
            compartment, the step and the flows out of it.
        ValueError
            If a parameter or initial condition has no value, the positions are
            not sorted and unique, or the engine does not report at one of them;
            the message is the engine's.
        RuntimeError
            If a rate cannot be evaluated or is not a number; the message
            names the flow and the position.
        """
        positions = [float(position) for position in at]
        self._validate_all_parameters_calibrated(self.model_definition)
        self._sync_engine()
        used_seed = self._seed_run(seed)
        self._engine.reset()
        rows = self._engine.run_at(positions)
        return SimulationResults.from_rows(
            positions,
            self._simulation_outputs,
            rows,
            seed=used_seed,
            model=self.model_definition,
        )

    def replicates(self, count: int, *, seed: int | None = None) -> "ReplicateRuns":
        """
        Replicates of the runs of a stochastic engine.

        The returned object has ``run`` and ``run_at``, as `Simulation`, and
        they return `SimulationReplicates`: ``count`` runs, each with its own seed,
        run in parallel in the engine. Replicate ``i`` takes the seed derived
        from ``seed`` and ``i`` (the SplitMix64 of the SplitMix64 of ``seed``
        plus ``i``), so two seeds share no replicate, and a replicate gives the
        same run however many replicates run.

        Parameters
        ----------
        count : int
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
            ``count`` is not a whole number or is below 1, or the seed is not
            valid.
        """
        if not self.is_stochastic:
            raise ValueError(
                f"The {self.model_definition.engine.type} engine is deterministic: "
                "every replicate would be equal. Replicates need a stochastic engine."
            )
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"count is a whole number, got {count!r}")
        return ReplicateRuns(self, count, None if seed is None else check_seed(seed))

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
        return outputs_for(self.model_definition, id)

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
        Sum the given outputs at each position.

        Parameters
        ----------
        results : Mapping[str, Sequence[float]]
            Simulation results, such as those of `run` or `run_at`.
        ids : Iterable[str]
            Bin and accumulator ids, each for all its outputs
            (`outputs_for`), and output names, each for itself. Their outputs
            are summed together.

        Returns
        -------
        list[float]
            One value per position of `results`.

        Raises
        ------
        KeyError
            If an id is not a bin, accumulator or output of the model, or an
            expected output is missing from `results`.
        """
        return total_series(results, series_names(self.model_definition, ids))
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
__init__(model: Model) -> None
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

Source code in `commol/_simulation/simulation.py`

```python
def __init__(self, model: Model) -> None:
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
    logger.info(f"Initializing Simulation with model: '{model.name}'")
    self.model_definition: Model = model
    self._build_engine(engine_json(model))
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

| Type                | Description                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `SimulationResults` | One value per position for each output, the positions, the seed of a stochastic run, and the model. |

Raises:

| Type                   | Description                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `InfeasibleStateError` | If a step leaves an amount below zero; the message names the compartment, the step and the flows out of it. |
| `ValueError`           | If a parameter or initial condition has no value, or the seed is not a whole number from 0 to 2\*\*64 - 1.  |
| `RuntimeError`         | If a rate cannot be evaluated or is not a number; the message names the flow and the position.              |

Source code in `commol/_simulation/simulation.py`

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
        One value per position for each output, the positions, the seed
        of a stochastic run, and the model.

    Raises
    ------
    InfeasibleStateError
        If a step leaves an amount below zero; the message names the
        compartment, the step and the flows out of it.
    ValueError
        If a parameter or initial condition has no value, or the seed is
        not a whole number from 0 to ``2**64 - 1``.
    RuntimeError
        If a rate cannot be evaluated or is not a number; the message
        names the flow and the position.
    """
    self._validate_all_parameters_calibrated(self.model_definition)
    self._sync_engine()
    used_seed = self._seed_run(seed)
    self._engine.reset()
    logger.info(f"Running simulation for {num_steps} steps.")
    start = time.time()
    positions, rows = self._engine.run(num_steps)
    logger.info(f"Simulation complete. It took {time.time() - start} seconds.")
    return SimulationResults.from_rows(
        positions,
        self._simulation_outputs,
        rows,
        seed=used_seed,
        model=self.model_definition,
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

| Type                | Description                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `SimulationResults` | One value per position for each output, the positions, the seed of a stochastic run, and the model. |

Raises:

| Type                   | Description                                                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InfeasibleStateError` | If a step leaves an amount below zero; the message names the compartment, the step and the flows out of it.                                                           |
| `ValueError`           | If a parameter or initial condition has no value, the positions are not sorted and unique, or the engine does not report at one of them; the message is the engine's. |
| `RuntimeError`         | If a rate cannot be evaluated or is not a number; the message names the flow and the position.                                                                        |

Source code in `commol/_simulation/simulation.py`

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
        One value per position for each output, the positions, the seed
        of a stochastic run, and the model.

    Raises
    ------
    InfeasibleStateError
        If a step leaves an amount below zero; the message names the
        compartment, the step and the flows out of it.
    ValueError
        If a parameter or initial condition has no value, the positions are
        not sorted and unique, or the engine does not report at one of them;
        the message is the engine's.
    RuntimeError
        If a rate cannot be evaluated or is not a number; the message
        names the flow and the position.
    """
    positions = [float(position) for position in at]
    self._validate_all_parameters_calibrated(self.model_definition)
    self._sync_engine()
    used_seed = self._seed_run(seed)
    self._engine.reset()
    rows = self._engine.run_at(positions)
    return SimulationResults.from_rows(
        positions,
        self._simulation_outputs,
        rows,
        seed=used_seed,
        model=self.model_definition,
    )
```

#### replicates

```python
replicates(count: int, *, seed: int | None = None) -> ReplicateRuns
```

Replicates of the runs of a stochastic engine.

The returned object has `run` and `run_at`, as `Simulation`, and they return `SimulationReplicates`: `count` runs, each with its own seed, run in parallel in the engine. Replicate `i` takes the seed derived from `seed` and `i` (the SplitMix64 of the SplitMix64 of `seed` plus `i`), so two seeds share no replicate, and a replicate gives the same run however many replicates run.

Parameters:

| Name    | Type  | Description                      | Default                                                                                                                    |
| ------- | ----- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `count` | `int` | How many replicates, at least 1. | *required*                                                                                                                 |
| `seed`  | \`int | None\`                           | The seed the replicate seeds derive from, a whole number from 0 to 2\*\*64 - 1. None draws one, which the replicates keep. |

Returns:

| Type            | Description                   |
| --------------- | ----------------------------- |
| `ReplicateRuns` | The runner of the replicates. |

Raises:

| Type         | Description                                                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the engine is deterministic (every replicate would be equal), count is not a whole number or is below 1, or the seed is not valid. |

Source code in `commol/_simulation/simulation.py`

```python
def replicates(self, count: int, *, seed: int | None = None) -> "ReplicateRuns":
    """
    Replicates of the runs of a stochastic engine.

    The returned object has ``run`` and ``run_at``, as `Simulation`, and
    they return `SimulationReplicates`: ``count`` runs, each with its own seed,
    run in parallel in the engine. Replicate ``i`` takes the seed derived
    from ``seed`` and ``i`` (the SplitMix64 of the SplitMix64 of ``seed``
    plus ``i``), so two seeds share no replicate, and a replicate gives the
    same run however many replicates run.

    Parameters
    ----------
    count : int
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
        ``count`` is not a whole number or is below 1, or the seed is not
        valid.
    """
    if not self.is_stochastic:
        raise ValueError(
            f"The {self.model_definition.engine.type} engine is deterministic: "
            "every replicate would be equal. Replicates need a stochastic engine."
        )
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"count is a whole number, got {count!r}")
    return ReplicateRuns(self, count, None if seed is None else check_seed(seed))
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

Source code in `commol/_simulation/simulation.py`

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
    return outputs_for(self.model_definition, id)
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

Source code in `commol/_simulation/simulation.py`

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

Sum the given outputs at each position.

Parameters:

| Name      | Type                            | Description                                                                                                                            | Default    |
| --------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `results` | `Mapping[str, Sequence[float]]` | Simulation results, such as those of run or run_at.                                                                                    | *required* |
| `ids`     | `Iterable[str]`                 | Bin and accumulator ids, each for all its outputs (outputs_for), and output names, each for itself. Their outputs are summed together. | *required* |

Returns:

| Type          | Description                        |
| ------------- | ---------------------------------- |
| `list[float]` | One value per position of results. |

Raises:

| Type       | Description                                                                                               |
| ---------- | --------------------------------------------------------------------------------------------------------- |
| `KeyError` | If an id is not a bin, accumulator or output of the model, or an expected output is missing from results. |

Source code in `commol/_simulation/simulation.py`

```python
def total_series(
    self,
    results: Mapping[str, Sequence[float]],
    ids: Iterable[str],
) -> list[float]:
    """
    Sum the given outputs at each position.

    Parameters
    ----------
    results : Mapping[str, Sequence[float]]
        Simulation results, such as those of `run` or `run_at`.
    ids : Iterable[str]
        Bin and accumulator ids, each for all its outputs
        (`outputs_for`), and output names, each for itself. Their outputs
        are summed together.

    Returns
    -------
    list[float]
        One value per position of `results`.

    Raises
    ------
    KeyError
        If an id is not a bin, accumulator or output of the model, or an
        expected output is missing from `results`.
    """
    return total_series(results, series_names(self.model_definition, ids))
```

## SimulationResults

Bases: `ComparedByValue`, `dict[str, list[float]]`

Values of every output at each position on the axis.

A `dict` from output name to one value per position, so `results["A"]`, `json.dumps(results)` and `pandas.DataFrame(results)` work as for any mapping. The positions the values were read at are in :attr:`positions`.

Two results are equal when their values (a NaN equal to a NaN), their positions, their seed and their model are (`ComparedByValue`); a plain `dict` is not a result, so it is not equal to one. Results can change, so they have no hash.

Attributes:

| Name        | Type          | Description                                                                              |
| ----------- | ------------- | ---------------------------------------------------------------------------------------- |
| `positions` | `list[float]` | Positions on the axis, ascending. results[name][i] is the value of name at positions[i]. |
| `seed`      | \`int         | None\`                                                                                   |
| `model`     | \`Model       | None\`                                                                                   |

Source code in `commol/_simulation/results.py`

```python
class SimulationResults(ComparedByValue, dict[str, list[float]]):
    """
    Values of every output at each position on the axis.

    A ``dict`` from output name to one value per position, so
    ``results["A"]``, ``json.dumps(results)`` and
    ``pandas.DataFrame(results)`` work as for any mapping. The positions the
    values were read at are in :attr:`positions`.

    Two results are equal when their values (a NaN equal to a NaN), their
    positions, their seed and their model are (`ComparedByValue`); a plain
    ``dict`` is not a result, so it is not equal to one. Results can change,
    so they have no hash.

    Attributes
    ----------
    positions : list[float]
        Positions on the axis, ascending. ``results[name][i]`` is the value of
        ``name`` at ``positions[i]``.
    seed : int | None
        The seed of the run of a stochastic engine, which repeats it; None for
        a deterministic engine, whose runs depend on no seed.
    model : Model | None
        The model of the run, which `Simulation.run` and `Simulation.run_at`
        set; None for results built by hand. A report reads the outputs of
        each bin and accumulator from its structure, never the values of its
        parameters.
    """

    positions: list[float]
    seed: int | None
    model: "Model | None"

    def __init__(
        self,
        positions: Iterable[float],
        values: Mapping[str, Sequence[float]] | None = None,
        seed: int | None = None,
        model: "Model | None" = None,
    ) -> None:
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
        model : Model | None, optional
            The model of the run, or None when no model is known.

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
        self.model = model
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
        model: "Model | None" = None,
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
        model : Model | None, optional
            The model of the run, or None when no model is known.

        Returns
        -------
        SimulationResults
            One series per output name.
        """
        columns = list(zip(*rows)) if rows else [() for _ in names]
        return cls(
            positions, dict(zip(names, columns, strict=True)), seed=seed, model=model
        )

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

    def quantities(self, quantities: "Sequence[QuantityBase]") -> dict[str, float]:
        """
        The quantities of interest of this run, as a `Propagation` reduces
        each of its runs.

        The run must hold every position the quantities read: the results of
        ``Simulation.run_at(quantity_schedule(quantities))`` hold them all.
        Each position is found within ``1e-9``, as an observation finds its
        row (`index_at` of `crates/commol-calibration/src/observation.rs`), and
        Rust reduces the rows with the code of `Propagation`, so the values
        equal those of a `Propagation` of a sample of one draw with the
        parameters of this run.

        On the ODE engine, `Simulation.run` reports a position that is not a
        breakpoint by interpolation, so a value read from it can differ from
        that of a run that ends there by about the relative tolerance of the
        solver, and `PeakPosition` can change on a near-tie.

        Parameters
        ----------
        quantities : Sequence[quantity]
            The quantities; a `Trajectory` gives one scalar per position.

        Returns
        -------
        dict[str, float]
            One value per label of the quantities, in their order.

        Raises
        ------
        ValueError
            If there is no quantity, two scalars have one label, a quantity
            names an output the results do not have, or the run lacks a
            position a quantity reads; the message lists every such position
            with its quantity and says to call `Simulation.run_at`.
        """
        return quantities_of_runs([self], self.positions, quantities)[0]

    def _total(self, ids: Iterable[str]) -> list[float]:
        """The sum of the outputs the ids stand for, at each position: of a
        source id, all its outputs, with the model of the run; of an output
        name, itself."""
        names = list(ids) if self.model is None else series_names(self.model, ids)
        return total_series(self, names)

    def sample(self, ids: Iterable[str], at: Iterable[float]) -> list[float]:
        """
        Value of the given bins and accumulators at each position of `at`.

        A bin holds an amount at a position, so the value is read at that
        position, not over the window before it; `windowed_totals` reads the
        gain of an accumulator over a window. Each position is found with the
        tolerance of the axis, as an observation finds its row (`index_at` of
        `crates/commol-calibration/src/observation.rs`), so ``0.1 + 0.2``
        reads the value at ``0.3``.

        Parameters
        ----------
        ids : Iterable[str]
            Bin and accumulator ids, each for all its outputs, and output
            names, each for itself. Their outputs are summed before they are
            read. Results without a model read output names only.
        at : Iterable[float]
            Positions to read, each a position of the results.

        Returns
        -------
        list[float]
            One value per position in `at`.

        Raises
        ------
        KeyError
            If an id is not a bin, accumulator or output of the model, or an
            output is missing from the results.
        ValueError
            If the results have no value at a position in `at`.
        """
        return _sample(self._total(ids), self.positions, at)

    def windowed_totals(
        self,
        ids: Iterable[str],
        window: float,
        at: Iterable[float] | None = None,
    ) -> list[float]:
        """
        Amount the given accumulators gained over each window.

        Each value is the sum of the outputs at the end of the window minus
        the sum at its start, ``at - window``, each found with the tolerance
        of the axis: the rule of a windowed observation (`window_increments`
        of `crates/commol-calibration/src/observation.rs`), so the value is
        the prediction a calibration compares an observation against when
        that observation sets ``window``.

        Parameters
        ----------
        ids : Iterable[str]
            Accumulator ids, each for all its outputs, and output names, each
            for itself. Their outputs are summed before windowing. Results
            without a model read output names only.
        window : float
            Length of one window on the axis.
        at : Iterable[float] | None, optional
            Positions at which windows close. Defaults to every complete
            window of the results (`window_ends` of the engine, as
            `AxisUnits.window_ends` gives it). Pass the observation positions
            to reproduce exactly the values a calibration used.

        Returns
        -------
        list[float]
            One increment per window.

        Raises
        ------
        KeyError
            If an id is not a bin, accumulator or output of the model, or an
            output is missing from the results.
        ValueError
            If `window` is not a finite, positive length (the message of the
            engine), a window starts before position 0, or the results have
            no value at the end or the start of a requested window.
        """
        return _windowed_totals(self._total(ids), self.positions, window, at)

    @property
    def plot(self) -> ResultsPlot:
        """
        The figures of the run: ``plot.series()`` and ``plot.cumulative()``.

        They name each series from `model`; results with no model draw only
        their outputs.
        """
        return ResultsPlot(self)

    def report(
        self,
        positions: Sequence[float],
        window: float,
        windows: int,
        observed_data: "Iterable[ObservedDataPoint]" = (),
        accumulators: Iterable[str] = (),
        prediction_intervals: "PredictionIntervalsData | None" = None,
    ) -> "SimulationReport":
        """
        Report the observed data, compartment series and accumulators of the run.

        The blocks share one window axis, so the observed values, the windowed
        accumulator increments and the sampled series can be read against
        each other. Add whatever envelope you need around them, such as a run
        label, axis labels or summary figures in the terms of your domain.

        Parameters
        ----------
        positions : Sequence[float]
            Positions the series and cumulative accumulators are sampled at;
            each must be a position of the results.
        window : float
            Length of one window on the axis.
        windows : int
            Number of windows on the observed and windowed axes.
        observed_data : Iterable[ObservedDataPoint], optional
            The calibration targets to report alongside the run.
        accumulators : Iterable[str], optional
            Accumulator ids to report; their outputs come from `model`.
        prediction_intervals : PredictionIntervalsData | None, optional
            Prediction intervals of a conformal method
            (``ConformalResult.intervals``), reported as they are.

        Returns
        -------
        SimulationReport
            Keys ``observed``, ``series`` and ``accumulators``, then
            ``prediction_intervals`` when they are given.

        Raises
        ------
        ValueError
            If the results have no value at a position the report reads, or
            they have no `model` and an accumulator is reported.
        KeyError
            If an accumulator is not a bin or accumulator of the model.
        """
        from commol._simulation.reports import _ReportRuns, _simulation_report

        return _simulation_report(
            _ReportRuns(model=self.model, runs=[self]),
            positions,
            window,
            windows,
            observed_data,
            accumulators,
            prediction_intervals,
        )

    @override
    def compared_values(self) -> object:
        """The values, the positions, the seed and the model."""
        return dict(self), self.positions, self.seed, self.model

    def __repr__(self) -> str:
        """The positions, the seed and the values."""
        seed = "" if self.seed is None else f"seed={self.seed!r}, "
        return (
            f"SimulationResults(positions={self.positions!r}, {seed}"
            f"{dict.__repr__(self)})"
        )
```

### Attributes

#### plot

```python
plot: ResultsPlot
```

The figures of the run: `plot.series()` and `plot.cumulative()`.

They name each series from `model`; results with no model draw only their outputs.

### Methods:

#### __init__

```python
__init__(positions: Iterable[float], values: Mapping[str, Sequence[float]] | None = None, seed: int | None = None, model: Model | None = None) -> None
```

Results at the given positions.

Parameters:

| Name        | Type                              | Description                       | Default                                                  |
| ----------- | --------------------------------- | --------------------------------- | -------------------------------------------------------- |
| `positions` | `Iterable[float]`                 | Positions on the axis, ascending. | *required*                                               |
| `values`    | \`Mapping\[str, Sequence[float]\] | None\`                            | One value per position for each output.                  |
| `seed`      | \`int                             | None\`                            | The seed of the run, or None when it depends on no seed. |
| `model`     | \`Model                           | None\`                            | The model of the run, or None when no model is known.    |

Raises:

| Type         | Description                                        |
| ------------ | -------------------------------------------------- |
| `ValueError` | If an output does not have one value per position. |

Source code in `commol/_simulation/results.py`

```python
def __init__(
    self,
    positions: Iterable[float],
    values: Mapping[str, Sequence[float]] | None = None,
    seed: int | None = None,
    model: "Model | None" = None,
) -> None:
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
    model : Model | None, optional
        The model of the run, or None when no model is known.

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
    self.model = model
    for name, series in self.items():
        if len(series) != len(self.positions):
            raise ValueError(
                f"Output '{name}' has {len(series)} values for "
                f"{len(self.positions)} positions."
            )
```

#### from_rows

```python
from_rows(positions: Sequence[float], names: Sequence[str], rows: Sequence[Sequence[float]], seed: int | None = None, model: Model | None = None) -> SimulationResults
```

Results from one row of output values per position.

Parameters:

| Name        | Type                        | Description                             | Default                                                  |
| ----------- | --------------------------- | --------------------------------------- | -------------------------------------------------------- |
| `positions` | `Sequence[float]`           | Positions on the axis, ascending.       | *required*                                               |
| `names`     | `Sequence[str]`             | Output names, in the order of each row. | *required*                                               |
| `rows`      | `Sequence[Sequence[float]]` | One row per position.                   | *required*                                               |
| `seed`      | \`int                       | None\`                                  | The seed of the run, or None when it depends on no seed. |
| `model`     | \`Model                     | None\`                                  | The model of the run, or None when no model is known.    |

Returns:

| Type                | Description                 |
| ------------------- | --------------------------- |
| `SimulationResults` | One series per output name. |

Source code in `commol/_simulation/results.py`

```python
@classmethod
def from_rows(
    cls,
    positions: Sequence[float],
    names: Sequence[str],
    rows: Sequence[Sequence[float]],
    seed: int | None = None,
    model: "Model | None" = None,
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
    model : Model | None, optional
        The model of the run, or None when no model is known.

    Returns
    -------
    SimulationResults
        One series per output name.
    """
    columns = list(zip(*rows)) if rows else [() for _ in names]
    return cls(
        positions, dict(zip(names, columns, strict=True)), seed=seed, model=model
    )
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

Source code in `commol/_simulation/results.py`

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

#### quantities

```python
quantities(quantities: Sequence[QuantityBase]) -> dict[str, float]
```

The quantities of interest of this run, as a `Propagation` reduces each of its runs.

The run must hold every position the quantities read: the results of `Simulation.run_at(quantity_schedule(quantities))` hold them all. Each position is found within `1e-9`, as an observation finds its row (`index_at` of `crates/commol-calibration/src/observation.rs`), and Rust reduces the rows with the code of `Propagation`, so the values equal those of a `Propagation` of a sample of one draw with the parameters of this run.

On the ODE engine, `Simulation.run` reports a position that is not a breakpoint by interpolation, so a value read from it can differ from that of a run that ends there by about the relative tolerance of the solver, and `PeakPosition` can change on a near-tie.

Parameters:

| Name         | Type                 | Description                                                 | Default    |
| ------------ | -------------------- | ----------------------------------------------------------- | ---------- |
| `quantities` | `Sequence[quantity]` | The quantities; a Trajectory gives one scalar per position. | *required* |

Returns:

| Type               | Description                                            |
| ------------------ | ------------------------------------------------------ |
| `dict[str, float]` | One value per label of the quantities, in their order. |

Raises:

| Type         | Description                                                                                                                                                                                                                                        |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If there is no quantity, two scalars have one label, a quantity names an output the results do not have, or the run lacks a position a quantity reads; the message lists every such position with its quantity and says to call Simulation.run_at. |

Source code in `commol/_simulation/results.py`

```python
def quantities(self, quantities: "Sequence[QuantityBase]") -> dict[str, float]:
    """
    The quantities of interest of this run, as a `Propagation` reduces
    each of its runs.

    The run must hold every position the quantities read: the results of
    ``Simulation.run_at(quantity_schedule(quantities))`` hold them all.
    Each position is found within ``1e-9``, as an observation finds its
    row (`index_at` of `crates/commol-calibration/src/observation.rs`), and
    Rust reduces the rows with the code of `Propagation`, so the values
    equal those of a `Propagation` of a sample of one draw with the
    parameters of this run.

    On the ODE engine, `Simulation.run` reports a position that is not a
    breakpoint by interpolation, so a value read from it can differ from
    that of a run that ends there by about the relative tolerance of the
    solver, and `PeakPosition` can change on a near-tie.

    Parameters
    ----------
    quantities : Sequence[quantity]
        The quantities; a `Trajectory` gives one scalar per position.

    Returns
    -------
    dict[str, float]
        One value per label of the quantities, in their order.

    Raises
    ------
    ValueError
        If there is no quantity, two scalars have one label, a quantity
        names an output the results do not have, or the run lacks a
        position a quantity reads; the message lists every such position
        with its quantity and says to call `Simulation.run_at`.
    """
    return quantities_of_runs([self], self.positions, quantities)[0]
```

#### sample

```python
sample(ids: Iterable[str], at: Iterable[float]) -> list[float]
```

Value of the given bins and accumulators at each position of `at`.

A bin holds an amount at a position, so the value is read at that position, not over the window before it; `windowed_totals` reads the gain of an accumulator over a window. Each position is found with the tolerance of the axis, as an observation finds its row (`index_at` of `crates/commol-calibration/src/observation.rs`), so `0.1 + 0.2` reads the value at `0.3`.

Parameters:

| Name  | Type              | Description                                                                                                                                                                          | Default    |
| ----- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `ids` | `Iterable[str]`   | Bin and accumulator ids, each for all its outputs, and output names, each for itself. Their outputs are summed before they are read. Results without a model read output names only. | *required* |
| `at`  | `Iterable[float]` | Positions to read, each a position of the results.                                                                                                                                   | *required* |

Returns:

| Type          | Description                   |
| ------------- | ----------------------------- |
| `list[float]` | One value per position in at. |

Raises:

| Type         | Description                                                                                          |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| `KeyError`   | If an id is not a bin, accumulator or output of the model, or an output is missing from the results. |
| `ValueError` | If the results have no value at a position in at.                                                    |

Source code in `commol/_simulation/results.py`

```python
def sample(self, ids: Iterable[str], at: Iterable[float]) -> list[float]:
    """
    Value of the given bins and accumulators at each position of `at`.

    A bin holds an amount at a position, so the value is read at that
    position, not over the window before it; `windowed_totals` reads the
    gain of an accumulator over a window. Each position is found with the
    tolerance of the axis, as an observation finds its row (`index_at` of
    `crates/commol-calibration/src/observation.rs`), so ``0.1 + 0.2``
    reads the value at ``0.3``.

    Parameters
    ----------
    ids : Iterable[str]
        Bin and accumulator ids, each for all its outputs, and output
        names, each for itself. Their outputs are summed before they are
        read. Results without a model read output names only.
    at : Iterable[float]
        Positions to read, each a position of the results.

    Returns
    -------
    list[float]
        One value per position in `at`.

    Raises
    ------
    KeyError
        If an id is not a bin, accumulator or output of the model, or an
        output is missing from the results.
    ValueError
        If the results have no value at a position in `at`.
    """
    return _sample(self._total(ids), self.positions, at)
```

#### windowed_totals

```python
windowed_totals(ids: Iterable[str], window: float, at: Iterable[float] | None = None) -> list[float]
```

Amount the given accumulators gained over each window.

Each value is the sum of the outputs at the end of the window minus the sum at its start, `at - window`, each found with the tolerance of the axis: the rule of a windowed observation (`window_increments` of `crates/commol-calibration/src/observation.rs`), so the value is the prediction a calibration compares an observation against when that observation sets `window`.

Parameters:

| Name     | Type              | Description                                                                                                                                                              | Default                                                                                                                                                                                                                               |
| -------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ids`    | `Iterable[str]`   | Accumulator ids, each for all its outputs, and output names, each for itself. Their outputs are summed before windowing. Results without a model read output names only. | *required*                                                                                                                                                                                                                            |
| `window` | `float`           | Length of one window on the axis.                                                                                                                                        | *required*                                                                                                                                                                                                                            |
| `at`     | \`Iterable[float] | None\`                                                                                                                                                                   | Positions at which windows close. Defaults to every complete window of the results (window_ends of the engine, as AxisUnits.window_ends gives it). Pass the observation positions to reproduce exactly the values a calibration used. |

Returns:

| Type          | Description               |
| ------------- | ------------------------- |
| `list[float]` | One increment per window. |

Raises:

| Type         | Description                                                                                                                                                                            |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KeyError`   | If an id is not a bin, accumulator or output of the model, or an output is missing from the results.                                                                                   |
| `ValueError` | If window is not a finite, positive length (the message of the engine), a window starts before position 0, or the results have no value at the end or the start of a requested window. |

Source code in `commol/_simulation/results.py`

```python
def windowed_totals(
    self,
    ids: Iterable[str],
    window: float,
    at: Iterable[float] | None = None,
) -> list[float]:
    """
    Amount the given accumulators gained over each window.

    Each value is the sum of the outputs at the end of the window minus
    the sum at its start, ``at - window``, each found with the tolerance
    of the axis: the rule of a windowed observation (`window_increments`
    of `crates/commol-calibration/src/observation.rs`), so the value is
    the prediction a calibration compares an observation against when
    that observation sets ``window``.

    Parameters
    ----------
    ids : Iterable[str]
        Accumulator ids, each for all its outputs, and output names, each
        for itself. Their outputs are summed before windowing. Results
        without a model read output names only.
    window : float
        Length of one window on the axis.
    at : Iterable[float] | None, optional
        Positions at which windows close. Defaults to every complete
        window of the results (`window_ends` of the engine, as
        `AxisUnits.window_ends` gives it). Pass the observation positions
        to reproduce exactly the values a calibration used.

    Returns
    -------
    list[float]
        One increment per window.

    Raises
    ------
    KeyError
        If an id is not a bin, accumulator or output of the model, or an
        output is missing from the results.
    ValueError
        If `window` is not a finite, positive length (the message of the
        engine), a window starts before position 0, or the results have
        no value at the end or the start of a requested window.
    """
    return _windowed_totals(self._total(ids), self.positions, window, at)
```

#### report

```python
report(positions: Sequence[float], window: float, windows: int, observed_data: Iterable[ObservedDataPoint] = (), accumulators: Iterable[str] = (), prediction_intervals: PredictionIntervalsData | None = None) -> SimulationReport
```

Report the observed data, compartment series and accumulators of the run.

The blocks share one window axis, so the observed values, the windowed accumulator increments and the sampled series can be read against each other. Add whatever envelope you need around them, such as a run label, axis labels or summary figures in the terms of your domain.

Parameters:

| Name                   | Type                          | Description                                                                                              | Default                                                                                       |
| ---------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `positions`            | `Sequence[float]`             | Positions the series and cumulative accumulators are sampled at; each must be a position of the results. | *required*                                                                                    |
| `window`               | `float`                       | Length of one window on the axis.                                                                        | *required*                                                                                    |
| `windows`              | `int`                         | Number of windows on the observed and windowed axes.                                                     | *required*                                                                                    |
| `observed_data`        | `Iterable[ObservedDataPoint]` | The calibration targets to report alongside the run.                                                     | `()`                                                                                          |
| `accumulators`         | `Iterable[str]`               | Accumulator ids to report; their outputs come from model.                                                | `()`                                                                                          |
| `prediction_intervals` | \`PredictionIntervalsData     | None\`                                                                                                   | Prediction intervals of a conformal method (ConformalResult.intervals), reported as they are. |

Returns:

| Type               | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `SimulationReport` | Keys observed, series and accumulators, then prediction_intervals when they are given. |

Raises:

| Type         | Description                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------------------ |
| `ValueError` | If the results have no value at a position the report reads, or they have no model and an accumulator is reported. |
| `KeyError`   | If an accumulator is not a bin or accumulator of the model.                                                        |

Source code in `commol/_simulation/results.py`

```python
def report(
    self,
    positions: Sequence[float],
    window: float,
    windows: int,
    observed_data: "Iterable[ObservedDataPoint]" = (),
    accumulators: Iterable[str] = (),
    prediction_intervals: "PredictionIntervalsData | None" = None,
) -> "SimulationReport":
    """
    Report the observed data, compartment series and accumulators of the run.

    The blocks share one window axis, so the observed values, the windowed
    accumulator increments and the sampled series can be read against
    each other. Add whatever envelope you need around them, such as a run
    label, axis labels or summary figures in the terms of your domain.

    Parameters
    ----------
    positions : Sequence[float]
        Positions the series and cumulative accumulators are sampled at;
        each must be a position of the results.
    window : float
        Length of one window on the axis.
    windows : int
        Number of windows on the observed and windowed axes.
    observed_data : Iterable[ObservedDataPoint], optional
        The calibration targets to report alongside the run.
    accumulators : Iterable[str], optional
        Accumulator ids to report; their outputs come from `model`.
    prediction_intervals : PredictionIntervalsData | None, optional
        Prediction intervals of a conformal method
        (``ConformalResult.intervals``), reported as they are.

    Returns
    -------
    SimulationReport
        Keys ``observed``, ``series`` and ``accumulators``, then
        ``prediction_intervals`` when they are given.

    Raises
    ------
    ValueError
        If the results have no value at a position the report reads, or
        they have no `model` and an accumulator is reported.
    KeyError
        If an accumulator is not a bin or accumulator of the model.
    """
    from commol._simulation.reports import _ReportRuns, _simulation_report

    return _simulation_report(
        _ReportRuns(model=self.model, runs=[self]),
        positions,
        window,
        windows,
        observed_data,
        accumulators,
        prediction_intervals,
    )
```

#### compared_values

```python
compared_values() -> object
```

The values, the positions, the seed and the model.

Source code in `commol/_simulation/results.py`

```python
@override
def compared_values(self) -> object:
    """The values, the positions, the seed and the model."""
    return dict(self), self.positions, self.seed, self.model
```

#### __repr__

```python
__repr__() -> str
```

The positions, the seed and the values.

Source code in `commol/_simulation/results.py`

```python
def __repr__(self) -> str:
    """The positions, the seed and the values."""
    seed = "" if self.seed is None else f"seed={self.seed!r}, "
    return (
        f"SimulationResults(positions={self.positions!r}, {seed}"
        f"{dict.__repr__(self)})"
    )
```

## ResultsPlot

The figures of one run, read from `results.plot`.

Each figure has one panel per series, or per observed series for `observations`, and returns the matplotlib `Figure`. Given `axes=`, the `Figure` of an earlier figure of commol, it draws each series on the panel of that series instead.

Source code in `commol/_simulation/plots.py`

```python
class ResultsPlot:
    """
    The figures of one run, read from ``results.plot``.

    Each figure has one panel per series, or per observed series for
    `observations`, and returns the matplotlib `Figure`. Given ``axes=``, the
    `Figure` of an earlier figure of commol, it draws each series on the
    panel of that series instead.
    """

    def __init__(self, results: "SimulationResults") -> None:
        """
        The figures of `results`.

        Parameters
        ----------
        results : SimulationResults
            The run.
        """
        self._results = results

    def series(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        prediction_intervals: "PredictionIntervalsData | None" = None,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw each series along the axis, one panel per series.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, each drawn as a point on the panel of its label.
            When the observations of a series share one sum of outputs, one
            factor and one scale and have no window, the line covers every
            position of the results. Otherwise the model of each observation
            is read through its own operator (outputs, window, scale, factor)
            at its own position, as the loss reads it, and the line joins
            these values.
        calibration_result : CalibrationResult | None
            Its ``best_parameters`` give the values of the factors of the
            observations, which multiply the line; observed values stay in
            their own units.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        series : list[str] | None
            What to draw, one panel each: an output name (such as ``A_g1``),
            or a bin or accumulator id, which draws the sum of its outputs.
            An exact output name wins over an id. If None, every output.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position on the axis into a tick label. If None, a model
            with an origin (``ModelBuilder(origin=...)``) labels each tick
            with its date: ``2020-01-02``, with the time when it is not
            midnight. Otherwise the ticks show the position.
        tick_every : float | None
            Labels the plotted positions `tick_every` apart, counted from the
            first plotted position. If None, every plotted position, or the
            ticks matplotlib places on a dated axis.
        x_label : str
            The label of the x axis.
        prediction_intervals : PredictionIntervalsData | None
            Prediction intervals of a conformal method
            (``ConformalResult.intervals``). The panel of each of their
            series, by label, shows the band between their lower and upper
            ends and their point prediction, dashed; an unbounded interval
            is left out of the band.
        axes : Figure | None
            A figure of commol to draw on: each series is drawn on its panel,
            the `Axes` whose label is the id of the series. If None, a new
            figure.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the run, such as
            ``linewidth``, ``alpha`` or ``linestyle``.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If an entry of ``series`` is a bin or accumulator id of results
            with no model; the results have no value at the position of an
            observation read on its own or at the start of its window; or
            ``axes`` has no panel of a series.
        KeyError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id of the model.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return run_series(
            self._results,
            output_file=output_file,
            observed_data=observed_data,
            calibration_result=calibration_result,
            config=config,
            series=series,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            prediction_intervals=prediction_intervals,
            axes=axes,
            **kwargs,
        )

    def cumulative(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the running total of each series, one panel per series.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, drawn as their running total on the panel of their
            label. Each observation is divided by its scale and by the value
            of its factor in ``calibration_result``.
        calibration_result : CalibrationResult | None
            Its ``best_parameters`` give the values of the factors. A factor
            it does not have counts as 1, with one warning for each factor.
        config : PlotConfig | None
            Layout, size and style. If None, the defaults.
        series : list[str] | None
            What to draw, one panel each, as in `series`. If None, every
            output.
        axes : Figure | None
            A figure of commol to draw on, as in `series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the run.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a series.
        KeyError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id of the model.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return run_cumulative(
            self._results,
            output_file=output_file,
            observed_data=observed_data,
            calibration_result=calibration_result,
            config=config,
            series=series,
            axes=axes,
            **kwargs,
        )

    def observations(
        self,
        observed_data: "list[ObservedDataPoint]",
        output_file: str | None = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw each observed series with the model read through its
        observation operator, one panel per label.

        The model is read only at the observations: the value of each
        observation through its own operator (the sum of its outputs, its
        window, its scale and its factor), as the loss reads it. A value with
        a window is a total over ``(at - window, at]``, drawn as a step over
        its window with a marker at ``at``. A value with no window is a
        marker, joined to the next only as a guide. The run along the whole
        axis is `series`, on the same figure with ``axes=``.

        Parameters
        ----------
        observed_data : list[ObservedDataPoint]
            The observations. Each label is one panel, in the order of its
            first observation, with the `Axes` label of the label.
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        calibration_result : CalibrationResult | None
            Its ``best_parameters`` give the values of the factors of the
            observations, which multiply the model; observed values stay in
            their own units.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `series`.
        tick_every : float | None
            The distance between labelled observation positions, as in
            `series`.
        x_label : str
            The label of the x axis.
        axes : Figure | None
            A figure of commol to draw on: each observed series is drawn on
            the `Axes` whose label is its label. If None, a new figure.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the steps and the guide of the model,
            such as ``linewidth`` or ``alpha``.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If there are no observations; the results have no value at the
            position of an observation or at the start of its window; or
            ``axes`` has no panel of a label.
        KeyError
            If an observation names an output the results do not have.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return run_observations(
            self._results,
            observed_data,
            output_file=output_file,
            calibration_result=calibration_result,
            config=config,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            axes=axes,
            **kwargs,
        )

    def _panel_model(self) -> "Model | None":
        """The model of the run."""
        return self._results.model

    def _panel_outputs(self) -> PanelOutputs:
        """The outputs of the run."""
        return PanelOutputs(self._results, self._results.model)

    def _series_panels(
        self,
        confidence_level: float,  # noqa: ARG002
        observed_data: "list[ObservedDataPoint]",
        series: list[str],
    ) -> list[Panel]:
        """The panels of `series`, as `series` draws them."""
        return run_panels(self._results, observed_data, series=series)

    def _window_trace(
        self,
        outputs: PanelOutputs,
        series_id: str,
        confidence_level: float,  # noqa: ARG002
        window: float,
    ) -> tuple[Trace, Band | None]:
        """The gain of a series over each complete window."""
        return run_window(self._results, outputs, series_id, window)

    def _observation_panels(
        self,
        confidence_level: float,  # noqa: ARG002
        observed_data: "list[ObservedDataPoint]",
    ) -> list[ObservationPanel]:
        """The panels of the observed series, as `observations` draws them."""
        return run_observation_panels(self._results, observed_data)
```

### Methods:

#### __init__

```python
__init__(results: SimulationResults) -> None
```

The figures of `results`.

Parameters:

| Name      | Type                | Description | Default    |
| --------- | ------------------- | ----------- | ---------- |
| `results` | `SimulationResults` | The run.    | *required* |

Source code in `commol/_simulation/plots.py`

```python
def __init__(self, results: "SimulationResults") -> None:
    """
    The figures of `results`.

    Parameters
    ----------
    results : SimulationResults
        The run.
    """
    self._results = results
```

#### series

```python
series(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, series: list[str] | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', prediction_intervals: PredictionIntervalsData | None = None, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw each series along the axis, one panel per series.

Parameters:

| Name                   | Type                       | Description                          | Default                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------- | -------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `output_file`          | \`str                      | None\`                               | Path to save the figure to. If None, the figure is only returned.                                                                                                                                                                                                                                                                                                                                            |
| `observed_data`        | \`list[ObservedDataPoint]  | None\`                               | Observations, each drawn as a point on the panel of its label. When the observations of a series share one sum of outputs, one factor and one scale and have no window, the line covers every position of the results. Otherwise the model of each observation is read through its own operator (outputs, window, scale, factor) at its own position, as the loss reads it, and the line joins these values. |
| `calibration_result`   | \`CalibrationResult        | None\`                               | Its best_parameters give the values of the factors of the observations, which multiply the line; observed values stay in their own units.                                                                                                                                                                                                                                                                    |
| `config`               | \`PlotConfig               | None\`                               | Layout and style. If None, the defaults.                                                                                                                                                                                                                                                                                                                                                                     |
| `series`               | \`list[str]                | None\`                               | What to draw, one panel each: an output name (such as A_g1), or a bin or accumulator id, which draws the sum of its outputs. An exact output name wins over an id. If None, every output.                                                                                                                                                                                                                    |
| `show_legend`          | `bool`                     | Whether each panel shows its legend. | `True`                                                                                                                                                                                                                                                                                                                                                                                                       |
| `position_label`       | \`Callable\[[float], str\] | None\`                               | Turns a position on the axis into a tick label. If None, a model with an origin (ModelBuilder(origin=...)) labels each tick with its date: 2020-01-02, with the time when it is not midnight. Otherwise the ticks show the position.                                                                                                                                                                         |
| `tick_every`           | \`float                    | None\`                               | Labels the plotted positions tick_every apart, counted from the first plotted position. If None, every plotted position, or the ticks matplotlib places on a dated axis.                                                                                                                                                                                                                                     |
| `x_label`              | `str`                      | The label of the x axis.             | `'Step'`                                                                                                                                                                                                                                                                                                                                                                                                     |
| `prediction_intervals` | \`PredictionIntervalsData  | None\`                               | Prediction intervals of a conformal method (ConformalResult.intervals). The panel of each of their series, by label, shows the band between their lower and upper ends and their point prediction, dashed; an unbounded interval is left out of the band.                                                                                                                                                    |
| `axes`                 | \`Figure                   | None\`                               | A figure of commol to draw on: each series is drawn on its panel, the Axes whose label is the id of the series. If None, a new figure.                                                                                                                                                                                                                                                                       |
| `**kwargs`             | \`str                      | int                                  | float                                                                                                                                                                                                                                                                                                                                                                                                        |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`  | If an entry of series is a bin or accumulator id of results with no model; the results have no value at the position of an observation read on its own or at the start of its window; or axes has no panel of a series. |
| `KeyError`    | If an entry of series is neither an output nor a bin or accumulator id of the model.                                                                                                                                    |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                                                                                                            |

Source code in `commol/_simulation/plots.py`

```python
def series(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervalsData | None" = None,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw each series along the axis, one panel per series.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, each drawn as a point on the panel of its label.
        When the observations of a series share one sum of outputs, one
        factor and one scale and have no window, the line covers every
        position of the results. Otherwise the model of each observation
        is read through its own operator (outputs, window, scale, factor)
        at its own position, as the loss reads it, and the line joins
        these values.
    calibration_result : CalibrationResult | None
        Its ``best_parameters`` give the values of the factors of the
        observations, which multiply the line; observed values stay in
        their own units.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    series : list[str] | None
        What to draw, one panel each: an output name (such as ``A_g1``),
        or a bin or accumulator id, which draws the sum of its outputs.
        An exact output name wins over an id. If None, every output.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position on the axis into a tick label. If None, a model
        with an origin (``ModelBuilder(origin=...)``) labels each tick
        with its date: ``2020-01-02``, with the time when it is not
        midnight. Otherwise the ticks show the position.
    tick_every : float | None
        Labels the plotted positions `tick_every` apart, counted from the
        first plotted position. If None, every plotted position, or the
        ticks matplotlib places on a dated axis.
    x_label : str
        The label of the x axis.
    prediction_intervals : PredictionIntervalsData | None
        Prediction intervals of a conformal method
        (``ConformalResult.intervals``). The panel of each of their
        series, by label, shows the band between their lower and upper
        ends and their point prediction, dashed; an unbounded interval
        is left out of the band.
    axes : Figure | None
        A figure of commol to draw on: each series is drawn on its panel,
        the `Axes` whose label is the id of the series. If None, a new
        figure.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the run, such as
        ``linewidth``, ``alpha`` or ``linestyle``.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If an entry of ``series`` is a bin or accumulator id of results
        with no model; the results have no value at the position of an
        observation read on its own or at the start of its window; or
        ``axes`` has no panel of a series.
    KeyError
        If an entry of ``series`` is neither an output nor a bin or
        accumulator id of the model.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    return run_series(
        self._results,
        output_file=output_file,
        observed_data=observed_data,
        calibration_result=calibration_result,
        config=config,
        series=series,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        prediction_intervals=prediction_intervals,
        axes=axes,
        **kwargs,
    )
```

#### cumulative

```python
cumulative(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, series: list[str] | None = None, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the running total of each series, one panel per series.

Parameters:

| Name                 | Type                      | Description | Default                                                                                                                                                                |
| -------------------- | ------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_file`        | \`str                     | None\`      | Path to save the figure to. If None, the figure is only returned.                                                                                                      |
| `observed_data`      | \`list[ObservedDataPoint] | None\`      | Observations, drawn as their running total on the panel of their label. Each observation is divided by its scale and by the value of its factor in calibration_result. |
| `calibration_result` | \`CalibrationResult       | None\`      | Its best_parameters give the values of the factors. A factor it does not have counts as 1, with one warning for each factor.                                           |
| `config`             | \`PlotConfig              | None\`      | Layout, size and style. If None, the defaults.                                                                                                                         |
| `series`             | \`list[str]               | None\`      | What to draw, one panel each, as in series. If None, every output.                                                                                                     |
| `axes`               | \`Figure                  | None\`      | A figure of commol to draw on, as in series.                                                                                                                           |
| `**kwargs`           | \`str                     | int         | float                                                                                                                                                                  |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                          |
| ------------- | ------------------------------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a series.                                                    |
| `KeyError`    | If an entry of series is neither an output nor a bin or accumulator id of the model. |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                         |

Source code in `commol/_simulation/plots.py`

```python
def cumulative(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the running total of each series, one panel per series.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, drawn as their running total on the panel of their
        label. Each observation is divided by its scale and by the value
        of its factor in ``calibration_result``.
    calibration_result : CalibrationResult | None
        Its ``best_parameters`` give the values of the factors. A factor
        it does not have counts as 1, with one warning for each factor.
    config : PlotConfig | None
        Layout, size and style. If None, the defaults.
    series : list[str] | None
        What to draw, one panel each, as in `series`. If None, every
        output.
    axes : Figure | None
        A figure of commol to draw on, as in `series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the run.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a series.
    KeyError
        If an entry of ``series`` is neither an output nor a bin or
        accumulator id of the model.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    return run_cumulative(
        self._results,
        output_file=output_file,
        observed_data=observed_data,
        calibration_result=calibration_result,
        config=config,
        series=series,
        axes=axes,
        **kwargs,
    )
```

#### observations

```python
observations(observed_data: list[ObservedDataPoint], output_file: str | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw each observed series with the model read through its observation operator, one panel per label.

The model is read only at the observations: the value of each observation through its own operator (the sum of its outputs, its window, its scale and its factor), as the loss reads it. A value with a window is a total over `(at - window, at]`, drawn as a step over its window with a marker at `at`. A value with no window is a marker, joined to the next only as a guide. The run along the whole axis is `series`, on the same figure with `axes=`.

Parameters:

| Name                 | Type                       | Description                                                                                                         | Default                                                                                                                                    |
| -------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `observed_data`      | `list[ObservedDataPoint]`  | The observations. Each label is one panel, in the order of its first observation, with the Axes label of the label. | *required*                                                                                                                                 |
| `output_file`        | \`str                      | None\`                                                                                                              | Path to save the figure to. If None, the figure is only returned.                                                                          |
| `calibration_result` | \`CalibrationResult        | None\`                                                                                                              | Its best_parameters give the values of the factors of the observations, which multiply the model; observed values stay in their own units. |
| `config`             | \`PlotConfig               | None\`                                                                                                              | Layout and style. If None, the defaults.                                                                                                   |
| `show_legend`        | `bool`                     | Whether each panel shows its legend.                                                                                | `True`                                                                                                                                     |
| `position_label`     | \`Callable\[[float], str\] | None\`                                                                                                              | Turns a position into a tick label, as in series.                                                                                          |
| `tick_every`         | \`float                    | None\`                                                                                                              | The distance between labelled observation positions, as in series.                                                                         |
| `x_label`            | `str`                      | The label of the x axis.                                                                                            | `'Step'`                                                                                                                                   |
| `axes`               | \`Figure                   | None\`                                                                                                              | A figure of commol to draw on: each observed series is drawn on the Axes whose label is its label. If None, a new figure.                  |
| `**kwargs`           | \`str                      | int                                                                                                                 | float                                                                                                                                      |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`  | If there are no observations; the results have no value at the position of an observation or at the start of its window; or axes has no panel of a label. |
| `KeyError`    | If an observation names an output the results do not have.                                                                                                |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                                              |

Source code in `commol/_simulation/plots.py`

```python
def observations(
    self,
    observed_data: "list[ObservedDataPoint]",
    output_file: str | None = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw each observed series with the model read through its
    observation operator, one panel per label.

    The model is read only at the observations: the value of each
    observation through its own operator (the sum of its outputs, its
    window, its scale and its factor), as the loss reads it. A value with
    a window is a total over ``(at - window, at]``, drawn as a step over
    its window with a marker at ``at``. A value with no window is a
    marker, joined to the next only as a guide. The run along the whole
    axis is `series`, on the same figure with ``axes=``.

    Parameters
    ----------
    observed_data : list[ObservedDataPoint]
        The observations. Each label is one panel, in the order of its
        first observation, with the `Axes` label of the label.
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    calibration_result : CalibrationResult | None
        Its ``best_parameters`` give the values of the factors of the
        observations, which multiply the model; observed values stay in
        their own units.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position into a tick label, as in `series`.
    tick_every : float | None
        The distance between labelled observation positions, as in
        `series`.
    x_label : str
        The label of the x axis.
    axes : Figure | None
        A figure of commol to draw on: each observed series is drawn on
        the `Axes` whose label is its label. If None, a new figure.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the steps and the guide of the model,
        such as ``linewidth`` or ``alpha``.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If there are no observations; the results have no value at the
        position of an observation or at the start of its window; or
        ``axes`` has no panel of a label.
    KeyError
        If an observation names an output the results do not have.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    return run_observations(
        self._results,
        observed_data,
        output_file=output_file,
        calibration_result=calibration_result,
        config=config,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        axes=axes,
        **kwargs,
    )
```

## ReplicateRuns

Runs replicates of a stochastic model; `Simulation.replicates` makes it.

Attributes:

| Name         | Type         | Description                                    |
| ------------ | ------------ | ---------------------------------------------- |
| `simulation` | `Simulation` | The simulation whose model the replicates run. |
| `count`      | `int`        | How many replicates.                           |
| `seed`       | \`int        | None\`                                         |

Source code in `commol/_simulation/simulation.py`

```python
class ReplicateRuns:
    """
    Runs replicates of a stochastic model; `Simulation.replicates` makes it.

    Attributes
    ----------
    simulation : Simulation
        The simulation whose model the replicates run.
    count : int
        How many replicates.
    seed : int | None
        The seed the replicate seeds derive from; None draws one at each run,
        which the replicates keep (`SimulationReplicates.seed`).
    """

    def __init__(self, simulation: Simulation, count: int, seed: int | None) -> None:
        """
        The replicates of a simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation whose model the replicates run.
        count : int
            How many replicates.
        seed : int | None
            The seed the replicate seeds derive from; None draws one at each
            run.

        Raises
        ------
        ValueError
            With the message of the engine, if ``count`` is below 1 or the
            seed is outside its range.
        """
        commol_rs.core.check_ranges("ReplicateRuns", {"count": count, "seed": seed})
        self.simulation = simulation
        self.count = count
        self.seed = seed

    def _replicates(self, grid: int | list[float]) -> SimulationReplicates:
        """The replicates on ``grid``: a number of steps or the positions."""
        simulation = self.simulation
        simulation._validate_all_parameters_calibrated(simulation.model_definition)
        engine = simulation.engine
        seed, positions, seeds, runs = engine.run_replicates(
            self.seed, self.count, grid
        )
        outputs = simulation.simulation_outputs
        model = simulation.model_definition
        return SimulationReplicates(
            (
                SimulationResults.from_rows(
                    positions, outputs, rows, seed=replicate_seed, model=model
                )
                for replicate_seed, rows in zip(seeds, runs, strict=True)
            ),
            seed=seed,
            positions=positions,
            model=model,
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
        return self._replicates(num_steps)

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
        return self._replicates([float(position) for position in at])
```

### Methods:

#### __init__

```python
__init__(simulation: Simulation, count: int, seed: int | None) -> None
```

The replicates of a simulation.

Parameters:

| Name         | Type         | Description                                    | Default                                                               |
| ------------ | ------------ | ---------------------------------------------- | --------------------------------------------------------------------- |
| `simulation` | `Simulation` | The simulation whose model the replicates run. | *required*                                                            |
| `count`      | `int`        | How many replicates.                           | *required*                                                            |
| `seed`       | \`int        | None\`                                         | The seed the replicate seeds derive from; None draws one at each run. |

Raises:

| Type         | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| `ValueError` | With the message of the engine, if count is below 1 or the seed is outside its range. |

Source code in `commol/_simulation/simulation.py`

```python
def __init__(self, simulation: Simulation, count: int, seed: int | None) -> None:
    """
    The replicates of a simulation.

    Parameters
    ----------
    simulation : Simulation
        The simulation whose model the replicates run.
    count : int
        How many replicates.
    seed : int | None
        The seed the replicate seeds derive from; None draws one at each
        run.

    Raises
    ------
    ValueError
        With the message of the engine, if ``count`` is below 1 or the
        seed is outside its range.
    """
    commol_rs.core.check_ranges("ReplicateRuns", {"count": count, "seed": seed})
    self.simulation = simulation
    self.count = count
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

Source code in `commol/_simulation/simulation.py`

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
    return self._replicates(num_steps)
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

Source code in `commol/_simulation/simulation.py`

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
    return self._replicates([float(position) for position in at])
```

## SimulationReplicates

Bases: `ComparedByValue`, `list[SimulationResults]`

Replicates of a run of a stochastic engine, one `SimulationResults` each.

A `list` of runs at the same positions, each with its own seed: `replicates[i].seed` repeats replicate `i`. The statistics reduce the replicates position by position, and `member_statistics` reduces them to a mean, a median, the extremes and an interval.

Two replicates are equal when their runs, their seed, their positions and their model are (`ComparedByValue`); a plain `list` of the same runs is not equal to them. They have no hash.

Attributes:

| Name        | Type          | Description                                       |
| ----------- | ------------- | ------------------------------------------------- |
| `seed`      | `int`         | The seed the replicate seeds derive from.         |
| `positions` | `list[float]` | Positions on the axis, shared by every replicate. |
| `model`     | \`Model       | None\`                                            |

Source code in `commol/_simulation/results.py`

```python
class SimulationReplicates(ComparedByValue, list[SimulationResults]):
    """
    Replicates of a run of a stochastic engine, one `SimulationResults` each.

    A ``list`` of runs at the same positions, each with its own seed:
    ``replicates[i].seed`` repeats replicate ``i``. The statistics reduce the
    replicates position by position, and `member_statistics` reduces them to a
    mean, a median, the extremes and an interval.

    Two replicates are equal when their runs, their seed, their positions and
    their model are (`ComparedByValue`); a plain ``list`` of the same runs is
    not equal to them. They have no hash.

    Attributes
    ----------
    seed : int
        The seed the replicate seeds derive from.
    positions : list[float]
        Positions on the axis, shared by every replicate.
    model : Model | None
        The model of the replicates, which `Simulation.replicates` sets; None
        for replicates built by hand.
    """

    seed: int
    positions: list[float]
    model: "Model | None"

    def __init__(
        self,
        runs: Iterable[SimulationResults],
        seed: int,
        positions: Sequence[float],
        model: "Model | None" = None,
    ) -> None:
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
        model : Model | None, optional
            The model of the replicates, or None when no model is known.

        Raises
        ------
        ValueError
            If a run is not at `positions`.
        """
        super().__init__(runs)
        self.seed = seed
        self.positions = [float(position) for position in positions]
        self.model = model
        for index, run in enumerate(self):
            if run.positions != self.positions:
                raise ValueError(
                    f"Replicate {index} is at other positions than the replicates."
                )

    @property
    def seeds(self) -> list[int]:
        """The seed of each replicate, in order."""
        return [run.seed for run in self if run.seed is not None]

    @property
    def plot(self) -> ReplicatesPlot:
        """
        The figures of the replicates: ``plot.series()``, their mean and band,
        and ``plot.cumulative()``.
        """
        return ReplicatesPlot(self)

    def _values(self, name: str) -> np.ndarray:
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
            model=self.model,
        )

    def quantiles(self, levels: Sequence[float]) -> dict[float, SimulationResults]:
        """
        Quantiles of the replicates at each position, for each output.

        The quantile is the one convention of commol (`weighted_quantiles`
        with equal weights, NumPy's ``method="hazen"``).

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
        commol_rs.inference.check_levels(list(levels))
        if not self:
            raise ValueError("There are no replicates.")
        names = list(self[0])
        per_name = {
            name: [
                quantiles(column.tolist(), levels) for column in self._values(name).T
            ]
            for name in names
        }
        return {
            level: SimulationResults(
                self.positions,
                {
                    name: [at_position[index] for at_position in per_name[name]]
                    for name in names
                },
                model=self.model,
            )
            for index, level in enumerate(levels)
        }

    def quantities(
        self, quantities: "Sequence[QuantityBase]"
    ) -> list[dict[str, float]]:
        """
        The quantities of interest of each replicate, as
        `SimulationResults.quantities` reads one run.

        Each replicate is reduced on its own, with no mean: a summary across
        the replicates is a reduction of these values, such as
        `member_statistics`.

        A stochastic `Propagation` gives draw ``j`` the seed
        ``draw_seeds[j]`` and runs its replicate ``i`` with the seed replicate
        ``i`` of ``Simulation.replicates(count, seed=draw_seeds[j])`` takes. So
        those replicates, of the model at the values of draw ``j``, give the
        quantities of the runs of that draw.

        Parameters
        ----------
        quantities : Sequence[quantity]
            The quantities; a `Trajectory` gives one scalar per position.

        Returns
        -------
        list[dict[str, float]]
            One dict per replicate, in seed order, with one value per label of
            the quantities.

        Raises
        ------
        ValueError
            If there are no replicates, there is no quantity, two scalars have
            one label, a quantity names an output the replicates do not have,
            or they lack a position a quantity reads.
        """
        if not self:
            raise ValueError("There are no replicates.")
        return quantities_of_runs(self, self.positions, quantities)

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
        if self and output not in self[0]:
            raise KeyError(
                f"'{output}' is not an output of the replicates. "
                f"Outputs: {sorted(self[0])}"
            )
        return spread([run[output] for run in self], confidence_level)

    def report(
        self,
        confidence_level: float,
        positions: Sequence[float],
        window: float,
        windows: int,
        observed_data: "Iterable[ObservedDataPoint]" = (),
        accumulators: Iterable[str] = (),
        prediction_intervals: "PredictionIntervalsData | None" = None,
    ) -> "SimulationReport":
        """
        Report the replicates as `SimulationResults.report` reports one run.

        Every number becomes a ``{mean, median, ci_lower, ci_upper, min,
        max}`` block, reduced replicate by replicate; the layout is the one of
        the report of a run.

        Parameters
        ----------
        confidence_level : float
            Confidence level of the reported intervals, strictly between 0
            and 1.
        positions : Sequence[float]
            Positions the series and cumulative accumulators are sampled at;
            each must be a position of the replicates.
        window : float
            Length of one window on the axis.
        windows : int
            Number of windows on the observed and windowed axes.
        observed_data : Iterable[ObservedDataPoint], optional
            The calibration targets to report alongside the replicates.
        accumulators : Iterable[str], optional
            Accumulator ids to report; their outputs come from `model`.
        prediction_intervals : PredictionIntervalsData | None, optional
            Prediction intervals of a conformal method
            (``ConformalResult.intervals``), reported as they are.

        Returns
        -------
        SimulationReport
            Keys ``observed``, ``series`` and ``accumulators``, then
            ``prediction_intervals`` when they are given.

        Raises
        ------
        ValueError
            If there are no replicates, the confidence level is out of range,
            the replicates have no value at a position the report reads, or
            they have no `model` and an accumulator is reported.
        KeyError
            If an accumulator is not a bin or accumulator of the model.
        """
        from commol._simulation.reports import _ReportRuns, _simulation_report

        if not self:
            raise ValueError("There are no replicates.")
        return _simulation_report(
            _ReportRuns(
                model=self.model,
                runs=list(self),
                confidence_level=confidence_level,
            ),
            positions,
            window,
            windows,
            observed_data,
            accumulators,
            prediction_intervals,
        )

    @override
    def compared_values(self) -> object:
        """The runs, the seed, the positions and the model."""
        return list(self), self.seed, self.positions, self.model

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

#### plot

```python
plot: ReplicatesPlot
```

The figures of the replicates: `plot.series()`, their mean and band, and `plot.cumulative()`.

### Methods:

#### __init__

```python
__init__(runs: Iterable[SimulationResults], seed: int, positions: Sequence[float], model: Model | None = None) -> None
```

Replicates of one run.

Parameters:

| Name        | Type                          | Description                               | Default                                                      |
| ----------- | ----------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| `runs`      | `Iterable[SimulationResults]` | One run per replicate, at positions.      | *required*                                                   |
| `seed`      | `int`                         | The seed the replicate seeds derive from. | *required*                                                   |
| `positions` | `Sequence[float]`             | Positions on the axis, ascending.         | *required*                                                   |
| `model`     | \`Model                       | None\`                                    | The model of the replicates, or None when no model is known. |

Raises:

| Type         | Description                   |
| ------------ | ----------------------------- |
| `ValueError` | If a run is not at positions. |

Source code in `commol/_simulation/results.py`

```python
def __init__(
    self,
    runs: Iterable[SimulationResults],
    seed: int,
    positions: Sequence[float],
    model: "Model | None" = None,
) -> None:
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
    model : Model | None, optional
        The model of the replicates, or None when no model is known.

    Raises
    ------
    ValueError
        If a run is not at `positions`.
    """
    super().__init__(runs)
    self.seed = seed
    self.positions = [float(position) for position in positions]
    self.model = model
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

Source code in `commol/_simulation/results.py`

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
        model=self.model,
    )
```

#### quantiles

```python
quantiles(levels: Sequence[float]) -> dict[float, SimulationResults]
```

Quantiles of the replicates at each position, for each output.

The quantile is the one convention of commol (`weighted_quantiles` with equal weights, NumPy's `method="hazen"`).

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

Source code in `commol/_simulation/results.py`

```python
def quantiles(self, levels: Sequence[float]) -> dict[float, SimulationResults]:
    """
    Quantiles of the replicates at each position, for each output.

    The quantile is the one convention of commol (`weighted_quantiles`
    with equal weights, NumPy's ``method="hazen"``).

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
    commol_rs.inference.check_levels(list(levels))
    if not self:
        raise ValueError("There are no replicates.")
    names = list(self[0])
    per_name = {
        name: [
            quantiles(column.tolist(), levels) for column in self._values(name).T
        ]
        for name in names
    }
    return {
        level: SimulationResults(
            self.positions,
            {
                name: [at_position[index] for at_position in per_name[name]]
                for name in names
            },
            model=self.model,
        )
        for index, level in enumerate(levels)
    }
```

#### quantities

```python
quantities(quantities: Sequence[QuantityBase]) -> list[dict[str, float]]
```

The quantities of interest of each replicate, as `SimulationResults.quantities` reads one run.

Each replicate is reduced on its own, with no mean: a summary across the replicates is a reduction of these values, such as `member_statistics`.

A stochastic `Propagation` gives draw `j` the seed `draw_seeds[j]` and runs its replicate `i` with the seed replicate `i` of `Simulation.replicates(count, seed=draw_seeds[j])` takes. So those replicates, of the model at the values of draw `j`, give the quantities of the runs of that draw.

Parameters:

| Name         | Type                 | Description                                                 | Default    |
| ------------ | -------------------- | ----------------------------------------------------------- | ---------- |
| `quantities` | `Sequence[quantity]` | The quantities; a Trajectory gives one scalar per position. | *required* |

Returns:

| Type                     | Description                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `list[dict[str, float]]` | One dict per replicate, in seed order, with one value per label of the quantities. |

Raises:

| Type         | Description                                                                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ValueError` | If there are no replicates, there is no quantity, two scalars have one label, a quantity names an output the replicates do not have, or they lack a position a quantity reads. |

Source code in `commol/_simulation/results.py`

```python
def quantities(
    self, quantities: "Sequence[QuantityBase]"
) -> list[dict[str, float]]:
    """
    The quantities of interest of each replicate, as
    `SimulationResults.quantities` reads one run.

    Each replicate is reduced on its own, with no mean: a summary across
    the replicates is a reduction of these values, such as
    `member_statistics`.

    A stochastic `Propagation` gives draw ``j`` the seed
    ``draw_seeds[j]`` and runs its replicate ``i`` with the seed replicate
    ``i`` of ``Simulation.replicates(count, seed=draw_seeds[j])`` takes. So
    those replicates, of the model at the values of draw ``j``, give the
    quantities of the runs of that draw.

    Parameters
    ----------
    quantities : Sequence[quantity]
        The quantities; a `Trajectory` gives one scalar per position.

    Returns
    -------
    list[dict[str, float]]
        One dict per replicate, in seed order, with one value per label of
        the quantities.

    Raises
    ------
    ValueError
        If there are no replicates, there is no quantity, two scalars have
        one label, a quantity names an output the replicates do not have,
        or they lack a position a quantity reads.
    """
    if not self:
        raise ValueError("There are no replicates.")
    return quantities_of_runs(self, self.positions, quantities)
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

Source code in `commol/_simulation/results.py`

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
    if self and output not in self[0]:
        raise KeyError(
            f"'{output}' is not an output of the replicates. "
            f"Outputs: {sorted(self[0])}"
        )
    return spread([run[output] for run in self], confidence_level)
```

#### report

```python
report(confidence_level: float, positions: Sequence[float], window: float, windows: int, observed_data: Iterable[ObservedDataPoint] = (), accumulators: Iterable[str] = (), prediction_intervals: PredictionIntervalsData | None = None) -> SimulationReport
```

Report the replicates as `SimulationResults.report` reports one run.

Every number becomes a `{mean, median, ci_lower, ci_upper, min, max}` block, reduced replicate by replicate; the layout is the one of the report of a run.

Parameters:

| Name                   | Type                          | Description                                                                                                 | Default                                                                                       |
| ---------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `confidence_level`     | `float`                       | Confidence level of the reported intervals, strictly between 0 and 1.                                       | *required*                                                                                    |
| `positions`            | `Sequence[float]`             | Positions the series and cumulative accumulators are sampled at; each must be a position of the replicates. | *required*                                                                                    |
| `window`               | `float`                       | Length of one window on the axis.                                                                           | *required*                                                                                    |
| `windows`              | `int`                         | Number of windows on the observed and windowed axes.                                                        | *required*                                                                                    |
| `observed_data`        | `Iterable[ObservedDataPoint]` | The calibration targets to report alongside the replicates.                                                 | `()`                                                                                          |
| `accumulators`         | `Iterable[str]`               | Accumulator ids to report; their outputs come from model.                                                   | `()`                                                                                          |
| `prediction_intervals` | \`PredictionIntervalsData     | None\`                                                                                                      | Prediction intervals of a conformal method (ConformalResult.intervals), reported as they are. |

Returns:

| Type               | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `SimulationReport` | Keys observed, series and accumulators, then prediction_intervals when they are given. |

Raises:

| Type         | Description                                                                                                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ValueError` | If there are no replicates, the confidence level is out of range, the replicates have no value at a position the report reads, or they have no model and an accumulator is reported. |
| `KeyError`   | If an accumulator is not a bin or accumulator of the model.                                                                                                                          |

Source code in `commol/_simulation/results.py`

```python
def report(
    self,
    confidence_level: float,
    positions: Sequence[float],
    window: float,
    windows: int,
    observed_data: "Iterable[ObservedDataPoint]" = (),
    accumulators: Iterable[str] = (),
    prediction_intervals: "PredictionIntervalsData | None" = None,
) -> "SimulationReport":
    """
    Report the replicates as `SimulationResults.report` reports one run.

    Every number becomes a ``{mean, median, ci_lower, ci_upper, min,
    max}`` block, reduced replicate by replicate; the layout is the one of
    the report of a run.

    Parameters
    ----------
    confidence_level : float
        Confidence level of the reported intervals, strictly between 0
        and 1.
    positions : Sequence[float]
        Positions the series and cumulative accumulators are sampled at;
        each must be a position of the replicates.
    window : float
        Length of one window on the axis.
    windows : int
        Number of windows on the observed and windowed axes.
    observed_data : Iterable[ObservedDataPoint], optional
        The calibration targets to report alongside the replicates.
    accumulators : Iterable[str], optional
        Accumulator ids to report; their outputs come from `model`.
    prediction_intervals : PredictionIntervalsData | None, optional
        Prediction intervals of a conformal method
        (``ConformalResult.intervals``), reported as they are.

    Returns
    -------
    SimulationReport
        Keys ``observed``, ``series`` and ``accumulators``, then
        ``prediction_intervals`` when they are given.

    Raises
    ------
    ValueError
        If there are no replicates, the confidence level is out of range,
        the replicates have no value at a position the report reads, or
        they have no `model` and an accumulator is reported.
    KeyError
        If an accumulator is not a bin or accumulator of the model.
    """
    from commol._simulation.reports import _ReportRuns, _simulation_report

    if not self:
        raise ValueError("There are no replicates.")
    return _simulation_report(
        _ReportRuns(
            model=self.model,
            runs=list(self),
            confidence_level=confidence_level,
        ),
        positions,
        window,
        windows,
        observed_data,
        accumulators,
        prediction_intervals,
    )
```

#### compared_values

```python
compared_values() -> object
```

The runs, the seed, the positions and the model.

Source code in `commol/_simulation/results.py`

```python
@override
def compared_values(self) -> object:
    """The runs, the seed, the positions and the model."""
    return list(self), self.seed, self.positions, self.model
```

#### __repr__

```python
__repr__() -> str
```

The number of replicates, their seed and their positions.

Source code in `commol/_simulation/results.py`

```python
def __repr__(self) -> str:
    """The number of replicates, their seed and their positions."""
    return (
        f"SimulationReplicates({len(self)} replicates, seed={self.seed!r}, "
        f"positions={self.positions!r})"
    )
```

## ReplicatesPlot

The figures of the replicates of a stochastic engine, read from `replicates.plot`.

The figures are those of `ResultsPlot` for the mean of the replicates; `series` adds the band of the replicates.

Source code in `commol/_simulation/plots.py`

```python
class ReplicatesPlot:
    """
    The figures of the replicates of a stochastic engine, read from
    ``replicates.plot``.

    The figures are those of `ResultsPlot` for the mean of the replicates;
    `series` adds the band of the replicates.
    """

    def __init__(self, replicates: "SimulationReplicates") -> None:
        """
        The figures of `replicates`.

        Parameters
        ----------
        replicates : SimulationReplicates
            The replicates.
        """
        self._replicates = replicates

    def series(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        prediction_intervals: "PredictionIntervalsData | None" = None,
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the mean of each series and the band of the replicates, one
        panel per series.

        Each replicate is reduced to the series first, through the operator
        of its observations, then the band takes the interval of
        `confidence_level` across them (`member_statistics`).

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, as in `ResultsPlot.series`.
        calibration_result : CalibrationResult | None
            The values of the factors, as in `ResultsPlot.series`.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        series : list[str] | None
            What to draw, as in `ResultsPlot.series`.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `ResultsPlot.series`.
        tick_every : float | None
            The distance between labelled positions, as in
            `ResultsPlot.series`.
        x_label : str
            The label of the x axis.
        prediction_intervals : PredictionIntervalsData | None
            Prediction intervals of a conformal method, as in
            `ResultsPlot.series`.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the mean.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            As `ResultsPlot.series`; or if there are no replicates or the
            confidence level is out of range.
        KeyError
            As `ResultsPlot.series`.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return replicates_series(
            self._replicates,
            output_file=output_file,
            observed_data=observed_data,
            calibration_result=calibration_result,
            config=config,
            series=series,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            prediction_intervals=prediction_intervals,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )

    def cumulative(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the running total of the mean of the replicates, as
        `ResultsPlot.cumulative` draws one run.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, as in `ResultsPlot.cumulative`.
        calibration_result : CalibrationResult | None
            The values of the factors, as in `ResultsPlot.cumulative`.
        config : PlotConfig | None
            Layout, size and style. If None, the defaults.
        series : list[str] | None
            What to draw, as in `ResultsPlot.series`.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the mean.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            As `ResultsPlot.cumulative`; or if there are no replicates.
        KeyError
            As `ResultsPlot.cumulative`.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return replicates_cumulative(
            self._replicates,
            output_file=output_file,
            observed_data=observed_data,
            calibration_result=calibration_result,
            config=config,
            series=series,
            axes=axes,
            **kwargs,
        )

    def observations(
        self,
        observed_data: "list[ObservedDataPoint]",
        output_file: str | None = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw each observed series with the mean of the replicates and their
        band, read through its observation operator, one panel per label.

        Each replicate is read through the operator of each observation
        first, as `ResultsPlot.observations` reads a run, then the band takes
        the interval of `confidence_level` across them
        (`member_statistics`). A band over a window is a rectangle over that
        window. The band is a band of the model, with no observation noise,
        not a prediction interval, and its legend says so.

        Parameters
        ----------
        observed_data : list[ObservedDataPoint]
            The observations, as in `ResultsPlot.observations`.
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        calibration_result : CalibrationResult | None
            The values of the factors, as in `ResultsPlot.observations`.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `ResultsPlot.series`.
        tick_every : float | None
            The distance between labelled positions, as in
            `ResultsPlot.series`.
        x_label : str
            The label of the x axis.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.observations`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the steps and the guide of the mean.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            As `ResultsPlot.observations`; or if there are no replicates or
            the confidence level is out of range.
        KeyError
            If an observation names an output the replicates do not have.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return replicates_observations(
            self._replicates,
            observed_data,
            output_file=output_file,
            calibration_result=calibration_result,
            config=config,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )

    def _panel_model(self) -> "Model | None":
        """The model of the replicates."""
        return self._replicates.model

    def _panel_outputs(self) -> PanelOutputs:
        """The outputs of the replicates."""
        return PanelOutputs(self._replicates.mean(), self._replicates.model)

    def _series_panels(
        self,
        confidence_level: float,
        observed_data: "list[ObservedDataPoint]",
        series: list[str],
    ) -> list[Panel]:
        """The panels of `series`, as `series` draws them."""
        return replicate_panels(
            self._replicates, confidence_level, observed_data, series=series
        )

    def _window_trace(
        self,
        outputs: PanelOutputs,
        series_id: str,
        confidence_level: float,
        window: float,
    ) -> tuple[Trace, Band | None]:
        """The mean gain of a series over each complete window, and its band."""
        return replicates_window(
            self._replicates, outputs, series_id, confidence_level, window
        )

    def _observation_panels(
        self, confidence_level: float, observed_data: "list[ObservedDataPoint]"
    ) -> list[ObservationPanel]:
        """The panels of the observed series, as `observations` draws them."""
        return replicate_observation_panels(
            self._replicates, confidence_level, observed_data
        )
```

### Methods:

#### __init__

```python
__init__(replicates: SimulationReplicates) -> None
```

The figures of `replicates`.

Parameters:

| Name         | Type                   | Description     | Default    |
| ------------ | ---------------------- | --------------- | ---------- |
| `replicates` | `SimulationReplicates` | The replicates. | *required* |

Source code in `commol/_simulation/plots.py`

```python
def __init__(self, replicates: "SimulationReplicates") -> None:
    """
    The figures of `replicates`.

    Parameters
    ----------
    replicates : SimulationReplicates
        The replicates.
    """
    self._replicates = replicates
```

#### series

```python
series(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, series: list[str] | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', prediction_intervals: PredictionIntervalsData | None = None, confidence_level: float = 0.95, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the mean of each series and the band of the replicates, one panel per series.

Each replicate is reduced to the series first, through the operator of its observations, then the band takes the interval of `confidence_level` across them (`member_statistics`).

Parameters:

| Name                   | Type                       | Description                                      | Default                                                               |
| ---------------------- | -------------------------- | ------------------------------------------------ | --------------------------------------------------------------------- |
| `output_file`          | \`str                      | None\`                                           | Path to save the figure to. If None, the figure is only returned.     |
| `observed_data`        | \`list[ObservedDataPoint]  | None\`                                           | Observations, as in ResultsPlot.series.                               |
| `calibration_result`   | \`CalibrationResult        | None\`                                           | The values of the factors, as in ResultsPlot.series.                  |
| `config`               | \`PlotConfig               | None\`                                           | Layout and style. If None, the defaults.                              |
| `series`               | \`list[str]                | None\`                                           | What to draw, as in ResultsPlot.series.                               |
| `show_legend`          | `bool`                     | Whether each panel shows its legend.             | `True`                                                                |
| `position_label`       | \`Callable\[[float], str\] | None\`                                           | Turns a position into a tick label, as in ResultsPlot.series.         |
| `tick_every`           | \`float                    | None\`                                           | The distance between labelled positions, as in ResultsPlot.series.    |
| `x_label`              | `str`                      | The label of the x axis.                         | `'Step'`                                                              |
| `prediction_intervals` | \`PredictionIntervalsData  | None\`                                           | Prediction intervals of a conformal method, as in ResultsPlot.series. |
| `confidence_level`     | `float`                    | The level of the band, strictly between 0 and 1. | `0.95`                                                                |
| `axes`                 | \`Figure                   | None\`                                           | A figure of commol to draw on, as in ResultsPlot.series.              |
| `**kwargs`             | \`str                      | int                                              | float                                                                 |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------- |
| `ValueError`  | As ResultsPlot.series; or if there are no replicates or the confidence level is out of range. |
| `KeyError`    | As ResultsPlot.series.                                                                        |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                  |

Source code in `commol/_simulation/plots.py`

```python
def series(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervalsData | None" = None,
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the mean of each series and the band of the replicates, one
    panel per series.

    Each replicate is reduced to the series first, through the operator
    of its observations, then the band takes the interval of
    `confidence_level` across them (`member_statistics`).

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, as in `ResultsPlot.series`.
    calibration_result : CalibrationResult | None
        The values of the factors, as in `ResultsPlot.series`.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    series : list[str] | None
        What to draw, as in `ResultsPlot.series`.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position into a tick label, as in `ResultsPlot.series`.
    tick_every : float | None
        The distance between labelled positions, as in
        `ResultsPlot.series`.
    x_label : str
        The label of the x axis.
    prediction_intervals : PredictionIntervalsData | None
        Prediction intervals of a conformal method, as in
        `ResultsPlot.series`.
    confidence_level : float
        The level of the band, strictly between 0 and 1.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the mean.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        As `ResultsPlot.series`; or if there are no replicates or the
        confidence level is out of range.
    KeyError
        As `ResultsPlot.series`.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    return replicates_series(
        self._replicates,
        output_file=output_file,
        observed_data=observed_data,
        calibration_result=calibration_result,
        config=config,
        series=series,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        prediction_intervals=prediction_intervals,
        confidence_level=confidence_level,
        axes=axes,
        **kwargs,
    )
```

#### cumulative

```python
cumulative(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, series: list[str] | None = None, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the running total of the mean of the replicates, as `ResultsPlot.cumulative` draws one run.

Parameters:

| Name                 | Type                      | Description | Default                                                           |
| -------------------- | ------------------------- | ----------- | ----------------------------------------------------------------- |
| `output_file`        | \`str                     | None\`      | Path to save the figure to. If None, the figure is only returned. |
| `observed_data`      | \`list[ObservedDataPoint] | None\`      | Observations, as in ResultsPlot.cumulative.                       |
| `calibration_result` | \`CalibrationResult       | None\`      | The values of the factors, as in ResultsPlot.cumulative.          |
| `config`             | \`PlotConfig              | None\`      | Layout, size and style. If None, the defaults.                    |
| `series`             | \`list[str]               | None\`      | What to draw, as in ResultsPlot.series.                           |
| `axes`               | \`Figure                  | None\`      | A figure of commol to draw on, as in ResultsPlot.series.          |
| `**kwargs`           | \`str                     | int         | float                                                             |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | As ResultsPlot.cumulative; or if there are no replicates.    |
| `KeyError`    | As ResultsPlot.cumulative.                                   |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/_simulation/plots.py`

```python
def cumulative(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the running total of the mean of the replicates, as
    `ResultsPlot.cumulative` draws one run.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, as in `ResultsPlot.cumulative`.
    calibration_result : CalibrationResult | None
        The values of the factors, as in `ResultsPlot.cumulative`.
    config : PlotConfig | None
        Layout, size and style. If None, the defaults.
    series : list[str] | None
        What to draw, as in `ResultsPlot.series`.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the mean.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        As `ResultsPlot.cumulative`; or if there are no replicates.
    KeyError
        As `ResultsPlot.cumulative`.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    return replicates_cumulative(
        self._replicates,
        output_file=output_file,
        observed_data=observed_data,
        calibration_result=calibration_result,
        config=config,
        series=series,
        axes=axes,
        **kwargs,
    )
```

#### observations

```python
observations(observed_data: list[ObservedDataPoint], output_file: str | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', confidence_level: float = 0.95, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw each observed series with the mean of the replicates and their band, read through its observation operator, one panel per label.

Each replicate is read through the operator of each observation first, as `ResultsPlot.observations` reads a run, then the band takes the interval of `confidence_level` across them (`member_statistics`). A band over a window is a rectangle over that window. The band is a band of the model, with no observation noise, not a prediction interval, and its legend says so.

Parameters:

| Name                 | Type                       | Description                                       | Default                                                            |
| -------------------- | -------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| `observed_data`      | `list[ObservedDataPoint]`  | The observations, as in ResultsPlot.observations. | *required*                                                         |
| `output_file`        | \`str                      | None\`                                            | Path to save the figure to. If None, the figure is only returned.  |
| `calibration_result` | \`CalibrationResult        | None\`                                            | The values of the factors, as in ResultsPlot.observations.         |
| `config`             | \`PlotConfig               | None\`                                            | Layout and style. If None, the defaults.                           |
| `show_legend`        | `bool`                     | Whether each panel shows its legend.              | `True`                                                             |
| `position_label`     | \`Callable\[[float], str\] | None\`                                            | Turns a position into a tick label, as in ResultsPlot.series.      |
| `tick_every`         | \`float                    | None\`                                            | The distance between labelled positions, as in ResultsPlot.series. |
| `x_label`            | `str`                      | The label of the x axis.                          | `'Step'`                                                           |
| `confidence_level`   | `float`                    | The level of the band, strictly between 0 and 1.  | `0.95`                                                             |
| `axes`               | \`Figure                   | None\`                                            | A figure of commol to draw on, as in ResultsPlot.observations.     |
| `**kwargs`           | \`str                      | int                                               | float                                                              |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                         |
| ------------- | --------------------------------------------------------------------------------------------------- |
| `ValueError`  | As ResultsPlot.observations; or if there are no replicates or the confidence level is out of range. |
| `KeyError`    | If an observation names an output the replicates do not have.                                       |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                        |

Source code in `commol/_simulation/plots.py`

```python
def observations(
    self,
    observed_data: "list[ObservedDataPoint]",
    output_file: str | None = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw each observed series with the mean of the replicates and their
    band, read through its observation operator, one panel per label.

    Each replicate is read through the operator of each observation
    first, as `ResultsPlot.observations` reads a run, then the band takes
    the interval of `confidence_level` across them
    (`member_statistics`). A band over a window is a rectangle over that
    window. The band is a band of the model, with no observation noise,
    not a prediction interval, and its legend says so.

    Parameters
    ----------
    observed_data : list[ObservedDataPoint]
        The observations, as in `ResultsPlot.observations`.
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    calibration_result : CalibrationResult | None
        The values of the factors, as in `ResultsPlot.observations`.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position into a tick label, as in `ResultsPlot.series`.
    tick_every : float | None
        The distance between labelled positions, as in
        `ResultsPlot.series`.
    x_label : str
        The label of the x axis.
    confidence_level : float
        The level of the band, strictly between 0 and 1.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.observations`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the steps and the guide of the mean.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        As `ResultsPlot.observations`; or if there are no replicates or
        the confidence level is out of range.
    KeyError
        If an observation names an output the replicates do not have.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    return replicates_observations(
        self._replicates,
        observed_data,
        output_file=output_file,
        calibration_result=calibration_result,
        config=config,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        confidence_level=confidence_level,
        axes=axes,
        **kwargs,
    )
```

## InfeasibleStateError

```python
InfeasibleStateError = _engine.InfeasibleStateError
```

A run left the values its model allows, such as an amount below zero.

A difference-equations step or an impulse that leaves a compartment below zero raises it. A stochastic engine raises it for a rate it cannot draw, such as a negative rate, an impulse fraction outside 0 to 1, or an absolute impulse that is not a whole number. It is a `ValueError`; a calibration treats such a run as infeasible.
