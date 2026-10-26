# Sensitivity API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The sensitivity analysis, what it varies, the quantities of interest it reads from each run, and the result of each method. Guide: [Sensitivity analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/index.md).

## SensitivityAnalysis

The forward questions of a model: how uncertainty in its inputs moves its outputs, and which inputs matter most.

A calibration asks which inputs fit the data; an analysis needs no data. It holds the simulation, what varies (`InputSpace`), the replicates of a stochastic engine and the seed of every random number, and every method of one analysis reads the same inputs. The forward model is built in Rust once, when the analysis is created: the values of the inputs reach the engine as a calibration writes them.

Source code in `commol/api/sensitivity.py`

```python
class SensitivityAnalysis:
    """
    The forward questions of a model: how uncertainty in its inputs moves its
    outputs, and which inputs matter most.

    A calibration asks which inputs fit the data; an analysis needs no data.
    It holds the simulation, what varies (`InputSpace`), the replicates of a
    stochastic engine and the seed of every random number, and every method
    of one analysis reads the same inputs. The forward model is built in Rust
    once, when the analysis is created: the values of the inputs reach the
    engine as a calibration writes them.
    """

    def __init__(
        self,
        simulation: Simulation,
        inputs: InputSpace,
        *,
        replicates: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Build the analysis of `inputs` on `simulation`.

        Parameters
        ----------
        simulation : Simulation
            The model to analyse.
        inputs : InputSpace
            What varies.
        replicates : int | None, optional
            For a stochastic engine, required, at least 1: how many replicates
            each point runs. A point of the design of PRCC, Morris, Sobol or
            polynomial chaos holds the mean of its replicates, on the same
            seeds at every point; a propagation keeps every replicate, each
            with a seed of its own. A deterministic engine does not use it, so
            the same analysis runs on every engine.
        seed : int | None, optional
            The seed of the designs and the replicates. Required with a
            stochastic engine; otherwise None has the engine draw one, which
            `seed` reports and the analysis logs.

        Raises
        ------
        ValueError
            If an input is not a value of the model, the replicates or the
            seed do not fit the engine, `replicates` is below 1 (a negative
            number included), or the seed is not a whole number from 0 to
            ``2**64 - 1``.
        """
        self.simulation: Simulation = simulation
        self.inputs: InputSpace = inputs
        self.replicates: int | None = replicates
        check_values(
            simulation.model_definition,
            simulation.engine,
            inputs.parameters,
            inputs.initial_condition_remainder,
        )
        self._rust: "ForwardAnalysisProtocol" = commol_rs.inference.ForwardAnalysis(
            simulation.engine,
            engine_json(inputs),
            **engine_handles(inputs),
            replicates=replicates,
            seed=None if seed is None else check_seed(seed),
        )
        if seed is None:
            logger.info(
                "No seed set; drew the seed %d, which repeats this analysis.",
                self._rust.seed,
            )

    def _predictive(
        self, sample: ParameterSample, positions: list[float]
    ) -> "PredictiveProtocol":
        """
        The run of every draw of `sample` at `positions`, for a `Predictive`:
        each replicate of a stochastic engine has a seed of its own, derived
        from the seed of the analysis, as in a propagation.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the order of the inputs.
        positions : list[float]
            Positions to report at, sorted and unique.

        Returns
        -------
        PredictiveProtocol
            The runs, computed in Rust.
        """
        return self._rust.predictive(sample._rust, positions)

    def with_model(self, model: "Model") -> "SensitivityAnalysis":
        """
        The same analysis on another model, with the same seed.

        The draws of a predictive of the copy take the seeds they take here,
        so `Scenarios` runs every scenario on common random numbers.

        Parameters
        ----------
        model : Model
            The model, such as a copy of this one with other parameter values.

        Returns
        -------
        SensitivityAnalysis
            An analysis of `model`, with `inputs`, `replicates` and `seed`.

        Raises
        ------
        ValueError
            If an input is not a value of `model`.
        """
        return SensitivityAnalysis(
            Simulation(model), self.inputs, replicates=self.replicates, seed=self.seed
        )

    def _observation_draws(
        self,
        sample: ParameterSample,
        observed_data: list[ObservedDataPoint],
        noise: _ObservationNoise,
    ) -> "ObservationDrawsProtocol":
        """
        Every draw's value of `observed_data`, for a `Predictive`.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the order of the inputs.
        observed_data : list[ObservedDataPoint]
            The observations.
        noise : NoNoise | ResidualNoise | DistributionNoise
            The observation noise; only ``NoNoise()``.

        Returns
        -------
        ObservationDrawsProtocol
            The draws, computed in Rust.

        Raises
        ------
        ValueError
            If the noise is not ``NoNoise()``: residual noise needs the
            residuals of a fit, distribution noise the observation models of a
            problem, and an analysis has neither.
        """
        if not isinstance(noise, NoNoise):
            raise ValueError(
                f"{type(noise).__name__} needs a calibration: residual noise reads "
                "the residuals of a fit and distribution noise the observation "
                "models of a problem, and a sensitivity analysis has no data. Read "
                "the observations with NoNoise()."
            )
        return self._rust.observation_draws(sample._rust, engine_json(observed_data))

    def _log_score(
        self,
        sample: ParameterSample,  # noqa: ARG002
        observed_data: list[ObservedDataPoint],  # noqa: ARG002
    ) -> list[float]:
        """
        Refuse the log score, which `commol.scores.log_score` asks for.

        Parameters
        ----------
        sample : ParameterSample
            The draws.
        observed_data : list[ObservedDataPoint]
            The observations.

        Raises
        ------
        ValueError
            Always: the log score reads the observation models of a problem,
            and an analysis has no data.
        """
        raise ValueError(
            "The log score needs the observation models of a calibration problem, "
            "and a sensitivity analysis has no data."
        )

    @property
    def seed(self) -> int:
        """
        The seed of the analysis' random numbers.

        Returns
        -------
        int
            The seed given, or the seed the engine drew when the analysis
            was built, which it logs.
        """
        return self._rust.seed

    def run(self, method: SensitivityMethod[ResultT]) -> ResultT:
        """
        Run a sensitivity method on the inputs of this analysis.

        Parameters
        ----------
        method : SensitivityMethod[ResultT]
            The configuration of the method; its type sets the result type.

        Returns
        -------
        ResultT
            The method's result.

        Raises
        ------
        TypeError
            If `method` is not a sensitivity method; the message names its
            entry point.
        ValueError
            If the method does not take these inputs, such as a method of
            independent inputs on inputs from a sample, or a setting, a
            quantity or the model does not fit the method; the message says
            why.
        RuntimeError
            If a run of the design fails.
        InfeasibleStateError
            If a point of the design leaves the values its model allows.
        """
        return run_method(self, method, SensitivityMethod)
```

### Attributes

#### seed

```python
seed: int
```

The seed of the analysis' random numbers.

Returns:

| Type  | Description                                                                             |
| ----- | --------------------------------------------------------------------------------------- |
| `int` | The seed given, or the seed the engine drew when the analysis was built, which it logs. |

### Methods:

#### __init__

```python
__init__(simulation: Simulation, inputs: InputSpace, *, replicates: int | None = None, seed: int | None = None) -> None
```

Build the analysis of `inputs` on `simulation`.

Parameters:

| Name         | Type         | Description           | Default                                                                                                                                                                                                                                                                                                                                                                       |
| ------------ | ------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `simulation` | `Simulation` | The model to analyse. | *required*                                                                                                                                                                                                                                                                                                                                                                    |
| `inputs`     | `InputSpace` | What varies.          | *required*                                                                                                                                                                                                                                                                                                                                                                    |
| `replicates` | \`int        | None\`                | For a stochastic engine, required, at least 1: how many replicates each point runs. A point of the design of PRCC, Morris, Sobol or polynomial chaos holds the mean of its replicates, on the same seeds at every point; a propagation keeps every replicate, each with a seed of its own. A deterministic engine does not use it, so the same analysis runs on every engine. |
| `seed`       | \`int        | None\`                | The seed of the designs and the replicates. Required with a stochastic engine; otherwise None has the engine draw one, which seed reports and the analysis logs.                                                                                                                                                                                                              |

Raises:

| Type         | Description                                                                                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If an input is not a value of the model, the replicates or the seed do not fit the engine, replicates is below 1 (a negative number included), or the seed is not a whole number from 0 to 2\*\*64 - 1. |

Source code in `commol/api/sensitivity.py`

```python
def __init__(
    self,
    simulation: Simulation,
    inputs: InputSpace,
    *,
    replicates: int | None = None,
    seed: int | None = None,
) -> None:
    """
    Build the analysis of `inputs` on `simulation`.

    Parameters
    ----------
    simulation : Simulation
        The model to analyse.
    inputs : InputSpace
        What varies.
    replicates : int | None, optional
        For a stochastic engine, required, at least 1: how many replicates
        each point runs. A point of the design of PRCC, Morris, Sobol or
        polynomial chaos holds the mean of its replicates, on the same
        seeds at every point; a propagation keeps every replicate, each
        with a seed of its own. A deterministic engine does not use it, so
        the same analysis runs on every engine.
    seed : int | None, optional
        The seed of the designs and the replicates. Required with a
        stochastic engine; otherwise None has the engine draw one, which
        `seed` reports and the analysis logs.

    Raises
    ------
    ValueError
        If an input is not a value of the model, the replicates or the
        seed do not fit the engine, `replicates` is below 1 (a negative
        number included), or the seed is not a whole number from 0 to
        ``2**64 - 1``.
    """
    self.simulation: Simulation = simulation
    self.inputs: InputSpace = inputs
    self.replicates: int | None = replicates
    check_values(
        simulation.model_definition,
        simulation.engine,
        inputs.parameters,
        inputs.initial_condition_remainder,
    )
    self._rust: "ForwardAnalysisProtocol" = commol_rs.inference.ForwardAnalysis(
        simulation.engine,
        engine_json(inputs),
        **engine_handles(inputs),
        replicates=replicates,
        seed=None if seed is None else check_seed(seed),
    )
    if seed is None:
        logger.info(
            "No seed set; drew the seed %d, which repeats this analysis.",
            self._rust.seed,
        )
```

#### with_model

```python
with_model(model: Model) -> SensitivityAnalysis
```

The same analysis on another model, with the same seed.

The draws of a predictive of the copy take the seeds they take here, so `Scenarios` runs every scenario on common random numbers.

Parameters:

| Name    | Type    | Description                                                        | Default    |
| ------- | ------- | ------------------------------------------------------------------ | ---------- |
| `model` | `Model` | The model, such as a copy of this one with other parameter values. | *required* |

Returns:

| Type                  | Description                                             |
| --------------------- | ------------------------------------------------------- |
| `SensitivityAnalysis` | An analysis of model, with inputs, replicates and seed. |

Raises:

| Type         | Description                          |
| ------------ | ------------------------------------ |
| `ValueError` | If an input is not a value of model. |

Source code in `commol/api/sensitivity.py`

```python
def with_model(self, model: "Model") -> "SensitivityAnalysis":
    """
    The same analysis on another model, with the same seed.

    The draws of a predictive of the copy take the seeds they take here,
    so `Scenarios` runs every scenario on common random numbers.

    Parameters
    ----------
    model : Model
        The model, such as a copy of this one with other parameter values.

    Returns
    -------
    SensitivityAnalysis
        An analysis of `model`, with `inputs`, `replicates` and `seed`.

    Raises
    ------
    ValueError
        If an input is not a value of `model`.
    """
    return SensitivityAnalysis(
        Simulation(model), self.inputs, replicates=self.replicates, seed=self.seed
    )
```

#### run

```python
run(method: SensitivityMethod[ResultT]) -> ResultT
```

Run a sensitivity method on the inputs of this analysis.

Parameters:

| Name     | Type                         | Description                                                     | Default    |
| -------- | ---------------------------- | --------------------------------------------------------------- | ---------- |
| `method` | `SensitivityMethod[ResultT]` | The configuration of the method; its type sets the result type. | *required* |

Returns:

| Type      | Description          |
| --------- | -------------------- |
| `ResultT` | The method's result. |

Raises:

| Type                   | Description                                                                                                                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TypeError`            | If method is not a sensitivity method; the message names its entry point.                                                                                                                      |
| `ValueError`           | If the method does not take these inputs, such as a method of independent inputs on inputs from a sample, or a setting, a quantity or the model does not fit the method; the message says why. |
| `RuntimeError`         | If a run of the design fails.                                                                                                                                                                  |
| `InfeasibleStateError` | If a point of the design leaves the values its model allows.                                                                                                                                   |

Source code in `commol/api/sensitivity.py`

```python
def run(self, method: SensitivityMethod[ResultT]) -> ResultT:
    """
    Run a sensitivity method on the inputs of this analysis.

    Parameters
    ----------
    method : SensitivityMethod[ResultT]
        The configuration of the method; its type sets the result type.

    Returns
    -------
    ResultT
        The method's result.

    Raises
    ------
    TypeError
        If `method` is not a sensitivity method; the message names its
        entry point.
    ValueError
        If the method does not take these inputs, such as a method of
        independent inputs on inputs from a sample, or a setting, a
        quantity or the model does not fit the method; the message says
        why.
    RuntimeError
        If a run of the design fails.
    InfeasibleStateError
        If a point of the design leaves the values its model allows.
    """
    return run_method(self, method, SensitivityMethod)
```

## InputSpace

Bases: `BaseModel`

What a sensitivity analysis varies.

Each input is a `CalibrationParameter`, a model parameter or an initial condition with its bounds, as a calibration sets it. The inputs follow independent distributions, one per input, truncated to its bounds, or are the draws of a sample, with their weights; the inputs of a sample may depend on each other, so only the methods that allow it take one. Build it with `independent`, `uniform_on_bounds` or `from_sample`.

Attributes:

| Name                          | Type                         | Description                                                                                                                 |
| ----------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `parameters`                  | `list[CalibrationParameter]` | The inputs, in order. A factor or a noise parameter is not an input: both belong to observations, and an analysis has none. |
| `distributions`               | \`dict[str, distribution]    | None\`                                                                                                                      |
| `sample`                      | \`ParameterSample            | None\`                                                                                                                      |
| `initial_condition_remainder` | \`str                        | None\`                                                                                                                      |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class InputSpace(BaseModel):
    """
    What a sensitivity analysis varies.

    Each input is a `CalibrationParameter`, a model parameter or an initial
    condition with its bounds, as a calibration sets it. The inputs follow
    independent distributions, one per input, truncated to its bounds, or
    are the draws of a sample, with their weights; the inputs of a sample may
    depend on each other, so only the methods that allow it take one. Build
    it with `independent`, `uniform_on_bounds` or `from_sample`.

    Attributes
    ----------
    parameters : list[CalibrationParameter]
        The inputs, in order. A factor or a noise parameter is not an input:
        both belong to observations, and an analysis has none.
    distributions : dict[str, distribution] | None
        The distribution of each input, for independent inputs.
    sample : ParameterSample | None
        The draws of the inputs, for inputs from a sample.
    initial_condition_remainder : str | None
        The bin or compartment that takes what the initial conditions leave
        of the sum the model states, as in a calibration problem.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    _engine_handles: ClassVar[tuple[str, ...]] = ("sample",)

    parameters: list[CalibrationParameter] = Field(
        min_length=1, description="The inputs, in order."
    )
    distributions: dict[str, _Distribution] | None = Field(
        default=None, description="The distribution of each independent input."
    )
    sample: ParameterSample | None = Field(
        default=None, description="The draws of inputs from a sample."
    )
    initial_condition_remainder: str | None = Field(
        default=None,
        description="Bin or compartment that takes what the initial conditions leave.",
    )

    @model_validator(mode="after")
    def _check_inputs(self) -> Self:
        """Check the inputs against the distributions or the sample."""
        ids = [parameter.id for parameter in self.parameters]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"Inputs repeat the ids {repeated}.")
        observed = {
            "factor": ("a factor, which multiplies an observation"),
            "noise": ("a noise parameter, which belongs to an observation model"),
        }
        for parameter in self.parameters:
            what = observed.get(parameter.parameter_type)
            if what is not None:
                raise ValueError(
                    f"'{parameter.id}' is {what}; a sensitivity analysis has no "
                    "observations."
                )
        if (self.distributions is None) == (self.sample is None):
            raise ValueError("Give distributions or a sample, not both.")
        if self.distributions is not None:
            missing = [i for i in ids if i not in self.distributions]
            extra = sorted(set(self.distributions) - set(ids))
            if missing:
                raise ValueError(
                    f"The inputs {missing} have no distribution: give one per "
                    "input, or InputSpace.uniform_on_bounds(parameters)."
                )
            if extra:
                raise ValueError(f"The distributions of {extra} name no input.")
        if self.sample is not None:
            bounds = [(p.min_bound, p.max_bound) for p in self.parameters]
            if self.sample.names != ids or list(self.sample.bounds) != bounds:
                raise ValueError(
                    f"The sample has the inputs {self.sample.names} with the bounds "
                    f"{list(self.sample.bounds)}; the parameters are {ids} with the "
                    f"bounds {bounds}, in this order."
                )
        return self

    @classmethod
    def independent(
        cls,
        parameters: Sequence[CalibrationParameter],
        distributions: Mapping[str, _Distribution],
        *,
        initial_condition_remainder: str | None = None,
    ) -> Self:
        """
        Independent inputs, each with its distribution.

        Parameters
        ----------
        parameters : Sequence[CalibrationParameter]
            The inputs, in order.
        distributions : Mapping[str, distribution]
            The distribution of each input, truncated to its bounds.
        initial_condition_remainder : str | None, optional
            The bin or compartment that takes what the initial conditions
            leave.

        Returns
        -------
        InputSpace
            The input space.

        Raises
        ------
        ValueError
            If an input has no distribution or is a factor, a distribution
            names no input, or an id repeats.
        """
        return cls(
            parameters=list(parameters),
            distributions=dict(distributions),
            initial_condition_remainder=initial_condition_remainder,
        )

    @classmethod
    def uniform_on_bounds(
        cls,
        parameters: Sequence[CalibrationParameter],
        *,
        initial_condition_remainder: str | None = None,
    ) -> Self:
        """
        Independent inputs, each uniform on its bounds; they must be finite.

        Parameters
        ----------
        parameters : Sequence[CalibrationParameter]
            The inputs, in order.
        initial_condition_remainder : str | None, optional
            The bin or compartment that takes what the initial conditions
            leave.

        Returns
        -------
        InputSpace
            The input space.
        """
        return cls.independent(
            parameters,
            {parameter.id: Uniform() for parameter in parameters},
            initial_condition_remainder=initial_condition_remainder,
        )

    @classmethod
    def from_sample(
        cls,
        sample: ParameterSample,
        parameters: Sequence[CalibrationParameter],
        *,
        initial_condition_remainder: str | None = None,
    ) -> Self:
        """
        The draws of a sample, with their weights, such as a posterior.

        Its inputs may depend on each other, so only PRCC and propagation
        take it. The parameters give the type of each value, which a sample
        does not hold.

        Parameters
        ----------
        sample : ParameterSample
            The draws.
        parameters : Sequence[CalibrationParameter]
            The inputs, in the order and with the bounds of the sample.
        initial_condition_remainder : str | None, optional
            The bin or compartment that takes what the initial conditions
            leave.

        Returns
        -------
        InputSpace
            The input space.

        Raises
        ------
        ValueError
            If the sample's names or bounds are not the parameters'.
        """
        return cls(
            parameters=list(parameters),
            sample=sample,
            initial_condition_remainder=initial_condition_remainder,
        )

    @property
    def names(self) -> list[str]:
        """The ids of the inputs, in order."""
        return [parameter.id for parameter in self.parameters]
```

### Attributes

#### names

```python
names: list[str]
```

The ids of the inputs, in order.

### Methods:

#### independent

```python
independent(parameters: Sequence[CalibrationParameter], distributions: Mapping[str, _Distribution], *, initial_condition_remainder: str | None = None) -> Self
```

Independent inputs, each with its distribution.

Parameters:

| Name                          | Type                             | Description                                              | Default                                                              |
| ----------------------------- | -------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------- |
| `parameters`                  | `Sequence[CalibrationParameter]` | The inputs, in order.                                    | *required*                                                           |
| `distributions`               | `Mapping[str, distribution]`     | The distribution of each input, truncated to its bounds. | *required*                                                           |
| `initial_condition_remainder` | \`str                            | None\`                                                   | The bin or compartment that takes what the initial conditions leave. |

Returns:

| Type         | Description      |
| ------------ | ---------------- |
| `InputSpace` | The input space. |

Raises:

| Type         | Description                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------ |
| `ValueError` | If an input has no distribution or is a factor, a distribution names no input, or an id repeats. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
@classmethod
def independent(
    cls,
    parameters: Sequence[CalibrationParameter],
    distributions: Mapping[str, _Distribution],
    *,
    initial_condition_remainder: str | None = None,
) -> Self:
    """
    Independent inputs, each with its distribution.

    Parameters
    ----------
    parameters : Sequence[CalibrationParameter]
        The inputs, in order.
    distributions : Mapping[str, distribution]
        The distribution of each input, truncated to its bounds.
    initial_condition_remainder : str | None, optional
        The bin or compartment that takes what the initial conditions
        leave.

    Returns
    -------
    InputSpace
        The input space.

    Raises
    ------
    ValueError
        If an input has no distribution or is a factor, a distribution
        names no input, or an id repeats.
    """
    return cls(
        parameters=list(parameters),
        distributions=dict(distributions),
        initial_condition_remainder=initial_condition_remainder,
    )
```

#### uniform_on_bounds

```python
uniform_on_bounds(parameters: Sequence[CalibrationParameter], *, initial_condition_remainder: str | None = None) -> Self
```

Independent inputs, each uniform on its bounds; they must be finite.

Parameters:

| Name                          | Type                             | Description           | Default                                                              |
| ----------------------------- | -------------------------------- | --------------------- | -------------------------------------------------------------------- |
| `parameters`                  | `Sequence[CalibrationParameter]` | The inputs, in order. | *required*                                                           |
| `initial_condition_remainder` | \`str                            | None\`                | The bin or compartment that takes what the initial conditions leave. |

Returns:

| Type         | Description      |
| ------------ | ---------------- |
| `InputSpace` | The input space. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
@classmethod
def uniform_on_bounds(
    cls,
    parameters: Sequence[CalibrationParameter],
    *,
    initial_condition_remainder: str | None = None,
) -> Self:
    """
    Independent inputs, each uniform on its bounds; they must be finite.

    Parameters
    ----------
    parameters : Sequence[CalibrationParameter]
        The inputs, in order.
    initial_condition_remainder : str | None, optional
        The bin or compartment that takes what the initial conditions
        leave.

    Returns
    -------
    InputSpace
        The input space.
    """
    return cls.independent(
        parameters,
        {parameter.id: Uniform() for parameter in parameters},
        initial_condition_remainder=initial_condition_remainder,
    )
```

#### from_sample

```python
from_sample(sample: ParameterSample, parameters: Sequence[CalibrationParameter], *, initial_condition_remainder: str | None = None) -> Self
```

The draws of a sample, with their weights, such as a posterior.

Its inputs may depend on each other, so only PRCC and propagation take it. The parameters give the type of each value, which a sample does not hold.

Parameters:

| Name                          | Type                             | Description                                                 | Default                                                              |
| ----------------------------- | -------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------- |
| `sample`                      | `ParameterSample`                | The draws.                                                  | *required*                                                           |
| `parameters`                  | `Sequence[CalibrationParameter]` | The inputs, in the order and with the bounds of the sample. | *required*                                                           |
| `initial_condition_remainder` | \`str                            | None\`                                                      | The bin or compartment that takes what the initial conditions leave. |

Returns:

| Type         | Description      |
| ------------ | ---------------- |
| `InputSpace` | The input space. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the sample's names or bounds are not the parameters'. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
@classmethod
def from_sample(
    cls,
    sample: ParameterSample,
    parameters: Sequence[CalibrationParameter],
    *,
    initial_condition_remainder: str | None = None,
) -> Self:
    """
    The draws of a sample, with their weights, such as a posterior.

    Its inputs may depend on each other, so only PRCC and propagation
    take it. The parameters give the type of each value, which a sample
    does not hold.

    Parameters
    ----------
    sample : ParameterSample
        The draws.
    parameters : Sequence[CalibrationParameter]
        The inputs, in the order and with the bounds of the sample.
    initial_condition_remainder : str | None, optional
        The bin or compartment that takes what the initial conditions
        leave.

    Returns
    -------
    InputSpace
        The input space.

    Raises
    ------
    ValueError
        If the sample's names or bounds are not the parameters'.
    """
    return cls(
        parameters=list(parameters),
        sample=sample,
        initial_condition_remainder=initial_condition_remainder,
    )
```

## Propagation

Bases: `SensitivityMethod[PropagationResult]`, `RangeChecked`

Monte Carlo propagation: run the draws of the inputs and reduce each run to the quantities.

With independent inputs the draws are a Latin hypercube of their distributions; with inputs from a sample, the sample is the design, with its weights.

Attributes:

| Name         | Type             | Description                                                                |
| ------------ | ---------------- | -------------------------------------------------------------------------- |
| `draws`      | \`int            | None\`                                                                     |
| `quantities` | `list[quantity]` | What each run is reduced to; none gives only the sample, for a Predictive. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class Propagation(SensitivityMethod[PropagationResult], RangeChecked):
    """
    Monte Carlo propagation: run the draws of the inputs and reduce each run
    to the quantities.

    With independent inputs the draws are a Latin hypercube of their
    distributions; with inputs from a sample, the sample is the design, with
    its weights.

    Attributes
    ----------
    draws : int | None
        The number of points of the Latin hypercube; required with
        independent inputs, and left unset with inputs from a sample.
    quantities : list[quantity]
        What each run is reduced to; none gives only the sample, for a
        `Predictive`.
    """

    draws: int | None = Field(
        default=None, description="Points of the Latin hypercube."
    )
    quantities: list[_Quantity] = Field(
        default_factory=list, description="What each run is reduced to."
    )
```

## PropagationResult

Bases: `ResultModel`

The draws of the inputs and what each run gives.

Its `sample` is a `ParameterSample`, so `Predictive(analysis, result.sample, positions)` gives the bands, plots and reports of the propagation, as for a calibration.

Attributes:

| Name                                                                          | Type                                                                   | Description                                                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample`                                                                      | `ParameterSample`                                                      | The draws of the inputs: a Latin hypercube of their distributions, or the sample of the input space itself.                                                                                                                                                       |
| `labels`                                                                      | `list[str]`                                                            | The label of each quantity.                                                                                                                                                                                                                                       |
| `values`                                                                      | `list[list[float]]`                                                    | The quantities of each run, [run]\[quantity\]: one run per draw, or with a stochastic engine one per replicate of each draw.                                                                                                                                      |
| `weights`                                                                     | `list[float]`                                                          | The weight of each run; with a stochastic engine each replicate has an equal share of its draw's weight. They sum to 1.                                                                                                                                           |
| `draw_seeds`                                                                  | `list[int]`                                                            | The seed of each draw of sample, in its order. With a stochastic engine, replicate i of draw j runs with the seed replicate i of Simulation.replicates(count, seed=draw_seeds[j]) takes, so those replicates of the model at that draw give the runs of the draw. |
| `` A quantity that reads a NaN is NaN in `values`, and `summary` raises on `` |                                                                        |                                                                                                                                                                                                                                                                   |
| `it`                                                                          | `` `where` keeps the runs a condition chooses, which conditions the `` |                                                                                                                                                                                                                                                                   |
| `result on that condition.`                                                   |                                                                        |                                                                                                                                                                                                                                                                   |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class PropagationResult(ResultModel):
    """
    The draws of the inputs and what each run gives.

    Its `sample` is a `ParameterSample`, so `Predictive(analysis,
    result.sample, positions)` gives the bands, plots and reports of the
    propagation, as for a calibration.

    Attributes
    ----------
    sample : ParameterSample
        The draws of the inputs: a Latin hypercube of their distributions, or
        the sample of the input space itself.
    labels : list[str]
        The label of each quantity.
    values : list[list[float]]
        The quantities of each run, ``[run][quantity]``: one run per draw, or
        with a stochastic engine one per replicate of each draw.
    weights : list[float]
        The weight of each run; with a stochastic engine each replicate has
        an equal share of its draw's weight. They sum to 1.
    draw_seeds : list[int]
        The seed of each draw of `sample`, in its order. With a stochastic
        engine, replicate ``i`` of draw ``j`` runs with the seed replicate
        ``i`` of ``Simulation.replicates(count, seed=draw_seeds[j])`` takes, so
        those replicates of the model at that draw give the runs of the draw.

    A quantity that reads a NaN is NaN in `values`, and `summary` raises on
    it: `where` keeps the runs a condition chooses, which conditions the
    result on that condition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: ParameterSample
    labels: list[str]
    values: list[list[float]]
    weights: list[float]
    draw_seeds: list[int]

    @property
    def effective_size(self) -> float:
        """
        Kish's effective size of the weights of the runs, ``(Σw)²/Σw²``: the
        number of equally weighted runs that carry as much information.
        """
        return commol_rs.inference.effective_size(self.weights)

    def where(self, mask: ArrayLike) -> "PropagationResult":
        """
        The runs where `mask` is true, with their weights renormalised to sum
        to 1.

        A filter conditions the result: a summary of the runs it keeps is the
        distribution of each quantity given the condition of the mask, such
        as ``np.isfinite(values[:, i])`` or a compartment above a threshold,
        not the distribution over every run. `effective_size` says how many
        runs the kept weights are worth. `sample` and `draw_seeds` stay those
        of the draws of the inputs.

        Parameters
        ----------
        mask : ArrayLike
            One boolean per run, in the order of `values`.

        Returns
        -------
        PropagationResult
            The kept runs, their values and their renormalised weights.

        Raises
        ------
        ValueError
            If `mask` has not one boolean per run, keeps no run, or keeps
            only runs of weight 0.
        """
        keep = np.asarray(mask)
        if keep.dtype != np.bool_ or keep.shape != (len(self.values),):
            raise ValueError(
                f"The mask has one boolean per run: {len(self.values)} runs, got "
                f"an array of {keep.dtype} with the shape {keep.shape}"
            )
        weights = np.asarray(self.weights, dtype=float)[keep]
        if weights.sum() <= 0.0:
            raise ValueError(
                "The mask keeps no run of positive weight: a summary needs at least one"
            )
        return self.model_copy(
            update={
                "values": [row for row, kept in zip(self.values, keep) if kept],
                "weights": (weights / weights.sum()).tolist(),
            }
        )

    def summary(self, confidence_level: float) -> list[QuantitySummary]:
        """
        The weighted distribution of each quantity across the runs.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        list[QuantitySummary]
            One summary per quantity, in the order of `labels`.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1, or a
            quantity is NaN in a run: `where` keeps the runs a condition
            chooses.
        """
        # The level is checked before the values, also when there are none.
        commol_rs.inference.interval_levels(confidence_level)
        summaries = []
        for index, label in enumerate(self.labels):
            column = [row[index] for row in self.values]
            try:
                summary = commol_rs.inference.weighted_summary(
                    column, self.weights, confidence_level
                )
            except ValueError as error:
                raise ValueError(
                    f"Quantity '{label}', in the runs of values counted from 0: "
                    f"{error}. where(mask) keeps the runs a condition chooses."
                ) from error
            summaries.append(
                QuantitySummary(
                    label=label,
                    confidence_level=confidence_level,
                    mean=summary.mean,
                    median=summary.median,
                    lower=summary.lower,
                    upper=summary.upper,
                )
            )
        return summaries
```

### Attributes

#### effective_size

```python
effective_size: float
```

Kish's effective size of the weights of the runs, `(Σw)²/Σw²`: the number of equally weighted runs that carry as much information.

### Methods:

#### where

```python
where(mask: ArrayLike) -> PropagationResult
```

The runs where `mask` is true, with their weights renormalised to sum to 1.

A filter conditions the result: a summary of the runs it keeps is the distribution of each quantity given the condition of the mask, such as `np.isfinite(values[:, i])` or a compartment above a threshold, not the distribution over every run. `effective_size` says how many runs the kept weights are worth. `sample` and `draw_seeds` stay those of the draws of the inputs.

Parameters:

| Name   | Type        | Description                                  | Default    |
| ------ | ----------- | -------------------------------------------- | ---------- |
| `mask` | `ArrayLike` | One boolean per run, in the order of values. | *required* |

Returns:

| Type                | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `PropagationResult` | The kept runs, their values and their renormalised weights. |

Raises:

| Type         | Description                                                                        |
| ------------ | ---------------------------------------------------------------------------------- |
| `ValueError` | If mask has not one boolean per run, keeps no run, or keeps only runs of weight 0. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def where(self, mask: ArrayLike) -> "PropagationResult":
    """
    The runs where `mask` is true, with their weights renormalised to sum
    to 1.

    A filter conditions the result: a summary of the runs it keeps is the
    distribution of each quantity given the condition of the mask, such
    as ``np.isfinite(values[:, i])`` or a compartment above a threshold,
    not the distribution over every run. `effective_size` says how many
    runs the kept weights are worth. `sample` and `draw_seeds` stay those
    of the draws of the inputs.

    Parameters
    ----------
    mask : ArrayLike
        One boolean per run, in the order of `values`.

    Returns
    -------
    PropagationResult
        The kept runs, their values and their renormalised weights.

    Raises
    ------
    ValueError
        If `mask` has not one boolean per run, keeps no run, or keeps
        only runs of weight 0.
    """
    keep = np.asarray(mask)
    if keep.dtype != np.bool_ or keep.shape != (len(self.values),):
        raise ValueError(
            f"The mask has one boolean per run: {len(self.values)} runs, got "
            f"an array of {keep.dtype} with the shape {keep.shape}"
        )
    weights = np.asarray(self.weights, dtype=float)[keep]
    if weights.sum() <= 0.0:
        raise ValueError(
            "The mask keeps no run of positive weight: a summary needs at least one"
        )
    return self.model_copy(
        update={
            "values": [row for row, kept in zip(self.values, keep) if kept],
            "weights": (weights / weights.sum()).tolist(),
        }
    )
```

#### summary

```python
summary(confidence_level: float) -> list[QuantitySummary]
```

The weighted distribution of each quantity across the runs.

Parameters:

| Name               | Type    | Description               | Default    |
| ------------------ | ------- | ------------------------- | ---------- |
| `confidence_level` | `float` | Strictly between 0 and 1. | *required* |

Returns:

| Type                    | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `list[QuantitySummary]` | One summary per quantity, in the order of labels. |

Raises:

| Type         | Description                                                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the confidence level is not strictly between 0 and 1, or a quantity is NaN in a run: where keeps the runs a condition chooses. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def summary(self, confidence_level: float) -> list[QuantitySummary]:
    """
    The weighted distribution of each quantity across the runs.

    Parameters
    ----------
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    list[QuantitySummary]
        One summary per quantity, in the order of `labels`.

    Raises
    ------
    ValueError
        If the confidence level is not strictly between 0 and 1, or a
        quantity is NaN in a run: `where` keeps the runs a condition
        chooses.
    """
    # The level is checked before the values, also when there are none.
    commol_rs.inference.interval_levels(confidence_level)
    summaries = []
    for index, label in enumerate(self.labels):
        column = [row[index] for row in self.values]
        try:
            summary = commol_rs.inference.weighted_summary(
                column, self.weights, confidence_level
            )
        except ValueError as error:
            raise ValueError(
                f"Quantity '{label}', in the runs of values counted from 0: "
                f"{error}. where(mask) keeps the runs a condition chooses."
            ) from error
        summaries.append(
            QuantitySummary(
                label=label,
                confidence_level=confidence_level,
                mean=summary.mean,
                median=summary.median,
                lower=summary.lower,
                upper=summary.upper,
            )
        )
    return summaries
```

## QuantitySummary

Bases: `ResultModel`

The weighted distribution of one quantity across the runs of a propagation.

Attributes:

| Name               | Type    | Description                        |
| ------------------ | ------- | ---------------------------------- |
| `label`            | `str`   | The quantity.                      |
| `confidence_level` | `float` | The level of the central interval. |
| `mean`             | `float` | The weighted mean.                 |
| `median`           | `float` | The weighted median.               |
| `lower`            | `float` | The quantile (1 − c)/2.            |
| `upper`            | `float` | The quantile (1 + c)/2.            |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class QuantitySummary(ResultModel):
    """
    The weighted distribution of one quantity across the runs of a
    propagation.

    Attributes
    ----------
    label : str
        The quantity.
    confidence_level : float
        The level of the central interval.
    mean : float
        The weighted mean.
    median : float
        The weighted median.
    lower : float
        The quantile ``(1 − c)/2``.
    upper : float
        The quantile ``(1 + c)/2``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    confidence_level: float
    mean: float
    median: float
    lower: float
    upper: float
```

## PRCC

Bases: `SensitivityMethod[PRCCResult]`, `RangeChecked`

Latin hypercube sampling with partial rank correlation coefficients (Marino et al. 2008): how steadily each quantity rises or falls with each input.

With independent inputs the draws are a Latin hypercube of their distributions; with inputs from a sample, the sample is the design, and the ranks, the correlations and the test read its weights. A `Trajectory` gives the coefficients over the axis.

Attributes:

| Name         | Type             | Description                  |
| ------------ | ---------------- | ---------------------------- |
| `draws`      | \`int            | None\`                       |
| `quantities` | `list[quantity]` | What each run is reduced to. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class PRCC(SensitivityMethod[PRCCResult], RangeChecked):
    """
    Latin hypercube sampling with partial rank correlation coefficients
    (Marino et al. 2008): how steadily each quantity rises or falls with each
    input.

    With independent inputs the draws are a Latin hypercube of their
    distributions; with inputs from a sample, the sample is the design, and
    the ranks, the correlations and the test read its weights. A
    `Trajectory` gives the coefficients over the axis.

    Attributes
    ----------
    draws : int | None
        The number of points of the Latin hypercube; required with
        independent inputs, and left unset with inputs from a sample.
    quantities : list[quantity]
        What each run is reduced to.
    """

    draws: int | None = Field(
        default=None, description="Points of the Latin hypercube."
    )
    quantities: list[_Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
```

## PRCCResult

Bases: `ResultModel`

The partial rank correlation of each input with each quantity.

A coefficient near 1 or −1 says the quantity rises or falls steadily with the input once the other inputs are accounted for; near 0, that it does not move steadily with it. The p-value tests the coefficient against 0 with Kish's effective size of the draws.

Attributes:

| Name             | Type                | Description                                                                                                   |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------------------------------- |
| `sample`         | `ParameterSample`   | The draws the coefficients come from.                                                                         |
| `inputs`         | `list[str]`         | The inputs, in order.                                                                                         |
| `labels`         | `list[str]`         | The label of each quantity.                                                                                   |
| `series`         | `list[str]`         | The series of each quantity: its label, or the name of its Trajectory.                                        |
| `positions`      | \`list\[float       | None\]\`                                                                                                      |
| `coefficients`   | `list[list[float]]` | The PRCC of each input for each quantity, [quantity][input]; NaN for a quantity with one value in every draw. |
| `p_values`       | `list[list[float]]` | The two-sided p-value of each coefficient, [quantity][input].                                                 |
| `effective_size` | `float`             | Kish's effective size of the draws: their number with equal weights.                                          |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class PRCCResult(ResultModel):
    """
    The partial rank correlation of each input with each quantity.

    A coefficient near 1 or −1 says the quantity rises or falls steadily
    with the input once the other inputs are accounted for; near 0, that it
    does not move steadily with it. The p-value tests the coefficient against
    0 with Kish's effective size of the draws.

    Attributes
    ----------
    sample : ParameterSample
        The draws the coefficients come from.
    inputs : list[str]
        The inputs, in order.
    labels : list[str]
        The label of each quantity.
    series : list[str]
        The series of each quantity: its label, or the name of its
        `Trajectory`.
    positions : list[float | None]
        The position of each quantity of a `Trajectory`, None for the others.
    coefficients : list[list[float]]
        The PRCC of each input for each quantity, ``[quantity][input]``; NaN
        for a quantity with one value in every draw.
    p_values : list[list[float]]
        The two-sided p-value of each coefficient, ``[quantity][input]``.
    effective_size : float
        Kish's effective size of the draws: their number with equal weights.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: ParameterSample
    inputs: list[str]
    labels: list[str]
    series: list[str]
    positions: list[float | None]
    coefficients: list[list[float]]
    p_values: list[list[float]]
    effective_size: float

    def plot(
        self,
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        x_label: str = "Step",
        axes: "Figure | None" = None,
    ) -> "Figure":
        """
        Draw the PRCC of each input: one panel per series.

        A series of a `Trajectory` draws one line per input over its
        positions; any other quantity draws one bar per input.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        x_label : str
            The label of the x axis of a series over positions.
        axes : Figure | None
            A figure of commol to draw on: each panel is the `Axes` whose
            label is the name of its series. If None, a new figure.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a series.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _prcc

        return _prcc(
            self, output_file=output_file, config=config, x_label=x_label, axes=axes
        )
```

### Methods:

#### plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, x_label: str = 'Step', axes: Figure | None = None) -> Figure
```

Draw the PRCC of each input: one panel per series.

A series of a `Trajectory` draws one line per input over its positions; any other quantity draws one bar per input.

Parameters:

| Name          | Type         | Description                                         | Default                                                                                                             |
| ------------- | ------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`                                              | Path to save the figure to. If None, the figure is only returned.                                                   |
| `config`      | \`PlotConfig | None\`                                              | Layout and style. If None, the defaults.                                                                            |
| `x_label`     | `str`        | The label of the x axis of a series over positions. | `'Step'`                                                                                                            |
| `axes`        | \`Figure     | None\`                                              | A figure of commol to draw on: each panel is the Axes whose label is the name of its series. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a series.                            |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    x_label: str = "Step",
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw the PRCC of each input: one panel per series.

    A series of a `Trajectory` draws one line per input over its
    positions; any other quantity draws one bar per input.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    x_label : str
        The label of the x axis of a series over positions.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the name of its series. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a series.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _prcc

    return _prcc(
        self, output_file=output_file, config=config, x_label=x_label, axes=axes
    )
```

## Morris

Bases: `SensitivityMethod[MorrisResult]`, `RangeChecked`

Morris screening (Morris 1991), with the `μ*` of Campolongo, Cariboni and Saltelli (2007): which inputs matter, in few runs.

Each trajectory starts at a random level of every input and moves the inputs one at a time, by half the levels; the levels are the centres of strata of equal probability of each distribution. It needs independent inputs: a step that moves one input breaks the dependence of a sample.

Attributes:

| Name           | Type             | Description                                                                 |
| -------------- | ---------------- | --------------------------------------------------------------------------- |
| `trajectories` | `int`            | The number of trajectories; each runs one more point than there are inputs. |
| `levels`       | `int`            | The number of levels of each input; even, at least 2.                       |
| `quantities`   | `list[quantity]` | What each run is reduced to.                                                |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class Morris(SensitivityMethod[MorrisResult], RangeChecked):
    """
    Morris screening (Morris 1991), with the ``μ*`` of Campolongo, Cariboni
    and Saltelli (2007): which inputs matter, in few runs.

    Each trajectory starts at a random level of every input and moves the
    inputs one at a time, by half the levels; the levels are the centres of
    strata of equal probability of each distribution. It needs independent
    inputs: a step that moves one input breaks the dependence of a sample.

    Attributes
    ----------
    trajectories : int
        The number of trajectories; each runs one more point than there are
        inputs.
    levels : int
        The number of levels of each input; even, at least 2.
    quantities : list[quantity]
        What each run is reduced to.
    """

    trajectories: int = Field(description="The number of trajectories.")
    levels: int = Field(description="The number of levels; even.")
    quantities: list[_Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
```

## MorrisResult

Bases: `ResultModel`

The Morris indices of each input for each quantity.

The elementary effects are on the scale of probabilities: a step moves an input by half the range of `F(x)`, so the indices of inputs of different units compare. Rank the inputs by `mu_star`; a `sigma` near or above `mu_star` says the effect of the input depends on the other inputs or is not linear.

Attributes:

| Name      | Type                | Description                                                                                   |
| --------- | ------------------- | --------------------------------------------------------------------------------------------- |
| `inputs`  | `list[str]`         | The inputs, in order.                                                                         |
| `labels`  | `list[str]`         | The label of each quantity.                                                                   |
| `mu`      | `list[list[float]]` | The mean elementary effect, [quantity][input].                                                |
| `mu_star` | `list[list[float]]` | The mean absolute elementary effect, [quantity][input].                                       |
| `sigma`   | `list[list[float]]` | The standard deviation of the elementary effects, [quantity][input]; NaN with one trajectory. |
| `runs`    | `int`               | The number of points run: one more than the inputs per trajectory.                            |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class MorrisResult(ResultModel):
    """
    The Morris indices of each input for each quantity.

    The elementary effects are on the scale of probabilities: a step moves
    an input by half the range of `F(x)`, so the indices of inputs of
    different units compare. Rank the inputs by `mu_star`; a `sigma` near or
    above `mu_star` says the effect of the input depends on the other inputs
    or is not linear.

    Attributes
    ----------
    inputs : list[str]
        The inputs, in order.
    labels : list[str]
        The label of each quantity.
    mu : list[list[float]]
        The mean elementary effect, ``[quantity][input]``.
    mu_star : list[list[float]]
        The mean absolute elementary effect, ``[quantity][input]``.
    sigma : list[list[float]]
        The standard deviation of the elementary effects,
        ``[quantity][input]``; NaN with one trajectory.
    runs : int
        The number of points run: one more than the inputs per trajectory.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: list[str]
    labels: list[str]
    mu: list[list[float]]
    mu_star: list[list[float]]
    sigma: list[list[float]]
    runs: int

    def plot(
        self,
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        axes: "Figure | None" = None,
    ) -> "Figure":
        """
        Draw ``σ`` against ``μ*`` of each input: one panel per quantity,
        one labelled point per input.

        An input far to the right matters; one high up acts through
        interactions or a non-linear effect.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        axes : Figure | None
            A figure of commol to draw on: each panel is the `Axes` whose
            label is the label of its quantity. If None, a new figure.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a quantity.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _morris

        return _morris(self, output_file=output_file, config=config, axes=axes)
```

### Methods:

#### plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, axes: Figure | None = None) -> Figure
```

Draw `σ` against `μ*` of each input: one panel per quantity, one labelled point per input.

An input far to the right matters; one high up acts through interactions or a non-linear effect.

Parameters:

| Name          | Type         | Description | Default                                                                                                                |
| ------------- | ------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`      | Path to save the figure to. If None, the figure is only returned.                                                      |
| `config`      | \`PlotConfig | None\`      | Layout and style. If None, the defaults.                                                                               |
| `axes`        | \`Figure     | None\`      | A figure of commol to draw on: each panel is the Axes whose label is the label of its quantity. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a quantity.                          |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw ``σ`` against ``μ*`` of each input: one panel per quantity,
    one labelled point per input.

    An input far to the right matters; one high up acts through
    interactions or a non-linear effect.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the label of its quantity. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a quantity.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _morris

    return _morris(self, output_file=output_file, config=config, axes=axes)
```

## Sobol

Bases: `SensitivityMethod[SobolResult]`, `RangeChecked`

Sobol indices on the design of Saltelli (2010): first-order indices by the estimator of Saltelli (2010) and total indices by the estimator of Jansen (1999), with intervals from a bootstrap of the rows.

The base matrices hold independent Monte Carlo draws, so the bootstrap intervals are valid. It needs independent inputs: with dependent inputs the indices lose their meaning and no longer add up; PRCC and propagation take a sample.

Attributes:

| Name               | Type             | Description                                                                  |
| ------------------ | ---------------- | ---------------------------------------------------------------------------- |
| `draws`            | `int`            | The rows of each base matrix; the analysis runs draws · (inputs + 2) points. |
| `replicates`       | `int`            | The bootstrap replicates of the intervals, at least 2.                       |
| `confidence_level` | `float`          | The level of the intervals, strictly between 0 and 1.                        |
| `quantities`       | `list[quantity]` | What each run is reduced to.                                                 |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class Sobol(SensitivityMethod[SobolResult], RangeChecked):
    """
    Sobol indices on the design of Saltelli (2010): first-order indices by
    the estimator of Saltelli (2010) and total indices by the estimator of
    Jansen (1999), with intervals from a bootstrap of the rows.

    The base matrices hold independent Monte Carlo draws, so the bootstrap
    intervals are valid. It needs independent inputs: with dependent inputs
    the indices lose their meaning and no longer add up; PRCC and propagation
    take a sample.

    Attributes
    ----------
    draws : int
        The rows of each base matrix; the analysis runs
        ``draws · (inputs + 2)`` points.
    replicates : int
        The bootstrap replicates of the intervals, at least 2.
    confidence_level : float
        The level of the intervals, strictly between 0 and 1.
    quantities : list[quantity]
        What each run is reduced to.
    """

    draws: int = Field(description="Rows of each base matrix.")
    replicates: int = Field(description="Bootstrap replicates.")
    confidence_level: float = Field(description="Level of the intervals.")
    quantities: list[_Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
```

## SobolResult

Bases: `ResultModel`

The Sobol indices of each input for each quantity, with their bootstrap intervals.

The first-order index is the share of the variance of the quantity the input explains alone; the total index adds every interaction the input takes part in. A total index near 0 says the input can be fixed; a total index well above the first-order one says the input acts through interactions.

Attributes:

| Name                       | Type                | Description                                                                                                                             |
| -------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `inputs`                   | `list[str]`         | The inputs, in order.                                                                                                                   |
| `labels`                   | `list[str]`         | The label of each quantity.                                                                                                             |
| `confidence_level`         | `float`             | The level of the intervals.                                                                                                             |
| `first, total`             | `list[list[float]]` | The first-order and total indices, [quantity][input]; NaN for a quantity of no variance.                                                |
| `first_lower, first_upper` | `list[list[float]]` | The ends of the interval of each first-order index; NaN when a bootstrap replicate of the rows has no variance, which more draws avoid. |
| `total_lower, total_upper` | `list[list[float]]` | The ends of the interval of each total index.                                                                                           |
| `runs`                     | `int`               | The number of points run: draws · (inputs + 2).                                                                                         |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class SobolResult(ResultModel):
    """
    The Sobol indices of each input for each quantity, with their bootstrap
    intervals.

    The first-order index is the share of the variance of the quantity the
    input explains alone; the total index adds every interaction the input
    takes part in. A total index near 0 says the input can be fixed; a total
    index well above the first-order one says the input acts through
    interactions.

    Attributes
    ----------
    inputs : list[str]
        The inputs, in order.
    labels : list[str]
        The label of each quantity.
    confidence_level : float
        The level of the intervals.
    first, total : list[list[float]]
        The first-order and total indices, ``[quantity][input]``; NaN for a
        quantity of no variance.
    first_lower, first_upper : list[list[float]]
        The ends of the interval of each first-order index; NaN when a
        bootstrap replicate of the rows has no variance, which more
        `draws` avoid.
    total_lower, total_upper : list[list[float]]
        The ends of the interval of each total index.
    runs : int
        The number of points run: ``draws · (inputs + 2)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: list[str]
    labels: list[str]
    confidence_level: float
    first: list[list[float]]
    total: list[list[float]]
    first_lower: list[list[float]]
    first_upper: list[list[float]]
    total_lower: list[list[float]]
    total_upper: list[list[float]]
    runs: int

    def plot(
        self,
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        axes: "Figure | None" = None,
    ) -> "Figure":
        """
        Draw the first-order and total Sobol indices: one panel per
        quantity, one pair of bars per input, with their intervals.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        axes : Figure | None
            A figure of commol to draw on: each panel is the `Axes` whose
            label is the label of its quantity. If None, a new figure.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a quantity.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _sobol

        return _sobol(self, output_file=output_file, config=config, axes=axes)
```

### Methods:

#### plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, axes: Figure | None = None) -> Figure
```

Draw the first-order and total Sobol indices: one panel per quantity, one pair of bars per input, with their intervals.

Parameters:

| Name          | Type         | Description | Default                                                                                                                |
| ------------- | ------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`      | Path to save the figure to. If None, the figure is only returned.                                                      |
| `config`      | \`PlotConfig | None\`      | Layout and style. If None, the defaults.                                                                               |
| `axes`        | \`Figure     | None\`      | A figure of commol to draw on: each panel is the Axes whose label is the label of its quantity. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a quantity.                          |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw the first-order and total Sobol indices: one panel per
    quantity, one pair of bars per input, with their intervals.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the label of its quantity. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a quantity.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _sobol

    return _sobol(self, output_file=output_file, config=config, axes=axes)
```

## PolynomialChaos

Bases: `SensitivityMethod[ChaosResult]`, `RangeChecked`

A polynomial chaos expansion (Sudret 2008): a polynomial surrogate of each quantity, and the Sobol indices of its coefficients, in far fewer runs than `Sobol` when the quantity is smooth.

Every input goes to `u = F(x)`, uniform on `[0, 1]` through its distribution, so one basis of orthonormal Legendre polynomials fits every distribution, truncated ones included. The coefficients come from least squares on a Latin hypercube; the leave-one-out error says how well the expansion predicts the model. It needs independent inputs.

The basis keeps every term whose degrees `α` have `Σ α_i^q ≤ degree^q` and that holds at most `max_interaction` inputs. The defaults keep the total-degree basis, of `(k + degree)! / (k! degree!)` terms in `k` inputs; a truncation keeps the terms of low interaction order, so a basis of many inputs needs far fewer terms and runs. The number of terms is known before any run.

Attributes:

| Name              | Type             | Description                                                                                                                      |
| ----------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `degree`          | `int`            | The degree of the basis: the largest degree of an input in a term, and the total degree of a term when q_norm is 1.              |
| `q_norm`          | `float`          | The q of the hyperbolic truncation, in (0, 1\]: below 1, a term of several inputs keeps a lower total degree than a term of one. |
| `max_interaction` | \`int            | None\`                                                                                                                           |
| `draws`           | `int`            | The points of the Latin hypercube; more than the terms, and two or three times as many is the usual choice.                      |
| `quantities`      | `list[quantity]` | What each run is reduced to.                                                                                                     |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class PolynomialChaos(SensitivityMethod[ChaosResult], RangeChecked):
    """
    A polynomial chaos expansion (Sudret 2008): a polynomial surrogate of
    each quantity, and the Sobol indices of its coefficients, in far fewer
    runs than `Sobol` when the quantity is smooth.

    Every input goes to ``u = F(x)``, uniform on ``[0, 1]`` through its
    distribution, so one basis of orthonormal Legendre polynomials fits every
    distribution, truncated ones included. The coefficients come from least
    squares on a Latin hypercube; the leave-one-out error says how well the
    expansion predicts the model. It needs independent inputs.

    The basis keeps every term whose degrees ``α`` have
    ``Σ α_i^q ≤ degree^q`` and that holds at most ``max_interaction``
    inputs. The defaults keep the total-degree basis, of
    ``(k + degree)! / (k! degree!)`` terms in ``k`` inputs; a truncation
    keeps the terms of low interaction order, so a basis of many inputs
    needs far fewer terms and runs. The number of terms is known before any
    run.

    Attributes
    ----------
    degree : int
        The degree of the basis: the largest degree of an input in a term,
        and the total degree of a term when ``q_norm`` is 1.
    q_norm : float
        The q of the hyperbolic truncation, in ``(0, 1]``: below 1, a term
        of several inputs keeps a lower total degree than a term of one.
    max_interaction : int | None
        The largest number of inputs of a term; None for no limit.
    draws : int
        The points of the Latin hypercube; more than the terms, and two or
        three times as many is the usual choice.
    quantities : list[quantity]
        What each run is reduced to.
    """

    degree: int = Field(description="The degree of the basis.")
    q_norm: float = Field(default=1.0, description="The q of the q-norm of the basis.")
    max_interaction: int | None = Field(
        default=None, description="The largest number of inputs of a term."
    )
    draws: int = Field(description="Points of the Latin hypercube.")
    quantities: list[_Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
```

## ChaosResult

Bases: `ResultModel`

A polynomial chaos expansion of each quantity, its accuracy and the Sobol indices of its coefficients.

The expansion is a surrogate of the model: `predict` gives the quantities at any values of the inputs inside their bounds, at the cost of a polynomial. Read `loo_error` first: the indices are those of the expansion, and they are those of the model only as far as the expansion predicts it.

Attributes:

| Name              | Type                | Description                                                                                                        |
| ----------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `inputs`          | `list[str]`         | The inputs, in order.                                                                                              |
| `input_space`     | `InputSpace`        | The inputs and their distributions, which predict maps values through.                                             |
| `labels`          | `list[str]`         | The label of each quantity.                                                                                        |
| `degree`          | `int`               | The degree of the basis: the bound of the q-norm of every term.                                                    |
| `q_norm`          | `float`             | The q of the q-norm of the basis; 1 for the total-degree basis.                                                    |
| `max_interaction` | \`int               | None\`                                                                                                             |
| `indices`         | `list[list[int]]`   | The degree of each input in each term, [term][input]; term 0 is the constant.                                      |
| `coefficients`    | `list[list[float]]` | The coefficient of each term for each quantity, [quantity][term], on orthonormal Legendre polynomials of u = F(x). |
| `loo_error`       | `list[float]`       | The leave-one-out error of each quantity, relative to its variance; NaN for a quantity of no variance.             |
| `mean, variance`  | `list[float]`       | The mean and the variance of each quantity under the expansion.                                                    |
| `first, total`    | `list[list[float]]` | The first-order and total Sobol indices, [quantity][input]; NaN for a quantity of no variance.                     |
| `runs`            | `int`               | The number of points run.                                                                                          |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class ChaosResult(ResultModel):
    """
    A polynomial chaos expansion of each quantity, its accuracy and the
    Sobol indices of its coefficients.

    The expansion is a surrogate of the model: `predict` gives the
    quantities at any values of the inputs inside their bounds, at the cost
    of a polynomial. Read
    `loo_error` first: the indices are those of the expansion, and they are
    those of the model only as far as the expansion predicts it.

    Attributes
    ----------
    inputs : list[str]
        The inputs, in order.
    input_space : InputSpace
        The inputs and their distributions, which `predict` maps values
        through.
    labels : list[str]
        The label of each quantity.
    degree : int
        The degree of the basis: the bound of the q-norm of every term.
    q_norm : float
        The q of the q-norm of the basis; 1 for the total-degree basis.
    max_interaction : int | None
        The largest number of inputs of a term; None for no limit.
    indices : list[list[int]]
        The degree of each input in each term, ``[term][input]``; term 0 is
        the constant.
    coefficients : list[list[float]]
        The coefficient of each term for each quantity, ``[quantity][term]``,
        on orthonormal Legendre polynomials of ``u = F(x)``.
    loo_error : list[float]
        The leave-one-out error of each quantity, relative to its variance;
        NaN for a quantity of no variance.
    mean, variance : list[float]
        The mean and the variance of each quantity under the expansion.
    first, total : list[list[float]]
        The first-order and total Sobol indices, ``[quantity][input]``; NaN
        for a quantity of no variance.
    runs : int
        The number of points run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: list[str]
    input_space: InputSpace
    labels: list[str]
    degree: int
    q_norm: float
    max_interaction: int | None
    indices: list[list[int]]
    coefficients: list[list[float]]
    loo_error: list[float]
    mean: list[float]
    variance: list[float]
    first: list[list[float]]
    total: list[list[float]]
    runs: int

    def plot(
        self,
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        axes: "Figure | None" = None,
    ) -> "Figure":
        """
        Draw the first-order and total Sobol indices of the expansion: one
        panel per quantity, one pair of bars per input.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        axes : Figure | None
            A figure of commol to draw on: each panel is the `Axes` whose
            label is the label of its quantity. If None, a new figure.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a quantity.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _sobol

        return _sobol(self, output_file=output_file, config=config, axes=axes)

    def predict(self, values: Sequence[Sequence[float]]) -> list[list[float]]:
        """
        The quantities the expansion gives at each set of input values.

        Parameters
        ----------
        values : Sequence[Sequence[float]]
            One value per input, in the order of `inputs`, for each point,
            each inside the bounds of its input.

        Returns
        -------
        list[list[float]]
            The value of each quantity at each point, ``[point][quantity]``.

        Raises
        ------
        ValueError
            If a point has not one value per input or a value outside the
            bounds of its input, the truncation is not valid, or the
            coefficients do not fit the basis.
        """
        return commol_rs.inference.chaos_evaluate(
            engine_json(self.input_space),
            self.degree,
            self.q_norm,
            self.max_interaction,
            self.coefficients,
            [list(point) for point in values],
        )
```

### Methods:

#### plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, axes: Figure | None = None) -> Figure
```

Draw the first-order and total Sobol indices of the expansion: one panel per quantity, one pair of bars per input.

Parameters:

| Name          | Type         | Description | Default                                                                                                                |
| ------------- | ------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`      | Path to save the figure to. If None, the figure is only returned.                                                      |
| `config`      | \`PlotConfig | None\`      | Layout and style. If None, the defaults.                                                                               |
| `axes`        | \`Figure     | None\`      | A figure of commol to draw on: each panel is the Axes whose label is the label of its quantity. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a quantity.                          |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw the first-order and total Sobol indices of the expansion: one
    panel per quantity, one pair of bars per input.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the label of its quantity. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a quantity.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _sobol

    return _sobol(self, output_file=output_file, config=config, axes=axes)
```

#### predict

```python
predict(values: Sequence[Sequence[float]]) -> list[list[float]]
```

The quantities the expansion gives at each set of input values.

Parameters:

| Name     | Type                        | Description                                                                                       | Default    |
| -------- | --------------------------- | ------------------------------------------------------------------------------------------------- | ---------- |
| `values` | `Sequence[Sequence[float]]` | One value per input, in the order of inputs, for each point, each inside the bounds of its input. | *required* |

Returns:

| Type                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `list[list[float]]` | The value of each quantity at each point, [point][quantity]. |

Raises:

| Type         | Description                                                                                                                                               |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a point has not one value per input or a value outside the bounds of its input, the truncation is not valid, or the coefficients do not fit the basis. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def predict(self, values: Sequence[Sequence[float]]) -> list[list[float]]:
    """
    The quantities the expansion gives at each set of input values.

    Parameters
    ----------
    values : Sequence[Sequence[float]]
        One value per input, in the order of `inputs`, for each point,
        each inside the bounds of its input.

    Returns
    -------
    list[list[float]]
        The value of each quantity at each point, ``[point][quantity]``.

    Raises
    ------
    ValueError
        If a point has not one value per input or a value outside the
        bounds of its input, the truncation is not valid, or the
        coefficients do not fit the basis.
    """
    return commol_rs.inference.chaos_evaluate(
        engine_json(self.input_space),
        self.degree,
        self.q_norm,
        self.max_interaction,
        self.coefficients,
        [list(point) for point in values],
    )
```

## Level

Bases: `_Outputs`

The level of some outputs: their sum at each position.

A quantity given a name or a list of names reads this series.

Attributes:

| Name      | Type        | Description                                    |
| --------- | ----------- | ---------------------------------------------- |
| `outputs` | `list[str]` | The outputs summed; one name is a list of one. |

Source code in `commol/context/inference/quantities.py`

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

    @property
    def text(self) -> str:
        """The series as a label reads it: the outputs."""
        return self.output_name

    def check_ends(self, positions: Sequence[float]) -> None:
        """A level reads any position."""
```

### Attributes

#### text

```python
text: str
```

The series as a label reads it: the outputs.

### Methods:

#### check_ends

```python
check_ends(positions: Sequence[float]) -> None
```

A level reads any position.

Source code in `commol/context/inference/quantities.py`

```python
def check_ends(self, positions: Sequence[float]) -> None:
    """A level reads any position."""
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

Source code in `commol/context/inference/quantities.py`

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

    @property
    def text(self) -> str:
        """The series as a label reads it: the gain over windows of a width."""
        return f"{self.output_name} gain over windows of {_position_text(self.window)}"

    def check_ends(self, positions: Sequence[float]) -> None:
        """
        No window starts before 0, with the rule of the engine
        (`check_window_end` of `crates/commol-calibration/src/observation.rs`).

        Raises
        ------
        ValueError
            With the message of the engine, if a window starts before 0.
        """
        for at in positions:
            commol_rs.calibration.check_window_end(at, self.window)
```

### Attributes

#### text

```python
text: str
```

The series as a label reads it: the gain over windows of a width.

### Methods:

#### check_ends

```python
check_ends(positions: Sequence[float]) -> None
```

No window starts before 0, with the rule of the engine (`check_window_end` of `crates/commol-calibration/src/observation.rs`).

Raises:

| Type         | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `ValueError` | With the message of the engine, if a window starts before 0. |

Source code in `commol/context/inference/quantities.py`

```python
def check_ends(self, positions: Sequence[float]) -> None:
    """
    No window starts before 0, with the rule of the engine
    (`check_window_end` of `crates/commol-calibration/src/observation.rs`).

    Raises
    ------
    ValueError
        With the message of the engine, if a window starts before 0.
    """
    for at in positions:
        commol_rs.calibration.check_window_end(at, self.window)
```

## ValueAt

Bases: `_OfSeries`, `_OneScalar`, `RangeChecked`

The value of a series at a position: the level of the outputs, or their gain over the window that ends there.

Attributes:

| Name     | Type    | Description                                           |
| -------- | ------- | ----------------------------------------------------- |
| `series` | \`Level | Increments\`                                          |
| `at`     | `float` | The position; with Increments, the end of the window. |
| `label`  | \`str   | None\`                                                |

Source code in `commol/context/inference/quantities.py`

```python
class ValueAt(_OfSeries, _OneScalar, RangeChecked):
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

    @model_validator(mode="after")
    def _window_after_zero(self) -> Self:
        """The window of increments ends at least its width after 0."""
        self.series.check_ends([self.at])
        return self

    def default_label(self) -> str:
        """The series at `at`."""
        return f"{self.series.text} at {_position_text(self.at)}"
```

### Methods:

#### default_label

```python
default_label() -> str
```

The series at `at`.

Source code in `commol/context/inference/quantities.py`

```python
def default_label(self) -> str:
    """The series at `at`."""
    return f"{self.series.text} at {_position_text(self.at)}"
```

## Gain

Bases: `_Outputs`, `_OneScalar`, `RangeChecked`

The value at `end` minus the value at `start`: what entered a cumulative output over `(start, end]`, as an observation with the window `end - start` that starts at exactly `start` reads it.

Attributes:

| Name      | Type        | Description                                    |
| --------- | ----------- | ---------------------------------------------- |
| `outputs` | `list[str]` | The outputs summed; one name is a list of one. |
| `start`   | `float`     | The start of the window.                       |
| `end`     | `float`     | The end of the window, after start.            |
| `label`   | \`str       | None\`                                         |

Source code in `commol/context/inference/quantities.py`

```python
class Gain(_Outputs, _OneScalar, RangeChecked):
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

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        """A window ends after it starts."""
        if self.end <= self.start:
            raise ValueError(
                f"A gain ends after it starts, got start {self.start} and end "
                f"{self.end}"
            )
        return self

    def default_label(self) -> str:
        """The gain of `outputs` over the window."""
        start, end = _position_text(self.start), _position_text(self.end)
        return f"{self.output_name} gain over ({start}, {end}]"
```

### Methods:

#### default_label

```python
default_label() -> str
```

The gain of `outputs` over the window.

Source code in `commol/context/inference/quantities.py`

```python
def default_label(self) -> str:
    """The gain of `outputs` over the window."""
    start, end = _position_text(self.start), _position_text(self.end)
    return f"{self.output_name} gain over ({start}, {end}]"
```

## Peak

Bases: `_AtPositions`, `_OneScalar`, `RangeChecked`

The largest value of a series at the positions.

The series decides what the peak is: the peak of `Increments(acc, window=7)` at the ends 7, 14, ... is the largest gain of a window of 7, while the peak of the level of an accumulator, which never falls, is its value at the last position. A NaN at a position makes the peak NaN.

Attributes:

| Name        | Type          | Description                                                      |
| ----------- | ------------- | ---------------------------------------------------------------- |
| `series`    | \`Level       | Increments\`                                                     |
| `positions` | `list[float]` | The positions compared, ascending; with Increments, window ends. |
| `label`     | \`str         | None\`                                                           |

Source code in `commol/context/inference/quantities.py`

```python
class Peak(_AtPositions, _OneScalar, RangeChecked):
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

    def default_label(self) -> str:
        """The peak of the series."""
        return f"peak of {self.series.text}"
```

### Methods:

#### default_label

```python
default_label() -> str
```

The peak of the series.

Source code in `commol/context/inference/quantities.py`

```python
def default_label(self) -> str:
    """The peak of the series."""
    return f"peak of {self.series.text}"
```

## PeakPosition

Bases: `_AtPositions`, `_OneScalar`, `RangeChecked`

The first of the positions where the value of a series is largest; NaN when the value at a position is NaN.

Attributes:

| Name        | Type          | Description                                                      |
| ----------- | ------------- | ---------------------------------------------------------------- |
| `series`    | \`Level       | Increments\`                                                     |
| `positions` | `list[float]` | The positions compared, ascending; with Increments, window ends. |
| `label`     | \`str         | None\`                                                           |

Source code in `commol/context/inference/quantities.py`

```python
class PeakPosition(_AtPositions, _OneScalar, RangeChecked):
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

    def default_label(self) -> str:
        """The position of the peak of the series."""
        return f"position of the peak of {self.series.text}"
```

### Methods:

#### default_label

```python
default_label() -> str
```

The position of the peak of the series.

Source code in `commol/context/inference/quantities.py`

```python
def default_label(self) -> str:
    """The position of the peak of the series."""
    return f"position of the peak of {self.series.text}"
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

Source code in `commol/context/inference/quantities.py`

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

    def _name(self) -> str:
        """The name of the series."""
        return self.label or self.series.text

    def labels(self) -> list[str]:
        """The name and each position."""
        return [f"{self._name()} at {_position_text(at)}" for at in self.positions]

    def axes(self) -> list[tuple[str, float | None]]:
        """Its name and each position."""
        return [(self._name(), at) for at in self.positions]
```

### Methods:

#### labels

```python
labels() -> list[str]
```

The name and each position.

Source code in `commol/context/inference/quantities.py`

```python
def labels(self) -> list[str]:
    """The name and each position."""
    return [f"{self._name()} at {_position_text(at)}" for at in self.positions]
```

#### axes

```python
axes() -> list[tuple[str, float | None]]
```

Its name and each position.

Source code in `commol/context/inference/quantities.py`

```python
def axes(self) -> list[tuple[str, float | None]]:
    """Its name and each position."""
    return [(self._name(), at) for at in self.positions]
```

## Ratio

Bases: `_OneScalar`

The ratio of two quantities of one value each, numerator N over denominator D.

The ratio of a run whose denominator is 0 is NaN, also when the numerator is not 0, and follows the NaN rule of every method. A method that reduces the points of a design (`PRCC`, `Morris`, `Sobol`, `PolynomialChaos`) reads one value per point; with a stochastic engine, `replicate_rule` says how the replicates of a point combine. A `Propagation`, the `Scenarios` and the quantities of one run keep the ratio of each run.

Attributes:

| Name             | Type                                | Description                                                                                                                                                                                     |
| ---------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `numerator`      | \`ValueAt                           | Gain                                                                                                                                                                                            |
| `denominator`    | \`ValueAt                           | Gain                                                                                                                                                                                            |
| `replicate_rule` | `{within, within_positive, pooled}` | "within": the mean of the ratio of each replicate, E[N/D]; a replicate with D = 0 leaves the point with no value. "within_positive": the mean of the ratio of the replicates with D > 0, E\[N/D |
| `label`          | \`str                               | None\`                                                                                                                                                                                          |

Source code in `commol/context/inference/quantities.py`

```python
class Ratio(_OneScalar):
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

    @field_validator("numerator", "denominator")
    @classmethod
    def _one_value(cls, value: QuantityBase) -> QuantityBase:
        """An operand gives one value per run."""
        if isinstance(value, PeakPosition | Trajectory):
            raise ValueError(
                f"A Ratio divides quantities of one value each (ValueAt, Gain, "
                f"Peak), got the {type(value).__name__} '{value.labels()[0]}'"
            )
        return value

    def default_label(self) -> str:
        """The numerator over the denominator."""
        numerator = self.numerator.labels()[0]
        denominator = self.denominator.labels()[0]
        return f"({numerator}) / ({denominator})"
```

### Methods:

#### default_label

```python
default_label() -> str
```

The numerator over the denominator.

Source code in `commol/context/inference/quantities.py`

```python
def default_label(self) -> str:
    """The numerator over the denominator."""
    numerator = self.numerator.labels()[0]
    denominator = self.denominator.labels()[0]
    return f"({numerator}) / ({denominator})"
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

Source code in `commol/context/inference/quantities.py`

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
    if not quantities:
        raise ValueError("Give at least one quantity of interest.")
    return commol_rs.inference.quantity_schedule(engine_json(list(quantities)))
```
