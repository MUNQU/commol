# Sensitivity API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The sensitivity analysis, what it varies, the quantities of interest it reads from each run, and the base of its methods. Guide: [Sensitivity analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/index.md).

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
    ):
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
            the same analysis runs on either engine.
        seed : int | None, optional
            The seed of the designs and the replicates. Required with a
            stochastic engine; otherwise None draws one from the operating
            system, which the analysis logs.

        Raises
        ------
        ValueError
            If an input is not a value of the model, the replicates or the
            seed do not fit the engine, `replicates` is below 1, or the seed
            is not a whole number from 0 to ``2**64 - 1``.
        """
        engine_type = simulation.model_definition.engine.type
        stochastic = commol_rs.engine.EngineType(engine_type).is_stochastic
        if replicates is not None and replicates < 1:
            raise ValueError(f"replicates is at least 1, got {replicates}")
        if stochastic and replicates is None:
            raise ValueError(
                f"The {engine_type} engine is stochastic: set replicates, how many "
                "replicates each point of a design runs."
            )
        if stochastic and seed is None:
            raise ValueError(
                f"The {engine_type} engine is stochastic: set seed, so the "
                "replicates and the analysis are reproducible."
            )
        self.simulation: Simulation = simulation
        self.inputs: InputSpace = inputs
        self.replicates: int | None = replicates
        if seed is not None:
            check_seed(seed)
        self._seed: int | None = seed
        check_values(
            simulation.model_definition,
            simulation.engine,
            inputs.parameters,
            inputs.initial_condition_remainder,
        )
        self._rust: "ForwardAnalysisProtocol" = commol_rs.inference.ForwardAnalysis(
            simulation.engine,
            engine_state.rust_parameters(inputs.parameters),
            distributions=inputs.rust_distributions(),
            sample=None if inputs.sample is None else inputs.sample.rust_sample,
            initial_condition_remainder=inputs.initial_condition_remainder,
            replicates=replicates if stochastic else None,
            seed=self.seed,
        )

    @property
    def rust_analysis(self) -> "ForwardAnalysisProtocol":
        """
        The forward analysis built in Rust, for the runners of methods.

        Returns
        -------
        ForwardAnalysisProtocol
            The forward model and the inputs every method reads.
        """
        return self._rust

    def rust_predictive(
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
        return self._rust.predictive(sample.rust_sample, positions)

    def rust_observation_draws(
        self,
        sample: ParameterSample,
        observed_data: list[ObservedDataPoint],
        noise: ObservationNoise,
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
        return self._rust.observation_draws(
            sample.rust_sample, engine_state.rust_observed_data(observed_data)
        )

    def rust_log_score(
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
            The seed given, or a seed drawn from the operating system once
            per analysis and logged.
        """
        if self._seed is not None:
            return self._seed
        if not hasattr(self, "_drawn_seed"):
            self._drawn_seed = draw_seed()
            logger.info(
                "No seed set; drew the seed %d, which repeats this analysis.",
                self._drawn_seed,
            )
        return self._drawn_seed

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
            independent inputs on inputs from a sample.
        """
        return run_method(self, method, SensitivityMethod)
```

### Attributes

#### rust_analysis

```python
rust_analysis: ForwardAnalysisProtocol
```

The forward analysis built in Rust, for the runners of methods.

Returns:

| Type                      | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `ForwardAnalysisProtocol` | The forward model and the inputs every method reads. |

#### seed

```python
seed: int
```

The seed of the analysis' random numbers.

Returns:

| Type  | Description                                                                             |
| ----- | --------------------------------------------------------------------------------------- |
| `int` | The seed given, or a seed drawn from the operating system once per analysis and logged. |

### Methods:

#### __init__

```python
__init__(simulation: Simulation, inputs: InputSpace, *, replicates: int | None = None, seed: int | None = None)
```

Build the analysis of `inputs` on `simulation`.

Parameters:

| Name         | Type         | Description           | Default                                                                                                                                                                                                                                                                                                                                                                        |
| ------------ | ------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `simulation` | `Simulation` | The model to analyse. | *required*                                                                                                                                                                                                                                                                                                                                                                     |
| `inputs`     | `InputSpace` | What varies.          | *required*                                                                                                                                                                                                                                                                                                                                                                     |
| `replicates` | \`int        | None\`                | For a stochastic engine, required, at least 1: how many replicates each point runs. A point of the design of PRCC, Morris, Sobol or polynomial chaos holds the mean of its replicates, on the same seeds at every point; a propagation keeps every replicate, each with a seed of its own. A deterministic engine does not use it, so the same analysis runs on either engine. |
| `seed`       | \`int        | None\`                | The seed of the designs and the replicates. Required with a stochastic engine; otherwise None draws one from the operating system, which the analysis logs.                                                                                                                                                                                                                    |

Raises:

| Type         | Description                                                                                                                                                                |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If an input is not a value of the model, the replicates or the seed do not fit the engine, replicates is below 1, or the seed is not a whole number from 0 to 2\*\*64 - 1. |

Source code in `commol/api/sensitivity.py`

```python
def __init__(
    self,
    simulation: Simulation,
    inputs: InputSpace,
    *,
    replicates: int | None = None,
    seed: int | None = None,
):
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
        the same analysis runs on either engine.
    seed : int | None, optional
        The seed of the designs and the replicates. Required with a
        stochastic engine; otherwise None draws one from the operating
        system, which the analysis logs.

    Raises
    ------
    ValueError
        If an input is not a value of the model, the replicates or the
        seed do not fit the engine, `replicates` is below 1, or the seed
        is not a whole number from 0 to ``2**64 - 1``.
    """
    engine_type = simulation.model_definition.engine.type
    stochastic = commol_rs.engine.EngineType(engine_type).is_stochastic
    if replicates is not None and replicates < 1:
        raise ValueError(f"replicates is at least 1, got {replicates}")
    if stochastic and replicates is None:
        raise ValueError(
            f"The {engine_type} engine is stochastic: set replicates, how many "
            "replicates each point of a design runs."
        )
    if stochastic and seed is None:
        raise ValueError(
            f"The {engine_type} engine is stochastic: set seed, so the "
            "replicates and the analysis are reproducible."
        )
    self.simulation: Simulation = simulation
    self.inputs: InputSpace = inputs
    self.replicates: int | None = replicates
    if seed is not None:
        check_seed(seed)
    self._seed: int | None = seed
    check_values(
        simulation.model_definition,
        simulation.engine,
        inputs.parameters,
        inputs.initial_condition_remainder,
    )
    self._rust: "ForwardAnalysisProtocol" = commol_rs.inference.ForwardAnalysis(
        simulation.engine,
        engine_state.rust_parameters(inputs.parameters),
        distributions=inputs.rust_distributions(),
        sample=None if inputs.sample is None else inputs.sample.rust_sample,
        initial_condition_remainder=inputs.initial_condition_remainder,
        replicates=replicates if stochastic else None,
        seed=self.seed,
    )
```

#### rust_predictive

```python
rust_predictive(sample: ParameterSample, positions: list[float]) -> PredictiveProtocol
```

The run of every draw of `sample` at `positions`, for a `Predictive`: each replicate of a stochastic engine has a seed of its own, derived from the seed of the analysis, as in a propagation.

Parameters:

| Name        | Type              | Description                                | Default    |
| ----------- | ----------------- | ------------------------------------------ | ---------- |
| `sample`    | `ParameterSample` | The draws, in the order of the inputs.     | *required* |
| `positions` | `list[float]`     | Positions to report at, sorted and unique. | *required* |

Returns:

| Type                 | Description                 |
| -------------------- | --------------------------- |
| `PredictiveProtocol` | The runs, computed in Rust. |

Source code in `commol/api/sensitivity.py`

```python
def rust_predictive(
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
    return self._rust.predictive(sample.rust_sample, positions)
```

#### rust_observation_draws

```python
rust_observation_draws(sample: ParameterSample, observed_data: list[ObservedDataPoint], noise: ObservationNoise) -> ObservationDrawsProtocol
```

Every draw's value of `observed_data`, for a `Predictive`.

Parameters:

| Name            | Type                      | Description                            | Default             |
| --------------- | ------------------------- | -------------------------------------- | ------------------- |
| `sample`        | `ParameterSample`         | The draws, in the order of the inputs. | *required*          |
| `observed_data` | `list[ObservedDataPoint]` | The observations.                      | *required*          |
| `noise`         | \`NoNoise                 | ResidualNoise                          | DistributionNoise\` |

Returns:

| Type                       | Description                  |
| -------------------------- | ---------------------------- |
| `ObservationDrawsProtocol` | The draws, computed in Rust. |

Raises:

| Type         | Description                                                                                                                                                      |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the noise is not NoNoise(): residual noise needs the residuals of a fit, distribution noise the observation models of a problem, and an analysis has neither. |

Source code in `commol/api/sensitivity.py`

```python
def rust_observation_draws(
    self,
    sample: ParameterSample,
    observed_data: list[ObservedDataPoint],
    noise: ObservationNoise,
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
    return self._rust.observation_draws(
        sample.rust_sample, engine_state.rust_observed_data(observed_data)
    )
```

#### rust_log_score

```python
rust_log_score(sample: ParameterSample, observed_data: list[ObservedDataPoint]) -> list[float]
```

Refuse the log score, which `commol.scores.log_score` asks for.

Parameters:

| Name            | Type                      | Description       | Default    |
| --------------- | ------------------------- | ----------------- | ---------- |
| `sample`        | `ParameterSample`         | The draws.        | *required* |
| `observed_data` | `list[ObservedDataPoint]` | The observations. | *required* |

Raises:

| Type         | Description                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------- |
| `ValueError` | Always: the log score reads the observation models of a problem, and an analysis has no data. |

Source code in `commol/api/sensitivity.py`

```python
def rust_log_score(
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

| Type         | Description                                                                                               |
| ------------ | --------------------------------------------------------------------------------------------------------- |
| `TypeError`  | If method is not a sensitivity method; the message names its entry point.                                 |
| `ValueError` | If the method does not take these inputs, such as a method of independent inputs on inputs from a sample. |

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
        independent inputs on inputs from a sample.
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
| `distributions`               | \`dict[str, Distribution]    | None\`                                                                                                                      |
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
    distributions : dict[str, Distribution] | None
        The distribution of each input, for independent inputs.
    sample : ParameterSample | None
        The draws of the inputs, for inputs from a sample.
    initial_condition_remainder : str | None
        The bin or compartment that takes what the initial conditions leave
        of the sum the model states, as in a calibration problem.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: list[CalibrationParameter] = Field(
        min_length=1, description="The inputs, in order."
    )
    distributions: dict[str, Distribution] | None = Field(
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
    def check_inputs(self) -> Self:
        """Check the inputs against the distributions or the sample."""
        ids = [parameter.id for parameter in self.parameters]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"Inputs repeat the ids {repeated}.")
        observed = {
            CalibrationParameterType.FACTOR: (
                "a factor, which multiplies an observation"
            ),
            CalibrationParameterType.NOISE: (
                "a noise parameter, which belongs to an observation model"
            ),
        }
        for parameter in self.parameters:
            what = observed.get(CalibrationParameterType(parameter.parameter_type))
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
        distributions: Mapping[str, Distribution],
        *,
        initial_condition_remainder: str | None = None,
    ) -> Self:
        """
        Independent inputs, each with its distribution.

        Parameters
        ----------
        parameters : Sequence[CalibrationParameter]
            The inputs, in order.
        distributions : Mapping[str, Distribution]
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

    def rust_distributions(self) -> list[tuple[str, str, list[float]]] | None:
        """The distributions as the Rust runners read them, or None."""
        if self.distributions is None:
            return None
        return [self.distributions[name].rust_prior(name) for name in self.names]
```

### Attributes

#### names

```python
names: list[str]
```

The ids of the inputs, in order.

### Methods:

#### check_inputs

```python
check_inputs() -> Self
```

Check the inputs against the distributions or the sample.

Source code in `commol/context/inference/global_sensitivity.py`

```python
@model_validator(mode="after")
def check_inputs(self) -> Self:
    """Check the inputs against the distributions or the sample."""
    ids = [parameter.id for parameter in self.parameters]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"Inputs repeat the ids {repeated}.")
    observed = {
        CalibrationParameterType.FACTOR: (
            "a factor, which multiplies an observation"
        ),
        CalibrationParameterType.NOISE: (
            "a noise parameter, which belongs to an observation model"
        ),
    }
    for parameter in self.parameters:
        what = observed.get(CalibrationParameterType(parameter.parameter_type))
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
```

#### independent

```python
independent(parameters: Sequence[CalibrationParameter], distributions: Mapping[str, Distribution], *, initial_condition_remainder: str | None = None) -> Self
```

Independent inputs, each with its distribution.

Parameters:

| Name                          | Type                             | Description                                              | Default                                                              |
| ----------------------------- | -------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------- |
| `parameters`                  | `Sequence[CalibrationParameter]` | The inputs, in order.                                    | *required*                                                           |
| `distributions`               | `Mapping[str, Distribution]`     | The distribution of each input, truncated to its bounds. | *required*                                                           |
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
    distributions: Mapping[str, Distribution],
    *,
    initial_condition_remainder: str | None = None,
) -> Self:
    """
    Independent inputs, each with its distribution.

    Parameters
    ----------
    parameters : Sequence[CalibrationParameter]
        The inputs, in order.
    distributions : Mapping[str, Distribution]
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

#### rust_distributions

```python
rust_distributions() -> list[tuple[str, str, list[float]]] | None
```

The distributions as the Rust runners read them, or None.

Source code in `commol/context/inference/global_sensitivity.py`

```python
def rust_distributions(self) -> list[tuple[str, str, list[float]]] | None:
    """The distributions as the Rust runners read them, or None."""
    if self.distributions is None:
        return None
    return [self.distributions[name].rust_prior(name) for name in self.names]
```

## SensitivityMethod

Bases: `InferenceMethod[ResultT]`, `Generic[ResultT]`

The configuration of a sensitivity method, whose result is `ResultT`.

`SensitivityAnalysis.run(method)` runs it on the inputs of the analysis, with no data, and returns its result type.

Source code in `commol/context/inference/method.py`

```python
class SensitivityMethod(InferenceMethod[ResultT], Generic[ResultT]):
    """
    The configuration of a sensitivity method, whose result is `ResultT`.

    ``SensitivityAnalysis.run(method)`` runs it on the inputs of the
    analysis, with no data, and returns its result type.
    """
```

## Propagation

Bases: `SensitivityMethod[PropagationResult]`

Monte Carlo propagation: run the draws of the inputs and reduce each run to the quantities.

With independent inputs the draws are a Latin hypercube of their distributions; with inputs from a sample, the sample is the design, with its weights.

Attributes:

| Name         | Type             | Description                                                                |
| ------------ | ---------------- | -------------------------------------------------------------------------- |
| `draws`      | \`int            | None\`                                                                     |
| `quantities` | `list[Quantity]` | What each run is reduced to; none gives only the sample, for a Predictive. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class Propagation(SensitivityMethod[PropagationResult]):
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
    quantities : list[Quantity]
        What each run is reduced to; none gives only the sample, for a
        `Predictive`.
    """

    draws: int | None = Field(
        default=None, ge=1, description="Points of the Latin hypercube."
    )
    quantities: list[Quantity] = Field(
        default_factory=list, description="What each run is reduced to."
    )
```

## PropagationResult

Bases: `BaseModel`

The draws of the inputs and what each run gives.

Its `sample` is a `ParameterSample`, so `Predictive(analysis, result.sample, positions)` gives the bands, plots and reports of the propagation, as for a calibration.

Attributes:

| Name      | Type                | Description                                                                                                                  |
| --------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `sample`  | `ParameterSample`   | The draws of the inputs: a Latin hypercube of their distributions, or the sample of the input space itself.                  |
| `labels`  | `list[str]`         | The label of each quantity.                                                                                                  |
| `values`  | `list[list[float]]` | The quantities of each run, [run]\[quantity\]: one run per draw, or with a stochastic engine one per replicate of each draw. |
| `weights` | `list[float]`       | The weight of each run; with a stochastic engine each replicate has an equal share of its draw's weight. They sum to 1.      |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class PropagationResult(BaseModel):
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
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: ParameterSample
    labels: list[str]
    values: list[list[float]]
    weights: list[float]

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
            If the confidence level is not strictly between 0 and 1.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"The confidence level is strictly between 0 and 1, got "
                f"{confidence_level}"
            )
        levels = [0.5, (1 - confidence_level) / 2, (1 + confidence_level) / 2]
        summaries = []
        for index, label in enumerate(self.labels):
            column = [row[index] for row in self.values]
            median, lower, upper = commol_rs.inference.weighted_quantiles(
                column, self.weights, levels
            )
            summaries.append(
                QuantitySummary(
                    label=label,
                    confidence_level=confidence_level,
                    mean=commol_rs.inference.weighted_mean(column, self.weights),
                    median=median,
                    lower=lower,
                    upper=upper,
                )
            )
        return summaries
```

### Methods:

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

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the confidence level is not strictly between 0 and 1. |

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
        If the confidence level is not strictly between 0 and 1.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"The confidence level is strictly between 0 and 1, got "
            f"{confidence_level}"
        )
    levels = [0.5, (1 - confidence_level) / 2, (1 + confidence_level) / 2]
    summaries = []
    for index, label in enumerate(self.labels):
        column = [row[index] for row in self.values]
        median, lower, upper = commol_rs.inference.weighted_quantiles(
            column, self.weights, levels
        )
        summaries.append(
            QuantitySummary(
                label=label,
                confidence_level=confidence_level,
                mean=commol_rs.inference.weighted_mean(column, self.weights),
                median=median,
                lower=lower,
                upper=upper,
            )
        )
    return summaries
```

## QuantitySummary

Bases: `BaseModel`

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
class QuantitySummary(BaseModel):
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

Bases: `SensitivityMethod[PRCCResult]`

Latin hypercube sampling with partial rank correlation coefficients (Marino et al. 2008): how steadily each quantity rises or falls with each input.

With independent inputs the draws are a Latin hypercube of their distributions; with inputs from a sample, the sample is the design, and the ranks, the correlations and the test read its weights. A `Trajectory` gives the coefficients over the axis.

Attributes:

| Name         | Type             | Description                  |
| ------------ | ---------------- | ---------------------------- |
| `draws`      | \`int            | None\`                       |
| `quantities` | `list[Quantity]` | What each run is reduced to. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class PRCC(SensitivityMethod[PRCCResult]):
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
    quantities : list[Quantity]
        What each run is reduced to.
    """

    draws: int | None = Field(
        default=None, ge=1, description="Points of the Latin hypercube."
    )
    quantities: list[Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
```

## PRCCResult

Bases: `BaseModel`

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
class PRCCResult(BaseModel):
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
```

## Morris

Bases: `SensitivityMethod[MorrisResult]`

Morris screening (Morris 1991), with the `μ*` of Campolongo, Cariboni and Saltelli (2007): which inputs matter, in few runs.

Each trajectory starts at a random level of every input and moves the inputs one at a time, by half the levels; the levels are the centres of strata of equal probability of each distribution. It needs independent inputs: a step that moves one input breaks the dependence of a sample.

Attributes:

| Name           | Type             | Description                                                                 |
| -------------- | ---------------- | --------------------------------------------------------------------------- |
| `trajectories` | `int`            | The number of trajectories; each runs one more point than there are inputs. |
| `levels`       | `int`            | The number of levels of each input; even, at least 2.                       |
| `quantities`   | `list[Quantity]` | What each run is reduced to.                                                |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class Morris(SensitivityMethod[MorrisResult]):
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
    quantities : list[Quantity]
        What each run is reduced to.
    """

    trajectories: int = Field(ge=1, description="The number of trajectories.")
    levels: int = Field(ge=2, description="The number of levels; even.")
    quantities: list[Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
```

## MorrisResult

Bases: `BaseModel`

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
class MorrisResult(BaseModel):
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
```

## Sobol

Bases: `SensitivityMethod[SobolResult]`

Sobol indices on the design of Saltelli (2010): first-order indices by the estimator of Saltelli (2010) and total indices by the estimator of Jansen (1999), with intervals from a bootstrap of the rows.

The base matrices hold independent Monte Carlo draws, so the bootstrap intervals are valid. It needs independent inputs: with dependent inputs the indices lose their meaning and no longer add up; PRCC and propagation take a sample.

Attributes:

| Name               | Type             | Description                                                                       |
| ------------------ | ---------------- | --------------------------------------------------------------------------------- |
| `base_draws`       | `int`            | The rows of each base matrix; the analysis runs base_draws · (inputs + 2) points. |
| `bootstrap`        | `int`            | The bootstrap replicates of the intervals, at least 2.                            |
| `confidence_level` | `float`          | The level of the intervals, strictly between 0 and 1.                             |
| `quantities`       | `list[Quantity]` | What each run is reduced to.                                                      |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class Sobol(SensitivityMethod[SobolResult]):
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
    base_draws : int
        The rows of each base matrix; the analysis runs
        ``base_draws · (inputs + 2)`` points.
    bootstrap : int
        The bootstrap replicates of the intervals, at least 2.
    confidence_level : float
        The level of the intervals, strictly between 0 and 1.
    quantities : list[Quantity]
        What each run is reduced to.
    """

    base_draws: int = Field(ge=2, description="Rows of each base matrix.")
    bootstrap: int = Field(ge=2, description="Bootstrap replicates.")
    confidence_level: float = Field(
        gt=0.0, lt=1.0, description="Level of the intervals."
    )
    quantities: list[Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
```

## SobolResult

Bases: `BaseModel`

The Sobol indices of each input for each quantity, with their bootstrap intervals.

The first-order index is the share of the variance of the quantity the input explains alone; the total index adds every interaction the input takes part in. A total index near 0 says the input can be fixed; a total index well above the first-order one says the input acts through interactions.

Attributes:

| Name                       | Type                | Description                                                                                                                                  |
| -------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `inputs`                   | `list[str]`         | The inputs, in order.                                                                                                                        |
| `labels`                   | `list[str]`         | The label of each quantity.                                                                                                                  |
| `confidence_level`         | `float`             | The level of the intervals.                                                                                                                  |
| `first, total`             | `list[list[float]]` | The first-order and total indices, [quantity][input]; NaN for a quantity of no variance.                                                     |
| `first_lower, first_upper` | `list[list[float]]` | The ends of the interval of each first-order index; NaN when a bootstrap replicate of the rows has no variance, which more base_draws avoid. |
| `total_lower, total_upper` | `list[list[float]]` | The ends of the interval of each total index.                                                                                                |
| `runs`                     | `int`               | The number of points run: base_draws · (inputs + 2).                                                                                         |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class SobolResult(BaseModel):
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
        `base_draws` avoid.
    total_lower, total_upper : list[list[float]]
        The ends of the interval of each total index.
    runs : int
        The number of points run: ``base_draws · (inputs + 2)``.
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
```

## PolynomialChaos

Bases: `SensitivityMethod[ChaosResult]`

A polynomial chaos expansion (Sudret 2008): a polynomial surrogate of each quantity, and the Sobol indices of its coefficients, in far fewer runs than `Sobol` when the quantity is smooth.

Every input goes to `u = F(x)`, uniform on `[0, 1]` through its distribution, so one basis of orthonormal Legendre polynomials fits every distribution, truncated ones included. The coefficients come from least squares on a Latin hypercube; the leave-one-out error says how well the expansion predicts the model. It needs independent inputs.

Attributes:

| Name         | Type             | Description                                                                                                 |
| ------------ | ---------------- | ----------------------------------------------------------------------------------------------------------- |
| `degree`     | `int`            | The total degree of the basis: in k inputs it has (k + degree)! / (k! degree!) terms.                       |
| `draws`      | `int`            | The points of the Latin hypercube; more than the terms, and two or three times as many is the usual choice. |
| `quantities` | `list[Quantity]` | What each run is reduced to.                                                                                |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class PolynomialChaos(SensitivityMethod[ChaosResult]):
    """
    A polynomial chaos expansion (Sudret 2008): a polynomial surrogate of
    each quantity, and the Sobol indices of its coefficients, in far fewer
    runs than `Sobol` when the quantity is smooth.

    Every input goes to ``u = F(x)``, uniform on ``[0, 1]`` through its
    distribution, so one basis of orthonormal Legendre polynomials fits every
    distribution, truncated ones included. The coefficients come from least
    squares on a Latin hypercube; the leave-one-out error says how well the
    expansion predicts the model. It needs independent inputs.

    Attributes
    ----------
    degree : int
        The total degree of the basis: in ``k`` inputs it has
        ``(k + degree)! / (k! degree!)`` terms.
    draws : int
        The points of the Latin hypercube; more than the terms, and two or
        three times as many is the usual choice.
    quantities : list[Quantity]
        What each run is reduced to.
    """

    degree: int = Field(ge=1, description="The total degree of the basis.")
    draws: int = Field(ge=2, description="Points of the Latin hypercube.")
    quantities: list[Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
```

## ChaosResult

Bases: `BaseModel`

A polynomial chaos expansion of each quantity, its accuracy and the Sobol indices of its coefficients.

The expansion is a surrogate of the model: `predict` gives the quantities at any values of the inputs inside their bounds, at the cost of a polynomial. Read `loo_error` first: the indices are those of the expansion, and they are those of the model only as far as the expansion predicts it.

Attributes:

| Name             | Type                | Description                                                                                                        |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `inputs`         | `list[str]`         | The inputs, in order.                                                                                              |
| `input_space`    | `InputSpace`        | The inputs and their distributions, which predict maps values through.                                             |
| `labels`         | `list[str]`         | The label of each quantity.                                                                                        |
| `degree`         | `int`               | The total degree of the basis.                                                                                     |
| `indices`        | `list[list[int]]`   | The degree of each input in each term, [term][input]; term 0 is the constant.                                      |
| `coefficients`   | `list[list[float]]` | The coefficient of each term for each quantity, [quantity][term], on orthonormal Legendre polynomials of u = F(x). |
| `loo_error`      | `list[float]`       | The leave-one-out error of each quantity, relative to its variance; NaN for a quantity of no variance.             |
| `mean, variance` | `list[float]`       | The mean and the variance of each quantity under the expansion.                                                    |
| `first, total`   | `list[list[float]]` | The first-order and total Sobol indices, [quantity][input]; NaN for a quantity of no variance.                     |
| `runs`           | `int`               | The number of points run.                                                                                          |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class ChaosResult(BaseModel):
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
        The total degree of the basis.
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
    indices: list[list[int]]
    coefficients: list[list[float]]
    loo_error: list[float]
    mean: list[float]
    variance: list[float]
    first: list[list[float]]
    total: list[list[float]]
    runs: int

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
            bounds of its input, or the coefficients do not fit the degree.
        """
        return commol_rs.inference.chaos_evaluate(
            engine_state.rust_parameters(self.input_space.parameters),
            self.input_space.rust_distributions() or [],
            self.degree,
            self.coefficients,
            [list(point) for point in values],
        )
```

### Methods:

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

| Type         | Description                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a point has not one value per input or a value outside the bounds of its input, or the coefficients do not fit the degree. |

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
        bounds of its input, or the coefficients do not fit the degree.
    """
    return commol_rs.inference.chaos_evaluate(
        engine_state.rust_parameters(self.input_space.parameters),
        self.input_space.rust_distributions() or [],
        self.degree,
        self.coefficients,
        [list(point) for point in values],
    )
```

## ValueAt

Bases: `_OneScalar`

The value at a position.

Attributes:

| Name | Type    | Description   |
| ---- | ------- | ------------- |
| `at` | `float` | The position. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class ValueAt(_OneScalar):
    """
    The value at a position.

    Attributes
    ----------
    at : float
        The position.
    """

    rust_kind: ClassVar[str] = "value_at"
    kind: Literal["value_at"] = "value_at"
    at: Position = Field(description="The position.")

    def default_label(self) -> str:
        """`outputs` at `at`."""
        return f"{self.output_name} at {self.at:g}"

    def rust_quantity(self) -> RustQuantity:
        """The kind, the outputs and the position."""
        return (self.rust_kind, self.output_list, [self.at])
```

### Methods:

#### default_label

```python
default_label() -> str
```

`outputs` at `at`.

Source code in `commol/context/inference/global_sensitivity.py`

```python
def default_label(self) -> str:
    """`outputs` at `at`."""
    return f"{self.output_name} at {self.at:g}"
```

#### rust_quantity

```python
rust_quantity() -> RustQuantity
```

The kind, the outputs and the position.

Source code in `commol/context/inference/global_sensitivity.py`

```python
def rust_quantity(self) -> RustQuantity:
    """The kind, the outputs and the position."""
    return (self.rust_kind, self.output_list, [self.at])
```

## Gain

Bases: `_OneScalar`

The value at `end` minus the value at `start`: what entered a cumulative output over `(start, end]`, as an observation with the window `end - start` reads it.

Attributes:

| Name    | Type    | Description                         |
| ------- | ------- | ----------------------------------- |
| `start` | `float` | The start of the window.            |
| `end`   | `float` | The end of the window, after start. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class Gain(_OneScalar):
    """
    The value at `end` minus the value at `start`: what entered a cumulative
    output over ``(start, end]``, as an observation with the window
    ``end - start`` reads it.

    Attributes
    ----------
    start : float
        The start of the window.
    end : float
        The end of the window, after `start`.
    """

    rust_kind: ClassVar[str] = "gain"
    kind: Literal["gain"] = "gain"
    start: Position = Field(description="The start of the window.")
    end: Position = Field(description="The end of the window.")

    @model_validator(mode="after")
    def end_after_start(self) -> Self:
        """A window ends after it starts."""
        if self.end <= self.start:
            raise ValueError(
                f"A gain ends after it starts, got start {self.start} and end "
                f"{self.end}"
            )
        return self

    def default_label(self) -> str:
        """The gain of `outputs` over the window."""
        return f"{self.output_name} gain over ({self.start:g}, {self.end:g}]"

    def rust_quantity(self) -> RustQuantity:
        """The kind, the outputs, the start and the end."""
        return (self.rust_kind, self.output_list, [self.start, self.end])
```

### Methods:

#### end_after_start

```python
end_after_start() -> Self
```

A window ends after it starts.

Source code in `commol/context/inference/global_sensitivity.py`

```python
@model_validator(mode="after")
def end_after_start(self) -> Self:
    """A window ends after it starts."""
    if self.end <= self.start:
        raise ValueError(
            f"A gain ends after it starts, got start {self.start} and end "
            f"{self.end}"
        )
    return self
```

#### default_label

```python
default_label() -> str
```

The gain of `outputs` over the window.

Source code in `commol/context/inference/global_sensitivity.py`

```python
def default_label(self) -> str:
    """The gain of `outputs` over the window."""
    return f"{self.output_name} gain over ({self.start:g}, {self.end:g}]"
```

#### rust_quantity

```python
rust_quantity() -> RustQuantity
```

The kind, the outputs, the start and the end.

Source code in `commol/context/inference/global_sensitivity.py`

```python
def rust_quantity(self) -> RustQuantity:
    """The kind, the outputs, the start and the end."""
    return (self.rust_kind, self.output_list, [self.start, self.end])
```

## Peak

Bases: `_Peak`

The largest value at the positions.

Attributes:

| Name        | Type          | Description                        |
| ----------- | ------------- | ---------------------------------- |
| `positions` | `list[float]` | The positions compared, ascending. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class Peak(_Peak):
    """
    The largest value at the positions.

    Attributes
    ----------
    positions : list[float]
        The positions compared, ascending.
    """

    rust_kind: ClassVar[str] = "peak"
    kind: Literal["peak"] = "peak"

    def default_label(self) -> str:
        """The peak of `outputs`."""
        return f"peak of {self.output_name}"
```

### Methods:

#### default_label

```python
default_label() -> str
```

The peak of `outputs`.

Source code in `commol/context/inference/global_sensitivity.py`

```python
def default_label(self) -> str:
    """The peak of `outputs`."""
    return f"peak of {self.output_name}"
```

## PeakPosition

Bases: `_Peak`

The first of the positions where the value is largest.

Attributes:

| Name        | Type          | Description                        |
| ----------- | ------------- | ---------------------------------- |
| `positions` | `list[float]` | The positions compared, ascending. |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class PeakPosition(_Peak):
    """
    The first of the positions where the value is largest.

    Attributes
    ----------
    positions : list[float]
        The positions compared, ascending.
    """

    rust_kind: ClassVar[str] = "peak_position"
    kind: Literal["peak_position"] = "peak_position"

    def default_label(self) -> str:
        """The position of the peak of `outputs`."""
        return f"position of the peak of {self.output_name}"
```

### Methods:

#### default_label

```python
default_label() -> str
```

The position of the peak of `outputs`.

Source code in `commol/context/inference/global_sensitivity.py`

```python
def default_label(self) -> str:
    """The position of the peak of `outputs`."""
    return f"position of the peak of {self.output_name}"
```

## Trajectory

Bases: `_AtPositions`

The value at each position: one `ValueAt` per position, for an analysis over the axis.

Attributes:

| Name        | Type          | Description               |
| ----------- | ------------- | ------------------------- |
| `positions` | `list[float]` | The positions, ascending. |
| `label`     | \`str         | None\`                    |

Source code in `commol/context/inference/global_sensitivity.py`

```python
class Trajectory(_AtPositions):
    """
    The value at each position: one `ValueAt` per position, for an analysis
    over the axis.

    Attributes
    ----------
    positions : list[float]
        The positions, ascending.
    label : str | None
        The name of the series; each scalar is named after it and its
        position.
    """

    rust_kind: ClassVar[str] = "value_at"
    kind: Literal["trajectory"] = "trajectory"

    def series(self) -> list[tuple[str, float | None]]:
        """Its name and each position."""
        name = self.label or self.output_name
        return [(name, at) for at in self.positions]

    def expand(self) -> list[tuple[str, RustQuantity]]:
        """One `ValueAt` per position."""
        name = self.label or self.output_name
        return [
            (f"{name} at {at:g}", (self.rust_kind, self.output_list, [at]))
            for at in self.positions
        ]
```

### Methods:

#### series

```python
series() -> list[tuple[str, float | None]]
```

Its name and each position.

Source code in `commol/context/inference/global_sensitivity.py`

```python
def series(self) -> list[tuple[str, float | None]]:
    """Its name and each position."""
    name = self.label or self.output_name
    return [(name, at) for at in self.positions]
```

#### expand

```python
expand() -> list[tuple[str, RustQuantity]]
```

One `ValueAt` per position.

Source code in `commol/context/inference/global_sensitivity.py`

```python
def expand(self) -> list[tuple[str, RustQuantity]]:
    """One `ValueAt` per position."""
    name = self.label or self.output_name
    return [
        (f"{name} at {at:g}", (self.rust_kind, self.output_list, [at]))
        for at in self.positions
    ]
```
