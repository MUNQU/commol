# Calibration API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Fits a model to observations. Guide: [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md).

## Calibrator

Calibrate the parameters of a simulation against observed data.

The calibration problem is built in Rust once, when the calibrator is created, and every optimization, prediction and uncertainty method reads that one problem.

Source code in `commol/_calibration/calibrator.py`

```python
class Calibrator:
    """
    Calibrate the parameters of a simulation against observed data.

    The calibration problem is built in Rust once, when the calibrator is
    created, and every optimization, prediction and uncertainty method reads
    that one problem.
    """

    def __init__(
        self,
        simulation: Simulation,
        problem: CalibrationProblem,
    ) -> None:
        """
        Initializes the calibration from a Simulation and CalibrationProblem.

        Parameters
        ----------
        simulation : Simulation
            A fully initialized Simulation object with the model to calibrate.
        problem : CalibrationProblem
            A fully constructed and validated calibration problem definition.

        Raises
        ------
        ValueError
            If a parameter, observation, position or constraint does not fit
            the model, or a setting of the problem is outside its rule; the
            message names the path of the setting, such as
            ``optimization_config.inertia_factor``.
        """
        logger.info(
            f"Initializing Calibration for model: '{simulation.model_definition.name}'"
        )
        self.simulation: Simulation = simulation
        self.problem: CalibrationProblem = problem
        self._engine: "EngineProtocol" = simulation.engine

        logger.info(
            (
                f"Calibration initialized with {len(problem.parameters)} parameters "
                f"and {len(problem.observed_data)} observed data points."
            )
        )

        self._validate_calibration_parameters()
        self._warn_constraint_positions_beyond_observations()
        self._engine_problem: "CalibrationProblemProtocol" = (
            commol_rs.inference.CalibrationProblem(self._engine, engine_json(problem))
        )
        if problem.seed is None:
            logger.info(
                "No seed set; drew the seed %d, which repeats this calibration.",
                self._engine_problem.seed,
            )

    def _predictive(
        self, sample: ParameterSample, positions: list[float]
    ) -> "PredictiveProtocol":
        """
        The run of every draw of `sample` at `positions`, for a `Predictive`:
        each replicate of a stochastic engine has a seed of its own, derived
        from the calibrator's seed.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the problem's order.
        positions : list[float]
            Positions to report at, sorted and unique.

        Returns
        -------
        PredictiveProtocol
            The runs, computed in Rust.
        """
        return self._engine_problem.predictive(sample._rust, positions)

    def with_model(self, model: "Model") -> "Calibrator":
        """
        The same calibration problem on another model, with the same seed.

        The draws of a predictive of the copy take the seeds they take here,
        so `Scenarios` runs every scenario on common random numbers.

        Parameters
        ----------
        model : Model
            The model, such as a copy of this one with other parameter values.

        Returns
        -------
        Calibrator
            A calibrator of `model`, with `problem` and its seed set to
            `seed`.

        Raises
        ------
        ValueError
            If a parameter, observation, position or constraint of the problem
            does not fit `model`.
        """
        return Calibrator(
            Simulation(model), self.problem.model_copy(update={"seed": self.seed})
        )

    def point_sample(self, result: CalibrationResult) -> ParameterSample:
        """
        The point estimate of a fit as a sample of one draw.

        A `Predictive` or `Scenarios` of it runs the model at the fit, so a
        fit is read as any sample is.

        Parameters
        ----------
        result : CalibrationResult
            A fit of this calibration problem.

        Returns
        -------
        ParameterSample
            One draw at ``result.best_parameters``, with the names and bounds
            of the parameters of the problem and
            ``SampleMethod(name="point", settings={"seed": result.seed})``.

        Raises
        ------
        ValueError
            If the names of `result` are not those of the problem; the message
            names the names that differ.
        """
        names = self.parameter_names
        missing = [name for name in names if name not in result.best_parameters]
        extra = [name for name in result.best_parameters if name not in names]
        if missing or extra:
            raise ValueError(
                f"The result is not a fit of this problem: it lacks {missing} and "
                f"has {extra}, which the problem does not have."
            )
        return ParameterSample.equally_weighted(
            names,
            [(p.min_bound, p.max_bound) for p in self.problem.parameters],
            [[result.best_parameters[name] for name in names]],
            SampleMethod(name="point", settings={"seed": result.seed}),
        )

    def _observation_draws(
        self,
        sample: ParameterSample,
        observed_data: list[ObservedDataPoint],
        noise: _ObservationNoise,
    ) -> "ObservationDrawsProtocol":
        """
        Every draw's value of `observed_data` with `noise`, for a `Predictive`.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the problem's order.
        observed_data : list[ObservedDataPoint]
            The observations.
        noise : NoNoise | ResidualNoise | DistributionNoise
            The observation noise, drawn from the calibrator's seed.

        Returns
        -------
        ObservationDrawsProtocol
            The draws, computed in Rust.
        """
        return self._engine_problem.observation_draws(
            sample._rust, engine_json(observed_data), engine_json(noise)
        )

    def _check_observations(self, observed_data: list[ObservedDataPoint]) -> None:
        """
        Check `observed_data` against the problem in Rust, with no run.

        Parameters
        ----------
        observed_data : list[ObservedDataPoint]
            The observations.

        Raises
        ------
        ValueError
            If the engine rejects a position, or an observation names an
            unknown output or factor, or a parameter that is not a factor.
        """
        self._engine_problem.check_observations(engine_json(observed_data))

    def _log_score(
        self, sample: ParameterSample, observed_data: list[ObservedDataPoint]
    ) -> list[float]:
        """
        The log score of each observation under the predictive of `sample`,
        for `commol.scores.log_score`.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the problem's order.
        observed_data : list[ObservedDataPoint]
            The observations.

        Returns
        -------
        list[float]
            One log score per observation, computed in Rust.
        """
        return self._engine_problem.log_score(sample._rust, engine_json(observed_data))

    @property
    def seed(self) -> int:
        """
        The seed of the calibration's random numbers.

        Returns
        -------
        int
            ``problem.seed``, or the seed the engine drew when the
            calibrator was built, which it logs.
        """
        return self._engine_problem.seed

    def run(self, starts: int = 1) -> CalibrationResult:
        """
        Runs the calibration optimization from one or more starts.

        Start 0 is the initial guess of each parameter. Each further start is a
        row of a Latin hypercube over the bounds, drawn from `seed`, so the
        starts cover the parameter space; a particle swarm takes a seed of
        its own at each start, also from `seed`. They run in parallel, on the
        threads of `commol.parallelism.threads`; the result is the best of them, the
        first start of lowest loss, and lists every start. Distinct ends with
        equal losses are a sign that the data do not identify the
        parameters.

        Parameters
        ----------
        starts : int, optional
            Number of starts, at least 1 (default 1). A verbose optimizer
            shows one start, so it needs ``starts=1``.

        Returns
        -------
        CalibrationResult
            The best start: the optimized parameter values, in the order of
            the problem's parameters, its final loss and convergence, every
            start (``starts``), each start that failed (``failed``, also
            logged as warnings) and the seed (``seed``).

        Raises
        ------
        ValueError
            If `starts` is below 1, or a verbose optimizer has more than one
            start.
        RuntimeError
            If every start fails.
        """
        measure = (
            f"{self.problem.loss_function} loss function"
            if self.problem.loss_function is not None
            else "the negative log-likelihood of its observation models"
        )
        logger.info(
            f"Starting calibration with {self.problem.optimization_config.kind} "
            f"algorithm, {measure} and {starts} start(s)."
        )
        result = fit_result(
            self._engine_problem.optimize(starts),
            self.parameter_names,
            self.seed,
            logger,
        )
        logger.info(
            (
                f"Calibration finished after {result.iterations} iterations. "
                f"Final loss: {result.final_loss:.6f}"
            )
        )
        return result

    def run_with_history(self) -> CalibrationHistory:
        """
        Runs the calibration from the initial guess and keeps every evaluation.

        Returns
        -------
        CalibrationHistory
            The result, and every evaluation of a feasible point the optimizer
            made, in order. Infeasible points, which the objective penalizes,
            are not kept.

        Raises
        ------
        RuntimeError
            If optimization fails.
        """
        logger.info("Starting calibration with history tracking")
        history = self._engine_problem.optimize_with_history()
        names = self.parameter_names
        best = history.result
        result = CalibrationResult(
            best_parameters=dict(best.best_parameters),
            final_loss=best.final_loss,
            iterations=best.iterations,
            converged=best.converged,
            termination_reason=best.termination_reason,
            seed=self.seed,
        )
        evaluations = [
            CalibrationEvaluation(
                parameters=dict(zip(names, evaluation.parameters, strict=True)),
                loss=evaluation.loss,
            )
            for evaluation in history.evaluations
        ]
        logger.info(
            f"Calibration finished after {result.iterations} iterations. "
            f"Final loss: {result.final_loss:.6f}. "
            f"Collected {len(evaluations)} evaluations."
        )
        return CalibrationHistory(result=result, evaluations=evaluations)

    def uncertainty(self, method: UncertaintyMethod[ResultT]) -> ResultT:
        """
        Run an uncertainty method on this calibration problem.

        Each method is a configuration (a subclass of `UncertaintyMethod`)
        whose type sets the result type and declares its runner. The guide's
        table "Which method
        answers which question" lists them.

        Parameters
        ----------
        method : UncertaintyMethod[ResultT]
            The configuration of the method.

        Returns
        -------
        ResultT
            The result of the method.

        Raises
        ------
        TypeError
            If `method` is an identifiability method; the message names
            ``Calibrator.identifiability``.
        ValueError
            If the configuration's type declares no runner, or
            a setting, an input, the data or the model does not fit the
            method; the message says why.
        RuntimeError
            If a computation of the method that could start fails: an engine
            run, an optimizer, every replicate, or a sampler that does not
            reach its target.
        InfeasibleStateError
            If a draw of a prediction leaves the values its model allows.
        """
        return run_method(self, method, UncertaintyMethod)

    def identifiability(self, method: IdentifiabilityMethod[ResultT]) -> ResultT:
        """
        Run an identifiability method on this calibration problem.

        Each method answers which parameters the model and the data
        determine, from another kind of evidence: `StructuralRank`,
        `Collinearity`, `SampleIdentifiability` or `StartsIdentifiability`.
        Each returns an `IdentifiabilityReport`; `IdentifiabilityReport.merge`
        joins them.

        Parameters
        ----------
        method : IdentifiabilityMethod[ResultT]
            The configuration of the method.

        Returns
        -------
        ResultT
            The result of the method.

        Raises
        ------
        TypeError
            If `method` is an uncertainty method; the message names
            ``Calibrator.uncertainty``.
        ValueError
            If the configuration's type declares no runner, or
            a setting, an input, the data or the model does not fit the
            method; the message says why.
        RuntimeError
            If a computation of the method that could start fails: an engine
            run or an optimizer.
        """
        return run_method(self, method, IdentifiabilityMethod)

    @property
    def num_parameters(self) -> int:
        """Number of parameters being calibrated."""
        return len(self.problem.parameters)

    @property
    def num_observations(self) -> int:
        """Number of observed data points."""
        return len(self.problem.observed_data)

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters being calibrated."""
        return [param.id for param in self.problem.parameters]

    def _validate_calibration_parameters(self) -> None:
        """
        Validate that all calibration parameters exist in the model.

        Raises
        ------
        ValueError
            If a parameter ID doesn't exist in the model or if a bin ID is invalid.
        """
        check_values(
            self.simulation.model_definition,
            self._engine,
            self.problem.parameters,
            self.problem.initial_condition_remainder,
        )

    def _warn_constraint_positions_beyond_observations(self) -> None:
        """Warn when constraints at positions extend beyond observed data."""
        max_observed_position = max(obs.at for obs in self.problem.observed_data)
        for constraint in self.problem.constraints:
            if constraint.at and max(constraint.at) > max_observed_position:
                logger.warning(
                    "Constraint '%s' has positions beyond the maximum observed "
                    "position (%s); simulation will be extended to evaluate them.",
                    constraint.id,
                    max_observed_position,
                )
```

### Attributes

#### seed

```python
seed: int
```

The seed of the calibration's random numbers.

Returns:

| Type  | Description                                                                             |
| ----- | --------------------------------------------------------------------------------------- |
| `int` | problem.seed, or the seed the engine drew when the calibrator was built, which it logs. |

#### num_parameters

```python
num_parameters: int
```

Number of parameters being calibrated.

#### num_observations

```python
num_observations: int
```

Number of observed data points.

#### parameter_names

```python
parameter_names: list[str]
```

Names of parameters being calibrated.

### Methods:

#### __init__

```python
__init__(simulation: Simulation, problem: CalibrationProblem) -> None
```

Initializes the calibration from a Simulation and CalibrationProblem.

Parameters:

| Name         | Type                 | Description                                                        | Default    |
| ------------ | -------------------- | ------------------------------------------------------------------ | ---------- |
| `simulation` | `Simulation`         | A fully initialized Simulation object with the model to calibrate. | *required* |
| `problem`    | `CalibrationProblem` | A fully constructed and validated calibration problem definition.  | *required* |

Raises:

| Type         | Description                                                                                                                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a parameter, observation, position or constraint does not fit the model, or a setting of the problem is outside its rule; the message names the path of the setting, such as optimization_config.inertia_factor. |

Source code in `commol/_calibration/calibrator.py`

```python
def __init__(
    self,
    simulation: Simulation,
    problem: CalibrationProblem,
) -> None:
    """
    Initializes the calibration from a Simulation and CalibrationProblem.

    Parameters
    ----------
    simulation : Simulation
        A fully initialized Simulation object with the model to calibrate.
    problem : CalibrationProblem
        A fully constructed and validated calibration problem definition.

    Raises
    ------
    ValueError
        If a parameter, observation, position or constraint does not fit
        the model, or a setting of the problem is outside its rule; the
        message names the path of the setting, such as
        ``optimization_config.inertia_factor``.
    """
    logger.info(
        f"Initializing Calibration for model: '{simulation.model_definition.name}'"
    )
    self.simulation: Simulation = simulation
    self.problem: CalibrationProblem = problem
    self._engine: "EngineProtocol" = simulation.engine

    logger.info(
        (
            f"Calibration initialized with {len(problem.parameters)} parameters "
            f"and {len(problem.observed_data)} observed data points."
        )
    )

    self._validate_calibration_parameters()
    self._warn_constraint_positions_beyond_observations()
    self._engine_problem: "CalibrationProblemProtocol" = (
        commol_rs.inference.CalibrationProblem(self._engine, engine_json(problem))
    )
    if problem.seed is None:
        logger.info(
            "No seed set; drew the seed %d, which repeats this calibration.",
            self._engine_problem.seed,
        )
```

#### with_model

```python
with_model(model: Model) -> Calibrator
```

The same calibration problem on another model, with the same seed.

The draws of a predictive of the copy take the seeds they take here, so `Scenarios` runs every scenario on common random numbers.

Parameters:

| Name    | Type    | Description                                                        | Default    |
| ------- | ------- | ------------------------------------------------------------------ | ---------- |
| `model` | `Model` | The model, such as a copy of this one with other parameter values. | *required* |

Returns:

| Type         | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `Calibrator` | A calibrator of model, with problem and its seed set to seed. |

Raises:

| Type         | Description                                                                            |
| ------------ | -------------------------------------------------------------------------------------- |
| `ValueError` | If a parameter, observation, position or constraint of the problem does not fit model. |

Source code in `commol/_calibration/calibrator.py`

```python
def with_model(self, model: "Model") -> "Calibrator":
    """
    The same calibration problem on another model, with the same seed.

    The draws of a predictive of the copy take the seeds they take here,
    so `Scenarios` runs every scenario on common random numbers.

    Parameters
    ----------
    model : Model
        The model, such as a copy of this one with other parameter values.

    Returns
    -------
    Calibrator
        A calibrator of `model`, with `problem` and its seed set to
        `seed`.

    Raises
    ------
    ValueError
        If a parameter, observation, position or constraint of the problem
        does not fit `model`.
    """
    return Calibrator(
        Simulation(model), self.problem.model_copy(update={"seed": self.seed})
    )
```

#### point_sample

```python
point_sample(result: CalibrationResult) -> ParameterSample
```

The point estimate of a fit as a sample of one draw.

A `Predictive` or `Scenarios` of it runs the model at the fit, so a fit is read as any sample is.

Parameters:

| Name     | Type                | Description                        | Default    |
| -------- | ------------------- | ---------------------------------- | ---------- |
| `result` | `CalibrationResult` | A fit of this calibration problem. | *required* |

Returns:

| Type              | Description                                                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ParameterSample` | One draw at result.best_parameters, with the names and bounds of the parameters of the problem and SampleMethod(name="point", settings={"seed": result.seed}). |

Raises:

| Type         | Description                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------- |
| `ValueError` | If the names of result are not those of the problem; the message names the names that differ. |

Source code in `commol/_calibration/calibrator.py`

```python
def point_sample(self, result: CalibrationResult) -> ParameterSample:
    """
    The point estimate of a fit as a sample of one draw.

    A `Predictive` or `Scenarios` of it runs the model at the fit, so a
    fit is read as any sample is.

    Parameters
    ----------
    result : CalibrationResult
        A fit of this calibration problem.

    Returns
    -------
    ParameterSample
        One draw at ``result.best_parameters``, with the names and bounds
        of the parameters of the problem and
        ``SampleMethod(name="point", settings={"seed": result.seed})``.

    Raises
    ------
    ValueError
        If the names of `result` are not those of the problem; the message
        names the names that differ.
    """
    names = self.parameter_names
    missing = [name for name in names if name not in result.best_parameters]
    extra = [name for name in result.best_parameters if name not in names]
    if missing or extra:
        raise ValueError(
            f"The result is not a fit of this problem: it lacks {missing} and "
            f"has {extra}, which the problem does not have."
        )
    return ParameterSample.equally_weighted(
        names,
        [(p.min_bound, p.max_bound) for p in self.problem.parameters],
        [[result.best_parameters[name] for name in names]],
        SampleMethod(name="point", settings={"seed": result.seed}),
    )
```

#### run

```python
run(starts: int = 1) -> CalibrationResult
```

Runs the calibration optimization from one or more starts.

Start 0 is the initial guess of each parameter. Each further start is a row of a Latin hypercube over the bounds, drawn from `seed`, so the starts cover the parameter space; a particle swarm takes a seed of its own at each start, also from `seed`. They run in parallel, on the threads of `commol.parallelism.threads`; the result is the best of them, the first start of lowest loss, and lists every start. Distinct ends with equal losses are a sign that the data do not identify the parameters.

Parameters:

| Name     | Type  | Description                                                                                          | Default |
| -------- | ----- | ---------------------------------------------------------------------------------------------------- | ------- |
| `starts` | `int` | Number of starts, at least 1 (default 1). A verbose optimizer shows one start, so it needs starts=1. | `1`     |

Returns:

| Type                | Description                                                                                                                                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CalibrationResult` | The best start: the optimized parameter values, in the order of the problem's parameters, its final loss and convergence, every start (starts), each start that failed (failed, also logged as warnings) and the seed (seed). |

Raises:

| Type           | Description                                                           |
| -------------- | --------------------------------------------------------------------- |
| `ValueError`   | If starts is below 1, or a verbose optimizer has more than one start. |
| `RuntimeError` | If every start fails.                                                 |

Source code in `commol/_calibration/calibrator.py`

```python
def run(self, starts: int = 1) -> CalibrationResult:
    """
    Runs the calibration optimization from one or more starts.

    Start 0 is the initial guess of each parameter. Each further start is a
    row of a Latin hypercube over the bounds, drawn from `seed`, so the
    starts cover the parameter space; a particle swarm takes a seed of
    its own at each start, also from `seed`. They run in parallel, on the
    threads of `commol.parallelism.threads`; the result is the best of them, the
    first start of lowest loss, and lists every start. Distinct ends with
    equal losses are a sign that the data do not identify the
    parameters.

    Parameters
    ----------
    starts : int, optional
        Number of starts, at least 1 (default 1). A verbose optimizer
        shows one start, so it needs ``starts=1``.

    Returns
    -------
    CalibrationResult
        The best start: the optimized parameter values, in the order of
        the problem's parameters, its final loss and convergence, every
        start (``starts``), each start that failed (``failed``, also
        logged as warnings) and the seed (``seed``).

    Raises
    ------
    ValueError
        If `starts` is below 1, or a verbose optimizer has more than one
        start.
    RuntimeError
        If every start fails.
    """
    measure = (
        f"{self.problem.loss_function} loss function"
        if self.problem.loss_function is not None
        else "the negative log-likelihood of its observation models"
    )
    logger.info(
        f"Starting calibration with {self.problem.optimization_config.kind} "
        f"algorithm, {measure} and {starts} start(s)."
    )
    result = fit_result(
        self._engine_problem.optimize(starts),
        self.parameter_names,
        self.seed,
        logger,
    )
    logger.info(
        (
            f"Calibration finished after {result.iterations} iterations. "
            f"Final loss: {result.final_loss:.6f}"
        )
    )
    return result
```

#### run_with_history

```python
run_with_history() -> CalibrationHistory
```

Runs the calibration from the initial guess and keeps every evaluation.

Returns:

| Type                 | Description                                                                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CalibrationHistory` | The result, and every evaluation of a feasible point the optimizer made, in order. Infeasible points, which the objective penalizes, are not kept. |

Raises:

| Type           | Description            |
| -------------- | ---------------------- |
| `RuntimeError` | If optimization fails. |

Source code in `commol/_calibration/calibrator.py`

```python
def run_with_history(self) -> CalibrationHistory:
    """
    Runs the calibration from the initial guess and keeps every evaluation.

    Returns
    -------
    CalibrationHistory
        The result, and every evaluation of a feasible point the optimizer
        made, in order. Infeasible points, which the objective penalizes,
        are not kept.

    Raises
    ------
    RuntimeError
        If optimization fails.
    """
    logger.info("Starting calibration with history tracking")
    history = self._engine_problem.optimize_with_history()
    names = self.parameter_names
    best = history.result
    result = CalibrationResult(
        best_parameters=dict(best.best_parameters),
        final_loss=best.final_loss,
        iterations=best.iterations,
        converged=best.converged,
        termination_reason=best.termination_reason,
        seed=self.seed,
    )
    evaluations = [
        CalibrationEvaluation(
            parameters=dict(zip(names, evaluation.parameters, strict=True)),
            loss=evaluation.loss,
        )
        for evaluation in history.evaluations
    ]
    logger.info(
        f"Calibration finished after {result.iterations} iterations. "
        f"Final loss: {result.final_loss:.6f}. "
        f"Collected {len(evaluations)} evaluations."
    )
    return CalibrationHistory(result=result, evaluations=evaluations)
```

#### uncertainty

```python
uncertainty(method: UncertaintyMethod[ResultT]) -> ResultT
```

Run an uncertainty method on this calibration problem.

Each method is a configuration (a subclass of `UncertaintyMethod`) whose type sets the result type and declares its runner. The guide's table "Which method answers which question" lists them.

Parameters:

| Name     | Type                         | Description                      | Default    |
| -------- | ---------------------------- | -------------------------------- | ---------- |
| `method` | `UncertaintyMethod[ResultT]` | The configuration of the method. | *required* |

Returns:

| Type      | Description               |
| --------- | ------------------------- |
| `ResultT` | The result of the method. |

Raises:

| Type                   | Description                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TypeError`            | If method is an identifiability method; the message names Calibrator.identifiability.                                                             |
| `ValueError`           | If the configuration's type declares no runner, or a setting, an input, the data or the model does not fit the method; the message says why.      |
| `RuntimeError`         | If a computation of the method that could start fails: an engine run, an optimizer, every replicate, or a sampler that does not reach its target. |
| `InfeasibleStateError` | If a draw of a prediction leaves the values its model allows.                                                                                     |

Source code in `commol/_calibration/calibrator.py`

```python
def uncertainty(self, method: UncertaintyMethod[ResultT]) -> ResultT:
    """
    Run an uncertainty method on this calibration problem.

    Each method is a configuration (a subclass of `UncertaintyMethod`)
    whose type sets the result type and declares its runner. The guide's
    table "Which method
    answers which question" lists them.

    Parameters
    ----------
    method : UncertaintyMethod[ResultT]
        The configuration of the method.

    Returns
    -------
    ResultT
        The result of the method.

    Raises
    ------
    TypeError
        If `method` is an identifiability method; the message names
        ``Calibrator.identifiability``.
    ValueError
        If the configuration's type declares no runner, or
        a setting, an input, the data or the model does not fit the
        method; the message says why.
    RuntimeError
        If a computation of the method that could start fails: an engine
        run, an optimizer, every replicate, or a sampler that does not
        reach its target.
    InfeasibleStateError
        If a draw of a prediction leaves the values its model allows.
    """
    return run_method(self, method, UncertaintyMethod)
```

#### identifiability

```python
identifiability(method: IdentifiabilityMethod[ResultT]) -> ResultT
```

Run an identifiability method on this calibration problem.

Each method answers which parameters the model and the data determine, from another kind of evidence: `StructuralRank`, `Collinearity`, `SampleIdentifiability` or `StartsIdentifiability`. Each returns an `IdentifiabilityReport`; `IdentifiabilityReport.merge` joins them.

Parameters:

| Name     | Type                             | Description                      | Default    |
| -------- | -------------------------------- | -------------------------------- | ---------- |
| `method` | `IdentifiabilityMethod[ResultT]` | The configuration of the method. | *required* |

Returns:

| Type      | Description               |
| --------- | ------------------------- |
| `ResultT` | The result of the method. |

Raises:

| Type           | Description                                                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `TypeError`    | If method is an uncertainty method; the message names Calibrator.uncertainty.                                                                |
| `ValueError`   | If the configuration's type declares no runner, or a setting, an input, the data or the model does not fit the method; the message says why. |
| `RuntimeError` | If a computation of the method that could start fails: an engine run or an optimizer.                                                        |

Source code in `commol/_calibration/calibrator.py`

```python
def identifiability(self, method: IdentifiabilityMethod[ResultT]) -> ResultT:
    """
    Run an identifiability method on this calibration problem.

    Each method answers which parameters the model and the data
    determine, from another kind of evidence: `StructuralRank`,
    `Collinearity`, `SampleIdentifiability` or `StartsIdentifiability`.
    Each returns an `IdentifiabilityReport`; `IdentifiabilityReport.merge`
    joins them.

    Parameters
    ----------
    method : IdentifiabilityMethod[ResultT]
        The configuration of the method.

    Returns
    -------
    ResultT
        The result of the method.

    Raises
    ------
    TypeError
        If `method` is an uncertainty method; the message names
        ``Calibrator.uncertainty``.
    ValueError
        If the configuration's type declares no runner, or
        a setting, an input, the data or the model does not fit the
        method; the message says why.
    RuntimeError
        If a computation of the method that could start fails: an engine
        run or an optimizer.
    """
    return run_method(self, method, IdentifiabilityMethod)
```

## CalibrationProblem

Bases: `RangeChecked`

Defines a complete calibration problem.

This class encapsulates all the information needed to calibrate model parameters against observed data. It provides validation of the calibration setup but delegates the actual optimization to the Rust backend.

Attributes:

| Name                          | Type                           | Description                                                                                                                                                                                               |
| ----------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observed_data`               | `list[ObservedDataPoint]`      | List of observed data points to fit against                                                                                                                                                               |
| `parameters`                  | `list[CalibrationParameter]`   | List of parameters to calibrate with their bounds                                                                                                                                                         |
| `constraints`                 | `list[CalibrationConstraint]`  | List of constraints on calibration parameters (optional, default: empty list)                                                                                                                             |
| `loss_function`               | \`{sse, rmse, mae}             | None\`                                                                                                                                                                                                    |
| `observation_models`          | \`dict[str, observation model] | None\`                                                                                                                                                                                                    |
| `normalize_observations`      | `bool`                         | When True, divide each residual by the RMS of its observation series before applying the loss metric (default: False). It needs a loss: observation models set the scale of each series with their noise. |
| `optimization_config`         | \`NelderMeadConfig             | ParticleSwarmConfig\`                                                                                                                                                                                     |
| `initial_condition_remainder` | \`str                          | None\`                                                                                                                                                                                                    |
| `replicates`                  | \`int                          | None\`                                                                                                                                                                                                    |
| `replicate_summary`           | \`{mean, median, crps}         | Quantile                                                                                                                                                                                                  |
| `seed`                        | \`int                          | None\`                                                                                                                                                                                                    |

Source code in `commol/_problem/problem.py`

```python
class CalibrationProblem(RangeChecked):
    """
    Defines a complete calibration problem.

    This class encapsulates all the information needed to calibrate model
    parameters against observed data. It provides validation of the calibration
    setup but delegates the actual optimization to the Rust backend.

    Attributes
    ----------
    observed_data : list[ObservedDataPoint]
        List of observed data points to fit against
    parameters : list[CalibrationParameter]
        List of parameters to calibrate with their bounds
    constraints : list[CalibrationConstraint]
        List of constraints on calibration parameters (optional, default: empty list)
    loss_function : {"sse", "rmse", "mae"} | None
        The loss that measures the fit, with the observation weights. A
        problem has a loss or ``observation_models``, exactly one, with no
        default: what to fit is a modelling choice.
    observation_models : dict[str, observation model] | None
        One observation model per series label (`NormalObservations`,
        `LogNormalObservations`, `StudentTObservations`,
        `PoissonObservations`, `NegativeBinomialObservations`,
        `BinomialObservations`): the problem fits by maximum likelihood, and
        its loss is the negative log-likelihood. Every observed series needs
        one, and every parameter of type ``"noise"`` must be an argument of
        one. It needs a deterministic engine.
    normalize_observations : bool
        When True, divide each residual by the RMS of its observation series
        before applying the loss metric (default: False). It needs a loss:
        observation models set the scale of each series with their noise.
    optimization_config : NelderMeadConfig | ParticleSwarmConfig
        Configuration for the optimization algorithm, by its ``kind``
    initial_condition_remainder : str | None
        Expanded compartment or bin that takes what the calibrated initial
        conditions leave of the amounts the model states (default: None). It
        is not calibrated itself: it takes the sum of the amounts the model
        states, less every other compartment, so that sum does not change.
        A value that would make it negative is penalized. Without a
        remainder, calibrated initial conditions are applied as given, and
        the other compartments keep their amounts.
    replicates : int | None
        For a stochastic engine: how many replicates each evaluation runs,
        with common random numbers (the same seeds, derived from ``seed``, at
        every evaluation), so the loss is a deterministic function of the
        parameters. Required with a stochastic engine; a deterministic engine
        does not use it.
    replicate_summary : {"mean", "median", "crps"} | Quantile | None
        For a stochastic engine: what each observation is compared with. The
        mean, the median or a `Quantile` of the replicates, with
        ``loss_function``; or ``"crps"``, the continuous ranked probability
        score of their distribution, whose loss is the weighted mean score.
        Required with a stochastic engine, with no default: what to fit is a
        modelling choice.
    seed : int | None
        Seed of every random number of a calibration (default: None, a seed
        the engine draws when the calibrator is built, which `Calibrator.seed`
        gives and the calibrator logs):
        the replicates of a stochastic engine, the starts of
        ``Calibrator.run(starts=n)``, a particle swarm, and the uncertainty
        methods. Required with a stochastic engine.
    """

    model_config = ConfigDict(extra="forbid")

    observed_data: list[ObservedDataPoint] = Field(
        default=..., min_length=1, description="Observed data points"
    )
    parameters: list[CalibrationParameter] = Field(
        default=..., min_length=1, description="Parameters to calibrate"
    )
    constraints: list[CalibrationConstraint] = Field(
        default_factory=list,
        description="Constraints on calibration parameters",
    )
    loss_function: Literal["sse", "rmse", "mae"] | None = Field(
        default=None,
        description="Loss function to use for measuring fit quality",
    )
    observation_models: dict[str, _ObservationModel] | None = Field(
        default=None,
        description="One observation model per series label",
    )
    normalize_observations: bool = Field(
        default=False,
        description=(
            "Normalize each residual by the RMS of its observation series before "
            "applying the loss metric. Keeps series measured on very different "
            "scales comparable in the loss."
        ),
    )
    optimization_config: Annotated[
        NelderMeadConfig | ParticleSwarmConfig, Field(discriminator="kind")
    ] = Field(default=..., description="Optimization algorithm configuration")
    initial_condition_remainder: str | None = Field(
        default=None,
        description=(
            "Expanded compartment or bin that takes what the calibrated "
            "initial conditions leave of the amounts the model states."
        ),
    )
    replicates: int | None = Field(
        default=None,
        description="Replicates per evaluation of a stochastic engine.",
    )
    replicate_summary: Literal["mean", "median", "crps"] | Quantile | None = Field(
        default=None,
        description="What the observations are compared with, for a stochastic engine.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Seed of every random number of a calibration: the replicates of a "
            "stochastic engine, the starts, a particle swarm and the methods"
        ),
    )

    @model_validator(mode="after")
    def _validate_loss_or_models(self) -> Self:
        """Require a loss or observation models, exactly one."""
        if self.loss_function is not None and self.observation_models is not None:
            raise ValueError(
                "A problem has a loss_function or observation_models, not both: "
                "the fit and the posterior would answer different questions."
            )
        if self.loss_function is None and self.observation_models is None:
            raise ValueError(
                "A problem needs a loss_function (sse, rmse or mae) or "
                "observation_models. There is no default: what to fit is a "
                "modelling choice."
            )
        if self.observation_models is not None and self.normalize_observations:
            raise ValueError(
                "normalize_observations needs a loss_function: the observation "
                "models set the scale of each series with their noise."
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_parameter_ids(self) -> Self:
        """Ensure parameter IDs are unique."""
        param_ids = [p.id for p in self.parameters]
        if len(param_ids) != len(set(param_ids)):
            duplicates = [id for id in set(param_ids) if param_ids.count(id) > 1]
            raise ValueError(f"Duplicate parameter IDs found: {duplicates}")
        return self
```

## Quantile

Bases: `BaseModel`

A quantile of the replicates of a stochastic engine, for `CalibrationProblem.replicate_summary`.

It is the one quantile of commol, the one of `weighted_quantiles` and `SimulationReplicates.quantiles`: with equal weights, NumPy's `method="hazen"`. `Quantile(0.25)` of the replicates 1, 2, 3, 4, 5 is 1.75.

Attributes:

| Name    | Type    | Description                                             |
| ------- | ------- | ------------------------------------------------------- |
| `level` | `float` | The level, between 0 and 1, such as 0.5 for the median. |

Source code in `commol/_problem/problem.py`

```python
class Quantile(BaseModel):
    """
    A quantile of the replicates of a stochastic engine, for
    ``CalibrationProblem.replicate_summary``.

    It is the one quantile of commol, the one of `weighted_quantiles` and
    `SimulationReplicates.quantiles`: with equal weights, NumPy's
    ``method="hazen"``. ``Quantile(0.25)`` of the replicates 1, 2, 3, 4, 5 is
    1.75.

    Attributes
    ----------
    level : float
        The level, between 0 and 1, such as 0.5 for the median.
    """

    model_config = ConfigDict(extra="forbid")

    level: Annotated[float, AfterValidator(check_quantile_level)] = Field(
        ..., description="Level of the quantile."
    )

    def __init__(self, level: float) -> None:
        """
        A quantile of the replicates.

        Parameters
        ----------
        level : float
            The level, between 0 and 1.
        """
        super().__init__(level=level)
```

### Methods:

#### __init__

```python
__init__(level: float) -> None
```

A quantile of the replicates.

Parameters:

| Name    | Type    | Description                 | Default    |
| ------- | ------- | --------------------------- | ---------- |
| `level` | `float` | The level, between 0 and 1. | *required* |

Source code in `commol/_problem/problem.py`

```python
def __init__(self, level: float) -> None:
    """
    A quantile of the replicates.

    Parameters
    ----------
    level : float
        The level, between 0 and 1.
    """
    super().__init__(level=level)
```

## ObservedDataPoint

Bases: `RangeChecked`

Represents a single observed data point for calibration.

An observation compares one output, or the sum of several outputs, with an observed value at a position on the axis.

Attributes:

| Name        | Type        | Description                                                                                                                                                                                                                                                                                                      |
| ----------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `at`        | `float`     | Position of the observation on the axis. The engine decides which positions its model type accepts: a DifferentialEquations model any non-negative position, a DifferenceEquations model only an integer step.                                                                                                   |
| `output`    | \`str       | None\`                                                                                                                                                                                                                                                                                                           |
| `outputs`   | \`list[str] | None\`                                                                                                                                                                                                                                                                                                           |
| `label`     | `str`       | Name of the observed series, used by reports, plots and the per-series normalization. Defaults to output.                                                                                                                                                                                                        |
| `value`     | `float`     | Observed value                                                                                                                                                                                                                                                                                                   |
| `weight`    | `float`     | How much this observation counts in the loss (default: 1.0). Every loss multiplies the squared or absolute residual by it: sse is Σ w·r², rmse is √(Σ w·r² / Σ w), mae is Σ w·                                                                                                                                   |
| `factor_id` | \`str       | None\`                                                                                                                                                                                                                                                                                                           |
| `scale`     | `float`     | A known positive number that multiplies the model output, with the factor if there is one, before comparison (default: 1.0): the known share of the model output that the observation sees, such as the share of the units that report. A count model then reads the observed count with the mean scale · model. |
| `window`    | \`float     | None\`                                                                                                                                                                                                                                                                                                           |
| `trials`    | \`int       | None\`                                                                                                                                                                                                                                                                                                           |

Source code in `commol/_problem/problem.py`

```python
class ObservedDataPoint(RangeChecked):
    """
    Represents a single observed data point for calibration.

    An observation compares one output, or the sum of several outputs, with an
    observed value at a position on the axis.

    Attributes
    ----------
    at : float
        Position of the observation on the axis. The engine decides which
        positions its model type accepts: a DifferentialEquations model any
        non-negative position, a DifferenceEquations model only an integer
        step.
    output : str | None
        The output being observed. Give either `output` or `outputs`.
    outputs : list[str] | None
        Outputs summed before comparing with this observation. Needs a
        `label`.
    label : str
        Name of the observed series, used by reports, plots and the
        per-series normalization. Defaults to `output`.
    value : float
        Observed value
    weight : float
        How much this observation counts in the loss (default: 1.0). Every
        loss multiplies the squared or absolute residual by it: ``sse`` is
        ``Σ w·r²``, ``rmse`` is ``√(Σ w·r² / Σ w)``, ``mae`` is
        ``Σ w·|r| / Σ w``. With observation models it is a power of the
        observation's likelihood, ``w·log p``: a weight of 2 counts the
        observation twice.
    factor_id : str | None
        Optional factor parameter ID that multiplies the model output before
        comparison
    scale : float
        A known positive number that multiplies the model output, with the
        factor if there is one, before comparison (default: 1.0): the known
        share of the model output that the observation sees, such as the
        share of the units that report. A count model then reads the
        observed count with the mean ``scale · model``.
    window : float | None
        Optional window length for a cumulative output: the prediction is the
        output at `at` minus the output at ``at - window``.
    trials : int | None
        The trials of this observation, for a `BinomialObservations` model
        with no ``trials`` of its own (default: None).
    """

    model_config = ConfigDict(extra="forbid")

    at: float = Field(default=..., description="Position on the axis")
    output: str | None = Field(
        default=None, min_length=1, description="The output being observed"
    )
    outputs: list[str] | None = Field(
        default=None,
        description="Outputs summed before comparison",
    )
    label: str = Field(
        default="",
        description="Name of the observed series; defaults to `output`",
    )
    value: float = Field(default=..., description="Observed value")
    weight: float = Field(
        default=1.0,
        description="Weight for this observation in the loss function",
    )
    factor_id: str | None = Field(
        default=None,
        description="Optional factor parameter ID that multiplies the model output",
    )
    scale: float = Field(
        default=1.0,
        description="Known positive number that multiplies the model output",
    )
    window: float | None = Field(
        default=None,
        description=(
            "Optional window length for cumulative-output observations. "
            "When set, prediction is value(at) - value(at - window)."
        ),
    )
    trials: int | None = Field(
        default=None,
        description="Trials of a binomial observation model that reads them here",
    )

    @model_validator(mode="after")
    def _validate_observed_outputs(self) -> Self:
        """Require one of `output` and `outputs`; name the series."""
        if (self.output is None) == (self.outputs is None):
            raise ValueError("Give exactly one of 'output' and 'outputs'.")
        if self.outputs is not None:
            if not self.outputs:
                raise ValueError("outputs must be non-empty when provided")
            if not self.label:
                raise ValueError(
                    "An observation of several outputs needs a 'label' that names "
                    "their sum."
                )
        elif not self.label:
            self.label = self.output or ""
        return self

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        """
        A window must not start before position 0, with the rule of the
        engine (`check_window_end` of
        `crates/commol-calibration/src/observation.rs`).
        """
        if self.window is not None:
            commol_rs.calibration.check_window_end(self.at, self.window)
        return self

    @property
    def observed_outputs(self) -> list[str]:
        """The outputs whose sum is compared: `outputs`, or `output` alone."""
        return list(self.outputs) if self.outputs is not None else [self.output or ""]
```

### Attributes

#### observed_outputs

```python
observed_outputs: list[str]
```

The outputs whose sum is compared: `outputs`, or `output` alone.

## Unknowns, optimizers and results

## CalibrationParameter

Bases: `BaseModel`

Defines a parameter or initial condition to be calibrated with its bounds.

Attributes:

| Name             | Type                                            | Description                                                                                                                                                                                                                                                                |
| ---------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`             | `str`                                           | Identifier (parameter ID for parameters, bin ID for initial conditions)                                                                                                                                                                                                    |
| `parameter_type` | `{parameter, initial_condition, factor, noise}` | What the value is: a parameter of the model; the initial amount of a compartment, a bin or a category; a factor that multiplies the model output of an observation; or an argument of an observation model, such as its standard deviation, which never reaches the engine |
| `min_bound`      | `float`                                         | Minimum allowed value for this parameter                                                                                                                                                                                                                                   |
| `max_bound`      | `float`                                         | Maximum allowed value for this parameter                                                                                                                                                                                                                                   |
| `initial_guess`  | \`float                                         | None\`                                                                                                                                                                                                                                                                     |

Source code in `commol/_problem/problem.py`

```python
class CalibrationParameter(BaseModel):
    """
    Defines a parameter or initial condition to be calibrated with its bounds.

    Attributes
    ----------
    id : str
        Identifier (parameter ID for parameters, bin ID for initial conditions)
    parameter_type : {"parameter", "initial_condition", "factor", "noise"}
        What the value is: a parameter of the model; the initial amount of a
        compartment, a bin or a category; a factor that multiplies the model
        output of an observation; or an argument of an observation model, such
        as its standard deviation, which never reaches the engine
    min_bound : float
        Minimum allowed value for this parameter
    max_bound : float
        Maximum allowed value for this parameter
    initial_guess : float | None
        Optional starting value for optimization (if None, midpoint is used)
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default=..., min_length=1, description="Parameter or bin identifier"
    )
    parameter_type: Literal["parameter", "initial_condition", "factor", "noise"] = (
        Field(
            default=...,
            description="Type of value being calibrated",
        )
    )
    min_bound: float = Field(default=..., description="Minimum allowed value")
    max_bound: float = Field(default=..., description="Maximum allowed value")
    initial_guess: float | None = Field(
        default=None, description="Optional starting value for optimization"
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Validate that max_bound > min_bound and initial_guess is within bounds."""
        if self.max_bound <= self.min_bound:
            raise ValueError(
                (
                    f"max_bound ({self.max_bound}) must be greater than "
                    f"min_bound ({self.min_bound}) for parameter '{self.id}'"
                )
            )
        if self.initial_guess is not None:
            if not (self.min_bound <= self.initial_guess <= self.max_bound):
                raise ValueError(
                    (
                        f"initial_guess ({self.initial_guess}) must be between "
                        f"min_bound ({self.min_bound}) and max_bound "
                        f"({self.max_bound}) for parameter '{self.id}'"
                    )
                )

        return self
```

## CalibrationConstraint

Bases: `RangeChecked`

A constraint on calibration parameters defined as a mathematical expression.

Constraints are mathematical expressions that must evaluate to >= 0 for the constraint to be satisfied. When the expression evaluates to < 0, the constraint is violated and a penalty is applied during optimization.

Attributes:

| Name          | Type          | Description                                                                                                                                                                                                                                                                                                                                                   |
| ------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`          | `str`         | Unique identifier for this constraint                                                                                                                                                                                                                                                                                                                         |
| `expression`  | `str`         | Mathematical expression that must evaluate >= 0 for constraint satisfaction. Can reference calibration parameters by their IDs, and the entries <M>_<i>_<j> and the spectral radius <M>\_radius of a matrix between categories, scaled by the values of the point. When at is specified, can also reference output values (A, B, C, etc.) at those positions. |
| `description` | \`str         | None\`                                                                                                                                                                                                                                                                                                                                                        |
| `weight`      | `float`       | Penalty weight multiplier (default: 1.0). Higher values make this constraint more important relative to others. The amount of a violation of this constraint is weight \*                                                                                                                                                                                     |
| `at`          | \`list[float] | None\`                                                                                                                                                                                                                                                                                                                                                        |

Examples:

```pycon
>>> # k1/k2 must be <= 5
>>> constraint = CalibrationConstraint(
...     id="ratio_bound",
...     expression="5.0 - k1/k2",
...     description="Ratio k1/k2 <= 5",
... )
```

```pycon
>>> # k1 must be >= k2
>>> constraint = CalibrationConstraint(
...     id="ordering",
...     expression="k1 - k2",
...     description="k1 >= k2",
... )
```

```pycon
>>> # Sum of parameters <= 1.0
>>> constraint = CalibrationConstraint(
...     id="sum_bound",
...     expression="1.0 - (param1 + param2 + param3)",
...     description="Sum of rates <= 1.0",
... )
```

```pycon
>>> # At positions: bin B stays at or below 500 at specific positions
>>> constraint = CalibrationConstraint(
...     id="peak_b",
...     expression="500.0 - B",
...     description="Peak B never exceeds 500",
...     at=[10, 20, 30, 40, 50],
... )
```

Source code in `commol/_problem/problem.py`

```python
class CalibrationConstraint(RangeChecked):
    """
    A constraint on calibration parameters defined as a mathematical expression.

    Constraints are mathematical expressions that must evaluate to >= 0 for the
    constraint to be satisfied. When the expression evaluates to < 0, the constraint
    is violated and a penalty is applied during optimization.

    Attributes
    ----------
    id : str
        Unique identifier for this constraint
    expression : str
        Mathematical expression that must evaluate >= 0 for constraint satisfaction.
        Can reference calibration parameters by their IDs, and the entries
        ``<M>_<i>_<j>`` and the spectral radius ``<M>_radius`` of a matrix
        between categories, scaled by the values of the point. When `at` is
        specified, can also reference output values (A, B, C, etc.) at those
        positions.
    description : str | None
        Human-readable description of the constraint (optional)
    weight : float
        Penalty weight multiplier (default: 1.0). Higher values make this constraint
        more important relative to others. The amount of a violation of this
        constraint is weight * |violation|, summed over its positions; the loss
        of a point that violates constraints is P * (1 + the sum of the amounts),
        with P much larger than any fit.
    at : list[float] | None
        Optional positions on the axis at which to evaluate this constraint.
        The engine decides which positions its model type accepts: a
        DifferenceEquations model only integer steps. If None, the constraint
        is evaluated once before simulation from the parameter values and the
        matrices at them. If given, it is evaluated at each position and can
        reference output values.

    Examples
    --------
    >>> # k1/k2 must be <= 5
    >>> constraint = CalibrationConstraint(
    ...     id="ratio_bound",
    ...     expression="5.0 - k1/k2",
    ...     description="Ratio k1/k2 <= 5",
    ... )

    >>> # k1 must be >= k2
    >>> constraint = CalibrationConstraint(
    ...     id="ordering",
    ...     expression="k1 - k2",
    ...     description="k1 >= k2",
    ... )

    >>> # Sum of parameters <= 1.0
    >>> constraint = CalibrationConstraint(
    ...     id="sum_bound",
    ...     expression="1.0 - (param1 + param2 + param3)",
    ...     description="Sum of rates <= 1.0",
    ... )

    >>> # At positions: bin B stays at or below 500 at specific positions
    >>> constraint = CalibrationConstraint(
    ...     id="peak_b",
    ...     expression="500.0 - B",
    ...     description="Peak B never exceeds 500",
    ...     at=[10, 20, 30, 40, 50],
    ... )
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default=..., min_length=1, description="Unique identifier for this constraint"
    )
    expression: str = Field(
        default=...,
        description="Mathematical expression that must evaluate >= 0",
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the constraint"
    )
    weight: float = Field(
        default=1.0,
        description="Penalty weight multiplier for violations",
    )
    at: list[float] | None = Field(
        default=None,
        description=(
            "Optional positions on the axis at which to evaluate the constraint; "
            "it can then reference output values."
        ),
    )

    @field_validator("expression")
    @classmethod
    def _validate_expression(cls, value: str) -> str:
        """Check the expression with the parser of the engine."""
        commol_rs.core.MathExpression(value).check()
        return value

    @model_validator(mode="after")
    def _validate_positions(self) -> Self:
        """Validate that the positions, when given, are not empty."""
        if self.at is not None and len(self.at) == 0:
            raise ValueError(
                f"at for constraint '{self.id}' must not be empty if specified"
            )
        return self
```

## NelderMeadConfig

Bases: `RangeChecked`

Configuration for the Nelder-Mead optimization algorithm.

The Nelder-Mead method is a simplex-based derivative-free optimization algorithm, suitable for problems where gradients are not available.

Attributes:

| Name                   | Type          | Description                                                                                                                                                                                                                                     |
| ---------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`                 | `nelder_mead` | The optimizer, which the engine reads.                                                                                                                                                                                                          |
| `max_iterations`       | `int`         | Maximum number of iterations (default: 1000)                                                                                                                                                                                                    |
| `sd_tolerance`         | `float`       | Convergence tolerance for standard deviation (default: 1e-6)                                                                                                                                                                                    |
| `initial_simplex_size` | `float`       | Size of the initial simplex, as a fraction of each parameter's bound width, in (0, 1\] (default: 0.1). Nelder-Mead searches an unbounded coordinate u and scores x = lo + (hi - lo)(sin u + 1)/2, so every point it tries is inside the bounds. |
| `alpha`                | \`float       | None\`                                                                                                                                                                                                                                          |
| `gamma`                | \`float       | None\`                                                                                                                                                                                                                                          |
| `rho`                  | \`float       | None\`                                                                                                                                                                                                                                          |
| `sigma`                | \`float       | None\`                                                                                                                                                                                                                                          |
| `verbose`              | `bool`        | Enable verbose output during optimization (default: False)                                                                                                                                                                                      |
| `header_interval`      | `int`         | Number of iterations between table header repeats in verbose output (default: 100)                                                                                                                                                              |

Source code in `commol/_problem/problem.py`

```python
class NelderMeadConfig(RangeChecked):
    """
    Configuration for the Nelder-Mead optimization algorithm.

    The Nelder-Mead method is a simplex-based derivative-free optimization
    algorithm, suitable for problems where gradients are not available.

    Attributes
    ----------
    kind : "nelder_mead"
        The optimizer, which the engine reads.
    max_iterations : int
        Maximum number of iterations (default: 1000)
    sd_tolerance : float
        Convergence tolerance for standard deviation (default: 1e-6)
    initial_simplex_size : float
        Size of the initial simplex, as a fraction of each parameter's bound
        width, in (0, 1] (default: 0.1). Nelder-Mead searches an unbounded
        coordinate ``u`` and scores ``x = lo + (hi - lo)(sin u + 1)/2``, so
        every point it tries is inside the bounds.
    alpha : float | None
        Reflection coefficient (default: None, uses argmin's default)
    gamma : float | None
        Expansion coefficient (default: None, uses argmin's default)
    rho : float | None
        Contraction coefficient (default: None, uses argmin's default)
    sigma : float | None
        Shrink coefficient (default: None, uses argmin's default)
    verbose : bool
        Enable verbose output during optimization (default: False)
    header_interval: int
        Number of iterations between table header repeats in verbose output
        (default: 100)
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["nelder_mead"] = Field(
        default="nelder_mead", description="The optimizer"
    )
    max_iterations: int = Field(
        default=1000, description="Maximum number of iterations"
    )
    sd_tolerance: float = Field(
        default=1e-6, description="Convergence tolerance for standard deviation"
    )
    initial_simplex_size: float = Field(
        default=0.1,
        description=(
            "Size of the initial simplex, as a fraction of each parameter's bound width"
        ),
    )
    alpha: float | None = Field(
        default=None,
        description="Reflection coefficient (default: None, uses argmin's default)",
    )
    gamma: float | None = Field(
        default=None,
        description="Expansion coefficient (default: None, uses argmin's default)",
    )
    rho: float | None = Field(
        default=None,
        description="Contraction coefficient (default: None, uses argmin's default)",
    )
    sigma: float | None = Field(
        default=None,
        description="Shrink coefficient (default: None, uses argmin's default)",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output during optimization (default: False)",
    )
    header_interval: int = Field(
        default=100,
        description=(
            "Number of iterations between table header repeats in verbose output "
            "(default: 100)"
        ),
    )
```

## ParticleSwarmConfig

Bases: `RangeChecked`

Configuration for the Particle Swarm Optimization algorithm.

Each strategy of the swarm is a name, and its numbers are fields of their own. A strategy left as None is that of canonical PSO; a number set without its strategy, or with another strategy, is an error, so a setting is never ignored.

Attributes:

| Name                                 | Type                                           | Description                                                                         |
| ------------------------------------ | ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| `kind`                               | `particle_swarm`                               | The optimizer, which the engine reads.                                              |
| `particles`                          | `int`                                          | Number of particles in the swarm (default: 20).                                     |
| `max_iterations`                     | `int`                                          | Maximum number of iterations (default: 1000).                                       |
| `verbose`                            | `bool`                                         | Enable verbose output (default: False).                                             |
| `header_interval`                    | `int`                                          | Number of iterations between table header repeats in verbose output (default: 100). |
| `initialization`                     | `{uniform, latin_hypercube, opposition_based}` | How the particles start (default: "uniform").                                       |
| `inertia`                            | \`{constant, chaotic}                          | None\`                                                                              |
| `inertia_factor`                     | \`float                                        | None\`                                                                              |
| `inertia_min`                        | \`float                                        | None\`                                                                              |
| `inertia_max`                        | \`float                                        | None\`                                                                              |
| `acceleration`                       | \`{constant, time_varying}                     | None\`                                                                              |
| `cognitive`                          | \`float                                        | None\`                                                                              |
| `social`                             | \`float                                        | None\`                                                                              |
| `cognitive_initial, cognitive_final` | \`float                                        | None\`                                                                              |
| `social_initial, social_final`       | \`float                                        | None\`                                                                              |
| `mutation`                           | \`{gaussian, cauchy}                           | None\`                                                                              |
| `mutation_scale`                     | \`float                                        | None\`                                                                              |
| `mutation_probability`               | \`float                                        | None\`                                                                              |
| `mutation_application`               | \`{global_best, all_particles, below_average}  | None\`                                                                              |
| `velocity_clamp_factor`              | \`float                                        | None\`                                                                              |
| `velocity_mutation_threshold`        | \`float                                        | None\`                                                                              |
| `improvement_window`                 | \`int                                          | None\`                                                                              |
| `improvement_tol_abs`                | \`float                                        | None\`                                                                              |
| `improvement_tol_rel`                | \`float                                        | None\`                                                                              |
| `radius_tol`                         | \`float                                        | None\`                                                                              |

References

Liu, B., Wang, L., Jin, Y. H., Tang, F., & Huang, D. X. (2005). Improved particle swarm optimization combined with chaos. Chaos, Solitons & Fractals, 25(5), 1261-1271.

Ratnaweera, A., Halgamuge, S. K., & Watson, H. C. (2004). Self-organizing hierarchical particle swarm optimizer with time-varying acceleration coefficients. IEEE Transactions on Evolutionary Computation, 8(3), 240-255.

Stacey, A., Jancic, M., & Grundy, I. (2003). Particle swarm optimization with mutation. In Proceedings of the 2003 Congress on Evolutionary Computation (CEC 2003) (Vol. 2, pp. 1425-1430).

Zielinski, K., & Laur, R. (2007). Stopping criteria for a constrained single-objective particle swarm optimization algorithm. Informatica, 31(1), 51-59.

Source code in `commol/_problem/problem.py`

```python
class ParticleSwarmConfig(RangeChecked):
    """
    Configuration for the Particle Swarm Optimization algorithm.

    Each strategy of the swarm is a name, and its numbers are fields of their
    own. A strategy left as None is that of canonical PSO; a number set
    without its strategy, or with another strategy, is an error, so a setting
    is never ignored.

    Attributes
    ----------
    kind : "particle_swarm"
        The optimizer, which the engine reads.
    particles : int
        Number of particles in the swarm (default: 20).
    max_iterations : int
        Maximum number of iterations (default: 1000).
    verbose : bool
        Enable verbose output (default: False).
    header_interval : int
        Number of iterations between table header repeats in verbose output
        (default: 100).
    initialization : {"uniform", "latin_hypercube", "opposition_based"}
        How the particles start (default: "uniform").
    inertia : {"constant", "chaotic"} | None
        The inertia weight: "constant" uses `inertia_factor`, "chaotic" moves
        between `inertia_min` and `inertia_max` on a logistic map (Liu et al.,
        2005). None is the weight of canonical PSO, 1/(2 ln 2) ≈ 0.72135.
    inertia_factor : float | None
        The constant weight, above 0; None is that of canonical PSO. Only with
        ``inertia="constant"``.
    inertia_min : float | None
        The lowest chaotic weight, above 0. Needed with ``inertia="chaotic"``.
    inertia_max : float | None
        The highest chaotic weight, above `inertia_min`. Needed with
        ``inertia="chaotic"``.
    acceleration : {"constant", "time_varying"} | None
        The acceleration coefficients: "constant" uses `cognitive` and
        `social`, "time_varying" moves each from its initial to its final
        value (Ratnaweera et al., 2004). None is the coefficients of canonical
        PSO, 0.5 + ln 2 ≈ 1.19315.
    cognitive : float | None
        The attraction to the best point of the particle (c1), above 0; None
        is that of canonical PSO. Only with ``acceleration="constant"``.
    social : float | None
        The attraction to the best point of the swarm (c2), above 0; None is
        that of canonical PSO. Only with ``acceleration="constant"``.
    cognitive_initial, cognitive_final : float | None
        The cognitive coefficient at the first and the last iteration, above
        0. Needed with ``acceleration="time_varying"``.
    social_initial, social_final : float | None
        The social coefficient at the first and the last iteration, above 0.
        Needed with ``acceleration="time_varying"``.
    mutation : {"gaussian", "cauchy"} | None
        The distribution of a mutation step, to escape a local optimum
        (Stacey et al., 2003); Cauchy has heavier tails. None mutates no
        particle.
    mutation_scale : float | None
        The standard deviation (gaussian) or the scale (cauchy) of a step,
        above 0. Needed with a mutation.
    mutation_probability : float | None
        The probability of a mutation at each iteration, from 0 to 1. Needed
        with a mutation.
    mutation_application : {"global_best", "all_particles", "below_average"} | None
        The particles a mutation moves. Needed with a mutation.
    velocity_clamp_factor : float | None
        The largest velocity, as a fraction of each bound width, from 0 to 1;
        None clamps no velocity.
    velocity_mutation_threshold : float | None
        The speed below which a velocity is drawn again, above 0; None draws
        none again.
    improvement_window : int | None
        Stop when the best loss improved by at most
        ``improvement_tol_abs + improvement_tol_rel * |best loss|`` over the
        last `improvement_window` iterations, and report ``converged=True``.
        Give the three fields or none. A swarm can stall for some iterations
        and then improve again, so a short window can stop it early.
    improvement_tol_abs : float | None
        The absolute tolerance of the improvement, at least 0.
    improvement_tol_rel : float | None
        The tolerance of the improvement relative to the best loss, at least
        0.
    radius_tol : float | None
        Stop when the largest distance of a particle from the best position,
        divided by the size of the search box (the norm of the widths of the
        bounds), is at most `radius_tol`, and report ``converged=True``
        (Zielinski and Laur, 2007). Mutation and velocity mutation scatter the
        swarm again, so with them the radius can stay above it.

    References
    ----------
    Liu, B., Wang, L., Jin, Y. H., Tang, F., & Huang, D. X. (2005).
    Improved particle swarm optimization combined with chaos.
    Chaos, Solitons & Fractals, 25(5), 1261-1271.

    Ratnaweera, A., Halgamuge, S. K., & Watson, H. C. (2004).
    Self-organizing hierarchical particle swarm optimizer with
    time-varying acceleration coefficients. IEEE Transactions on
    Evolutionary Computation, 8(3), 240-255.

    Stacey, A., Jancic, M., & Grundy, I. (2003). Particle swarm
    optimization with mutation. In Proceedings of the 2003 Congress
    on Evolutionary Computation (CEC 2003) (Vol. 2, pp. 1425-1430).

    Zielinski, K., & Laur, R. (2007). Stopping criteria for a constrained
    single-objective particle swarm optimization algorithm. Informatica,
    31(1), 51-59.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["particle_swarm"] = Field(
        default="particle_swarm", description="The optimizer"
    )
    particles: int = Field(default=20, description="Number of particles in the swarm")
    max_iterations: int = Field(
        default=1000, description="Maximum number of iterations"
    )
    verbose: bool = Field(
        default=False, description="Enable verbose output during optimization"
    )
    header_interval: int = Field(
        default=100,
        description="Number of iterations between table header repeats",
    )
    initialization: Literal["uniform", "latin_hypercube", "opposition_based"] = Field(
        default="uniform", description="How the particles start"
    )
    inertia: Literal["constant", "chaotic"] | None = Field(
        default=None, description="The inertia weight"
    )
    inertia_factor: float | None = Field(
        default=None, description="The constant inertia weight"
    )
    inertia_min: float | None = Field(
        default=None, description="The lowest chaotic inertia weight"
    )
    inertia_max: float | None = Field(
        default=None, description="The highest chaotic inertia weight"
    )
    acceleration: Literal["constant", "time_varying"] | None = Field(
        default=None, description="The acceleration coefficients"
    )
    cognitive: float | None = Field(
        default=None, description="The constant cognitive coefficient (c1)"
    )
    social: float | None = Field(
        default=None, description="The constant social coefficient (c2)"
    )
    cognitive_initial: float | None = Field(
        default=None, description="The cognitive coefficient at the first iteration"
    )
    cognitive_final: float | None = Field(
        default=None, description="The cognitive coefficient at the last iteration"
    )
    social_initial: float | None = Field(
        default=None, description="The social coefficient at the first iteration"
    )
    social_final: float | None = Field(
        default=None, description="The social coefficient at the last iteration"
    )
    mutation: Literal["gaussian", "cauchy"] | None = Field(
        default=None, description="The distribution of a mutation step"
    )
    mutation_scale: float | None = Field(
        default=None, description="The scale of a mutation step"
    )
    mutation_probability: float | None = Field(
        default=None, description="The probability of a mutation per iteration"
    )
    mutation_application: (
        Literal["global_best", "all_particles", "below_average"] | None
    ) = Field(default=None, description="The particles a mutation moves")
    velocity_clamp_factor: float | None = Field(
        default=None, description="The largest velocity, as a share of the bounds"
    )
    velocity_mutation_threshold: float | None = Field(
        default=None, description="The speed below which a velocity is drawn again"
    )
    improvement_window: int | None = Field(
        default=None, description="The iterations of the improvement stop rule"
    )
    improvement_tol_abs: float | None = Field(
        default=None, description="The absolute tolerance of the improvement"
    )
    improvement_tol_rel: float | None = Field(
        default=None, description="The relative tolerance of the improvement"
    )
    radius_tol: float | None = Field(
        default=None, description="The tolerance of the radius stop rule"
    )

    @model_validator(mode="after")
    def _check_strategies(self) -> Self:
        """
        Check each number against its strategy, as the engine reads them.

        Returns
        -------
        Self
            The configuration.

        Raises
        ------
        ValueError
            If a strategy lacks a number it needs, a number is set without
            its strategy or with another one, or the chaotic weights are not
            in order; the message is that of the engine.
        """
        commol_rs.json.as_read("ParticleSwarmConfig", engine_json(self))
        return self
```

## CalibrationResult

Bases: `ResultModel`, `RangeChecked`

Result of a calibration: the best of its starts.

Attributes:

| Name                 | Type                 | Description                                                                                                                                                                                 |
| -------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `best_parameters`    | `dict[str, float]`   | The calibrated value of each parameter, in the order of the problem's parameters.                                                                                                           |
| `final_loss`         | `float`              | Loss at the best parameters.                                                                                                                                                                |
| `iterations`         | `int`                | Iterations of the best start.                                                                                                                                                               |
| `converged`          | `bool`               | Whether the best start converged.                                                                                                                                                           |
| `termination_reason` | `str`                | Why the best start stopped.                                                                                                                                                                 |
| `starts`             | `list[StartSummary]` | Every start that finished, in order; the first is the initial guess.                                                                                                                        |
| `failed`             | `list[str]`          | Each start that failed: "Start <index> failed: <reason>", also logged as a warning.                                                                                                         |
| `seed`               | `int`                | The seed of the calibration's random numbers (Calibrator.seed): problem.seed, or the seed the calibrator drew. Set as problem.seed, it repeats the calibration and its uncertainty methods. |

Source code in `commol/_fit/result.py`

```python
class CalibrationResult(ResultModel, RangeChecked):
    """
    Result of a calibration: the best of its starts.

    Attributes
    ----------
    best_parameters : dict[str, float]
        The calibrated value of each parameter, in the order of the problem's
        parameters.
    final_loss : float
        Loss at the best parameters.
    iterations : int
        Iterations of the best start.
    converged : bool
        Whether the best start converged.
    termination_reason : str
        Why the best start stopped.
    starts : list[StartSummary]
        Every start that finished, in order; the first is the initial guess.
    failed : list[str]
        Each start that failed: ``"Start <index> failed: <reason>"``, also
        logged as a warning.
    seed : int
        The seed of the calibration's random numbers (`Calibrator.seed`):
        ``problem.seed``, or the seed the calibrator drew. Set as
        ``problem.seed``, it repeats the calibration and its uncertainty
        methods.
    """

    model_config = ConfigDict(extra="forbid")

    best_parameters: dict[str, float] = Field(
        default=..., description="Calibrated parameter values"
    )
    final_loss: float = Field(default=..., description="Final loss value")
    iterations: int = Field(default=..., description="Number of iterations performed")
    converged: bool = Field(default=..., description="Whether optimization converged")
    termination_reason: str = Field(
        default=..., description="Reason for optimization termination"
    )
    starts: list[StartSummary] = Field(
        default_factory=list, description="Every start that finished"
    )
    failed: list[str] = Field(
        default_factory=list, description="Each start that failed"
    )
    seed: int = Field(
        default=..., description="Seed of the calibration's random numbers"
    )

    @override
    def __str__(self) -> str:
        """String representation of calibration result."""
        return (
            f"CalibrationResult(\n"
            f"  converged={self.converged},\n"
            f"  final_loss={self.final_loss:.6f},\n"
            f"  iterations={self.iterations},\n"
            f"  best_parameters={self.best_parameters},\n"
            f"  termination_reason='{self.termination_reason}',\n"
            f"  starts={len(self.starts)},\n"
            f"  failed={len(self.failed)},\n"
            f"  seed={self.seed}\n"
            f")"
        )
```

### Methods:

#### __str__

```python
__str__() -> str
```

String representation of calibration result.

Source code in `commol/_fit/result.py`

```python
@override
def __str__(self) -> str:
    """String representation of calibration result."""
    return (
        f"CalibrationResult(\n"
        f"  converged={self.converged},\n"
        f"  final_loss={self.final_loss:.6f},\n"
        f"  iterations={self.iterations},\n"
        f"  best_parameters={self.best_parameters},\n"
        f"  termination_reason='{self.termination_reason}',\n"
        f"  starts={len(self.starts)},\n"
        f"  failed={len(self.failed)},\n"
        f"  seed={self.seed}\n"
        f")"
    )
```

## StartSummary

Bases: `ResultModel`

One start of a multi-start optimization.

Distinct ends with equal losses are a sign that the data do not identify the parameters.

Attributes:

| Name        | Type               | Description                                          |
| ----------- | ------------------ | ---------------------------------------------------- |
| `start`     | `dict[str, float]` | The point the optimizer started from.                |
| `end`       | `dict[str, float]` | The best point it found.                             |
| `loss`      | `float`            | The loss at the end.                                 |
| `converged` | `bool`             | Whether the optimizer met its convergence criterion. |

Source code in `commol/_fit/result.py`

```python
class StartSummary(ResultModel):
    """
    One start of a multi-start optimization.

    Distinct ends with equal losses are a sign that the data do not identify
    the parameters.

    Attributes
    ----------
    start : dict[str, float]
        The point the optimizer started from.
    end : dict[str, float]
        The best point it found.
    loss : float
        The loss at the end.
    converged : bool
        Whether the optimizer met its convergence criterion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: dict[str, float] = Field(description="Start point")
    end: dict[str, float] = Field(description="Best point found")
    loss: float = Field(description="Loss at the end")
    converged: bool = Field(description="Whether the optimizer converged")
```

## CalibrationHistory

Bases: `ResultModel`

A calibration and every evaluation of a feasible point it made.

Attributes:

| Name          | Type                          | Description                                                        |
| ------------- | ----------------------------- | ------------------------------------------------------------------ |
| `result`      | `CalibrationResult`           | The result of the calibration.                                     |
| `evaluations` | `list[CalibrationEvaluation]` | Every evaluation of a feasible point, in the order they were made. |

Source code in `commol/_fit/result.py`

```python
class CalibrationHistory(ResultModel):
    """
    A calibration and every evaluation of a feasible point it made.

    Attributes
    ----------
    result : CalibrationResult
        The result of the calibration.
    evaluations : list[CalibrationEvaluation]
        Every evaluation of a feasible point, in the order they were made.
    """

    model_config = ConfigDict(extra="forbid")

    result: CalibrationResult = Field(description="The result")
    evaluations: list[CalibrationEvaluation] = Field(
        description="Every evaluation of a feasible point"
    )
```

## CalibrationEvaluation

Bases: `ResultModel`

One evaluation of a feasible point during an optimization.

Attributes:

| Name         | Type               | Description                      |
| ------------ | ------------------ | -------------------------------- |
| `parameters` | `dict[str, float]` | The values the objective scored. |
| `loss`       | `float`            | The loss at these values.        |

Source code in `commol/_fit/result.py`

```python
class CalibrationEvaluation(ResultModel):
    """
    One evaluation of a feasible point during an optimization.

    Attributes
    ----------
    parameters : dict[str, float]
        The values the objective scored.
    loss : float
        The loss at these values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: dict[str, float] = Field(description="Values the objective scored")
    loss: float = Field(description="Loss at these values")
```
