# ModelBuilder API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Builds a `Model`. A `StratificationCondition` selects the compartments of a category, and a `StratifiedRate` gives a transition its rate in the compartments that match its conditions. Guide: [Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md), [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md).

## ModelBuilder

A fluent interface that writes the process of a model, once.

The builder adds bins, accumulators, stratifications, matrices, transitions and parameters: the process, which is the same for every kind of model. `build` joins it with a structure, which lays the process out (such as `Aggregate`, amounts per compartment), and an engine, which runs it. Each method that adds returns the builder, so the calls chain.

Source code in `commol/_model/builder.py`

```python
class ModelBuilder:
    """
    A fluent interface that writes the process of a model, once.

    The builder adds bins, accumulators, stratifications, matrices,
    transitions and parameters: the process, which is the same for every
    kind of model. `build` joins it with a structure, which lays the process
    out (such as `Aggregate`, amounts per compartment), and an engine, which
    runs it.
    Each method that adds returns the builder, so the calls chain.
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        version: str | None = None,
        bin_unit: str | None = None,
        step_unit: str | None = None,
        origin: date | None = None,
    ) -> None:
        """
        Initialize the ModelBuilder.

        Parameters
        ----------
        name : str
            The unique name that identifies the model.
        description : str | None, default=None
            A human-readable description of the model's purpose and function.
        version : str | None, default=None
            The version number of the model.
        bin_unit : str | None, default=None
            The default unit for all bins.
            Individual bins can override this with their own unit parameter.
            Units are optional but required for model.check_unit_consistency().
        step_unit : str | None, default=None
            Unit of one step on the axis, of any single dimension (for example
            ``"day"`` or ``"meter"``). The unit check requires every flow to be
            ``<bin unit>/<step_unit>``, and ``Model.axis_units()`` converts
            values to steps of it. None means ``step``, the default increment
            of any axis.
        origin : date | None, default=None
            Calendar moment of position 0: a `datetime.datetime` with no time
            zone, or a `datetime.date`, read as its midnight. With it,
            ``Model.axis_units()`` converts between dates and positions, and
            ``results.plot.series()`` labels its ticks with dates. It
            needs a ``step_unit`` of time whose length is a whole number of
            days or a whole fraction of a day, such as ``"hour"`` or
            ``"day"``; `build` raises ``ValueError`` otherwise, or for a
            moment with a time zone.
        """
        self._name: str = name
        self._description: str | None = description
        self._version: str | None = version
        self._bin_unit: str | None = bin_unit
        self._step_unit: str | None = step_unit
        self._origin: date | None = origin

        self._bins: list[Bin] = []
        self._accumulators: list[Accumulator] = []
        self._stratifications: list[Stratification] = []
        self._matrices: list[CategoryMatrix] = []
        self._transitions: list[Transition] = []
        self._parameters: list[Parameter] = []
        self._series_counter: int = 0

        logger.info(
            (
                f"Initialized ModelBuilder: name='{self._name}', "
                f"version='{self._version or 'N/A'}'"
            )
        )

    def add_bin(self, id: str, name: str, unit: str | None = None) -> Self:
        """
        Add a bin to the model.

        Parameters
        ----------
        id : str
            Unique identifier for the bin.
        name : str
            Human-readable name for the bin.
        unit : str | None, default=None
            Unit of measurement for this bin. If None, uses the model-level bin_unit.

        Returns
        -------
        ModelBuilder
            Self for method chaining.
        """
        # Use bin-specific unit, or fall back to model-level bin_unit
        final_unit = unit if unit is not None else self._bin_unit
        self._bins.append(Bin(id=id, name=name, unit=final_unit))
        logger.info(f"Added bin: id='{id}', name='{name}', unit='{final_unit}'")
        return self

    def add_accumulator(self, id: str, name: str) -> Self:
        """
        Add an accumulator: a cumulative event counter in the model outputs.

        A transition adds its flow to the accumulators it names
        (``add_transition(accumulators=...)``). An accumulator is an output,
        not a part of the state, so ``total`` leaves it out.

        Parameters
        ----------
        id : str
            Unique identifier for the accumulator. It names its output, which
            stratifications expand as they expand a bin (``acc_g1``,
            ``acc_g2``).
        name : str
            Human-readable name for the accumulator.

        Returns
        -------
        ModelBuilder
            Self for method chaining.
        """
        self._accumulators.append(Accumulator(id=id, name=name))
        logger.info(f"Added accumulator: id='{id}', name='{name}'")
        return self

    def add_stratification(
        self,
        id: str,
        categories: list[str],
        description: str | None = None,
        conditions: list[StratificationCondition] | None = None,
    ) -> Self:
        """
        Add a stratification to the model.

        Parameters
        ----------
        id : str
            Unique identifier for the stratification.
        categories : list[str]
            list of category identifiers within this stratification.
        description : str | None, default=None
            Human-readable description of the stratification.
        conditions : list[StratificationCondition] | None, default=None
            When set, this stratification only expands compartments whose
            already-applied categories satisfy ALL conditions. Each condition
            names a stratification declared before this one and its category.

            Example: only apply the `sub` stratification to `group1`::

                conditions = [
                    StratificationCondition(stratification="group", category="group1")
                ]

        Returns
        -------
        ModelBuilder
            Self for method chaining.

        Raises
        ------
        ValueError
            If any condition references a stratification not yet declared.
        """
        if conditions:
            declared_ids = {s.id for s in self._stratifications}
            for cond in conditions:
                if cond.stratification not in declared_ids:
                    raise ValueError(
                        f"Stratification '{id}': condition references "
                        f"'{cond.stratification}' which has not been declared yet. "
                        "Conditions may only reference stratifications declared before "
                        "this one."
                    )

        self._stratifications.append(
            Stratification(
                id=id,
                categories=categories,
                description=description,
                conditions=list(conditions) if conditions else None,
            )
        )
        logger.info(f"Added stratification: id='{id}', categories={categories}")
        return self

    def add_matrix(
        self,
        id: str,
        stratification: str,
        values: Mapping[str, Mapping[str, float]],
        unit: str | None = None,
        description: str | None = None,
    ) -> Self:
        """
        Add a matrix between the categories of a stratification.

        A rate reads it through ``matrix_sum(id, e)``: in the rate of a
        compartment of category ``i``, ``Σ_j values[i][j] · e_j``, where
        ``e_j`` is ``e`` with each bin read as its amount in category ``j``
        and ``total`` as the total of ``j``. A rate or a formula parameter
        also reads its entries as ``<id>_<i>_<j>`` and its spectral radius, the largest
        modulus of its eigenvalues, as ``<id>_radius``.

        Parameters
        ----------
        id : str
            Unique identifier for the matrix.
        stratification : str
            Id of the stratification whose categories are its rows and its
            columns.
        values : Mapping[str, Mapping[str, float]]
            ``values[i][j]``, the value of row ``i`` and column ``j``: one
            finite number for each pair of categories of the stratification.
        unit : str | None, default=None
            Unit of its values, which the unit check reads.
        description : str | None, default=None
            Human-readable description of the matrix.

        Returns
        -------
        ModelBuilder
            Self for method chaining.

        Notes
        -----
        `build` checks the matrix against the process: it raises
        ``ValueError`` when the stratification is not one of the process, a
        row or a value of a category is missing, a row or a column names no
        category, a value is not finite, or the id or a variable of the
        matrix has the name of another id or variable of the model.
        """
        self._matrices.append(
            CategoryMatrix(
                id=id,
                stratification=stratification,
                values={row: dict(columns) for row, columns in values.items()},
                unit=unit,
                description=description,
            )
        )
        logger.info(f"Added matrix: id='{id}', stratification='{stratification}'")
        return self

    def add_parameter(
        self,
        id: str,
        value: float | str | list[tuple[float, float]] | None,
        description: str | None = None,
        unit: str | None = None,
        mode: str | None = None,
    ) -> Self:
        """
        Add a global parameter to the model.

        Parameters
        ----------
        id : str
            Unique identifier for the parameter.
        value : float | str | list[tuple[float, float]] | None
            Value of the parameter. Can be:
            - float: A numerical constant value
            - str: A mathematical formula that can reference other parameters,
                   special variables (total, total_<category>, step, t, pi, e),
                   or contain mathematical expressions (e.g., "k1 * 2",
                   "total_g1 / total")
            - list[tuple[float, float]]: Empirical series as (position, value)
                   points; requires ``mode`` to be set.
            - None: Indicates that the parameter needs to be calibrated before use

        description : str | None, default=None
            Human-readable description of the parameter.
        unit : str | None, default=None
            Unit of the parameter (e.g., "1/day", "dimensionless", "item").
            Used for unit consistency checking in equations.
        mode : str | None, default=None
            Required when ``value`` is a list of (position, value) points.
            Selects how the series is read: ``"pulse"`` (the value on the step
            of a point, 0 on every other step; integer positions only),
            ``"step_function"`` (the value of the last point at or before
            ``t``), or ``"linear"`` (linear interpolation in ``t``).

        Returns
        -------
        ModelBuilder
            Self for method chaining.

        Raises
        ------
        ValueError
            If ``value`` is a list but ``mode`` is not provided, if
            ``mode`` is provided without a list ``value``, or if the parser
            of the engine rejects a formula ``value``, with its message.
        """
        if isinstance(value, list):
            if mode is None:
                raise ValueError(
                    f"Parameter '{id}': 'mode' is required when 'value' is a "
                    "series list. Use mode='pulse', 'step_function', or 'linear'."
                )
            param_value: float | str | SeriesValue | None = SeriesValue(
                data=value, mode=mode
            )
        else:
            if mode is not None:
                raise ValueError(
                    f"Parameter '{id}': 'mode' is only valid when 'value' is a "
                    "series list."
                )
            param_value = value
        self._ensure_parameter_id_available(id)
        self._parameters.append(
            Parameter(id=id, value=param_value, description=description, unit=unit)
        )
        logger.info(f"Added parameter: id='{id}', value={value}, unit='{unit}'")
        return self

    def _ensure_parameter_id_available(self, id: str) -> None:
        if any(parameter.id == id for parameter in self._parameters):
            raise ValueError(f"Parameter id '{id}' already exists")

    def _register_series_pattern(
        self,
        pattern: AxisPattern,
        *,
        transition_id: str,
        source_bin: str,
        target_bin: str,
    ) -> AxisPattern:
        """Auto-register a series parameter for pulse-type patterns.

        If ``pattern`` has numeric pulse data, registers a ``"pulse"`` series
        parameter and returns a ``_ComputedAxisPattern`` whose formula is the
        generated parameter name. Otherwise returns ``pattern`` unchanged.
        """
        data = pattern._as_series_data()
        if data is None:
            return pattern
        conditions = pattern.conditions
        if transition_id and conditions:
            src_cats = "_".join(c.category for c in conditions if c.category)
            tgt_cats = "_".join(
                target for c in conditions if (target := c.to or c.category)
            )
            src = f"{source_bin}_{src_cats}" if source_bin else src_cats
            tgt = f"{target_bin}_{tgt_cats}" if target_bin else tgt_cats
            param_name = f"{transition_id}_{src}_to_{tgt}"
        else:
            param_name = f"_series_{self._series_counter}"
            self._series_counter += 1
        self._ensure_parameter_id_available(param_name)
        self._parameters.append(
            Parameter(
                id=param_name,
                value=SeriesValue(data=data, mode="pulse"),
            )
        )
        return _ComputedAxisPattern(
            computed_formula=param_name,
            conditions=conditions,
            source_compartment=pattern.source_compartment,
        )

    def add_transition(
        self,
        id: str,
        source: list[str],
        target: list[str],
        rate: "str | float | AxisPattern | None" = None,
        stratified_rates: list[StratifiedRate] | None = None,
        per_compartment: bool | None = None,
        accumulators: list[str] | None = None,
        impulse_at: list[float] | None = None,
        impulse_every: tuple[float, float] | None = None,
    ) -> Self:
        """
        Add a transition between states to the model.

        This method supports two distinct behaviors:

        When you specify multiple sources without the $compartment placeholder,
        a single transition is created that affects all sources simultaneously.
        The rate is evaluated once, at each position the engine reads it, and
        that value applies to all source compartments at once. What the value
        means (an amount moved in one step or an instantaneous rate) is the
        engine's; see "What a rate means" in the Core Concepts guide.

        When you use the $compartment placeholder in the rate formula with multiple
        sources, the system automatically expands this into multiple independent
        transitions - one for each source compartment. Each transition has its own
        rate calculation where $compartment is replaced with the actual compartment
        name.

        Each transition's rate is evaluated independently, giving per-compartment
        dynamics. This suits an outflow such as ``k1 * $compartment``, where the
        flow of each compartment is proportional to its own amount.

        **$compartment Usage Rules**:
        - Only valid with several sources (and one or no target) or several
          targets (and one or no source)
        - Can be used with stratified_rates - the placeholder will be expanded in
          both the base rate and all stratified rate expressions
        - The placeholder $compartment will be replaced with each source compartment
          name

        **$compartment Restrictions**:
        - Using $compartment with a single source will raise an error (use the
          compartment name directly instead)
        - Using $compartment with multiple targets will raise an error (ambiguous
          which target corresponds to which source)

        Parameters
        ----------
        id : str
            Unique identifier for the transition. When using $compartment expansion,
            the system will append "__<compartment_name>" to create unique IDs for
            each expanded transition (e.g., "out__A", "out__B").
        source : list[str]
            List of source state/category identifiers. When using $compartment in
            the rate, each source will generate a separate transition.
        target : list[str]
            List of target state/category identifiers. When using $compartment
            with several targets, each target generates a separate transition.
        rate : str | float | AxisPattern | None, default=None
            Default mathematical formula, parameter reference, or constant value for
            the transition rate. Used when no stratified rate matches.
            Can be:
            - A parameter reference
            - A constant value
            - A mathematical formula
            - A formula with $compartment placeholder

            Special variables available in formulas:
            - total: The sum of every compartment, accumulators excluded
            - total_<category>…: The sum of the compartments with these
              categories, in the declaration order of the stratifications
            - t: The current position on the axis
            - step: The step that holds it, ``floor(t)``; in a
              DifferenceEquations model ``t`` and ``step`` are equal
            - pi, e: Mathematical constants
            - $compartment: Expands to each source compartment name (only with
              multiple sources)

        stratified_rates : list[StratifiedRate] | None, default=None
            The rates of the compartments that match some categories. A
            compartment that no stratified rate matches takes ``rate``.

        per_compartment : bool | None, default=None
            When True, each id of a source or target bin in a rate reads the
            compartment of the flow (``A_g1``) instead of the total of the bin.
            Any other bin keeps its total: in a transition from ``A`` to ``B``,
            ``k * A * (C + D) / total`` reads ``A_g1`` and the totals of ``C``
            and ``D``. It applies to the fallback rate, to every stratified
            rate, and to every transition that ``$compartment`` expands to,
            where ``$compartment`` reads the compartment that flows. A
            transition between categories ignores it.

        accumulators : list[str] | None, default=None
            Ids of accumulators that add up the flow of this transition.

        impulse_at : list[float] | None, default=None
            Make the transition an impulse at these positions on the axis. At
            each one, ``rate`` is the fraction of the source (proportional) or the
            amount (absolute) moved at once, before the flows of that step. A
            DifferenceEquations or StochasticSteps model needs integer
            positions.

        impulse_every : tuple[float, float] | None, default=None
            ``(period, offset)``: make the transition an impulse at ``offset``,
            ``offset + period``, ``offset + 2 * period``, ...

        Returns
        -------
        ModelBuilder
            Self for method chaining.

        Raises
        ------
        ValueError
            - If $compartment is used with only one source compartment
            - If $compartment is used with multiple targets
            - If rate is None when using $compartment
        """
        # If `rate` is a AxisPattern, extract its rate string and stratified
        # rates and route them into the existing pipeline. A AxisPattern fully
        # specifies how the rate changes along the axis, so passing `stratified_rates=`
        # alongside it is a conflict.
        if isinstance(rate, AxisPattern):
            if impulse_at is not None or impulse_every is not None:
                raise ValueError(
                    f"Transition '{id}': an impulse moves its rate at once; pass "
                    "a fraction or an amount as `rate=`, not a AxisPattern."
                )
            if stratified_rates is not None:
                raise ValueError(
                    f"Transition '{id}': pass either a AxisPattern as `rate=` "
                    "or `stratified_rates=`, not both."
                )
            src_bin = source[0] if source else ""
            tgt_bin = target[0] if target else ""
            if isinstance(rate, _ScheduleAxisPattern):
                resolved = rate.model_copy(
                    update={
                        "patterns": [
                            _GroupEntry(
                                pattern=self._register_series_pattern(
                                    entry.pattern,
                                    transition_id=id,
                                    source_bin=src_bin,
                                    target_bin=tgt_bin,
                                ),
                                absolute=entry.absolute,
                            )
                            for entry in rate.patterns
                        ]
                    }
                )
                rate = resolved
            else:
                rate = self._register_series_pattern(
                    rate, transition_id=id, source_bin=src_bin, target_bin=tgt_bin
                )
            stratified_rates = rate._builder_stratified_rates()
            rate = rate._builder_rate()

        # Convert rate to string if numeric
        if isinstance(rate, int) or isinstance(rate, float):
            rate = str(rate)

        # Check for $compartment placeholder and handle expansion
        if rate and "$compartment" in rate:
            self._validate_compartment_placeholder(id, source, target)
            self._expand_compartment_transition(
                id,
                source,
                target,
                rate,
                stratified_rates,
                per_compartment,
                accumulators,
                impulse_at,
                impulse_every,
            )
            return self

        stratified_rates_objects = _stratified_rate_objects(stratified_rates)
        self._transitions.append(
            Transition(
                id=id,
                source=source,
                target=target,
                accumulators=accumulators or [],
                rate=rate,
                stratified_rates=stratified_rates_objects,
                per_compartment=per_compartment,
                impulse_at=impulse_at,
                impulse_every=impulse_every,
            )
        )
        logger.info(
            (
                f"Added transition: id='{id}', source={source}, target={target}, "
                f"rate='{rate}', stratified_rates={
                    len(stratified_rates_objects) if stratified_rates_objects else 0
                }"
            )
        )
        return self

    def _validate_compartment_placeholder(
        self,
        id: str,
        source: list[str],
        target: list[str],
    ) -> None:
        """
        Validate that $compartment placeholder is used correctly.

        Parameters
        ----------
        id : str
            Transition identifier (for error messages)
        source : list[str]
            Source compartments
        target : list[str]
            Target compartments
        rate : str
            Rate formula
        stratified_rates : list[StratifiedRate] | None
            Stratified rates

        Raises
        ------
        ValueError
            If $compartment usage violates any rules
        """
        # Check that we have either multiple sources OR multiple targets, but not both
        has_multiple_sources = len(source) > 1
        has_multiple_targets = len(target) > 1

        if not has_multiple_sources and not has_multiple_targets:
            raise ValueError(
                (
                    f"Transition '{id}': $compartment placeholder requires either "
                    f"multiple source compartments or multiple target compartments "
                    f"(found {len(source)} source(s) and {len(target)} target(s)). "
                    f"If you have a single source/target, use the compartment name "
                    f"directly in the rate formula instead of $compartment."
                )
            )

        if has_multiple_sources and has_multiple_targets:
            raise ValueError(
                (
                    f"Transition '{id}': $compartment placeholder cannot be used "
                    f"with both multiple sources and multiple targets "
                    f"(found {len(source)} sources and {len(target)} targets). "
                    f"This would create ambiguous mappings. Use either multiple "
                    f"sources with single/empty target, or single/empty source with "
                    f"multiple targets."
                )
            )

    def _expand_compartment_transition(
        self,
        id: str,
        source: list[str],
        target: list[str],
        rate: str,
        stratified_rates: list[StratifiedRate] | None,
        per_compartment: bool | None,
        accumulators: list[str] | None,
        impulse_at: list[float] | None,
        impulse_every: tuple[float, float] | None,
    ) -> None:
        """
        Expand a transition with $compartment placeholder into multiple transitions.

        Parameters
        ----------
        id : str
            Base transition identifier
        source : list[str]
            Source compartments
            (will create one transition per source if multiple sources)
        target : list[str]
            Target compartments
            (will create one transition per target if multiple targets)
        rate : str
            Rate formula containing $compartment
        stratified_rates : list[StratifiedRate] | None
            Stratified rates (can contain $compartment which will be expanded)
        per_compartment, accumulators, impulse_at, impulse_every : optional
            The arguments of `add_transition` that every expanded transition
            keeps
        """
        # Determine if we're expanding over sources or targets
        has_multiple_sources = len(source) > 1

        if has_multiple_sources:
            compartments = source
            expand_type = "source"
        else:
            compartments = target
            expand_type = "target"

        logger.info(
            (
                f"Expanding transition '{id}' with $compartment placeholder: "
                f"{len(compartments)} {expand_type} compartments will create "
                f"{len(compartments)} separate transitions"
            )
        )

        for compartment in compartments:
            # Replace $compartment with actual compartment name in base rate
            expanded_rate = rate.replace("$compartment", compartment)

            expanded_stratified_rates = _stratified_rate_objects(
                stratified_rates, compartment
            )

            # Generate unique ID for this expanded transition
            expanded_id = f"{id}__{compartment}"

            # add_transition accepts several sources or several targets.
            if has_multiple_sources:
                expanded_source = [compartment]
                expanded_target = target
            else:
                expanded_source = source
                expanded_target = [compartment]

            self._transitions.append(
                Transition(
                    id=expanded_id,
                    source=expanded_source,
                    target=expanded_target,
                    accumulators=accumulators or [],
                    rate=expanded_rate,
                    stratified_rates=expanded_stratified_rates,
                    per_compartment=per_compartment,
                    impulse_at=impulse_at,
                    impulse_every=impulse_every,
                )
            )

            strat_info = (
                f", {len(expanded_stratified_rates)} stratified rates"
                if expanded_stratified_rates
                else ""
            )
            logger.info(
                (
                    f"  Created expanded transition: id='{expanded_id}', "
                    f"source=['{compartment}'], target={target}, "
                    f"rate='{expanded_rate}'{strat_info}"
                )
            )

        logger.info(
            f"Successfully expanded transition '{id}' into {len(compartments)} "
            "transitions"
        )

    def get_summary(self) -> dict[str, str | int | list[str] | None]:
        """
        Get a summary of the current model builder state.

        Returns
        -------
        dict[str, dict[str, str | int | list[str] | None]]
            Dictionary containing summary information about the model being built.
        """
        return {
            "name": self._name,
            "description": self._description,
            "version": self._version,
            "bins_count": len(self._bins),
            "bin_ids": [state.id for state in self._bins],
            "stratifications_count": len(self._stratifications),
            "stratification_ids": [strat.id for strat in self._stratifications],
            "matrices_count": len(self._matrices),
            "matrix_ids": [matrix.id for matrix in self._matrices],
            "transitions_count": len(self._transitions),
            "transition_ids": [trans.id for trans in self._transitions],
            "parameters_count": len(self._parameters),
            "parameter_ids": [param.id for param in self._parameters],
        }

    def clone(self) -> Self:
        """
        Create a deep copy of this ModelBuilder.

        Returns
        -------
        ModelBuilder
            A new ModelBuilder instance with the same configuration: every
            bin, accumulator, stratification, matrix, transition and
            parameter.
            Changes to one builder do not affect the other.
        """
        return copy.deepcopy(self)

    def reset(self) -> Self:
        """
        Reset the builder to empty state.

        The name, description, version, ``bin_unit``, ``step_unit`` and
        ``origin`` are kept. Every bin,
        accumulator, stratification, matrix, transition and parameter is
        removed.

        Returns
        -------
        ModelBuilder
            Self for method chaining.
        """
        self._bins.clear()
        self._accumulators.clear()
        self._stratifications.clear()
        self._matrices.clear()
        self._transitions.clear()
        self._parameters.clear()
        self._series_counter = 0
        return self

    def process(self) -> Process:
        """
        The process the builder has written so far.

        Returns
        -------
        Process
            The bins, accumulators, stratifications, matrices and
            transitions.

        Raises
        ------
        ValueError
            If ids repeat, or a transition or a stratified rate names an id
            the process does not have.
        """
        return Process(
            bins=self._bins,
            accumulators=self._accumulators,
            stratifications=self._stratifications,
            matrices=self._matrices,
            transitions=self._transitions,
        )

    def build(
        self,
        structure: StructureT,
        engine: EngineName,
        settings: EngineSettings | None = None,
    ) -> Model[StructureT]:
        """
        Join the process with a structure and an engine into a model.

        The process is written once; the structure lays it out and sets the
        kind of the model, so one builder builds every kind its process fits:

        ``builder.build(Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}),
        engine="DifferentialEquations")``

        Parameters
        ----------
        structure : ModelStructure
            How the process is laid out, such as `Aggregate` (amounts per
            compartment). The model is a ``Model`` of its class.
        engine : EngineName
            The engine that runs the model, which must run the kind of the
            structure: ``"DifferenceEquations"`` (discrete time),
            ``"DifferentialEquations"`` (continuous time), ``"StochasticEvents"``
            or ``"StochasticSteps"`` (at random).
        settings : EngineSettings | None, default=None
            Settings of that engine; None uses its defaults: `ODESolver` for
            ``"DifferentialEquations"``, `StochasticEventsSettings` and
            `StochasticStepsSettings` for the stochastic engines.

        Returns
        -------
        Model[StructureT]
            The model, of the class of ``structure``.

        Raises
        ------
        ValueError
            If validation fails, the structure does not fit the process (for
            example a fraction names no bin, or a rate reads ``neighbors_<bin>``
            and the structure has no ``neighbors``), the engine is unknown or
            does not run the kind of the structure, the settings are not
            settings of that engine, a series point or an impulse position is
            one the engine cannot run (a position between two steps for an
            engine that steps only at integers, or a ``"pulse"`` point between
            two steps), a per-compartment override names an unknown
            compartment, a stochastic engine gets an amount that is not whole,
            or the origin has a time zone or a step unit that is not of fixed
            calendar length.
        """
        model: Model[StructureT] = Model(
            name=self._name,
            description=self._description,
            version=self._version,
            parameters=self._parameters,
            engine=ModelEngine(type=engine, settings=settings),
            process=self.process(),
            structure=structure,
            step_unit=self._step_unit,
            origin=self._origin,
        )

        logger.info(f"Model '{self._name}' successfully built for the {engine} engine.")

        return model
```

### Methods:

#### __init__

```python
__init__(name: str, description: str | None = None, version: str | None = None, bin_unit: str | None = None, step_unit: str | None = None, origin: date | None = None) -> None
```

Initialize the ModelBuilder.

Parameters:

| Name          | Type   | Description                                | Default                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------- | ------ | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`        | `str`  | The unique name that identifies the model. | *required*                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `description` | \`str  | None\`                                     | A human-readable description of the model's purpose and function.                                                                                                                                                                                                                                                                                                                                                                        |
| `version`     | \`str  | None\`                                     | The version number of the model.                                                                                                                                                                                                                                                                                                                                                                                                         |
| `bin_unit`    | \`str  | None\`                                     | The default unit for all bins. Individual bins can override this with their own unit parameter. Units are optional but required for model.check_unit_consistency().                                                                                                                                                                                                                                                                      |
| `step_unit`   | \`str  | None\`                                     | Unit of one step on the axis, of any single dimension (for example "day" or "meter"). The unit check requires every flow to be <bin unit>/\<step_unit>, and Model.axis_units() converts values to steps of it. None means step, the default increment of any axis.                                                                                                                                                                       |
| `origin`      | \`date | None\`                                     | Calendar moment of position 0: a datetime.datetime with no time zone, or a datetime.date, read as its midnight. With it, Model.axis_units() converts between dates and positions, and results.plot.series() labels its ticks with dates. It needs a step_unit of time whose length is a whole number of days or a whole fraction of a day, such as "hour" or "day"; build raises ValueError otherwise, or for a moment with a time zone. |

Source code in `commol/_model/builder.py`

```python
def __init__(
    self,
    name: str,
    description: str | None = None,
    version: str | None = None,
    bin_unit: str | None = None,
    step_unit: str | None = None,
    origin: date | None = None,
) -> None:
    """
    Initialize the ModelBuilder.

    Parameters
    ----------
    name : str
        The unique name that identifies the model.
    description : str | None, default=None
        A human-readable description of the model's purpose and function.
    version : str | None, default=None
        The version number of the model.
    bin_unit : str | None, default=None
        The default unit for all bins.
        Individual bins can override this with their own unit parameter.
        Units are optional but required for model.check_unit_consistency().
    step_unit : str | None, default=None
        Unit of one step on the axis, of any single dimension (for example
        ``"day"`` or ``"meter"``). The unit check requires every flow to be
        ``<bin unit>/<step_unit>``, and ``Model.axis_units()`` converts
        values to steps of it. None means ``step``, the default increment
        of any axis.
    origin : date | None, default=None
        Calendar moment of position 0: a `datetime.datetime` with no time
        zone, or a `datetime.date`, read as its midnight. With it,
        ``Model.axis_units()`` converts between dates and positions, and
        ``results.plot.series()`` labels its ticks with dates. It
        needs a ``step_unit`` of time whose length is a whole number of
        days or a whole fraction of a day, such as ``"hour"`` or
        ``"day"``; `build` raises ``ValueError`` otherwise, or for a
        moment with a time zone.
    """
    self._name: str = name
    self._description: str | None = description
    self._version: str | None = version
    self._bin_unit: str | None = bin_unit
    self._step_unit: str | None = step_unit
    self._origin: date | None = origin

    self._bins: list[Bin] = []
    self._accumulators: list[Accumulator] = []
    self._stratifications: list[Stratification] = []
    self._matrices: list[CategoryMatrix] = []
    self._transitions: list[Transition] = []
    self._parameters: list[Parameter] = []
    self._series_counter: int = 0

    logger.info(
        (
            f"Initialized ModelBuilder: name='{self._name}', "
            f"version='{self._version or 'N/A'}'"
        )
    )
```

#### add_bin

```python
add_bin(id: str, name: str, unit: str | None = None) -> Self
```

Add a bin to the model.

Parameters:

| Name   | Type  | Description                      | Default                                                                   |
| ------ | ----- | -------------------------------- | ------------------------------------------------------------------------- |
| `id`   | `str` | Unique identifier for the bin.   | *required*                                                                |
| `name` | `str` | Human-readable name for the bin. | *required*                                                                |
| `unit` | \`str | None\`                           | Unit of measurement for this bin. If None, uses the model-level bin_unit. |

Returns:

| Type           | Description               |
| -------------- | ------------------------- |
| `ModelBuilder` | Self for method chaining. |

Source code in `commol/_model/builder.py`

```python
def add_bin(self, id: str, name: str, unit: str | None = None) -> Self:
    """
    Add a bin to the model.

    Parameters
    ----------
    id : str
        Unique identifier for the bin.
    name : str
        Human-readable name for the bin.
    unit : str | None, default=None
        Unit of measurement for this bin. If None, uses the model-level bin_unit.

    Returns
    -------
    ModelBuilder
        Self for method chaining.
    """
    # Use bin-specific unit, or fall back to model-level bin_unit
    final_unit = unit if unit is not None else self._bin_unit
    self._bins.append(Bin(id=id, name=name, unit=final_unit))
    logger.info(f"Added bin: id='{id}', name='{name}', unit='{final_unit}'")
    return self
```

#### add_accumulator

```python
add_accumulator(id: str, name: str) -> Self
```

Add an accumulator: a cumulative event counter in the model outputs.

A transition adds its flow to the accumulators it names (`add_transition(accumulators=...)`). An accumulator is an output, not a part of the state, so `total` leaves it out.

Parameters:

| Name   | Type  | Description                                                                                                                     | Default    |
| ------ | ----- | ------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `id`   | `str` | Unique identifier for the accumulator. It names its output, which stratifications expand as they expand a bin (acc_g1, acc_g2). | *required* |
| `name` | `str` | Human-readable name for the accumulator.                                                                                        | *required* |

Returns:

| Type           | Description               |
| -------------- | ------------------------- |
| `ModelBuilder` | Self for method chaining. |

Source code in `commol/_model/builder.py`

```python
def add_accumulator(self, id: str, name: str) -> Self:
    """
    Add an accumulator: a cumulative event counter in the model outputs.

    A transition adds its flow to the accumulators it names
    (``add_transition(accumulators=...)``). An accumulator is an output,
    not a part of the state, so ``total`` leaves it out.

    Parameters
    ----------
    id : str
        Unique identifier for the accumulator. It names its output, which
        stratifications expand as they expand a bin (``acc_g1``,
        ``acc_g2``).
    name : str
        Human-readable name for the accumulator.

    Returns
    -------
    ModelBuilder
        Self for method chaining.
    """
    self._accumulators.append(Accumulator(id=id, name=name))
    logger.info(f"Added accumulator: id='{id}', name='{name}'")
    return self
```

#### add_stratification

```python
add_stratification(id: str, categories: list[str], description: str | None = None, conditions: list[StratificationCondition] | None = None) -> Self
```

Add a stratification to the model.

Parameters:

| Name          | Type                            | Description                                              | Default                                                                                                                                                                                                                                                                                                                                      |
| ------------- | ------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`          | `str`                           | Unique identifier for the stratification.                | *required*                                                                                                                                                                                                                                                                                                                                   |
| `categories`  | `list[str]`                     | list of category identifiers within this stratification. | *required*                                                                                                                                                                                                                                                                                                                                   |
| `description` | \`str                           | None\`                                                   | Human-readable description of the stratification.                                                                                                                                                                                                                                                                                            |
| `conditions`  | \`list[StratificationCondition] | None\`                                                   | When set, this stratification only expands compartments whose already-applied categories satisfy ALL conditions. Each condition names a stratification declared before this one and its category. Example: only apply the sub stratification to group1:: conditions = [ StratificationCondition(stratification="group", category="group1") ] |

Returns:

| Type           | Description               |
| -------------- | ------------------------- |
| `ModelBuilder` | Self for method chaining. |

Raises:

| Type         | Description                                                    |
| ------------ | -------------------------------------------------------------- |
| `ValueError` | If any condition references a stratification not yet declared. |

Source code in `commol/_model/builder.py`

```python
def add_stratification(
    self,
    id: str,
    categories: list[str],
    description: str | None = None,
    conditions: list[StratificationCondition] | None = None,
) -> Self:
    """
    Add a stratification to the model.

    Parameters
    ----------
    id : str
        Unique identifier for the stratification.
    categories : list[str]
        list of category identifiers within this stratification.
    description : str | None, default=None
        Human-readable description of the stratification.
    conditions : list[StratificationCondition] | None, default=None
        When set, this stratification only expands compartments whose
        already-applied categories satisfy ALL conditions. Each condition
        names a stratification declared before this one and its category.

        Example: only apply the `sub` stratification to `group1`::

            conditions = [
                StratificationCondition(stratification="group", category="group1")
            ]

    Returns
    -------
    ModelBuilder
        Self for method chaining.

    Raises
    ------
    ValueError
        If any condition references a stratification not yet declared.
    """
    if conditions:
        declared_ids = {s.id for s in self._stratifications}
        for cond in conditions:
            if cond.stratification not in declared_ids:
                raise ValueError(
                    f"Stratification '{id}': condition references "
                    f"'{cond.stratification}' which has not been declared yet. "
                    "Conditions may only reference stratifications declared before "
                    "this one."
                )

    self._stratifications.append(
        Stratification(
            id=id,
            categories=categories,
            description=description,
            conditions=list(conditions) if conditions else None,
        )
    )
    logger.info(f"Added stratification: id='{id}', categories={categories}")
    return self
```

#### add_matrix

```python
add_matrix(id: str, stratification: str, values: Mapping[str, Mapping[str, float]], unit: str | None = None, description: str | None = None) -> Self
```

Add a matrix between the categories of a stratification.

A rate reads it through `matrix_sum(id, e)`: in the rate of a compartment of category `i`, `Σ_j values[i][j] · e_j`, where `e_j` is `e` with each bin read as its amount in category `j` and `total` as the total of `j`. A rate or a formula parameter also reads its entries as `<id>_<i>_<j>` and its spectral radius, the largest modulus of its eigenvalues, as `<id>_radius`.

Parameters:

| Name             | Type                                | Description                                                                                                         | Default                                         |
| ---------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `id`             | `str`                               | Unique identifier for the matrix.                                                                                   | *required*                                      |
| `stratification` | `str`                               | Id of the stratification whose categories are its rows and its columns.                                             | *required*                                      |
| `values`         | `Mapping[str, Mapping[str, float]]` | values[i][j], the value of row i and column j: one finite number for each pair of categories of the stratification. | *required*                                      |
| `unit`           | \`str                               | None\`                                                                                                              | Unit of its values, which the unit check reads. |
| `description`    | \`str                               | None\`                                                                                                              | Human-readable description of the matrix.       |

Returns:

| Type           | Description               |
| -------------- | ------------------------- |
| `ModelBuilder` | Self for method chaining. |

Notes

`build` checks the matrix against the process: it raises `ValueError` when the stratification is not one of the process, a row or a value of a category is missing, a row or a column names no category, a value is not finite, or the id or a variable of the matrix has the name of another id or variable of the model.

Source code in `commol/_model/builder.py`

```python
def add_matrix(
    self,
    id: str,
    stratification: str,
    values: Mapping[str, Mapping[str, float]],
    unit: str | None = None,
    description: str | None = None,
) -> Self:
    """
    Add a matrix between the categories of a stratification.

    A rate reads it through ``matrix_sum(id, e)``: in the rate of a
    compartment of category ``i``, ``Σ_j values[i][j] · e_j``, where
    ``e_j`` is ``e`` with each bin read as its amount in category ``j``
    and ``total`` as the total of ``j``. A rate or a formula parameter
    also reads its entries as ``<id>_<i>_<j>`` and its spectral radius, the largest
    modulus of its eigenvalues, as ``<id>_radius``.

    Parameters
    ----------
    id : str
        Unique identifier for the matrix.
    stratification : str
        Id of the stratification whose categories are its rows and its
        columns.
    values : Mapping[str, Mapping[str, float]]
        ``values[i][j]``, the value of row ``i`` and column ``j``: one
        finite number for each pair of categories of the stratification.
    unit : str | None, default=None
        Unit of its values, which the unit check reads.
    description : str | None, default=None
        Human-readable description of the matrix.

    Returns
    -------
    ModelBuilder
        Self for method chaining.

    Notes
    -----
    `build` checks the matrix against the process: it raises
    ``ValueError`` when the stratification is not one of the process, a
    row or a value of a category is missing, a row or a column names no
    category, a value is not finite, or the id or a variable of the
    matrix has the name of another id or variable of the model.
    """
    self._matrices.append(
        CategoryMatrix(
            id=id,
            stratification=stratification,
            values={row: dict(columns) for row, columns in values.items()},
            unit=unit,
            description=description,
        )
    )
    logger.info(f"Added matrix: id='{id}', stratification='{stratification}'")
    return self
```

#### add_parameter

```python
add_parameter(id: str, value: float | str | list[tuple[float, float]] | None, description: str | None = None, unit: str | None = None, mode: str | None = None) -> Self
```

Add a global parameter to the model.

Parameters:

| Name          | Type    | Description                          | Default                                                                                                                                                                                                                                                                                      |
| ------------- | ------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`          | `str`   | Unique identifier for the parameter. | *required*                                                                                                                                                                                                                                                                                   |
| `value`       | \`float | str                                  | list\[tuple[float, float]\]                                                                                                                                                                                                                                                                  |
| `description` | \`str   | None\`                               | Human-readable description of the parameter.                                                                                                                                                                                                                                                 |
| `unit`        | \`str   | None\`                               | Unit of the parameter (e.g., "1/day", "dimensionless", "item"). Used for unit consistency checking in equations.                                                                                                                                                                             |
| `mode`        | \`str   | None\`                               | Required when value is a list of (position, value) points. Selects how the series is read: "pulse" (the value on the step of a point, 0 on every other step; integer positions only), "step_function" (the value of the last point at or before t), or "linear" (linear interpolation in t). |

Returns:

| Type           | Description               |
| -------------- | ------------------------- |
| `ModelBuilder` | Self for method chaining. |

Raises:

| Type         | Description                                                                                                                                                      |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If value is a list but mode is not provided, if mode is provided without a list value, or if the parser of the engine rejects a formula value, with its message. |

Source code in `commol/_model/builder.py`

```python
def add_parameter(
    self,
    id: str,
    value: float | str | list[tuple[float, float]] | None,
    description: str | None = None,
    unit: str | None = None,
    mode: str | None = None,
) -> Self:
    """
    Add a global parameter to the model.

    Parameters
    ----------
    id : str
        Unique identifier for the parameter.
    value : float | str | list[tuple[float, float]] | None
        Value of the parameter. Can be:
        - float: A numerical constant value
        - str: A mathematical formula that can reference other parameters,
               special variables (total, total_<category>, step, t, pi, e),
               or contain mathematical expressions (e.g., "k1 * 2",
               "total_g1 / total")
        - list[tuple[float, float]]: Empirical series as (position, value)
               points; requires ``mode`` to be set.
        - None: Indicates that the parameter needs to be calibrated before use

    description : str | None, default=None
        Human-readable description of the parameter.
    unit : str | None, default=None
        Unit of the parameter (e.g., "1/day", "dimensionless", "item").
        Used for unit consistency checking in equations.
    mode : str | None, default=None
        Required when ``value`` is a list of (position, value) points.
        Selects how the series is read: ``"pulse"`` (the value on the step
        of a point, 0 on every other step; integer positions only),
        ``"step_function"`` (the value of the last point at or before
        ``t``), or ``"linear"`` (linear interpolation in ``t``).

    Returns
    -------
    ModelBuilder
        Self for method chaining.

    Raises
    ------
    ValueError
        If ``value`` is a list but ``mode`` is not provided, if
        ``mode`` is provided without a list ``value``, or if the parser
        of the engine rejects a formula ``value``, with its message.
    """
    if isinstance(value, list):
        if mode is None:
            raise ValueError(
                f"Parameter '{id}': 'mode' is required when 'value' is a "
                "series list. Use mode='pulse', 'step_function', or 'linear'."
            )
        param_value: float | str | SeriesValue | None = SeriesValue(
            data=value, mode=mode
        )
    else:
        if mode is not None:
            raise ValueError(
                f"Parameter '{id}': 'mode' is only valid when 'value' is a "
                "series list."
            )
        param_value = value
    self._ensure_parameter_id_available(id)
    self._parameters.append(
        Parameter(id=id, value=param_value, description=description, unit=unit)
    )
    logger.info(f"Added parameter: id='{id}', value={value}, unit='{unit}'")
    return self
```

#### add_transition

```python
add_transition(id: str, source: list[str], target: list[str], rate: str | float | AxisPattern | None = None, stratified_rates: list[StratifiedRate] | None = None, per_compartment: bool | None = None, accumulators: list[str] | None = None, impulse_at: list[float] | None = None, impulse_every: tuple[float, float] | None = None) -> Self
```

Add a transition between states to the model.

This method supports two distinct behaviors:

When you specify multiple sources without the $compartment placeholder, a single transition is created that affects all sources simultaneously. The rate is evaluated once, at each position the engine reads it, and that value applies to all source compartments at once. What the value means (an amount moved in one step or an instantaneous rate) is the engine's; see "What a rate means" in the Core Concepts guide.

When you use the $compartment placeholder in the rate formula with multiple sources, the system automatically expands this into multiple independent transitions - one for each source compartment. Each transition has its own rate calculation where $compartment is replaced with the actual compartment name.

Each transition's rate is evaluated independently, giving per-compartment dynamics. This suits an outflow such as `k1 * $compartment`, where the flow of each compartment is proportional to its own amount.

**$compartment Usage Rules**:

- Only valid with several sources (and one or no target) or several targets (and one or no source)
- Can be used with stratified_rates - the placeholder will be expanded in both the base rate and all stratified rate expressions
- The placeholder $compartment will be replaced with each source compartment name

**$compartment Restrictions**:

- Using $compartment with a single source will raise an error (use the compartment name directly instead)
- Using $compartment with multiple targets will raise an error (ambiguous which target corresponds to which source)

Parameters:

| Name               | Type                   | Description                                                                                                                                                                              | Default                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------ | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`               | `str`                  | Unique identifier for the transition. When using $compartment expansion, the system will append "\_\_" to create unique IDs for each expanded transition (e.g., "out\_\_A", "out\_\_B"). | *required*                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `source`           | `list[str]`            | List of source state/category identifiers. When using $compartment in the rate, each source will generate a separate transition.                                                         | *required*                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `target`           | `list[str]`            | List of target state/category identifiers. When using $compartment with several targets, each target generates a separate transition.                                                    | *required*                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `rate`             | \`str                  | float                                                                                                                                                                                    | AxisPattern                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `stratified_rates` | \`list[StratifiedRate] | None\`                                                                                                                                                                                   | The rates of the compartments that match some categories. A compartment that no stratified rate matches takes rate.                                                                                                                                                                                                                                                                                                                                                          |
| `per_compartment`  | \`bool                 | None\`                                                                                                                                                                                   | When True, each id of a source or target bin in a rate reads the compartment of the flow (A_g1) instead of the total of the bin. Any other bin keeps its total: in a transition from A to B, k * A * (C + D) / total reads A_g1 and the totals of C and D. It applies to the fallback rate, to every stratified rate, and to every transition that $compartment expands to, where $compartment reads the compartment that flows. A transition between categories ignores it. |
| `accumulators`     | \`list[str]            | None\`                                                                                                                                                                                   | Ids of accumulators that add up the flow of this transition.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `impulse_at`       | \`list[float]          | None\`                                                                                                                                                                                   | Make the transition an impulse at these positions on the axis. At each one, rate is the fraction of the source (proportional) or the amount (absolute) moved at once, before the flows of that step. A DifferenceEquations or StochasticSteps model needs integer positions.                                                                                                                                                                                                 |
| `impulse_every`    | \`tuple[float, float]  | None\`                                                                                                                                                                                   | (period, offset): make the transition an impulse at offset, offset + period, offset + 2 * period, ...                                                                                                                                                                                                                                                                                                                                                                        |

Returns:

| Type           | Description               |
| -------------- | ------------------------- |
| `ModelBuilder` | Self for method chaining. |

Raises:

| Type         | Description                                                                                                                                    |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If $compartment is used with only one source compartment If $compartment is used with multiple targets If rate is None when using $compartment |

Source code in `commol/_model/builder.py`

```python
def add_transition(
    self,
    id: str,
    source: list[str],
    target: list[str],
    rate: "str | float | AxisPattern | None" = None,
    stratified_rates: list[StratifiedRate] | None = None,
    per_compartment: bool | None = None,
    accumulators: list[str] | None = None,
    impulse_at: list[float] | None = None,
    impulse_every: tuple[float, float] | None = None,
) -> Self:
    """
    Add a transition between states to the model.

    This method supports two distinct behaviors:

    When you specify multiple sources without the $compartment placeholder,
    a single transition is created that affects all sources simultaneously.
    The rate is evaluated once, at each position the engine reads it, and
    that value applies to all source compartments at once. What the value
    means (an amount moved in one step or an instantaneous rate) is the
    engine's; see "What a rate means" in the Core Concepts guide.

    When you use the $compartment placeholder in the rate formula with multiple
    sources, the system automatically expands this into multiple independent
    transitions - one for each source compartment. Each transition has its own
    rate calculation where $compartment is replaced with the actual compartment
    name.

    Each transition's rate is evaluated independently, giving per-compartment
    dynamics. This suits an outflow such as ``k1 * $compartment``, where the
    flow of each compartment is proportional to its own amount.

    **$compartment Usage Rules**:
    - Only valid with several sources (and one or no target) or several
      targets (and one or no source)
    - Can be used with stratified_rates - the placeholder will be expanded in
      both the base rate and all stratified rate expressions
    - The placeholder $compartment will be replaced with each source compartment
      name

    **$compartment Restrictions**:
    - Using $compartment with a single source will raise an error (use the
      compartment name directly instead)
    - Using $compartment with multiple targets will raise an error (ambiguous
      which target corresponds to which source)

    Parameters
    ----------
    id : str
        Unique identifier for the transition. When using $compartment expansion,
        the system will append "__<compartment_name>" to create unique IDs for
        each expanded transition (e.g., "out__A", "out__B").
    source : list[str]
        List of source state/category identifiers. When using $compartment in
        the rate, each source will generate a separate transition.
    target : list[str]
        List of target state/category identifiers. When using $compartment
        with several targets, each target generates a separate transition.
    rate : str | float | AxisPattern | None, default=None
        Default mathematical formula, parameter reference, or constant value for
        the transition rate. Used when no stratified rate matches.
        Can be:
        - A parameter reference
        - A constant value
        - A mathematical formula
        - A formula with $compartment placeholder

        Special variables available in formulas:
        - total: The sum of every compartment, accumulators excluded
        - total_<category>…: The sum of the compartments with these
          categories, in the declaration order of the stratifications
        - t: The current position on the axis
        - step: The step that holds it, ``floor(t)``; in a
          DifferenceEquations model ``t`` and ``step`` are equal
        - pi, e: Mathematical constants
        - $compartment: Expands to each source compartment name (only with
          multiple sources)

    stratified_rates : list[StratifiedRate] | None, default=None
        The rates of the compartments that match some categories. A
        compartment that no stratified rate matches takes ``rate``.

    per_compartment : bool | None, default=None
        When True, each id of a source or target bin in a rate reads the
        compartment of the flow (``A_g1``) instead of the total of the bin.
        Any other bin keeps its total: in a transition from ``A`` to ``B``,
        ``k * A * (C + D) / total`` reads ``A_g1`` and the totals of ``C``
        and ``D``. It applies to the fallback rate, to every stratified
        rate, and to every transition that ``$compartment`` expands to,
        where ``$compartment`` reads the compartment that flows. A
        transition between categories ignores it.

    accumulators : list[str] | None, default=None
        Ids of accumulators that add up the flow of this transition.

    impulse_at : list[float] | None, default=None
        Make the transition an impulse at these positions on the axis. At
        each one, ``rate`` is the fraction of the source (proportional) or the
        amount (absolute) moved at once, before the flows of that step. A
        DifferenceEquations or StochasticSteps model needs integer
        positions.

    impulse_every : tuple[float, float] | None, default=None
        ``(period, offset)``: make the transition an impulse at ``offset``,
        ``offset + period``, ``offset + 2 * period``, ...

    Returns
    -------
    ModelBuilder
        Self for method chaining.

    Raises
    ------
    ValueError
        - If $compartment is used with only one source compartment
        - If $compartment is used with multiple targets
        - If rate is None when using $compartment
    """
    # If `rate` is a AxisPattern, extract its rate string and stratified
    # rates and route them into the existing pipeline. A AxisPattern fully
    # specifies how the rate changes along the axis, so passing `stratified_rates=`
    # alongside it is a conflict.
    if isinstance(rate, AxisPattern):
        if impulse_at is not None or impulse_every is not None:
            raise ValueError(
                f"Transition '{id}': an impulse moves its rate at once; pass "
                "a fraction or an amount as `rate=`, not a AxisPattern."
            )
        if stratified_rates is not None:
            raise ValueError(
                f"Transition '{id}': pass either a AxisPattern as `rate=` "
                "or `stratified_rates=`, not both."
            )
        src_bin = source[0] if source else ""
        tgt_bin = target[0] if target else ""
        if isinstance(rate, _ScheduleAxisPattern):
            resolved = rate.model_copy(
                update={
                    "patterns": [
                        _GroupEntry(
                            pattern=self._register_series_pattern(
                                entry.pattern,
                                transition_id=id,
                                source_bin=src_bin,
                                target_bin=tgt_bin,
                            ),
                            absolute=entry.absolute,
                        )
                        for entry in rate.patterns
                    ]
                }
            )
            rate = resolved
        else:
            rate = self._register_series_pattern(
                rate, transition_id=id, source_bin=src_bin, target_bin=tgt_bin
            )
        stratified_rates = rate._builder_stratified_rates()
        rate = rate._builder_rate()

    # Convert rate to string if numeric
    if isinstance(rate, int) or isinstance(rate, float):
        rate = str(rate)

    # Check for $compartment placeholder and handle expansion
    if rate and "$compartment" in rate:
        self._validate_compartment_placeholder(id, source, target)
        self._expand_compartment_transition(
            id,
            source,
            target,
            rate,
            stratified_rates,
            per_compartment,
            accumulators,
            impulse_at,
            impulse_every,
        )
        return self

    stratified_rates_objects = _stratified_rate_objects(stratified_rates)
    self._transitions.append(
        Transition(
            id=id,
            source=source,
            target=target,
            accumulators=accumulators or [],
            rate=rate,
            stratified_rates=stratified_rates_objects,
            per_compartment=per_compartment,
            impulse_at=impulse_at,
            impulse_every=impulse_every,
        )
    )
    logger.info(
        (
            f"Added transition: id='{id}', source={source}, target={target}, "
            f"rate='{rate}', stratified_rates={
                len(stratified_rates_objects) if stratified_rates_objects else 0
            }"
        )
    )
    return self
```

#### get_summary

```python
get_summary() -> dict[str, str | int | list[str] | None]
```

Get a summary of the current model builder state.

Returns:

| Type                        | Description |
| --------------------------- | ----------- |
| \`dict\[str, dict\[str, str | int         |

Source code in `commol/_model/builder.py`

```python
def get_summary(self) -> dict[str, str | int | list[str] | None]:
    """
    Get a summary of the current model builder state.

    Returns
    -------
    dict[str, dict[str, str | int | list[str] | None]]
        Dictionary containing summary information about the model being built.
    """
    return {
        "name": self._name,
        "description": self._description,
        "version": self._version,
        "bins_count": len(self._bins),
        "bin_ids": [state.id for state in self._bins],
        "stratifications_count": len(self._stratifications),
        "stratification_ids": [strat.id for strat in self._stratifications],
        "matrices_count": len(self._matrices),
        "matrix_ids": [matrix.id for matrix in self._matrices],
        "transitions_count": len(self._transitions),
        "transition_ids": [trans.id for trans in self._transitions],
        "parameters_count": len(self._parameters),
        "parameter_ids": [param.id for param in self._parameters],
    }
```

#### clone

```python
clone() -> Self
```

Create a deep copy of this ModelBuilder.

Returns:

| Type           | Description                                                                                                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModelBuilder` | A new ModelBuilder instance with the same configuration: every bin, accumulator, stratification, matrix, transition and parameter. Changes to one builder do not affect the other. |

Source code in `commol/_model/builder.py`

```python
def clone(self) -> Self:
    """
    Create a deep copy of this ModelBuilder.

    Returns
    -------
    ModelBuilder
        A new ModelBuilder instance with the same configuration: every
        bin, accumulator, stratification, matrix, transition and
        parameter.
        Changes to one builder do not affect the other.
    """
    return copy.deepcopy(self)
```

#### reset

```python
reset() -> Self
```

Reset the builder to empty state.

The name, description, version, `bin_unit`, `step_unit` and `origin` are kept. Every bin, accumulator, stratification, matrix, transition and parameter is removed.

Returns:

| Type           | Description               |
| -------------- | ------------------------- |
| `ModelBuilder` | Self for method chaining. |

Source code in `commol/_model/builder.py`

```python
def reset(self) -> Self:
    """
    Reset the builder to empty state.

    The name, description, version, ``bin_unit``, ``step_unit`` and
    ``origin`` are kept. Every bin,
    accumulator, stratification, matrix, transition and parameter is
    removed.

    Returns
    -------
    ModelBuilder
        Self for method chaining.
    """
    self._bins.clear()
    self._accumulators.clear()
    self._stratifications.clear()
    self._matrices.clear()
    self._transitions.clear()
    self._parameters.clear()
    self._series_counter = 0
    return self
```

#### process

```python
process() -> Process
```

The process the builder has written so far.

Returns:

| Type      | Description                                                        |
| --------- | ------------------------------------------------------------------ |
| `Process` | The bins, accumulators, stratifications, matrices and transitions. |

Raises:

| Type         | Description                                                                                |
| ------------ | ------------------------------------------------------------------------------------------ |
| `ValueError` | If ids repeat, or a transition or a stratified rate names an id the process does not have. |

Source code in `commol/_model/builder.py`

```python
def process(self) -> Process:
    """
    The process the builder has written so far.

    Returns
    -------
    Process
        The bins, accumulators, stratifications, matrices and
        transitions.

    Raises
    ------
    ValueError
        If ids repeat, or a transition or a stratified rate names an id
        the process does not have.
    """
    return Process(
        bins=self._bins,
        accumulators=self._accumulators,
        stratifications=self._stratifications,
        matrices=self._matrices,
        transitions=self._transitions,
    )
```

#### build

```python
build(structure: StructureT, engine: EngineName, settings: EngineSettings | None = None) -> Model[StructureT]
```

Join the process with a structure and an engine into a model.

The process is written once; the structure lays it out and sets the kind of the model, so one builder builds every kind its process fits:

`builder.build(Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}), engine="DifferentialEquations")`

Parameters:

| Name        | Type             | Description                                                                                                                                                                                                      | Default                                                                                                                                                                  |
| ----------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `structure` | `ModelStructure` | How the process is laid out, such as Aggregate (amounts per compartment). The model is a Model of its class.                                                                                                     | *required*                                                                                                                                                               |
| `engine`    | `EngineName`     | The engine that runs the model, which must run the kind of the structure: "DifferenceEquations" (discrete time), "DifferentialEquations" (continuous time), "StochasticEvents" or "StochasticSteps" (at random). | *required*                                                                                                                                                               |
| `settings`  | \`EngineSettings | None\`                                                                                                                                                                                                           | Settings of that engine; None uses its defaults: ODESolver for "DifferentialEquations", StochasticEventsSettings and StochasticStepsSettings for the stochastic engines. |

Returns:

| Type                | Description                           |
| ------------------- | ------------------------------------- |
| `Model[StructureT]` | The model, of the class of structure. |

Raises:

| Type         | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If validation fails, the structure does not fit the process (for example a fraction names no bin, or a rate reads neighbors\_<bin> and the structure has no neighbors), the engine is unknown or does not run the kind of the structure, the settings are not settings of that engine, a series point or an impulse position is one the engine cannot run (a position between two steps for an engine that steps only at integers, or a "pulse" point between two steps), a per-compartment override names an unknown compartment, a stochastic engine gets an amount that is not whole, or the origin has a time zone or a step unit that is not of fixed calendar length. |

Source code in `commol/_model/builder.py`

```python
def build(
    self,
    structure: StructureT,
    engine: EngineName,
    settings: EngineSettings | None = None,
) -> Model[StructureT]:
    """
    Join the process with a structure and an engine into a model.

    The process is written once; the structure lays it out and sets the
    kind of the model, so one builder builds every kind its process fits:

    ``builder.build(Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}),
    engine="DifferentialEquations")``

    Parameters
    ----------
    structure : ModelStructure
        How the process is laid out, such as `Aggregate` (amounts per
        compartment). The model is a ``Model`` of its class.
    engine : EngineName
        The engine that runs the model, which must run the kind of the
        structure: ``"DifferenceEquations"`` (discrete time),
        ``"DifferentialEquations"`` (continuous time), ``"StochasticEvents"``
        or ``"StochasticSteps"`` (at random).
    settings : EngineSettings | None, default=None
        Settings of that engine; None uses its defaults: `ODESolver` for
        ``"DifferentialEquations"``, `StochasticEventsSettings` and
        `StochasticStepsSettings` for the stochastic engines.

    Returns
    -------
    Model[StructureT]
        The model, of the class of ``structure``.

    Raises
    ------
    ValueError
        If validation fails, the structure does not fit the process (for
        example a fraction names no bin, or a rate reads ``neighbors_<bin>``
        and the structure has no ``neighbors``), the engine is unknown or
        does not run the kind of the structure, the settings are not
        settings of that engine, a series point or an impulse position is
        one the engine cannot run (a position between two steps for an
        engine that steps only at integers, or a ``"pulse"`` point between
        two steps), a per-compartment override names an unknown
        compartment, a stochastic engine gets an amount that is not whole,
        or the origin has a time zone or a step unit that is not of fixed
        calendar length.
    """
    model: Model[StructureT] = Model(
        name=self._name,
        description=self._description,
        version=self._version,
        parameters=self._parameters,
        engine=ModelEngine(type=engine, settings=settings),
        process=self.process(),
        structure=structure,
        step_unit=self._step_unit,
        origin=self._origin,
    )

    logger.info(f"Model '{self._name}' successfully built for the {engine} engine.")

    return model
```

## StratificationCondition

Bases: `BaseModel`

Specifies a category within a stratification for rate matching.

When `to` is set, the matched category in the source compartment is replaced with the `to` category in the target compartment name. This enables cross-category transitions within the same bin (e.g., from `g1` to `g2`).

Attributes:

| Name             | Type  | Description                                                |
| ---------------- | ----- | ---------------------------------------------------------- |
| `stratification` | `str` | The ID of the stratification (e.g., "group", "type")       |
| `category`       | `str` | The category within that stratification (e.g., "g1", "t1") |
| `to`             | \`str | None\`                                                     |

Source code in `commol/_model/stratification_condition.py`

```python
class StratificationCondition(BaseModel):
    """
    Specifies a category within a stratification for rate matching.

    When ``to`` is set, the matched category in the source compartment is
    replaced with the ``to`` category in the target compartment name. This
    enables cross-category transitions within the same bin (e.g., from ``g1``
    to ``g2``).

    Attributes
    ----------
    stratification : str
        The ID of the stratification (e.g., "group", "type")
    category : str
        The category within that stratification (e.g., "g1", "t1")
    to : str | None
        Target category override. When set, the target compartment will use
        this category instead of the source compartment's category for this
        stratification.
    """

    model_config = ConfigDict(extra="forbid")

    stratification: str = Field(default=..., description="ID of the stratification")
    category: str | None = Field(
        default=None,
        description=(
            "Category within the stratification. When None, the condition acts as a "
            "target-only override: it does not filter source compartments but still "
            "contributes a 'to' category to the computed target compartment name."
        ),
    )
    to: str | None = Field(
        default=None,
        description=(
            "Target category override. When set, the target compartment uses "
            "this category instead of the source's for this stratification."
        ),
    )
```

## StratifiedRate

Bases: `BaseModel`

The rate of the compartments of a transition that match some categories.

`ModelBuilder.add_transition` takes a list of them as `stratified_rates`. A compartment takes the rate of the stratified rate whose conditions it matches; the most specific match wins.

Attributes:

| Name         | Type                            | Description                                                                                   |
| ------------ | ------------------------------- | --------------------------------------------------------------------------------------------- |
| `conditions` | `list[StratificationCondition]` | The categories that a compartment must match, at least one, each stratification at most once. |
| `rate`       | \`str                           | float\`                                                                                       |
| `absolute`   | \`bool                          | None\`                                                                                        |

Source code in `commol/_model/transition.py`

```python
class StratifiedRate(BaseModel):
    """
    The rate of the compartments of a transition that match some categories.

    ``ModelBuilder.add_transition`` takes a list of them as
    ``stratified_rates``. A compartment takes the rate of the stratified rate
    whose conditions it matches; the most specific match wins.

    Attributes
    ----------
    conditions : list[StratificationCondition]
        The categories that a compartment must match, at least one, each
        stratification at most once.
    rate : str | float
        The rate of the matching compartments: a formula, a parameter or a
        number, which is kept as its string. It can read ``$compartment``
        when the transition expands over its compartments.
    absolute : bool | None
        True: the flow is the rate, not multiplied by the source amount.
        False: the flow is the rate times the source amount. None: the rate
        is absolute when it reads a compartment, a bin total or a
        ``total_…`` variable.
    """

    model_config = ConfigDict(extra="forbid")

    conditions: list[StratificationCondition] = Field(
        default=..., description="Stratification conditions that must match"
    )
    rate: str | float = Field(
        default=..., description="Rate expression for matching compartments"
    )
    absolute: bool | None = Field(
        default=None,
        description=(
            "When True, the flow is treated as absolute (not multiplied by the "
            "source amount). When None, inferred from whether the rate references "
            "any compartment variable."
        ),
    )

    @field_validator("conditions")
    @classmethod
    def _validate_not_empty(
        cls, v: list[StratificationCondition]
    ) -> list[StratificationCondition]:
        """Ensure at least one stratification condition is specified."""
        if not v:
            raise ValueError(
                "conditions must contain at least one stratification condition"
            )
        return v

    @model_validator(mode="after")
    def _validate_unique_stratifications(self) -> Self:
        """Ensure no duplicate stratification IDs."""
        strat_ids = [sc.stratification for sc in self.conditions]
        if len(strat_ids) != len(set(strat_ids)):
            duplicates = [s for s in strat_ids if strat_ids.count(s) > 1]
            raise ValueError(
                (
                    f"Duplicate stratifications found: {list(set(duplicates))}. "
                    f"Each stratification can only be specified once per rate."
                )
            )
        return self

    @field_validator("rate")
    @classmethod
    def _validate_rate(cls, value: str | float) -> str:
        """The rate as its string, once the parser of the engine accepts it as
        the builder writes it: with ``$compartment`` set to the name of a
        compartment."""
        text = str(value)
        _checked_rate(text.replace("$compartment", "compartment"))
        return text
```
