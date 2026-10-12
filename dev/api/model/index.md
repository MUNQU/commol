# Model API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The validated model definition that `ModelBuilder.build` returns, and the entries of `Model.initial_state()`. Guide: [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md), [Units and Equations](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md).

## Model

Bases: `BaseModel`

Root class of compartment model.

Attributes:

| Name          | Type              | Description                                                                 |
| ------------- | ----------------- | --------------------------------------------------------------------------- |
| `name`        | `str`             | A unique name that identifies the model.                                    |
| `description` | \`str             | None\`                                                                      |
| `version`     | \`str             | None\`                                                                      |
| `population`  | `Population`      | Population details, subpopulations, stratifications and initial conditions. |
| `parameters`  | `list[Parameter]` | A list of global model parameters.                                          |
| `dynamics`    | `Dynamics`        | The rules that govern system evolution.                                     |

Source code in `commol/context/model.py`

```python
class Model(BaseModel):
    """
    Root class of compartment model.

    Attributes
    ----------
    name : str
        A unique name that identifies the model.
    description : str | None
        A human-readable description of the model's purpose and function.
    version : str | None
        The version number of the model.
    population : Population
        Population details, subpopulations, stratifications and initial conditions.
    parameters : list[Parameter]
        A list of global model parameters.
    dynamics : Dynamics
        The rules that govern system evolution.
    """

    name: str = Field(..., description="Name which identifies the model.")
    description: str | None = Field(
        None,
        description="Human-readable description of the model's purpose and function.",
    )
    version: str | None = Field(None, description="Version number of the model.")

    population: Population
    parameters: list[Parameter]
    dynamics: Dynamics

    @classmethod
    def from_json(cls, file_path: str | Path) -> Self:
        """
        Loads a model from a JSON file.

        The method reads the specified JSON file, parses its content, and validates
        it against the Model schema.

        Parameters
        ----------
        file_path : str | Path
            The path to the JSON file.

        Returns
        -------
        Model
            A validated Model instance.

        Raises
        ------
        FileNotFoundError
            If the file at `file_path` does not exist.
        pydantic.ValidationError
            If the JSON content does not conform to the Model schema.
        """
        with open(file_path, "r") as f:
            json_data = f.read()

        return cls.model_validate_json(json_data)

    def to_json(self, file_path: str | Path, indent: int = 2) -> None:
        """
        Save the model to a JSON file.

        Every field is written, including those left unset, so the file always
        reloads through :meth:`from_json` to an equal model.

        Long numeric arrays, such as a time-series parameter, are written on a
        single line.

        Parameters
        ----------
        file_path : str | Path
            Path of the file to write.
        indent : int, optional
            Indentation width for nested structures.
        """
        payload = self.model_dump(mode="json")
        Path(file_path).write_text(
            render_json(payload, indent=indent), encoding="utf-8"
        )

    @model_validator(mode="after")
    def validate_unique_parameter_ids(self) -> Self:
        """
        Validates that parameter IDs are unique.
        """
        parameter_ids = [p.id for p in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            duplicates = [
                item for item in set(parameter_ids) if parameter_ids.count(item) > 1
            ]
            raise ValueError(f"Duplicate parameter IDs found: {duplicates}")
        return self

    @model_validator(mode="after")
    def validate_parameter_names_not_reserved(self) -> Self:
        """
        Validates that parameter IDs do not conflict with reserved variable names.

        Bin IDs are reserved variables that cannot be used as parameter names.
        When stratifications are present, base compartment names represent the
        sum of all stratified versions (e.g., S = S_young + S_old).

        Parameters cannot use these reserved names to avoid conflicts during
        rate expression evaluation.
        """
        bin_ids = {bin_item.id for bin_item in self.population.bins}
        parameter_ids = {p.id for p in self.parameters}

        conflicting_names = bin_ids & parameter_ids
        if conflicting_names:
            raise ValueError(
                f"Parameter IDs conflict with reserved compartment names: "
                f"{sorted(conflicting_names)}. "
                f"Bin IDs ({sorted(bin_ids)}) are reserved variables. "
                f"Please rename the conflicting parameters."
            )
        return self

    def update_parameters(self, parameter_values: Mapping[str, float | None]) -> None:
        """
        Update parameter values in the model.

        Parameters
        ----------
        parameter_values : Mapping[str, float | None]
            Dictionary mapping parameter IDs to their new values.

        Raises
        ------
        ValueError
            If a parameter ID in the dictionary doesn't exist in the model.
        """
        param_dict = {param.id: param for param in self.parameters}

        for param_id, value in parameter_values.items():
            if param_id not in param_dict:
                raise ValueError(
                    (
                        f"Parameter '{param_id}' not found in model. "
                        f"Available parameters: {', '.join(param_dict.keys())}"
                    )
                )
            param_dict[param_id].value = value

    def get_uncalibrated_parameters(self) -> list[str]:
        """
        Get a list of parameter IDs that have None values (need calibration).

        Returns
        -------
        list[str]
            List of parameter IDs that require calibration.
        """
        return [param.id for param in self.parameters if param.value is None]

    def get_uncalibrated_initial_conditions(self) -> list[str]:
        """
        Get a list of bin IDs that have None fractions (need calibration).

        Returns
        -------
        list[str]
            List of bin IDs with uncalibrated initial conditions.
        """
        return self.population.initial_conditions.get_uncalibrated_bins()

    def update_initial_conditions(
        self, bin_fractions: Mapping[str, float | None]
    ) -> None:
        """
        Update initial condition fractions for specified bins.

        Parameters
        ----------
        bin_fractions : Mapping[str, float | None]
            Dictionary mapping bin IDs to their new fraction values.

        Raises
        ------
        ValueError
            If a bin ID in the dictionary doesn't exist in the model.
        """
        self.population.initial_conditions.update_bin_fractions(bin_fractions)

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update initial stratification fractions for specified categories.

        Within each stratification, at most one category may be omitted; the
        omitted category receives the remaining fraction.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping category names to their new fraction values.

        Raises
        ------
        ValueError
            If a category doesn't exist in the model, a value lies outside
            [0.0, 1.0], a stratification has more than one category omitted, or
            the resulting fractions do not sum to 1.0.
        """
        self.population.initial_conditions.update_stratification_fractions(fractions)

    def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Upsert per-compartment initial condition overrides.

        A compartment already declared in the model definition has its
        fraction replaced; a new one is appended. The engine validates the
        result before it is stored, so a rejected call leaves the model
        unchanged.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping expanded compartment names to their new fraction
            of the total population.

        Raises
        ------
        ValueError
            If a name is not an expanded compartment of the model, a value lies
            outside ``[0.0, 1.0]``, or the resulting overrides make the initial
            population exceed ``population_size``.
        """
        candidate = self.population.initial_conditions.model_copy(deep=True)
        candidate.update_compartment_fractions(fractions)
        engine_state.initial_state(with_initial_conditions(self, candidate))
        self.population.initial_conditions = candidate

    def initial_state(self) -> list[InitialCompartment]:
        """
        Expanded compartments and the population each one starts with.

        The engine builds this state from the bin fractions, the stratification
        fractions and the per-compartment overrides; it is the population a
        simulation of this model starts from.

        Returns
        -------
        list[InitialCompartment]
            One entry per expanded compartment, in engine order.

        Raises
        ------
        ValueError
            If the per-compartment overrides are not valid for the model.
        """
        return engine_state.initial_state(self)

    def apply_calibration_parameters(
        self,
        result: CalibrationResult
        | ProbabilisticCalibrationResult
        | Mapping[str, float],
        problem: CalibrationProblem,
    ) -> None:
        """
        Write calibrated values back onto the model.

        Each value is routed by the ``parameter_type`` declared for it in
        ``problem``: ``parameter`` values update model parameters, and
        ``scale`` values are ignored because they apply to observations rather
        than to the model.

        ``initial_condition`` values are persisted so that the model rebuilds
        the initial population the engine scored for them. Bin and category
        values update the fractions they declare. Every compartment that those
        fractions do not reproduce, such as an exact compartment value or a
        value that overlaps a per-compartment override, is persisted as a
        per-compartment override.

        Parameters
        ----------
        result : CalibrationResult | ProbabilisticCalibrationResult | Mapping
            The calibration outcome, or a mapping of parameter id to value.
            For a probabilistic result the point parameters of the selected
            ensemble are applied.
        problem : CalibrationProblem
            The problem the result came from, used to resolve parameter types.

        Raises
        ------
        ValueError
            If an id is not declared in ``problem``, or the engine cannot
            resolve an initial condition id.
        """
        values = calibrated_values(result)
        types = {param.id: param.parameter_type for param in problem.parameters}
        unknown = sorted(set(values) - set(types))
        if unknown:
            raise ValueError(
                f"Cannot apply values with no declared parameter type: {unknown}. "
                f"Declared calibration parameters: {sorted(types)}"
            )

        parameters: dict[str, float | None] = {
            param_id: value
            for param_id, value in values.items()
            if types[param_id] == CalibrationParameterType.PARAMETER
        }
        # Computed before any mutation, so a rejected value leaves the model
        # unchanged.
        initial_conditions = calibrated_initial_conditions(self, problem, values)

        if parameters:
            self.update_parameters(parameters)
        self.population.initial_conditions = initial_conditions

    @model_validator(mode="after")
    def validate_transitions_with_engine(self) -> Self:
        """
        Validate the transitions against the flows the engine plans for them.

        The engine rejects a category transition without exactly one source and
        one target category, and a stratified rate whose conditions match none
        of the compartments the transition starts from.
        """
        engine_state.transition_flows(self)
        return self

    @model_validator(mode="after")
    def validate_compartment_fractions(self) -> Self:
        """Validate per-compartment overrides against the engine expansion."""
        if self.population.initial_conditions.compartment_fractions:
            engine_state.initial_state(self)
        return self

    @model_validator(mode="after")
    def validate_formula_variables(self) -> Self:
        """
        Validate that all variables in rate expressions are defined.
        This is done by gathering all valid identifiers and checking each
        transition's rate expressions against them.
        """
        valid_identifiers = self._get_valid_identifiers()

        for transition in self.dynamics.transitions:
            self._validate_transition_rates(transition, valid_identifiers)
        return self

    def _get_valid_identifiers(self) -> set[str]:
        """Gathers all valid identifiers for use in rate expressions."""
        special_vars = {"N", "step", "pi", "e", "t"}
        param_ids = {param.id for param in self.parameters}
        bin_ids = {bin_item.id for bin_item in self.population.bins}

        strat_category_ids: set[str] = {
            cat for strat in self.population.stratifications for cat in strat.categories
        }

        subpopulation_n_vars = self._get_subpopulation_n_vars()
        full_compartment_names = self._get_full_compartment_names()
        bin_subpopulation_vars = self._get_bin_subpopulation_vars()

        return (
            param_ids
            | bin_ids
            | strat_category_ids
            | special_vars
            | subpopulation_n_vars
            | full_compartment_names
            | bin_subpopulation_vars
        )

    def _get_subpopulation_n_vars(self) -> set[str]:
        """Generates all possible N_{category...} variable names."""
        if not self.population.stratifications:
            return set()

        subpopulation_n_vars: set[str] = set()
        category_groups = [s.categories for s in self.population.stratifications]

        full_category_combos = product(*category_groups)

        for combo_tuple in full_category_combos:
            for i in range(1, len(combo_tuple) + 1):
                for subset in combinations(combo_tuple, i):
                    var_name = f"N_{'_'.join(subset)}"
                    subpopulation_n_vars.add(var_name)

        return subpopulation_n_vars

    def get_conditioning_categories(self, stratification_id: str) -> tuple[str, ...]:
        """
        Get the categories a stratification is conditioned on.

        An unconditional stratification subdivides the whole population and has
        no conditioning categories. A conditional one subdivides only the
        subgroup its conditions select, and its fractions are relative to that
        subgroup rather than to the whole population.

        Parameters
        ----------
        stratification_id : str
            Id of a stratification of the model.

        Returns
        -------
        tuple[str, ...]
            Ids of the categories selecting the subgroup this stratification
            subdivides, empty for an unconditional stratification. A category
            name another stratification also declares is qualified, as in
            ``sex:female``.

        Raises
        ------
        ValueError
            If the stratification is not found.
        """
        initial_conditions = self.population.initial_conditions
        for stratification in self.population.stratifications:
            if stratification.id == stratification_id:
                return tuple(
                    initial_conditions.category_id(
                        condition.stratification, condition.category
                    )
                    for condition in (stratification.conditions or [])
                    if condition.category is not None
                )
        raise ValueError(
            f"Stratification '{stratification_id}' not found in model "
            f"'{self.name}'. Available: "
            f"{sorted(s.id for s in self.population.stratifications)}"
        )

    def subgroup_population(self, categories: Iterable[str] = ()) -> float:
        """
        Get the initial head count of a subgroup.

        See
        :meth:`~commol.context.initial_conditions.InitialConditions.subgroup_population`.

        Per-compartment overrides are folded into the sum for every compartment
        the category chain selects, not just when it selects one, so the head
        count matches the population the engine builds. Without overrides the
        declared fractions are multiplied exactly as before.

        Parameters
        ----------
        categories : Iterable[str], optional
            Category ids, bare or qualified (``sex:female``), forming a chain
            from the whole population down to the subgroup, outermost first.

        Returns
        -------
        float
            Number of individuals in the subgroup at step 0.

        Raises
        ------
        ValueError
            If a category is not found or a bare name is ambiguous.
        """
        selected = tuple(categories)
        initial_conditions = self.population.initial_conditions
        if not initial_conditions.compartment_fractions:
            return initial_conditions.subgroup_population(selected)
        resolved = [initial_conditions.resolve_category(c) for c in selected]
        return sum(
            compartment.population
            for compartment in self.initial_state()
            if all(
                compartment.categories.get(stratification) == name
                for stratification, name in resolved
            )
        )

    def get_outputs_by_source(self) -> dict[str, list[str]]:
        """
        Map each bin and accumulator id to the output names it expands into.

        A bin or accumulator produces one output per combination of the
        stratification categories that apply to it. Without stratifications it
        produces a single output named after the id itself.

        Returns
        -------
        dict[str, list[str]]
            Bin and accumulator ids, each mapped to its output names in
            declaration order.
        """
        from commol.context._model.helpers import ModelCompartmentHelper

        helper = ModelCompartmentHelper(self)
        outputs: dict[str, list[str]] = {
            source_id: []
            for source_id in (
                *(bin_item.id for bin_item in self.population.bins),
                *(accumulator.id for accumulator in self.population.accumulators),
            )
        }
        expanded = (
            *helper.generate_compartments(),
            *helper.generate_accumulator_outputs(),
        )
        for source_id, applied in expanded:
            name = "_".join(
                [source_id]
                + [
                    applied[stratification.id]
                    for stratification in self.population.stratifications
                    if stratification.id in applied
                ]
            )
            outputs[source_id].append(name)
        return outputs

    def _get_full_compartment_names(self) -> set[str]:
        """Returns all full stratified compartment names."""
        if not self.population.stratifications:
            return set()

        from commol.context._model.helpers import ModelCompartmentHelper

        helper = ModelCompartmentHelper(self)
        compartments = helper.generate_compartments()
        return {
            helper.compartment_to_string(comp, PrintEquationsOutputFormat.TEXT)
            for comp in compartments
        }

    def _get_bin_subpopulation_vars(self) -> set[str]:
        """Generates partial bin-stratification sum variable names.

        For bins and stratifications, generates the combinations. These represent
        partial sums over one or more stratification dimensions. Requires 2+
        stratifications (with 1, partial sums duplicate existing names).
        """
        if len(self.population.stratifications) < 2:
            return set()

        bin_strat_vars: set[str] = set()
        bin_ids = [bin_item.id for bin_item in self.population.bins]
        category_groups = [s.categories for s in self.population.stratifications]

        for combo_tuple in product(*category_groups):
            num_cats = len(combo_tuple)
            full_mask = (1 << num_cats) - 1
            for subset_mask in range(1, full_mask):
                subset = [
                    cat for k, cat in enumerate(combo_tuple) if (subset_mask >> k) & 1
                ]
                suffix = "_".join(subset)
                for bin_id in bin_ids:
                    bin_strat_vars.add(f"{bin_id}_{suffix}")

        return bin_strat_vars

    def _validate_transition_rates(
        self, transition: Transition, valid_identifiers: set[str]
    ) -> None:
        """Validates the rate expressions for a single transition."""
        if transition.rate:
            self._validate_rate_expression(
                transition.rate, transition.id, "rate", valid_identifiers
            )

        if transition.stratified_rates:
            for sr in transition.stratified_rates:
                self._validate_rate_expression(
                    sr.rate, transition.id, "stratified_rate", valid_identifiers
                )

    def _validate_rate_expression(
        self, rate: str, transition_id: str, context: str, valid_identifiers: set[str]
    ) -> None:
        """Validates variables in a single rate expression."""
        variables = get_expression_variables(rate)
        undefined_vars = [var for var in variables if var not in valid_identifiers]
        if undefined_vars:
            param_ids = {param.id for param in self.parameters}
            bin_ids = {bin_item.id for bin_item in self.population.bins}
            raise ValueError(
                (
                    f"Undefined variables in transition '{transition_id}' "
                    f"{context} '{rate}': {', '.join(undefined_vars)}. "
                    f"Available parameters: "
                    f"{', '.join(sorted(param_ids)) if param_ids else 'none'}. "
                    f"Available bins: "
                    f"{', '.join(sorted(bin_ids)) if bin_ids else 'none'}."
                )
            )

    @model_validator(mode="after")
    def validate_transition_ids(self) -> Self:
        """
        Validates that transition ids (source/target) are consistent in type
        and match the defined Bin IDs or Stratification Categories
        in the Population instance.
        """

        bin_ids = {bin_item.id for bin_item in self.population.bins}
        categories_ids = {
            cat for strat in self.population.stratifications for cat in strat.categories
        }
        bin_and_categories_ids = bin_ids.union(categories_ids)

        for transition in self.dynamics.transitions:
            source = set(transition.source)
            target = set(transition.target)
            transition_ids = source.union(target)

            if not transition_ids.issubset(bin_and_categories_ids):
                invalid_ids = transition_ids - bin_and_categories_ids
                raise ValueError(
                    (
                        f"Transition '{transition.id}' contains invalid ids: "
                        f"{invalid_ids}. Ids must be defined in Bin ids "
                        f"or Stratification Categories."
                    )
                )

            is_bin_flow = transition_ids.issubset(bin_ids)
            is_stratification_flow = transition_ids.issubset(categories_ids)

            if (not is_bin_flow) and (not is_stratification_flow):
                bin_elements = transition_ids.intersection(bin_ids)
                categories_elements = transition_ids.intersection(categories_ids)
                raise ValueError(
                    (
                        f"Transition '{transition.id}' mixes id types. "
                        f"Found Bin ids ({bin_elements}) and "
                        f"Stratification Categories ids ({categories_elements}). "
                        "Transitions must be purely Bin flow or purely "
                        f"Stratification flow."
                    )
                )

            if is_stratification_flow:
                category_to_stratification_map = {
                    cat: strat.id
                    for strat in self.population.stratifications
                    for cat in strat.categories
                }
                parent_stratification_ids = {
                    category_to_stratification_map[cat_id] for cat_id in transition_ids
                }
                if len(parent_stratification_ids) > 1:
                    mixed_strats = ", ".join(parent_stratification_ids)
                    raise ValueError(
                        (
                            f"Transition '{transition.id}' is a Stratification flow "
                            f"but involves categories from multiple stratifications: "
                            f"{mixed_strats}. A single transition must only move "
                            f"between categories belonging to the same parent "
                            f"stratification."
                        )
                    )

        return self

    # ------------------------------------------------------------------
    # Delegation to extracted modules
    # ------------------------------------------------------------------

    def print_equations(
        self,
        output_file: str | None = None,
        format: str = PrintEquationsOutputFormat.TEXT,
    ) -> None:
        """
        Prints the equations of the model in mathematical form.

        Displays model metadata and the system of equations in both
        compact (mathematical notation) and expanded (individual equations) forms.

        For DifferentialEquations models, displays equations as dX/dt = ...
        For DifferenceEquations models,
        displays equations as [X(t+Dt) - X(t)] / Dt = ...

        Parameters
        ----------
        output_file : str | None
            If provided, writes the equations to this file path instead of printing
            to console. If None, prints to console.
        format : str, default="text"
            Output format for equations. Must be one of:
            - "text": Plain text format (default)
            - "latex": LaTeX mathematical notation format

        Raises
        ------
        ValueError
            If format is not "text" or "latex"

        Examples
        --------
        >>> model.print_equations()  # Print to console in text format
        >>> model.print_equations(output_file="equations.txt")  # Save text format
        >>> model.print_equations(format="latex")  # Print LaTeX to console
        >>> model.print_equations(
        ...     output_file="equations.txt", format="latex"
        ... )  # Save LaTeX
        """
        from commol.context._model.equations import EquationPrinter

        EquationPrinter(self).print_equations(output_file, format)

    def check_unit_consistency(self, verbose: bool = False) -> None:
        """
        Check unit consistency of all equations in the model.

        This method validates that all transition rates have consistent units.
        It only performs the check if ALL parameters have units specified.
        If any parameter lacks a unit, the check is skipped.

        For difference equation models, all rates should have units that result in
        population change rates (e.g., "person/day" or "1/day" when multiplied by
        population).

        Parameters
        ----------
        verbose : bool, default=False
            If True, prints a success message when all units are consistent.

        Raises
        ------
        UnitConsistencyError
            If unit inconsistencies are found in any equation.
        ValueError
            If the model type doesn't support unit checking.

        Notes
        -----
        - Bin variables are assumed to have units of "person"
        - Predefined variables (N, N_young, etc.) have units of "person"
        - Time step variables (t, step) are dimensionless
        - Mathematical constants (pi, e) are dimensionless
        """
        from commol.context._model.units import UnitChecker

        UnitChecker(self).check_unit_consistency(verbose)
```

### Methods:

#### from_json

```python
from_json(file_path: str | Path) -> Self
```

Loads a model from a JSON file.

The method reads the specified JSON file, parses its content, and validates it against the Model schema.

Parameters:

| Name        | Type  | Description | Default                    |
| ----------- | ----- | ----------- | -------------------------- |
| `file_path` | \`str | Path\`      | The path to the JSON file. |

Returns:

| Type    | Description                 |
| ------- | --------------------------- |
| `Model` | A validated Model instance. |

Raises:

| Type                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `FileNotFoundError` | If the file at file_path does not exist.                  |
| `ValidationError`   | If the JSON content does not conform to the Model schema. |

Source code in `commol/context/model.py`

```python
@classmethod
def from_json(cls, file_path: str | Path) -> Self:
    """
    Loads a model from a JSON file.

    The method reads the specified JSON file, parses its content, and validates
    it against the Model schema.

    Parameters
    ----------
    file_path : str | Path
        The path to the JSON file.

    Returns
    -------
    Model
        A validated Model instance.

    Raises
    ------
    FileNotFoundError
        If the file at `file_path` does not exist.
    pydantic.ValidationError
        If the JSON content does not conform to the Model schema.
    """
    with open(file_path, "r") as f:
        json_data = f.read()

    return cls.model_validate_json(json_data)
```

#### to_json

```python
to_json(file_path: str | Path, indent: int = 2) -> None
```

Save the model to a JSON file.

Every field is written, including those left unset, so the file always reloads through :meth:`from_json` to an equal model.

Long numeric arrays, such as a time-series parameter, are written on a single line.

Parameters:

| Name        | Type  | Description                              | Default                    |
| ----------- | ----- | ---------------------------------------- | -------------------------- |
| `file_path` | \`str | Path\`                                   | Path of the file to write. |
| `indent`    | `int` | Indentation width for nested structures. | `2`                        |

Source code in `commol/context/model.py`

```python
def to_json(self, file_path: str | Path, indent: int = 2) -> None:
    """
    Save the model to a JSON file.

    Every field is written, including those left unset, so the file always
    reloads through :meth:`from_json` to an equal model.

    Long numeric arrays, such as a time-series parameter, are written on a
    single line.

    Parameters
    ----------
    file_path : str | Path
        Path of the file to write.
    indent : int, optional
        Indentation width for nested structures.
    """
    payload = self.model_dump(mode="json")
    Path(file_path).write_text(
        render_json(payload, indent=indent), encoding="utf-8"
    )
```

#### validate_unique_parameter_ids

```python
validate_unique_parameter_ids() -> Self
```

Validates that parameter IDs are unique.

Source code in `commol/context/model.py`

```python
@model_validator(mode="after")
def validate_unique_parameter_ids(self) -> Self:
    """
    Validates that parameter IDs are unique.
    """
    parameter_ids = [p.id for p in self.parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        duplicates = [
            item for item in set(parameter_ids) if parameter_ids.count(item) > 1
        ]
        raise ValueError(f"Duplicate parameter IDs found: {duplicates}")
    return self
```

#### validate_parameter_names_not_reserved

```python
validate_parameter_names_not_reserved() -> Self
```

Validates that parameter IDs do not conflict with reserved variable names.

Bin IDs are reserved variables that cannot be used as parameter names. When stratifications are present, base compartment names represent the sum of all stratified versions (e.g., S = S_young + S_old).

Parameters cannot use these reserved names to avoid conflicts during rate expression evaluation.

Source code in `commol/context/model.py`

```python
@model_validator(mode="after")
def validate_parameter_names_not_reserved(self) -> Self:
    """
    Validates that parameter IDs do not conflict with reserved variable names.

    Bin IDs are reserved variables that cannot be used as parameter names.
    When stratifications are present, base compartment names represent the
    sum of all stratified versions (e.g., S = S_young + S_old).

    Parameters cannot use these reserved names to avoid conflicts during
    rate expression evaluation.
    """
    bin_ids = {bin_item.id for bin_item in self.population.bins}
    parameter_ids = {p.id for p in self.parameters}

    conflicting_names = bin_ids & parameter_ids
    if conflicting_names:
        raise ValueError(
            f"Parameter IDs conflict with reserved compartment names: "
            f"{sorted(conflicting_names)}. "
            f"Bin IDs ({sorted(bin_ids)}) are reserved variables. "
            f"Please rename the conflicting parameters."
        )
    return self
```

#### update_parameters

```python
update_parameters(parameter_values: Mapping[str, float | None]) -> None
```

Update parameter values in the model.

Parameters:

| Name               | Type                  | Description | Default                                               |
| ------------------ | --------------------- | ----------- | ----------------------------------------------------- |
| `parameter_values` | \`Mapping\[str, float | None\]\`    | Dictionary mapping parameter IDs to their new values. |

Raises:

| Type         | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| `ValueError` | If a parameter ID in the dictionary doesn't exist in the model. |

Source code in `commol/context/model.py`

```python
def update_parameters(self, parameter_values: Mapping[str, float | None]) -> None:
    """
    Update parameter values in the model.

    Parameters
    ----------
    parameter_values : Mapping[str, float | None]
        Dictionary mapping parameter IDs to their new values.

    Raises
    ------
    ValueError
        If a parameter ID in the dictionary doesn't exist in the model.
    """
    param_dict = {param.id: param for param in self.parameters}

    for param_id, value in parameter_values.items():
        if param_id not in param_dict:
            raise ValueError(
                (
                    f"Parameter '{param_id}' not found in model. "
                    f"Available parameters: {', '.join(param_dict.keys())}"
                )
            )
        param_dict[param_id].value = value
```

#### get_uncalibrated_parameters

```python
get_uncalibrated_parameters() -> list[str]
```

Get a list of parameter IDs that have None values (need calibration).

Returns:

| Type        | Description                                     |
| ----------- | ----------------------------------------------- |
| `list[str]` | List of parameter IDs that require calibration. |

Source code in `commol/context/model.py`

```python
def get_uncalibrated_parameters(self) -> list[str]:
    """
    Get a list of parameter IDs that have None values (need calibration).

    Returns
    -------
    list[str]
        List of parameter IDs that require calibration.
    """
    return [param.id for param in self.parameters if param.value is None]
```

#### get_uncalibrated_initial_conditions

```python
get_uncalibrated_initial_conditions() -> list[str]
```

Get a list of bin IDs that have None fractions (need calibration).

Returns:

| Type        | Description                                           |
| ----------- | ----------------------------------------------------- |
| `list[str]` | List of bin IDs with uncalibrated initial conditions. |

Source code in `commol/context/model.py`

```python
def get_uncalibrated_initial_conditions(self) -> list[str]:
    """
    Get a list of bin IDs that have None fractions (need calibration).

    Returns
    -------
    list[str]
        List of bin IDs with uncalibrated initial conditions.
    """
    return self.population.initial_conditions.get_uncalibrated_bins()
```

#### update_initial_conditions

```python
update_initial_conditions(bin_fractions: Mapping[str, float | None]) -> None
```

Update initial condition fractions for specified bins.

Parameters:

| Name            | Type                  | Description | Default                                                  |
| --------------- | --------------------- | ----------- | -------------------------------------------------------- |
| `bin_fractions` | \`Mapping\[str, float | None\]\`    | Dictionary mapping bin IDs to their new fraction values. |

Raises:

| Type         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `ValueError` | If a bin ID in the dictionary doesn't exist in the model. |

Source code in `commol/context/model.py`

```python
def update_initial_conditions(
    self, bin_fractions: Mapping[str, float | None]
) -> None:
    """
    Update initial condition fractions for specified bins.

    Parameters
    ----------
    bin_fractions : Mapping[str, float | None]
        Dictionary mapping bin IDs to their new fraction values.

    Raises
    ------
    ValueError
        If a bin ID in the dictionary doesn't exist in the model.
    """
    self.population.initial_conditions.update_bin_fractions(bin_fractions)
```

#### update_stratification_fractions

```python
update_stratification_fractions(fractions: Mapping[str, float]) -> None
```

Update initial stratification fractions for specified categories.

Within each stratification, at most one category may be omitted; the omitted category receives the remaining fraction.

Parameters:

| Name        | Type                  | Description                                                     | Default    |
| ----------- | --------------------- | --------------------------------------------------------------- | ---------- |
| `fractions` | `Mapping[str, float]` | Dictionary mapping category names to their new fraction values. | *required* |

Raises:

| Type         | Description                                                                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a category doesn't exist in the model, a value lies outside [0.0, 1.0], a stratification has more than one category omitted, or the resulting fractions do not sum to 1.0. |

Source code in `commol/context/model.py`

```python
def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
    """
    Update initial stratification fractions for specified categories.

    Within each stratification, at most one category may be omitted; the
    omitted category receives the remaining fraction.

    Parameters
    ----------
    fractions : Mapping[str, float]
        Dictionary mapping category names to their new fraction values.

    Raises
    ------
    ValueError
        If a category doesn't exist in the model, a value lies outside
        [0.0, 1.0], a stratification has more than one category omitted, or
        the resulting fractions do not sum to 1.0.
    """
    self.population.initial_conditions.update_stratification_fractions(fractions)
```

#### update_compartment_fractions

```python
update_compartment_fractions(fractions: Mapping[str, float]) -> None
```

Upsert per-compartment initial condition overrides.

A compartment already declared in the model definition has its fraction replaced; a new one is appended. The engine validates the result before it is stored, so a rejected call leaves the model unchanged.

Parameters:

| Name        | Type                  | Description                                                                                  | Default    |
| ----------- | --------------------- | -------------------------------------------------------------------------------------------- | ---------- |
| `fractions` | `Mapping[str, float]` | Dictionary mapping expanded compartment names to their new fraction of the total population. | *required* |

Raises:

| Type         | Description                                                                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a name is not an expanded compartment of the model, a value lies outside [0.0, 1.0], or the resulting overrides make the initial population exceed population_size. |

Source code in `commol/context/model.py`

```python
def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
    """
    Upsert per-compartment initial condition overrides.

    A compartment already declared in the model definition has its
    fraction replaced; a new one is appended. The engine validates the
    result before it is stored, so a rejected call leaves the model
    unchanged.

    Parameters
    ----------
    fractions : Mapping[str, float]
        Dictionary mapping expanded compartment names to their new fraction
        of the total population.

    Raises
    ------
    ValueError
        If a name is not an expanded compartment of the model, a value lies
        outside ``[0.0, 1.0]``, or the resulting overrides make the initial
        population exceed ``population_size``.
    """
    candidate = self.population.initial_conditions.model_copy(deep=True)
    candidate.update_compartment_fractions(fractions)
    engine_state.initial_state(with_initial_conditions(self, candidate))
    self.population.initial_conditions = candidate
```

#### initial_state

```python
initial_state() -> list[InitialCompartment]
```

Expanded compartments and the population each one starts with.

The engine builds this state from the bin fractions, the stratification fractions and the per-compartment overrides; it is the population a simulation of this model starts from.

Returns:

| Type                       | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `list[InitialCompartment]` | One entry per expanded compartment, in engine order. |

Raises:

| Type         | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `ValueError` | If the per-compartment overrides are not valid for the model. |

Source code in `commol/context/model.py`

```python
def initial_state(self) -> list[InitialCompartment]:
    """
    Expanded compartments and the population each one starts with.

    The engine builds this state from the bin fractions, the stratification
    fractions and the per-compartment overrides; it is the population a
    simulation of this model starts from.

    Returns
    -------
    list[InitialCompartment]
        One entry per expanded compartment, in engine order.

    Raises
    ------
    ValueError
        If the per-compartment overrides are not valid for the model.
    """
    return engine_state.initial_state(self)
```

#### apply_calibration_parameters

```python
apply_calibration_parameters(result: CalibrationResult | ProbabilisticCalibrationResult | Mapping[str, float], problem: CalibrationProblem) -> None
```

Write calibrated values back onto the model.

Each value is routed by the `parameter_type` declared for it in `problem`: `parameter` values update model parameters, and `scale` values are ignored because they apply to observations rather than to the model.

`initial_condition` values are persisted so that the model rebuilds the initial population the engine scored for them. Bin and category values update the fractions they declare. Every compartment that those fractions do not reproduce, such as an exact compartment value or a value that overlaps a per-compartment override, is persisted as a per-compartment override.

Parameters:

| Name      | Type                 | Description                                                        | Default    |
| --------- | -------------------- | ------------------------------------------------------------------ | ---------- |
| `result`  | \`CalibrationResult  | ProbabilisticCalibrationResult                                     | Mapping\`  |
| `problem` | `CalibrationProblem` | The problem the result came from, used to resolve parameter types. | *required* |

Raises:

| Type         | Description                                                                                |
| ------------ | ------------------------------------------------------------------------------------------ |
| `ValueError` | If an id is not declared in problem, or the engine cannot resolve an initial condition id. |

Source code in `commol/context/model.py`

```python
def apply_calibration_parameters(
    self,
    result: CalibrationResult
    | ProbabilisticCalibrationResult
    | Mapping[str, float],
    problem: CalibrationProblem,
) -> None:
    """
    Write calibrated values back onto the model.

    Each value is routed by the ``parameter_type`` declared for it in
    ``problem``: ``parameter`` values update model parameters, and
    ``scale`` values are ignored because they apply to observations rather
    than to the model.

    ``initial_condition`` values are persisted so that the model rebuilds
    the initial population the engine scored for them. Bin and category
    values update the fractions they declare. Every compartment that those
    fractions do not reproduce, such as an exact compartment value or a
    value that overlaps a per-compartment override, is persisted as a
    per-compartment override.

    Parameters
    ----------
    result : CalibrationResult | ProbabilisticCalibrationResult | Mapping
        The calibration outcome, or a mapping of parameter id to value.
        For a probabilistic result the point parameters of the selected
        ensemble are applied.
    problem : CalibrationProblem
        The problem the result came from, used to resolve parameter types.

    Raises
    ------
    ValueError
        If an id is not declared in ``problem``, or the engine cannot
        resolve an initial condition id.
    """
    values = calibrated_values(result)
    types = {param.id: param.parameter_type for param in problem.parameters}
    unknown = sorted(set(values) - set(types))
    if unknown:
        raise ValueError(
            f"Cannot apply values with no declared parameter type: {unknown}. "
            f"Declared calibration parameters: {sorted(types)}"
        )

    parameters: dict[str, float | None] = {
        param_id: value
        for param_id, value in values.items()
        if types[param_id] == CalibrationParameterType.PARAMETER
    }
    # Computed before any mutation, so a rejected value leaves the model
    # unchanged.
    initial_conditions = calibrated_initial_conditions(self, problem, values)

    if parameters:
        self.update_parameters(parameters)
    self.population.initial_conditions = initial_conditions
```

#### validate_transitions_with_engine

```python
validate_transitions_with_engine() -> Self
```

Validate the transitions against the flows the engine plans for them.

The engine rejects a category transition without exactly one source and one target category, and a stratified rate whose conditions match none of the compartments the transition starts from.

Source code in `commol/context/model.py`

```python
@model_validator(mode="after")
def validate_transitions_with_engine(self) -> Self:
    """
    Validate the transitions against the flows the engine plans for them.

    The engine rejects a category transition without exactly one source and
    one target category, and a stratified rate whose conditions match none
    of the compartments the transition starts from.
    """
    engine_state.transition_flows(self)
    return self
```

#### validate_compartment_fractions

```python
validate_compartment_fractions() -> Self
```

Validate per-compartment overrides against the engine expansion.

Source code in `commol/context/model.py`

```python
@model_validator(mode="after")
def validate_compartment_fractions(self) -> Self:
    """Validate per-compartment overrides against the engine expansion."""
    if self.population.initial_conditions.compartment_fractions:
        engine_state.initial_state(self)
    return self
```

#### validate_formula_variables

```python
validate_formula_variables() -> Self
```

Validate that all variables in rate expressions are defined. This is done by gathering all valid identifiers and checking each transition's rate expressions against them.

Source code in `commol/context/model.py`

```python
@model_validator(mode="after")
def validate_formula_variables(self) -> Self:
    """
    Validate that all variables in rate expressions are defined.
    This is done by gathering all valid identifiers and checking each
    transition's rate expressions against them.
    """
    valid_identifiers = self._get_valid_identifiers()

    for transition in self.dynamics.transitions:
        self._validate_transition_rates(transition, valid_identifiers)
    return self
```

#### get_conditioning_categories

```python
get_conditioning_categories(stratification_id: str) -> tuple[str, ...]
```

Get the categories a stratification is conditioned on.

An unconditional stratification subdivides the whole population and has no conditioning categories. A conditional one subdivides only the subgroup its conditions select, and its fractions are relative to that subgroup rather than to the whole population.

Parameters:

| Name                | Type  | Description                          | Default    |
| ------------------- | ----- | ------------------------------------ | ---------- |
| `stratification_id` | `str` | Id of a stratification of the model. | *required* |

Returns:

| Type              | Description                                                                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tuple[str, ...]` | Ids of the categories selecting the subgroup this stratification subdivides, empty for an unconditional stratification. A category name another stratification also declares is qualified, as in sex:female. |

Raises:

| Type         | Description                         |
| ------------ | ----------------------------------- |
| `ValueError` | If the stratification is not found. |

Source code in `commol/context/model.py`

```python
def get_conditioning_categories(self, stratification_id: str) -> tuple[str, ...]:
    """
    Get the categories a stratification is conditioned on.

    An unconditional stratification subdivides the whole population and has
    no conditioning categories. A conditional one subdivides only the
    subgroup its conditions select, and its fractions are relative to that
    subgroup rather than to the whole population.

    Parameters
    ----------
    stratification_id : str
        Id of a stratification of the model.

    Returns
    -------
    tuple[str, ...]
        Ids of the categories selecting the subgroup this stratification
        subdivides, empty for an unconditional stratification. A category
        name another stratification also declares is qualified, as in
        ``sex:female``.

    Raises
    ------
    ValueError
        If the stratification is not found.
    """
    initial_conditions = self.population.initial_conditions
    for stratification in self.population.stratifications:
        if stratification.id == stratification_id:
            return tuple(
                initial_conditions.category_id(
                    condition.stratification, condition.category
                )
                for condition in (stratification.conditions or [])
                if condition.category is not None
            )
    raise ValueError(
        f"Stratification '{stratification_id}' not found in model "
        f"'{self.name}'. Available: "
        f"{sorted(s.id for s in self.population.stratifications)}"
    )
```

#### subgroup_population

```python
subgroup_population(categories: Iterable[str] = ()) -> float
```

Get the initial head count of a subgroup.

See :meth:`~commol.context.initial_conditions.InitialConditions.subgroup_population`.

Per-compartment overrides are folded into the sum for every compartment the category chain selects, not just when it selects one, so the head count matches the population the engine builds. Without overrides the declared fractions are multiplied exactly as before.

Parameters:

| Name         | Type            | Description                                                                                                                    | Default |
| ------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `categories` | `Iterable[str]` | Category ids, bare or qualified (sex:female), forming a chain from the whole population down to the subgroup, outermost first. | `()`    |

Returns:

| Type    | Description                                      |
| ------- | ------------------------------------------------ |
| `float` | Number of individuals in the subgroup at step 0. |

Raises:

| Type         | Description                                             |
| ------------ | ------------------------------------------------------- |
| `ValueError` | If a category is not found or a bare name is ambiguous. |

Source code in `commol/context/model.py`

```python
def subgroup_population(self, categories: Iterable[str] = ()) -> float:
    """
    Get the initial head count of a subgroup.

    See
    :meth:`~commol.context.initial_conditions.InitialConditions.subgroup_population`.

    Per-compartment overrides are folded into the sum for every compartment
    the category chain selects, not just when it selects one, so the head
    count matches the population the engine builds. Without overrides the
    declared fractions are multiplied exactly as before.

    Parameters
    ----------
    categories : Iterable[str], optional
        Category ids, bare or qualified (``sex:female``), forming a chain
        from the whole population down to the subgroup, outermost first.

    Returns
    -------
    float
        Number of individuals in the subgroup at step 0.

    Raises
    ------
    ValueError
        If a category is not found or a bare name is ambiguous.
    """
    selected = tuple(categories)
    initial_conditions = self.population.initial_conditions
    if not initial_conditions.compartment_fractions:
        return initial_conditions.subgroup_population(selected)
    resolved = [initial_conditions.resolve_category(c) for c in selected]
    return sum(
        compartment.population
        for compartment in self.initial_state()
        if all(
            compartment.categories.get(stratification) == name
            for stratification, name in resolved
        )
    )
```

#### get_outputs_by_source

```python
get_outputs_by_source() -> dict[str, list[str]]
```

Map each bin and accumulator id to the output names it expands into.

A bin or accumulator produces one output per combination of the stratification categories that apply to it. Without stratifications it produces a single output named after the id itself.

Returns:

| Type                   | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| `dict[str, list[str]]` | Bin and accumulator ids, each mapped to its output names in declaration order. |

Source code in `commol/context/model.py`

```python
def get_outputs_by_source(self) -> dict[str, list[str]]:
    """
    Map each bin and accumulator id to the output names it expands into.

    A bin or accumulator produces one output per combination of the
    stratification categories that apply to it. Without stratifications it
    produces a single output named after the id itself.

    Returns
    -------
    dict[str, list[str]]
        Bin and accumulator ids, each mapped to its output names in
        declaration order.
    """
    from commol.context._model.helpers import ModelCompartmentHelper

    helper = ModelCompartmentHelper(self)
    outputs: dict[str, list[str]] = {
        source_id: []
        for source_id in (
            *(bin_item.id for bin_item in self.population.bins),
            *(accumulator.id for accumulator in self.population.accumulators),
        )
    }
    expanded = (
        *helper.generate_compartments(),
        *helper.generate_accumulator_outputs(),
    )
    for source_id, applied in expanded:
        name = "_".join(
            [source_id]
            + [
                applied[stratification.id]
                for stratification in self.population.stratifications
                if stratification.id in applied
            ]
        )
        outputs[source_id].append(name)
    return outputs
```

#### validate_transition_ids

```python
validate_transition_ids() -> Self
```

Validates that transition ids (source/target) are consistent in type and match the defined Bin IDs or Stratification Categories in the Population instance.

Source code in `commol/context/model.py`

```python
@model_validator(mode="after")
def validate_transition_ids(self) -> Self:
    """
    Validates that transition ids (source/target) are consistent in type
    and match the defined Bin IDs or Stratification Categories
    in the Population instance.
    """

    bin_ids = {bin_item.id for bin_item in self.population.bins}
    categories_ids = {
        cat for strat in self.population.stratifications for cat in strat.categories
    }
    bin_and_categories_ids = bin_ids.union(categories_ids)

    for transition in self.dynamics.transitions:
        source = set(transition.source)
        target = set(transition.target)
        transition_ids = source.union(target)

        if not transition_ids.issubset(bin_and_categories_ids):
            invalid_ids = transition_ids - bin_and_categories_ids
            raise ValueError(
                (
                    f"Transition '{transition.id}' contains invalid ids: "
                    f"{invalid_ids}. Ids must be defined in Bin ids "
                    f"or Stratification Categories."
                )
            )

        is_bin_flow = transition_ids.issubset(bin_ids)
        is_stratification_flow = transition_ids.issubset(categories_ids)

        if (not is_bin_flow) and (not is_stratification_flow):
            bin_elements = transition_ids.intersection(bin_ids)
            categories_elements = transition_ids.intersection(categories_ids)
            raise ValueError(
                (
                    f"Transition '{transition.id}' mixes id types. "
                    f"Found Bin ids ({bin_elements}) and "
                    f"Stratification Categories ids ({categories_elements}). "
                    "Transitions must be purely Bin flow or purely "
                    f"Stratification flow."
                )
            )

        if is_stratification_flow:
            category_to_stratification_map = {
                cat: strat.id
                for strat in self.population.stratifications
                for cat in strat.categories
            }
            parent_stratification_ids = {
                category_to_stratification_map[cat_id] for cat_id in transition_ids
            }
            if len(parent_stratification_ids) > 1:
                mixed_strats = ", ".join(parent_stratification_ids)
                raise ValueError(
                    (
                        f"Transition '{transition.id}' is a Stratification flow "
                        f"but involves categories from multiple stratifications: "
                        f"{mixed_strats}. A single transition must only move "
                        f"between categories belonging to the same parent "
                        f"stratification."
                    )
                )

    return self
```

#### print_equations

```python
print_equations(output_file: str | None = None, format: str = TEXT) -> None
```

Prints the equations of the model in mathematical form.

Displays model metadata and the system of equations in both compact (mathematical notation) and expanded (individual equations) forms.

For DifferentialEquations models, displays equations as dX/dt = ... For DifferenceEquations models, displays equations as [X(t+Dt) - X(t)] / Dt = ...

Parameters:

| Name          | Type  | Description                                                                                                                      | Default                                                                                                         |
| ------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str | None\`                                                                                                                           | If provided, writes the equations to this file path instead of printing to console. If None, prints to console. |
| `format`      | `str` | Output format for equations. Must be one of: - "text": Plain text format (default) - "latex": LaTeX mathematical notation format | `"text"`                                                                                                        |

Raises:

| Type         | Description                        |
| ------------ | ---------------------------------- |
| `ValueError` | If format is not "text" or "latex" |

Examples:

```pycon
>>> model.print_equations()  # Print to console in text format
>>> model.print_equations(output_file="equations.txt")  # Save text format
>>> model.print_equations(format="latex")  # Print LaTeX to console
>>> model.print_equations(
...     output_file="equations.txt", format="latex"
... )  # Save LaTeX
```

Source code in `commol/context/model.py`

```python
def print_equations(
    self,
    output_file: str | None = None,
    format: str = PrintEquationsOutputFormat.TEXT,
) -> None:
    """
    Prints the equations of the model in mathematical form.

    Displays model metadata and the system of equations in both
    compact (mathematical notation) and expanded (individual equations) forms.

    For DifferentialEquations models, displays equations as dX/dt = ...
    For DifferenceEquations models,
    displays equations as [X(t+Dt) - X(t)] / Dt = ...

    Parameters
    ----------
    output_file : str | None
        If provided, writes the equations to this file path instead of printing
        to console. If None, prints to console.
    format : str, default="text"
        Output format for equations. Must be one of:
        - "text": Plain text format (default)
        - "latex": LaTeX mathematical notation format

    Raises
    ------
    ValueError
        If format is not "text" or "latex"

    Examples
    --------
    >>> model.print_equations()  # Print to console in text format
    >>> model.print_equations(output_file="equations.txt")  # Save text format
    >>> model.print_equations(format="latex")  # Print LaTeX to console
    >>> model.print_equations(
    ...     output_file="equations.txt", format="latex"
    ... )  # Save LaTeX
    """
    from commol.context._model.equations import EquationPrinter

    EquationPrinter(self).print_equations(output_file, format)
```

#### check_unit_consistency

```python
check_unit_consistency(verbose: bool = False) -> None
```

Check unit consistency of all equations in the model.

This method validates that all transition rates have consistent units. It only performs the check if ALL parameters have units specified. If any parameter lacks a unit, the check is skipped.

For difference equation models, all rates should have units that result in population change rates (e.g., "person/day" or "1/day" when multiplied by population).

Parameters:

| Name      | Type   | Description                                                      | Default |
| --------- | ------ | ---------------------------------------------------------------- | ------- |
| `verbose` | `bool` | If True, prints a success message when all units are consistent. | `False` |

Raises:

| Type                   | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `UnitConsistencyError` | If unit inconsistencies are found in any equation. |
| `ValueError`           | If the model type doesn't support unit checking.   |

Notes

- Bin variables are assumed to have units of "person"
- Predefined variables (N, N_young, etc.) have units of "person"
- Time step variables (t, step) are dimensionless
- Mathematical constants (pi, e) are dimensionless

Source code in `commol/context/model.py`

```python
def check_unit_consistency(self, verbose: bool = False) -> None:
    """
    Check unit consistency of all equations in the model.

    This method validates that all transition rates have consistent units.
    It only performs the check if ALL parameters have units specified.
    If any parameter lacks a unit, the check is skipped.

    For difference equation models, all rates should have units that result in
    population change rates (e.g., "person/day" or "1/day" when multiplied by
    population).

    Parameters
    ----------
    verbose : bool, default=False
        If True, prints a success message when all units are consistent.

    Raises
    ------
    UnitConsistencyError
        If unit inconsistencies are found in any equation.
    ValueError
        If the model type doesn't support unit checking.

    Notes
    -----
    - Bin variables are assumed to have units of "person"
    - Predefined variables (N, N_young, etc.) have units of "person"
    - Time step variables (t, step) are dimensionless
    - Mathematical constants (pi, e) are dimensionless
    """
    from commol.context._model.units import UnitChecker

    UnitChecker(self).check_unit_consistency(verbose)
```

## InitialCompartment

Bases: `NamedTuple`

One expanded compartment and the population it starts with.

Attributes:

| Name         | Type             | Description                                                                                                                                             |
| ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`       | `str`            | Expanded compartment name, for example IH_y60.                                                                                                          |
| `bin`        | `str`            | Bin the compartment expands from.                                                                                                                       |
| `categories` | `dict[str, str]` | Category applied to the compartment, keyed by stratification id. A conditional stratification whose conditions the compartment does not meet is absent. |
| `population` | `float`          | Initial head count, including per-compartment overrides.                                                                                                |

Source code in `commol/context/_model/engine_state.py`

```python
class InitialCompartment(NamedTuple):
    """
    One expanded compartment and the population it starts with.

    Attributes
    ----------
    name : str
        Expanded compartment name, for example ``IH_y60``.
    bin : str
        Bin the compartment expands from.
    categories : dict[str, str]
        Category applied to the compartment, keyed by stratification id. A
        conditional stratification whose conditions the compartment does not
        meet is absent.
    population : float
        Initial head count, including per-compartment overrides.
    """

    name: str
    bin: str
    categories: dict[str, str]
    population: float
```

## UnitConsistencyError

Bases: `Exception`

Raised when unit consistency check fails.

Source code in `commol/utils/equations.py`

```python
class UnitConsistencyError(Exception):
    """Raised when unit consistency check fails."""

    pass
```
