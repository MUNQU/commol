"""The dynamics of an aggregate model, as blocks of its description.

The flows come from the engine (``commol_rs.model.transition_flows``), so the
equations a description shows are the ones the engine simulates: each flow
moves an amount from one compartment to another, and a proportional rate is
shown multiplied by its source. An impulse is not part of the equations: it is
listed apart, with the positions where it moves its amount at once.
"""

import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from commol.commol_rs import _commol_rs as commol_rs

from commol.context._model import engine_state
from commol.context._model.engine_state import TransitionFlow
from commol.context._model.helpers import ModelCompartmentHelper
from commol.context.description import (
    Block,
    Cell,
    Equation,
    EquationGroup,
    Equations,
    GraphEdge,
    GraphNode,
    StateGraph,
    Table,
    Text,
)

if TYPE_CHECKING:
    from commol.context.model import Model
    from commol.context.transition import Transition


class DynamicsDescription:
    """
    Builds the blocks of the ``dynamics`` section of an aggregate model.

    The compact form lists each transition with the flows it creates between
    compartments; the expanded form gives one equation per compartment, its
    inflows minus its outflows; the impulses are listed apart. The state
    graphs draw the bins and the transitions, and the expanded compartments
    and their flows.
    """

    def __init__(self, model: "Model") -> None:
        self._model = model
        self._helper = ModelCompartmentHelper(model)
        self._flows: list[TransitionFlow] | None = None

    def _transition_flows(self) -> list[TransitionFlow]:
        if self._flows is None:
            self._flows = engine_state.transition_flows(
                self._model.process, self._model.structure
            )
        return self._flows

    def blocks(self) -> list[Block]:
        """
        The blocks of the dynamics: graphs, equations and impulses.

        Returns
        -------
        list[Block]
            The state graphs, the compact form, the expanded form when the
            engine has equations or the events when it draws them, and the
            impulses.
        """
        blocks: list[Block] = [self.state_graph(), self.expanded_graph()]
        blocks.extend(self.compact_form())
        engine = commol_rs.engine.EngineType(self._model.engine.type)
        if engine.equation_lhs("X", False):
            blocks.append(self.expanded_form())
        if engine.dynamics_form == "stochastic_events":
            blocks.extend(self.events(engine.rate_meaning == "continuous"))
        impulses = self.impulses()
        if impulses is not None:
            blocks.append(impulses)
        return blocks

    # ------------------------------------------------------------------
    # State graphs
    # ------------------------------------------------------------------

    def state_graph(self) -> StateGraph:
        """
        The bins, and one edge per transition, labelled with its rate.

        A transition between the categories of a stratification moves between
        nodes of those categories. A transition with several sources or
        targets has an edge from each source to each target; one with no
        source or no target has an edge from or to nowhere.

        Returns
        -------
        StateGraph
            The graph of the states of the model.
        """
        process = self._model.process
        nodes = [
            GraphNode(id=bin_item.id, label=bin_item.name or bin_item.id, group="bin")
            for bin_item in process.bins
        ]
        bin_ids = {bin_item.id for bin_item in process.bins}
        category_nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        for transition in process.transitions:
            ids = [*transition.source, *transition.target]
            if not set(ids) <= bin_ids:
                stratification = next(
                    strat
                    for strat in process.stratifications
                    if set(ids) <= set(strat.categories)
                )
                for category in ids:
                    node_id = f"{stratification.id}:{category}"
                    category_nodes.setdefault(
                        category,
                        GraphNode(id=node_id, label=category, group=stratification.id),
                    )
            label = f"{transition.id}: {_rate_text(transition)}"

            def node(item: str) -> str:
                found = category_nodes.get(item)
                return item if found is None else found.id

            sources = [node(item) for item in dict.fromkeys(transition.source)]
            targets = [node(item) for item in dict.fromkeys(transition.target)]
            for source in sources or [None]:
                for target in targets or [None]:
                    edges.append(GraphEdge(source=source, target=target, label=label))
        return StateGraph(
            heading="States and transitions",
            name="states",
            nodes=[*nodes, *category_nodes.values()],
            edges=edges,
        )

    def expanded_graph(self) -> StateGraph:
        """
        The expanded compartments, and one edge per flow, as the engine runs it.

        Returns
        -------
        StateGraph
            The graph of the compartments, marked as expanded.
        """
        compartments, _ = engine_state.expanded_outputs(self._model.process)
        nodes = [
            GraphNode(
                id=compartment.name, label=compartment.name, group=compartment.source
            )
            for compartment in compartments
        ]
        edges = [
            GraphEdge(
                source=flow.source,
                target=flow.target,
                label=f"{flow.transition}: {flow.term}",
            )
            for flow in self._transition_flows()
        ]
        return StateGraph(
            heading="Expanded compartments",
            name="compartments",
            nodes=nodes,
            edges=edges,
            expanded=True,
        )

    # ------------------------------------------------------------------
    # Compact form
    # ------------------------------------------------------------------

    def compact_form(self) -> list[Block]:
        """
        Each transition, with the flows it creates between compartments.

        Transitions are grouped as the definition declares them: between
        bins, or between the categories of one stratification. The last block
        gives the size of the system.

        Returns
        -------
        list[Block]
            One `Equations` block per group, then a `Text` with the size.
        """
        process = self._model.process
        units = self._variable_units()

        flows_by_transition: dict[str, list[TransitionFlow]] = {}
        for flow in self._transition_flows():
            flows_by_transition.setdefault(flow.transition, []).append(flow)

        bin_ids = {bin_item.id for bin_item in process.bins}
        groups: list[tuple[str, list["Transition"]]] = [("Bin Transitions", [])]
        by_stratification: dict[str, list["Transition"]] = {}
        for transition in process.transitions:
            states = set(transition.source) | set(transition.target)
            if states <= bin_ids:
                groups[0][1].append(transition)
                continue
            for strat in process.stratifications:
                if states <= set(strat.categories):
                    by_stratification.setdefault(strat.id, []).append(transition)
                    break
        groups.extend(
            (f"{strat_id.capitalize()} Stratification Transitions", transitions)
            for strat_id, transitions in by_stratification.items()
        )

        blocks: list[Block] = []
        for title, transitions in groups:
            if not transitions:
                continue
            blocks.append(
                Equations(
                    heading="Compact form" if not blocks else None,
                    title=title,
                    groups=[
                        self._transition_group(
                            transition,
                            flows_by_transition.get(transition.id, []),
                            units,
                        )
                        for transition in transitions
                    ],
                )
            )
        blocks.append(Text(text=self._system_size()))
        return blocks

    def _variable_units(self) -> dict[str, str] | None:
        """The unit of every variable, or None when the model has no units."""
        if not self._helper.has_all_units():
            return None
        return self._helper.build_variable_units()

    def _transition_group(
        self,
        transition: "Transition",
        flows: list[TransitionFlow],
        units: dict[str, str] | None,
    ) -> EquationGroup:
        source_str = ", ".join(sorted(set(transition.source))) or "none"
        target_str = ", ".join(sorted(set(transition.target))) or "none"
        schedule = f", {_impulse_schedule(transition)}" if transition.is_impulse else ""
        return EquationGroup(
            heading=f"{transition.id} ({source_str} -> {target_str}{schedule})",
            equations=[
                Equation(
                    left=_text_arrow(flow),
                    right=self._helper.format_rate_with_unit(flow.term, units),
                    left_latex=_latex_transition_arrow(
                        _display_name(flow.source, latex=True),
                        _display_name(flow.target, latex=True),
                    ),
                    right_latex=self._helper.format_rate_with_unit(
                        flow.term, units, latex=True
                    ),
                    relation="flow",
                )
                for flow in flows
            ],
        )

    def _system_size(self) -> str:
        process = self._model.process
        num_bins = len(process.bins)
        if not process.stratifications:
            return f"Total System: {num_bins} coupled equations ({num_bins} bins)"
        compartments, _ = engine_state.expanded_outputs(process)
        num_compartments = len(compartments)
        strat_details = [
            f"{len(strat.categories)} {strat.id}" for strat in process.stratifications
        ]
        return (
            f"Total System: {num_compartments} coupled equations "
            f"({num_bins} bins × {' × '.join(strat_details)})"
        )

    # ------------------------------------------------------------------
    # Expanded form
    # ------------------------------------------------------------------

    def expanded_form(self) -> Equations:
        """
        One equation per compartment: its inflows minus its outflows.

        Returns
        -------
        Equations
            One group with the equation of every compartment, in engine
            order.
        """
        compartments, _ = engine_state.expanded_outputs(self._model.process)
        # Net coefficient of each flow term per compartment, in first-seen
        # order: a piece that leaves and enters the same compartment cancels,
        # and a compartment a flow enters twice gains it twice.
        net: dict[str, dict[str, int]] = {
            compartment.name: {} for compartment in compartments
        }
        impulse_ids = {t.id for t in self._model.process.transitions if t.is_impulse}
        for flow in self._transition_flows():
            if flow.transition in impulse_ids:
                continue
            for name, sign in ((flow.source, -1), (flow.target, 1)):
                if name is not None:
                    terms = net[name]
                    terms[flow.term] = terms.get(flow.term, 0) + sign

        engine_type = self._model.engine.type
        equations = []
        for compartment in compartments:
            raw_terms = [
                (
                    1 if count > 0 else -1,
                    term if abs(count) == 1 else f"{abs(count)} * {term}",
                )
                for term, count in net[compartment.name].items()
                if count != 0
            ]
            equations.append(
                Equation(
                    left=_equation_lhs(compartment.name, False, engine_type),
                    right=_format_equation(raw_terms, latex=False),
                    left_latex=_equation_lhs(
                        latex_variable(compartment.name), True, engine_type
                    ),
                    right_latex=_format_equation(raw_terms, latex=True),
                )
            )
        return Equations(
            heading="Expanded form", groups=[EquationGroup(equations=equations)]
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def events(self, continuous: bool) -> list[Block]:
        """
        The event table: one row per flow, with the change of one event and
        its propensity (continuous time) or its draw (discrete time).

        An event moves one unit: it takes one from each source of the flow
        and adds one to each target. The pieces of a flow with several
        sources or targets are one event.

        Parameters
        ----------
        continuous : bool
            Whether the engine runs in continuous time, where a flow has a
            propensity, or steps, where it has a draw.

        Returns
        -------
        list[Block]
            The table, and the rule of the draws of a step.
        """
        impulses = {t.id for t in self._model.process.transitions if t.is_impulse}
        flows: dict[int, list[TransitionFlow]] = {}
        for flow in self._transition_flows():
            if flow.transition not in impulses:
                flows.setdefault(flow.flow, []).append(flow)
        rows: list[list[Cell]] = []
        for pieces in flows.values():
            first = pieces[0]
            change: dict[str, int] = {}
            for piece in pieces:
                if piece.source is not None:
                    change[piece.source] = change.get(piece.source, 0) - 1
                if piece.target is not None:
                    change[piece.target] = change.get(piece.target, 0) + 1
            rows.append(
                [
                    first.transition,
                    ", ".join(
                        f"{name} {units:+d}" for name, units in change.items() if units
                    )
                    or "none",
                    first.term if continuous else _draw(pieces),
                ]
            )
        table = Table(
            heading="Events",
            columns=["Transition", "Change", "Propensity" if continuous else "Draw"],
            rows=rows,
        )
        if continuous:
            rule = (
                "Each event moves one unit. A flow has no events while one of its "
                "sources holds fewer units than an event takes."
            )
        else:
            rule = (
                "The draws of a step read the state at its start. The proportional "
                "flows out of one compartment are one multinomial draw, so a unit "
                "leaves by one flow at most; an absolute flow is capped by what its "
                "sources still hold, in the order of the table."
            )
        return [table, Text(text=rule)]

    def impulses(self) -> Equations | None:
        """
        Each impulse transition: where it acts and the amounts it moves.

        Returns
        -------
        Equations | None
            One group per impulse transition, or None when there is none.
        """
        transitions = [t for t in self._model.process.transitions if t.is_impulse]
        if not transitions:
            return None
        groups = []
        for transition in transitions:
            equations = [
                Equation(
                    left=_text_arrow(flow),
                    right=flow.term,
                    left_latex=_latex_transition_arrow(
                        _display_name(flow.source, latex=True),
                        _display_name(flow.target, latex=True),
                    ),
                    right_latex=latex_rate_expression(flow.term),
                    relation="flow",
                )
                for flow in self._transition_flows()
                if flow.transition == transition.id
            ]
            groups.append(
                EquationGroup(
                    heading=f"{transition.id}, {_impulse_schedule(transition)}",
                    equations=equations,
                )
            )
        return Equations(heading="Impulses", groups=groups)


# ======================================================================
# Module-level functions (no model dependency)
# ======================================================================


def _rate_text(transition: "Transition") -> str:
    """The rate of a transition as written, or ``stratified`` when it has none."""
    if transition.rate is not None:
        return str(transition.rate)
    if transition.stratified_rates:
        return "stratified rates"
    return "no rate"


def _draw(pieces: list[TransitionFlow]) -> str:
    """How many units a flow moves in a step of the chain binomial."""
    first = pieces[0]
    if first.proportional_to is not None:
        return f"Binomial({first.proportional_to}, {first.rate})"
    sources = [piece.source for piece in pieces if piece.source is not None]
    if not sources:
        return f"Poisson({first.rate})"
    return f"Poisson({first.rate}), at most {', '.join(dict.fromkeys(sources))}"


def _text_arrow(flow: TransitionFlow) -> str:
    """Where a flow moves an amount, as ``A -> B``."""
    return f"{_display_name(flow.source)} -> {_display_name(flow.target)}"


def _display_name(name: str | None, latex: bool = False) -> str:
    if name is None:
        return "none"
    return latex_variable(name) if latex else name


def _format_equation(raw_terms: list[tuple[int, str]], latex: bool) -> str:
    terms = [
        f"{'+' if sign > 0 else '-'} ({latex_rate_expression(rate) if latex else rate})"
        for sign, rate in simplify_complementary_terms(raw_terms)
    ]
    if not terms:
        return "0"
    equation = " ".join(terms)
    return equation[2:] if equation.startswith("+ ") else equation


def _impulse_schedule(transition) -> str:
    """Where an impulse acts, in words: ``impulse at 1, 3`` or ``every 7 from 0``."""
    if transition.impulse_at is not None:
        return "impulse at " + ", ".join(f"{p:g}" for p in transition.impulse_at)
    period, offset = transition.impulse_every
    return f"impulse every {period:g} from {offset:g}"


def _equation_lhs(variable_name: str, latex: bool, engine_type: str) -> str:
    """The left-hand side of an equation, as the engine of the model writes it."""
    lhs = commol_rs.engine.EngineType(engine_type).equation_lhs(variable_name, latex)
    if lhs is None:
        raise ValueError(
            f"The {engine_type} engine has no equations: its dynamics are "
            f"{commol_rs.engine.EngineType(engine_type).dynamics_form}"
        )
    return lhs


def _latex_transition_arrow(source: str, target: str) -> str:
    """Format transition arrow for LaTeX."""
    formatted_source = "\\varnothing" if source == "none" else source
    formatted_target = "\\varnothing" if target == "none" else target
    return f"{formatted_source} \\to {formatted_target}"


# Greek letters commonly used in mathematical modeling
_GREEK_LETTERS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
}


def latex_variable(var_name: str) -> str:
    """Format variable name for LaTeX math mode."""
    if var_name in _GREEK_LETTERS:
        return f"\\{var_name}"

    if "_" in var_name:
        parts = var_name.split("_")
        base = parts[0]
        subscripts = parts[1:]

        if base in _GREEK_LETTERS:
            base = f"\\{base}"

        subscript_str = ",".join(subscripts)
        return f"{base}_{{{subscript_str}}}"

    return var_name


def latex_rate_expression(rate: str, units: Mapping[str, str] | None = None) -> str:
    """
    Convert a rate expression to LaTeX math mode.

    With `units`, each variable that has one is followed by it, as in
    ``k(\\text{1/day})``.
    """

    def latex(var: str) -> str:
        latex_var = latex_variable(var)
        if units is not None and var in units:
            latex_var += f"(\\text{{{units[var]}}})"
        return latex_var

    latex_rate = replace_variables(rate.replace(" * ", " \\cdot "), rate, latex)
    return _convert_division_to_frac(latex_rate)


def replace_variables(text: str, rate: str, replace: Callable[[str], str]) -> str:
    """
    `text` with each variable of `rate` replaced by ``replace(name)``.

    One pass replaces every variable, so a replacement is never read again:
    the category ``g1`` inside ``A_{g1}``, which replaces ``A_g1``, stays.
    """
    variables = engine_state.formula_inputs(rate)
    if not variables:
        return text
    names = "|".join(re.escape(variable) for variable in variables)
    return re.sub(rf"\b({names})\b", lambda match: replace(match.group(1)), text)


def _convert_division_to_frac(expr: str) -> str:
    """Convert division operations to LaTeX fractions."""
    if " / " not in expr:
        return expr

    divisions = []
    paren_depth = 0
    i = 0

    while i < len(expr):
        if expr[i] == "(":
            paren_depth += 1
        elif expr[i] == ")":
            paren_depth -= 1
        elif paren_depth == 0 and expr[i : i + 3] == " / ":
            divisions.append(i)
            i += 2
        i += 1

    if not divisions:
        return expr

    parts = []
    start = 0
    for div_pos in divisions:
        parts.append(expr[start:div_pos].strip())
        start = div_pos + 3
    parts.append(expr[start:].strip())

    result = _strip_outer_parens(parts[0])
    for i in range(1, len(parts)):
        denominator = _strip_outer_parens(parts[i])
        result = f"\\frac{{{result}}}{{{denominator}}}"

    return result


def _strip_outer_parens(expr: str) -> str:
    """Remove outer parentheses if they wrap the entire expression."""
    expr = expr.strip()
    if not expr.startswith("(") or not expr.endswith(")"):
        return expr

    depth = 0
    for i, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

        if depth == 0 and i < len(expr) - 1:
            return expr

    return expr[1:-1]


# ------------------------------------------------------------------
# Complementary term simplification
# ------------------------------------------------------------------


def tokenize_rate(rate: str) -> list[str]:
    """Tokenize a rate expression by ``*`` respecting parentheses."""
    tokens: list[str] = []
    depth = 0
    current_start = 0
    i = 0
    while i < len(rate):
        if rate[i] == "(":
            depth += 1
        elif rate[i] == ")":
            depth -= 1
        elif depth == 0 and rate[i : i + 3] == " * ":
            tokens.append(rate[current_start:i].strip())
            i += 3
            current_start = i
            continue
        i += 1
    tokens.append(rate[current_start:].strip())
    return tokens


def are_complementary_factors(varying: list[str]) -> bool:
    """Check if a list of varying tokens are complementary percentages."""
    if len(varying) < 2:
        return False

    complement_idx = None
    for i, token in enumerate(varying):
        if token.startswith("(1 - ") and token.endswith(")"):
            complement_idx = i
            break

    if complement_idx is None:
        return False

    complement_token = varying[complement_idx]
    other_tokens = [t for i, t in enumerate(varying) if i != complement_idx]

    expected = "(1 - " + " - ".join(sorted(other_tokens)) + ")"

    # The token starts with "(1 - ", so its first part is "1".
    parts = [p.strip() for p in complement_token[1:-1].split(" - ")]
    actual_sorted = "(1 - " + " - ".join(sorted(parts[1:])) + ")"

    return actual_sorted == expected


def simplify_complementary_terms(
    raw_terms: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """
    Merge the terms of one sign that differ only by complementary factors.

    ``k * p`` and ``k * (1 - p)`` share the factor ``k``, and ``p`` and
    ``(1 - p)`` are complementary, so they merge into ``k``. A merged term
    takes the place of the first term of its group.
    """
    merged: dict[int, tuple[int, str]] = {}
    used: set[int] = set()
    for sign in (-1, 1):
        # The terms that share the same other factors, by those factors.
        groups: dict[tuple[str, ...], list[tuple[int, str, list[str]]]] = {}
        for i, (term_sign, rate) in enumerate(raw_terms):
            if term_sign != sign:
                continue
            tokens = tokenize_rate(rate)
            for k, varying in enumerate(tokens):
                common = tokens[:k] + tokens[k + 1 :]
                if common:
                    groups.setdefault(tuple(sorted(common)), []).append(
                        (i, varying, common)
                    )
        for members in groups.values():
            # A term already merged, or a repeat of a factor, stays out.
            free: dict[str, tuple[int, str, list[str]]] = {}
            for member in members:
                if member[0] not in used:
                    free.setdefault(member[1], member)
            if len(free) >= 2 and are_complementary_factors(list(free)):
                first, _, common = next(iter(free.values()))
                merged[first] = (sign, " * ".join(common))
                used.update(i for i, _, _ in free.values())
    return [
        merged.get(i, term)
        for i, term in enumerate(raw_terms)
        if i not in used or i in merged
    ]
