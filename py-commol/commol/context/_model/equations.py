"""The dynamics of an aggregate model, as blocks of its description.

The flows come from the engine (``commol_rs.model.transition_flows``), so the
equations a description shows are the ones the engine simulates: each flow
moves an amount from one compartment to another, and a proportional rate is
shown multiplied by its source. An impulse is not part of the equations: it is
listed apart, with the positions where it moves its amount at once.

Each rate and each flow is written from the tree of the engine
(``engine_state.formula_tree``), never from its text: a rate gives the same
description with or without spaces.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from commol.commol_rs import _commol_rs as commol_rs

from commol.context._model import engine_state
from commol.context._model.engine_state import ExpandedOutput, TransitionFlow
from commol.context._model.formula_tree import (
    PRIMARY,
    PRODUCT,
    Chain,
    Constant,
    FormulaWriter,
    Node,
    binding,
    formula_text,
    product,
    variables,
)
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


type _Categories = tuple[tuple[str, str], ...]
"""The categories of a node of the graph of the strata, as (stratification,
category) pairs in the order of the stratifications."""


@dataclass
class _Moves:
    """
    What moves along one edge of a state graph.

    Attributes
    ----------
    transitions : dict[str, None]
        The ids of the transitions, in the order of the flows.
    rates : dict[str, None]
        The rates the engine evaluates for the flows of the edge.
    bins : set[str]
        The bins that move, for an edge between categories.
    """

    transitions: dict[str, None] = field(default_factory=dict)
    rates: dict[str, None] = field(default_factory=dict)
    bins: set[str] = field(default_factory=set)

    def add(self, flow: TransitionFlow, bin_id: str | None = None) -> None:
        self.transitions[flow.transition] = None
        self.rates[flow.rate] = None
        if bin_id is not None:
            self.bins.add(bin_id)


class _GraphMoves:
    """
    The edges of the graph of the bins and of the graph of the strata, with
    what moves along each.

    Attributes
    ----------
    bins : dict[tuple[str | None, str | None], _Moves]
        The edges between bins, by source and target; None is the outside.
    strata : dict[tuple[_Categories | None, _Categories | None], _Moves]
        The edges between categories, by source and target.
    """

    def __init__(self, stratifications: list[str]) -> None:
        self._stratifications = stratifications
        self.bins: dict[tuple[str | None, str | None], _Moves] = {}
        self.strata: dict[tuple[_Categories | None, _Categories | None], _Moves] = {}

    def add_piece(
        self,
        flow: TransitionFlow,
        leaves: ExpandedOutput | None,
        enters: ExpandedOutput | None,
    ) -> None:
        """
        A piece of `flow` from the compartment `leaves` to `enters`.

        A piece that changes categories is an edge between the categories it
        changes. A piece is an edge between bins unless it stays in one bin
        and changes categories.
        """
        changed = []
        if leaves is not None and enters is not None:
            changed = [
                strat
                for strat in self._stratifications
                if leaves.categories.get(strat) != enters.categories.get(strat)
            ]
            if changed:
                pair = (
                    _categories(leaves.categories, changed),
                    _categories(enters.categories, changed),
                )
                self.strata.setdefault(pair, _Moves()).add(flow, leaves.source)
        ends = (
            None if leaves is None else leaves.source,
            None if enters is None else enters.source,
        )
        if ends[0] != ends[1] or not changed:
            self.bins.setdefault(ends, _Moves()).add(flow)

    def add_declared(
        self, transition: "Transition", stratification: str | None
    ) -> None:
        """
        The edges of a transition with no flow, from its sources to its
        targets: bins, or the categories of `stratification`.
        """
        for source in dict.fromkeys(transition.source) or [None]:
            for target in dict.fromkeys(transition.target) or [None]:
                if stratification is None:
                    moves = self.bins.setdefault((source, target), _Moves())
                else:
                    pair = (
                        None if source is None else ((stratification, source),),
                        None if target is None else ((stratification, target),),
                    )
                    moves = self.strata.setdefault(pair, _Moves())
                moves.transitions[transition.id] = None


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
        self._moves: _GraphMoves | None = None

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
            The graph of the bins, the graph of the strata when the model
            moves amounts between categories, the graph of the compartments,
            the compact form, the expanded form when the engine has equations
            or the events when it draws them, and the impulses.
        """
        strata = self.strata_graph()
        blocks: list[Block] = [
            self.state_graph(),
            *([strata] if strata is not None else []),
            self.expanded_graph(),
        ]
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
        The bins, and one edge per pair of bins that an amount moves between.

        The edges come from the flows of the engine. A flow between two
        compartments of one bin is a move between categories: it is an edge of
        `strata_graph`, not of this graph. An inflow starts at no node and an
        outflow ends at no node. The label of an edge is `_label`.

        Returns
        -------
        StateGraph
            The graph of the bins of the model.
        """
        bins = self._graph_moves().bins
        return StateGraph(
            heading="States and transitions",
            name="states",
            nodes=[
                GraphNode(
                    id=bin_item.id, label=bin_item.name or bin_item.id, group="bin"
                )
                for bin_item in self._model.process.bins
            ],
            edges=[
                GraphEdge(source=source, target=target, label=self._label(moves))
                for (source, target), moves in bins.items()
            ],
        )

    def strata_graph(self) -> StateGraph | None:
        """
        The categories, and one edge per pair of categories that an amount
        moves between, in any bin.

        A node holds the categories of the stratifications that a move
        changes: ``group:g1``, or ``group:g1, size:s1`` for a move that changes
        two stratifications at once. The label of an edge names the bins that
        move, in the order of their declaration, then `_label`.

        Returns
        -------
        StateGraph | None
            The graph of the strata, or None when no amount moves between
            categories.
        """
        strata = self._graph_moves().strata
        if not strata:
            return None
        order = {
            (strat.id, category): (strat_index, category_index)
            for strat_index, strat in enumerate(self._model.process.stratifications)
            for category_index, category in enumerate(strat.categories)
        }
        ends = {end for pair in strata for end in pair if end is not None}
        nodes = [
            GraphNode(
                id=_strata_id(categories),
                label=", ".join(category for _, category in categories),
                group=", ".join(strat for strat, _ in categories),
            )
            for categories in sorted(
                ends, key=lambda node: [order[pair] for pair in node]
            )
        ]
        edges = []
        for (source, target), moves in strata.items():
            rates = self._label(moves)
            bins = ", ".join(
                bin_item.id
                for bin_item in self._model.process.bins
                if bin_item.id in moves.bins
            )
            edges.append(
                GraphEdge(
                    source=None if source is None else _strata_id(source),
                    target=None if target is None else _strata_id(target),
                    label=": ".join(text for text in (bins, rates) if text),
                )
            )
        return StateGraph(
            heading="Moves between categories",
            name="strata",
            nodes=nodes,
            edges=edges,
        )

    def _graph_moves(self) -> "_GraphMoves":
        """
        The edges of the graph of the bins and of the graph of the strata.

        Each piece of a flow goes from each source of the flow to each target
        (`_GraphMoves.add_piece`). A transition with no flow, such as one with
        no rate, has the edges of its sources and targets.
        """
        if self._moves is not None:
            return self._moves
        process = self._model.process
        compartments, _ = engine_state.expanded_outputs(process)
        found = {compartment.name: compartment for compartment in compartments}
        moves = _GraphMoves([strat.id for strat in process.stratifications])

        pieces: dict[int, list[TransitionFlow]] = {}
        for flow in self._transition_flows():
            pieces.setdefault(flow.flow, []).append(flow)
        for flow_pieces in pieces.values():
            sources = dict.fromkeys(f.source for f in flow_pieces if f.source)
            targets = dict.fromkeys(f.target for f in flow_pieces if f.target)
            for source in sources or [None]:
                for target in targets or [None]:
                    moves.add_piece(
                        flow_pieces[0],
                        None if source is None else found[source],
                        None if target is None else found[target],
                    )

        moved = {flow.transition for flow in self._transition_flows()}
        bin_ids = {bin_item.id for bin_item in process.bins}
        for transition in process.transitions:
            if transition.id in moved:
                continue
            ids = {*transition.source, *transition.target}
            moves.add_declared(
                transition,
                None
                if ids <= bin_ids
                else next(
                    strat.id
                    for strat in process.stratifications
                    if ids <= set(strat.categories)
                ),
            )
        self._moves = moves
        return moves

    def _label(self, moves: _Moves) -> str:
        """
        The rates of an edge: the formula of the rate of its one transition
        when the transition has one rate, and otherwise the parameters that
        the rates of the edge read, in the order of their declaration.
        """
        if len(moves.transitions) == 1:
            (transition_id,) = moves.transitions
            transition = next(
                t for t in self._model.process.transitions if t.id == transition_id
            )
            rates = _declared_rates(transition)
            if len(rates) == 1:
                return _formula(rates[0])
            if not rates:
                return "no rate"
        read = {
            name
            for rate in moves.rates
            for name in variables(engine_state.formula_tree(rate))
        }
        return ", ".join(
            parameter.id for parameter in self._model.parameters if parameter.id in read
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
                label=f"{flow.transition}: {formula_text(flow.term)}",
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
        net: dict[str, dict[Node, int]] = {
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
                    term if abs(count) == 1 else product(Constant(abs(count)), term),
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
                    formula_text(first.term) if continuous else _draw(pieces),
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
                    right=formula_text(flow.term),
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


def _declared_rates(transition: "Transition") -> list[str]:
    """The rates a transition declares: its rate and its stratified rates."""
    rates = [] if transition.rate is None else [transition.rate]
    rates.extend(rate.rate for rate in transition.stratified_rates or [])
    return rates


def _categories(
    categories: Mapping[str, str], changed: list[str]
) -> _Categories | None:
    """The categories of `changed` that a compartment has, or None when it has
    none of them."""
    pairs = tuple(
        (strat, categories[strat]) for strat in changed if strat in categories
    )
    return pairs or None


def _strata_id(categories: _Categories) -> str:
    """The id of a node of the graph of the strata, such as ``group:g1``."""
    return ", ".join(f"{strat}:{category}" for strat, category in categories)


def _formula(formula: str) -> str:
    """A formula as its tree writes it."""
    return formula_text(engine_state.formula_tree(formula))


def _draw(pieces: list[TransitionFlow]) -> str:
    """How many units a flow moves in a step of the chain binomial."""
    first = pieces[0]
    rate = _formula(first.rate)
    if first.proportional_to is not None:
        return f"Binomial({first.proportional_to}, {rate})"
    sources = [piece.source for piece in pieces if piece.source is not None]
    if not sources:
        return f"Poisson({rate})"
    return f"Poisson({rate}), at most {', '.join(dict.fromkeys(sources))}"


def _text_arrow(flow: TransitionFlow) -> str:
    """Where a flow moves an amount, as ``A -> B``."""
    return f"{_display_name(flow.source)} -> {_display_name(flow.target)}"


def _display_name(name: str | None, latex: bool = False) -> str:
    if name is None:
        return "none"
    return latex_variable(name) if latex else name


def _format_equation(raw_terms: list[tuple[int, Node]], latex: bool) -> str:
    terms = [
        f"{'+' if sign > 0 else '-'} "
        f"({latex_rate_expression(rate) if latex else formula_text(rate)})"
        for sign, rate in simplify_complementary_terms(raw_terms)
    ]
    if not terms:
        return "0"
    equation = " ".join(terms)
    return equation[2:] if equation.startswith("+ ") else equation


def _impulse_schedule(transition: "Transition") -> str:
    """Where an impulse acts, in words: ``impulse at 1, 3`` or ``every 7 from 0``."""
    if transition.impulse_every is not None:
        period, offset = transition.impulse_every
        return f"impulse every {period:g} from {offset:g}"
    positions = transition.impulse_at or []
    return "impulse at " + ", ".join(f"{p:g}" for p in positions)


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


class _LatexWriter(FormulaWriter):
    """
    Writes a tree in LaTeX: a product with ``\\cdot``, a division that no
    parenthesis holds as a fraction, a power with its exponent raised, and
    each variable as `latex_variable` writes it.
    """

    SYMBOLS = FormulaWriter.SYMBOLS | {
        "Mul": "\\cdot",
        "Mod": "\\bmod",
        "Le": "\\leq",
        "Ge": "\\geq",
        "Eq": "=",
        "Ne": "\\neq",
    }
    UNARY = {"Neg": "-", "Not": "\\lnot "}

    def chain(self, first: Node, rest: tuple[tuple[str, Node], ...], free: bool) -> str:
        if not free or all(op != "Div" for op, _ in rest):
            return super().chain(first, rest, free)
        # A fraction holds the whole product before it: its numerator and its
        # denominator need no parentheses.
        text = self.operand(first, PRODUCT, free=False)
        alone: Node | None = first
        for op, operand in rest:
            if op == "Div":
                numerator = text if alone is None else self.node(alone, free=False)
                denominator = self.node(operand, free=False)
                text = f"\\frac{{{numerator}}}{{{denominator}}}"
            else:
                text += f" {self.SYMBOLS[op]} {self.operand(operand, PRODUCT, False)}"
            alone = None
        return text

    def power(self, base: Node, exponent: Node) -> str:
        base_text = self.grouped(base, binding(base) < PRIMARY, free=False)
        return f"{base_text}^{{{self.node(exponent, free=False)}}}"


def latex_rate_expression(rate: Node, units: Mapping[str, str] | None = None) -> str:
    """
    The tree of a rate in LaTeX math mode.

    With `units`, each variable that has one is followed by it, as in
    ``k(\\text{1/day})``.
    """

    def latex(var: str) -> str:
        latex_var = latex_variable(var)
        if units is not None and var in units:
            latex_var += f"(\\text{{{units[var]}}})"
        return latex_var

    return _LatexWriter(latex).write(rate)


# ------------------------------------------------------------------
# Complementary term simplification
# ------------------------------------------------------------------


def _factors(rate: Node) -> list[Node]:
    """
    The factors of a product: its operands between two ``*``, each with the
    ``/`` that follow it. A rate that is no product, or a product with
    ``%``, which does not split, is one factor.
    """
    if not isinstance(rate, Chain) or binding(rate) != PRODUCT:
        return [rate]
    if any(op == "Mod" for op, _ in rate.rest):
        return [rate]
    factors: list[list[tuple[str, Node]]] = [[("Mul", rate.first)]]
    for op, operand in rate.rest:
        if op == "Mul":
            factors.append([(op, operand)])
        else:
            factors[-1].append((op, operand))
    return [
        first if not rest else Chain(first, tuple(rest))
        for (_, first), *rest in factors
    ]


def _product_of(factors: list[Node]) -> Node:
    """The product of `factors`, in order."""
    result = factors[0]
    for factor in factors[1:]:
        result = product(result, factor)
    return result


def _one_minus(factor: Node) -> Counter[Node] | None:
    """What ``1 - x - y`` subtracts from 1, or None for another factor."""
    if (
        isinstance(factor, Chain)
        and factor.first == Constant(1.0)
        and all(op == "Sub" for op, _ in factor.rest)
    ):
        return Counter(operand for _, operand in factor.rest)
    return None


def are_complementary_factors(varying: list[Node]) -> bool:
    """Whether one factor is one minus the others: ``p``, ``q`` and
    ``1 - p - q``."""
    if len(varying) < 2:
        return False
    for i, factor in enumerate(varying):
        subtracted = _one_minus(factor)
        if subtracted is not None:
            return subtracted == Counter(varying[:i] + varying[i + 1 :])
    return False


def simplify_complementary_terms(
    raw_terms: list[tuple[int, Node]],
) -> list[tuple[int, Node]]:
    """
    Merge the terms of one sign that differ only by complementary factors.

    ``k * p`` and ``k * (1 - p)`` share the factor ``k``, and ``p`` and
    ``(1 - p)`` are complementary, so they merge into ``k``. A merged term
    takes the place of the first term of its group.
    """
    merged: dict[int, tuple[int, Node]] = {}
    used: set[int] = set()
    for sign in (-1, 1):
        # The terms that share the same other factors, by those factors.
        groups: dict[
            frozenset[tuple[Node, int]], list[tuple[int, Node, list[Node]]]
        ] = {}
        for i, (term_sign, rate) in enumerate(raw_terms):
            if term_sign != sign:
                continue
            factors = _factors(rate)
            for k, varying in enumerate(factors):
                common = factors[:k] + factors[k + 1 :]
                if common:
                    key = frozenset(Counter(common).items())
                    groups.setdefault(key, []).append((i, varying, common))
        for members in groups.values():
            # A term already merged, or a repeat of a factor, stays out.
            free: dict[Node, tuple[int, Node, list[Node]]] = {}
            for member in members:
                if member[0] not in used:
                    free.setdefault(member[1], member)
            if len(free) >= 2 and are_complementary_factors(list(free)):
                first, _, common = next(iter(free.values()))
                merged[first] = (sign, _product_of(common))
                used.update(i for i, _, _ in free.values())
    return [
        merged.get(i, term)
        for i, term in enumerate(raw_terms)
        if i not in used or i in merged
    ]
