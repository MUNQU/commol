import re

from commol.constants import ModelTypes, PrintEquationsOutputFormat
from commol.context._model import engine_state
from commol.context._model.engine_state import TransitionFlow
from commol.context._model.helpers import ModelCompartmentHelper
from commol.utils.security import get_expression_variables


class EquationPrinter:
    """
    Formats and prints model equations in text or LaTeX.

    The flows come from the engine, so the printed equations are the ones the
    engine simulates: each flow moves population from one compartment to
    another, and a per-capita rate is shown multiplied by its source.
    """

    def __init__(self, model) -> None:
        self._model = model
        self._helper = ModelCompartmentHelper(model)
        self._flows: list[TransitionFlow] | None = None

    def _transition_flows(self) -> list[TransitionFlow]:
        if self._flows is None:
            self._flows = engine_state.transition_flows(self._model)
        return self._flows

    def print_equations(
        self,
        output_file: str | None = None,
        format: str = PrintEquationsOutputFormat.TEXT,
    ) -> None:
        """Print the equations of the model in mathematical form."""
        if format not in PrintEquationsOutputFormat:
            raise ValueError(
                f"Invalid format: {format}. "
                f"Must be one of {list(PrintEquationsOutputFormat)}"
            )

        lines = self._generate_model_header()
        lines.extend(self._generate_compact_form(format=format))
        lines.append("")
        lines.extend(self._generate_expanded_form(format=format))

        output = "\n".join(lines)
        _write_output(output, output_file)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _generate_model_header(self) -> list[str]:
        """Generate the header lines with model metadata."""
        lines: list[str] = []
        lines.append("=" * 40)
        lines.append("MODEL INFORMATION")
        lines.append("=" * 40)
        lines.append(f"Model: {self._model.name}")
        lines.append(f"Model Type: {self._model.dynamics.typology}")
        lines.append(f"Number of Bins: {len(self._model.population.bins)}")
        lines.append(
            f"Number of Stratifications: {len(self._model.population.stratifications)}"
        )
        lines.append(f"Number of Parameters: {len(self._model.parameters)}")
        lines.append(f"Number of Transitions: {len(self._model.dynamics.transitions)}")

        bin_ids = [bin_item.id for bin_item in self._model.population.bins]
        lines.append(f"Bins: {', '.join(bin_ids)}")

        if self._model.population.stratifications:
            lines.append("Stratifications:")
            for strat in self._model.population.stratifications:
                categories = ", ".join(strat.categories)
                lines.append(f"  - {strat.id}: [{categories}]")

        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # Compact form
    # ------------------------------------------------------------------

    def _generate_compact_form(self, format: str) -> list[str]:
        """Each transition, with the flows it creates between compartments."""
        lines: list[str] = []
        lines.append("=" * 40)
        lines.append("COMPACT FORM")
        lines.append("=" * 40)
        lines.append("")

        show_units = self._helper.has_all_units()
        variable_units = self._helper.build_variable_units() if show_units else None

        flows_by_transition: dict[str, list[TransitionFlow]] = {}
        for flow in self._transition_flows():
            flows_by_transition.setdefault(flow.transition, []).append(flow)

        # Transitions are grouped as the definition declares them: between
        # bins, or between the categories of one stratification.
        bin_ids = {bin_item.id for bin_item in self._model.population.bins}
        groups: list[tuple[str, list]] = [("Bin Transitions:", [])]
        by_stratification: dict[str, list] = {}
        for transition in self._model.dynamics.transitions:
            states = set(transition.source) | set(transition.target)
            if states <= bin_ids:
                groups[0][1].append(transition)
                continue
            for strat in self._model.population.stratifications:
                if states <= set(strat.categories):
                    by_stratification.setdefault(strat.id, []).append(transition)
                    break
        groups.extend(
            (f"{strat_id.capitalize()} Stratification Transitions:", transitions)
            for strat_id, transitions in by_stratification.items()
        )

        for heading, transitions in groups:
            if not transitions:
                continue
            lines.append(heading)
            for transition in transitions:
                lines.extend(
                    self._format_transition_flows(
                        transition,
                        flows_by_transition.get(transition.id, []),
                        variable_units,
                        show_units,
                        format,
                    )
                )

        num_compartments = len(self._helper.generate_compartments())
        lines.extend(
            self._format_total_system_size(
                [bin_item.id for bin_item in self._model.population.bins],
                num_compartments,
            )
        )

        return lines

    def _format_transition_flows(
        self,
        transition,
        flows: list[TransitionFlow],
        variable_units: dict[str, str] | None,
        show_units: bool,
        format: str,
    ) -> list[str]:
        source_str = ", ".join(sorted(set(transition.source))) or "none"
        target_str = ", ".join(sorted(set(transition.target))) or "none"
        lines = [f"{transition.id} ({source_str} -> {target_str}):"]
        if not flows:
            # The engine creates no flow, for example when the rate is missing.
            lines.append("  (no flows)")
        for flow in flows:
            source = self._display_name(flow.source, format)
            target = self._display_name(flow.target, format)
            term = self._helper.format_rate_with_unit(
                flow.term, variable_units, show_units, format
            )
            if format == PrintEquationsOutputFormat.LATEX:
                arrow = _latex_transition_arrow(source, target)
                lines.append(f"  ${arrow}: {term}$")
            else:
                lines.append(f"  {source} -> {target}: {term}")
        lines.append("")
        return lines

    def _format_total_system_size(
        self, bin_ids: list[str], num_compartments: int
    ) -> list[str]:
        num_bins = len(bin_ids)
        if not self._model.population.stratifications:
            return [f"Total System: {num_bins} coupled equations ({num_bins} bins)"]

        strat_details = [
            f"{len(strat.categories)} {strat.id}"
            for strat in self._model.population.stratifications
        ]
        return [
            f"Total System: {num_compartments} coupled equations "
            f"({num_bins} bins × {' × '.join(strat_details)})"
        ]

    # ------------------------------------------------------------------
    # Expanded form
    # ------------------------------------------------------------------

    def _generate_expanded_form(self, format: str) -> list[str]:
        """One equation per compartment: its inflows minus its outflows."""
        lines: list[str] = []
        lines.append("=" * 40)
        lines.append("EXPANDED FORM")
        lines.append("=" * 40)

        compartments, _ = engine_state.expanded_outputs(self._model)
        # Net coefficient of each flow term per compartment, in first-seen
        # order: a piece that leaves and enters the same compartment cancels,
        # and a compartment a flow enters twice gains it twice.
        net: dict[str, dict[str, int]] = {
            compartment.name: {} for compartment in compartments
        }
        for flow in self._transition_flows():
            for name, sign in ((flow.source, -1), (flow.target, 1)):
                if name is not None:
                    terms = net[name]
                    terms[flow.term] = terms.get(flow.term, 0) + sign

        for compartment in compartments:
            raw_terms = [
                (
                    1 if count > 0 else -1,
                    term if abs(count) == 1 else f"{abs(count)} * {term}",
                )
                for term, count in net[compartment.name].items()
                if count != 0
            ]
            equation = self._format_equation(raw_terms, format)
            lhs = _get_equation_lhs(
                self._display_name(compartment.name, format),
                format,
                self._model.dynamics.typology,
            )
            if format == PrintEquationsOutputFormat.LATEX:
                lines.append(f"\\[{lhs} = {equation}\\]")
            else:
                lines.append(f"{lhs} = {equation}")

        return lines

    def _format_equation(self, raw_terms: list[tuple[int, str]], format: str) -> str:
        terms = [
            f"{'+' if sign > 0 else '-'} ({self._format_rate_expr(rate, format)})"
            for sign, rate in simplify_complementary_terms(raw_terms)
        ]
        if not terms:
            return "0"
        equation = " ".join(terms)
        return equation[2:] if equation.startswith("+ ") else equation

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _display_name(name: str | None, format: str) -> str:
        if name is None:
            return "none"
        if format == PrintEquationsOutputFormat.LATEX:
            return latex_variable(name)
        return name

    @staticmethod
    def _format_rate_expr(rate: str, format: str) -> str:
        if format == PrintEquationsOutputFormat.LATEX:
            return latex_rate_expression(rate)
        return rate


# ======================================================================
# Module-level functions (no model dependency)
# ======================================================================


def _write_output(output: str, output_file: str | None) -> None:
    """Write output to file or console."""
    if output_file:
        with open(output_file, "w") as f:
            _ = f.write(output)
    else:
        print(output)


def _get_equation_lhs(variable_name: str, format: str, typology: str) -> str:
    """Get the left-hand side of an equation based on model type."""
    if format == PrintEquationsOutputFormat.LATEX:
        if typology == ModelTypes.DIFFERENTIAL_EQUATIONS:
            return f"\\frac{{d{variable_name}}}{{dt}}"
        else:
            return (
                f"\\frac{{{variable_name}(t+\\Delta t) - "
                f"{variable_name}(t)}}{{\\Delta t}}"
            )
    else:
        if typology == ModelTypes.DIFFERENTIAL_EQUATIONS:
            return f"d{variable_name}/dt"
        else:
            return f"[{variable_name}(t+Dt) - {variable_name}(t)] / Dt"


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


def latex_rate_expression(rate: str) -> str:
    """Convert rate expression to LaTeX math mode."""
    if not rate:
        return rate

    latex_rate = rate.replace(" * ", " \\cdot ")

    variables = get_expression_variables(rate)
    sorted_vars = sorted(variables, key=lambda x: len(x), reverse=True)

    for var in sorted_vars:
        latex_var = latex_variable(var)
        pattern = r"\b" + re.escape(var) + r"\b"
        latex_rate = re.sub(pattern, latex_var.replace("\\", "\\\\"), latex_rate)

    # Wrap units in \text{} — exclude content with LaTeX commands or math operators
    latex_rate = re.sub(
        r"\(([^)\\{}+\-]*[a-zA-Z][^)\\{}+\-]*)\)", r"(\\text{\1})", latex_rate
    )
    latex_rate = re.sub(
        r"\[([^\]\\{}+\-]*[a-zA-Z][^\]\\{}+\-]*)\]", r"[\\text{\1}]", latex_rate
    )

    latex_rate = _convert_division_to_frac(latex_rate)

    return latex_rate


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
        elif paren_depth == 0 and i + 3 <= len(expr) and expr[i : i + 3] == " / ":
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
        elif depth == 0 and i + 3 <= len(rate) and rate[i : i + 3] == " * ":
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

    inner = complement_token[1:-1]
    parts = [p.strip() for p in inner.split(" - ")]
    if not parts or parts[0] != "1":
        return False
    actual_sorted = "(1 - " + " - ".join(sorted(parts[1:])) + ")"

    return actual_sorted == expected


def simplify_complementary_terms(
    raw_terms: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Simplify groups of terms with complementary percentage factors."""
    if len(raw_terms) < 2:
        return raw_terms

    result: list[tuple[int, str]] = []
    used: set[int] = set()

    for sign_val in (-1, 1):
        indices = [
            i for i, (s, _) in enumerate(raw_terms) if s == sign_val and i not in used
        ]
        if len(indices) < 2:
            continue

        groups = find_common_factor_groups([raw_terms[i][1] for i in indices])

        for group_indices, common_tokens in groups:
            actual_indices = [indices[gi] for gi in group_indices]
            varying = []
            for gi in group_indices:
                rate_tokens = tokenize_rate(raw_terms[indices[gi]][1])
                diff = [t for t in rate_tokens if t not in common_tokens]
                if len(diff) == 1:
                    varying.append(diff[0])
                else:
                    varying = []
                    break

            if varying and are_complementary_factors(varying):
                simplified = " * ".join(common_tokens)
                result.append((sign_val, simplified))
                used.update(actual_indices)

    for i, term in enumerate(raw_terms):
        if i not in used:
            result.append(term)

    return _restore_term_order(raw_terms, result, used)


def find_common_factor_groups(
    rates: list[str],
) -> list[tuple[list[int], list[str]]]:
    """Find groups of rates that share common token factors."""
    tokenized = [tokenize_rate(r) for r in rates]
    groups: list[tuple[list[int], list[str]]] = []
    used_in_group: set[int] = set()

    for i in range(len(rates)):
        if i in used_in_group:
            continue

        group = _build_candidate_group(i, tokenized, used_in_group)

        if len(group) >= 2:
            common = _validate_group(group, tokenized)
            if common:
                groups.append((group, common))
                used_in_group.update(group)

    return groups


def _build_candidate_group(
    i: int,
    tokenized: list[list[str]],
    used: set[int],
) -> list[int]:
    group = [i]
    tokens_i = tokenized[i]

    for j in range(i + 1, len(tokenized)):
        if j in used:
            continue
        tokens_j = tokenized[j]
        common = [t for t in tokens_i if t in tokens_j]
        diff_i = [t for t in tokens_i if t not in common]
        diff_j = [t for t in tokens_j if t not in common]
        if len(diff_i) == 1 and len(diff_j) == 1 and common:
            group.append(j)

    return group


def _validate_group(
    group: list[int],
    tokenized: list[list[str]],
) -> list[str] | None:
    common = [t for t in tokenized[group[0]] if t in tokenized[group[1]]]
    for idx in group:
        diff = [t for t in tokenized[idx] if t not in common]
        if len(diff) != 1:
            return None
    return common


def _restore_term_order(
    original: list[tuple[int, str]],
    result: list[tuple[int, str]],
    used: set[int],
) -> list[tuple[int, str]]:
    """Restore term ordering after simplification."""
    if not used:
        return result

    original_set = set(original)
    simplified = [t for t in result if t not in original_set]
    non_simplified = [t for t in result if t in original_set]

    ordered: list[tuple[int, str]] = []
    simplified_iter = iter(simplified)
    non_simplified_iter = iter(non_simplified)
    simplified_placed = False

    for i, _term in enumerate(original):
        if i in used:
            if not simplified_placed:
                s = next(simplified_iter, None)
                if s is not None:
                    ordered.append(s)
                simplified_placed = True
        else:
            ns = next(non_simplified_iter, None)
            if ns is not None:
                ordered.append(ns)

    for s in simplified_iter:
        ordered.append(s)
    for ns in non_simplified_iter:
        ordered.append(ns)

    return ordered
