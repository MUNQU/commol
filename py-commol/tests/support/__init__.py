"""Model factories, a reference solver, oracles and assertions shared by the tests."""

from .assertions import (
    ENGINE_REL,
    ODE_REL,
    assert_close,
    assert_total_conserved,
)
from .flows import Flow, flow_model, increments
from .models import (
    a_to_b_builder,
    TIGHT,
    build,
    build_ode,
    conditional_pathway_builder,
    override_builder,
    seir_builder,
    shared_category_builder,
    sir_builder,
    stratify,
)
from .oracles import (
    assert_threshold,
    bisect,
    final_size,
    growth_ratio,
    settled,
    total_at,
)
from .problems import bounded, initial_condition_problem, observed
from .reference import SIR_START, reference_run, sir_flows, total
from .simulation import initial_state, run

__all__ = [
    "ENGINE_REL",
    "Flow",
    "ODE_REL",
    "SIR_START",
    "a_to_b_builder",
    "assert_close",
    "assert_total_conserved",
    "assert_threshold",
    "bisect",
    "bounded",
    "TIGHT",
    "build",
    "build_ode",
    "conditional_pathway_builder",
    "final_size",
    "flow_model",
    "growth_ratio",
    "increments",
    "initial_condition_problem",
    "initial_state",
    "observed",
    "override_builder",
    "reference_run",
    "run",
    "seir_builder",
    "settled",
    "shared_category_builder",
    "sir_builder",
    "sir_flows",
    "stratify",
    "total",
    "total_at",
]
