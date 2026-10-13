"""Model factories, a reference solver and assertions shared by the tests."""

from .assertions import ENGINE_REL, assert_close, assert_population_conserved
from .models import (
    a_to_b_builder,
    build,
    conditional_pathway_builder,
    override_builder,
    seir_builder,
    shared_category_builder,
    sir_builder,
    stratify,
)
from .problems import bounded, initial_condition_problem, observed
from .reference import SIR_START, reference_run, sir_flows, total
from .simulation import initial_population, run

__all__ = [
    "ENGINE_REL",
    "SIR_START",
    "a_to_b_builder",
    "assert_close",
    "assert_population_conserved",
    "bounded",
    "build",
    "conditional_pathway_builder",
    "initial_condition_problem",
    "initial_population",
    "observed",
    "override_builder",
    "reference_run",
    "run",
    "seir_builder",
    "shared_category_builder",
    "sir_builder",
    "sir_flows",
    "stratify",
    "total",
]
