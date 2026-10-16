# Commol

[![PyPI](https://img.shields.io/pypi/v/commol)](https://pypi.org/project/commol/)
[![Python](https://img.shields.io/pypi/pyversions/commol)](https://pypi.org/project/commol/)
[![Build](https://github.com/MUNQU/commol/actions/workflows/build.yaml/badge.svg)](https://github.com/MUNQU/commol/actions/workflows/build.yaml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://munqu.github.io/commol/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/MUNQU/commol/blob/main/LICENSE)

**Declarative dynamical modelling in Python.**

In Commol, a model is a description of a system rather than a set of
equations. You state what the system is made of and how it changes: its
states, the flows between them and the rates that drive them. Commol turns
that description into equations, solves them, and fits them to your data.

Writing the description instead of the equations keeps a model small as the
system grows: splitting every state into subgroups takes a few lines, not
dozens of new equations. And since the description says nothing about how it
is solved, the same model runs in discrete steps or continuously, and can be
calibrated against observations without extra code.

## What you can model

Commol supports compartmental models: quantities that flow between states at
rates written as formulas. A quantity can be a count, a mass or a
concentration, and each state can be split into as many subgroups as the
problem needs. This structure is common in epidemiology, pharmacokinetics,
ecology and chemical kinetics, and Commol assumes none of them.

Models are deterministic and evolve along a single variable, usually time.
Calibration is based on optimization, with ensembles of good fits to measure
uncertainty; Commol has no stochastic engine and no Bayesian inference.

## Installation

```bash
pip install commol
```

Commol requires Python 3.12 or later. Prebuilt wheels are available for Linux
and macOS (x86_64 and arm64) and for Windows (x86_64); on other platforms pip
builds Commol from source, which needs a Rust toolchain (see
[Installation](https://munqu.github.io/commol/latest/getting-started/installation/)).

Commol is under active development. Until version 1.0, a minor release
(`0.2` → `0.3`) may break the API; the
[changelog](https://munqu.github.io/commol/latest/about/changelog/) lists
every such change.

## Example

A model of three states with one unknown rate. Commol describes the model with
the equations it derived, fits the rate to five observations, and simulates
the result:

```python
from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)

# 1. Write the process (the states and the flows between them), and build it
#    with a structure (how much starts where) and an engine.
model = (
    ModelBuilder(name="Three states")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=None)  # unknown: calibration finds it
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .build(
        Aggregate(
            total=1000,
            bin_fractions={"A": 0.99, "B": 0.01, "C": 0.0},
        ),
        engine="DifferentialEquations",
    )  # or "DifferenceEquations"
)
print(model.describe().render("text"))  # the equations Commol derived, and more
model.describe().save("model.md")  # Markdown, with the state graph as a diagram

# 2. Fit the unknown to the observations of B.
observed = [
    ObservedDataPoint(at=at, output="B", value=value)
    for at, value in [(10, 150.8), (20, 404.7), (30, 239.0), (40, 104.0), (50, 42.6)]
]
problem = CalibrationProblem(
    observed_data=observed,
    parameters=[
        CalibrationParameter(
            id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0
        )
    ],
    optimization_config=NelderMeadConfig(max_iterations=500),
)
result = Calibrator(Simulation(model), problem).run()
model.apply_calibration_parameters(result, problem)

# 3. Simulate the calibrated model.
results = Simulation(model).run(num_steps=60)
print(f"k1 = {result.best_parameters['k1']:.2f}")  # k1 = 0.40
print(f"B at 60: {results['B'][-1]:.1f}")
```

## Features

- **[Subgroups](https://munqu.github.io/commol/latest/guide/core-concepts/#stratifications).**
  Split every state by any number of categories, nest a split inside one
  subgroup, and give each subgroup its own rates.
- **[Two engines](https://munqu.github.io/commol/latest/guide/simulations/#differential-equations).**
  Run the same model as difference equations or as ordinary differential
  equations, with solvers for stiff systems.
- **[Rates that change](https://munqu.github.io/commol/latest/guide/axis-patterns/).**
  Write rates as formulas of the state and the axis, or build them from data
  series, pulses, intervals, ramps and impulses.
- **[Calibration](https://munqu.github.io/commol/latest/guide/calibration/).**
  Fit parameters and initial conditions to observations of single outputs,
  sums or counts per window, under constraints.
- **[Uncertainty](https://munqu.github.io/commol/latest/guide/probabilistic-calibration/).**
  Build ensembles of good fits and read prediction intervals and parameter
  spreads from them.
- **[Checks and output](https://munqu.github.io/commol/latest/guide/units-and-equations/).**
  Check units, print the equations as text or LaTeX, and save models and
  reports as JSON.

## Documentation

The [documentation](https://munqu.github.io/commol/latest/) begins with a
[Quick Start](https://munqu.github.io/commol/latest/getting-started/quickstart/)
that builds, runs and calibrates a first model. The
[User Guide](https://munqu.github.io/commol/latest/guide/core-concepts/)
explains every concept and option, and the
[API Reference](https://munqu.github.io/commol/latest/api/) documents every
public class and function.

## Contributing

Bug reports, questions and suggestions are welcome in the
[issue tracker](https://github.com/MUNQU/commol/issues). Contributors to the
code should first read the
[contributing guide](https://munqu.github.io/commol/latest/development/contributing/),
which explains how to set up the project and open a pull request. AI agents
should begin with [`AGENTS.md`](https://github.com/MUNQU/commol/blob/develop/AGENTS.md).

## Citation

If you use Commol in your research, please cite it as follows:

```bibtex
@software{commol2025,
  title = {Commol: High-Performance Declarative Dynamical Modelling in Python},
  author = {
    Villanueva Micó, Rafael J.
    and Andreu Vilarroig, Carlos
    and Martínez Rodríguez, David
  },
  year = {2025},
  url = {https://github.com/MUNQU/commol}
}
```

## License and authors

Commol is licensed under the [MIT License](https://github.com/MUNQU/commol/blob/main/LICENSE).
It is developed by the MUNQU team:

- Rafael J. Villanueva Micó - [rjvillan@imm.upv.es](mailto:rjvillan@imm.upv.es)
- Carlos Andreu Vilarroig - [caranvi1@upv.es](mailto:caranvi1@upv.es)
- David Martínez Rodríguez - [damarro3@upv.es](mailto:damarro3@upv.es)
