# Commol

Build, simulate and calibrate compartment models in Python, with a fast Rust
engine. You describe the states, the flows between them and how the population
starts; Commol expands the model over your stratifications, compiles every rate
formula to machine code, and fits unknown values to your data.

> **Before version 1.0**: the API can change between minor versions
> (`0.2` → `0.3`). The [changelog](https://munqu.github.io/commol/latest/about/changelog/)
> lists every change that breaks code.

## Features

| Feature | What you get |
|---|---|
| Fluent model building | `ModelBuilder` with validation of every input |
| Stratifications | Subgroups with their own rates, including conditional stratifications and per-compartment initial conditions |
| Rate formulas | Parameters, compartments, totals, `step`, math functions and `if`, compiled with Cranelift |
| Axis patterns | Pulses, windows, ramps, seasons and schedules per subgroup |
| Unit checking | Dimensional analysis of every rate |
| Calibration | Nelder-Mead and particle swarm, with initial conditions, scale factors, windows and constraints |
| Probabilistic calibration | Ensembles, prediction intervals and parameter spreads |
| Reports and plots | JSON-ready reports and seaborn plots |

## Installation

```bash
pip install commol
```

Commol needs Python 3.12 or later. Wheels are published for Linux and macOS
(x86_64 and arm64) and Windows (x86_64). To build from source, see
[Installation](https://munqu.github.io/commol/latest/getting-started/installation/).

## Quick start

```python
from commol import ModelBuilder, Simulation

model = (
    ModelBuilder(name="Basic Model")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.9},
            {"bin": "B", "fraction": 0.1},
            {"bin": "C", "fraction": 0.0},
        ],
    )
    .build(typology="DifferenceEquations")
)

results = Simulation(model).run(num_steps=100)
print(f"C at step 100: {results['C'][-1]:.0f}")
```

## Documentation

| Where to go | What you will find |
|---|---|
| **[Quick Start](https://munqu.github.io/commol/latest/getting-started/quickstart/)** | Build, run, check and calibrate a first model. Start here. |
| [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/) | Bins, stratifications, transitions, rates and initial conditions |
| [Building Models](https://munqu.github.io/commol/latest/guide/building-models/) | Every builder option, transitions, expressions, axis patterns and units |
| [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/) | Outputs, totals, windows and plots |
| [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/) | Fitting parameters and initial conditions, and probabilistic calibration |
| [API Reference](https://munqu.github.io/commol/latest/api/) | Every public class and function |
| [Codebase Guide](https://munqu.github.io/commol/latest/codebase/) | How Commol is built, for contributors |
| [`AGENTS.md`](https://github.com/MUNQU/commol/blob/develop/AGENTS.md) | The entry point for AI agents working on this repository |

The site has two versions: `latest` (the released code) and `dev` (the
`develop` branch).

## Development

```bash
git clone https://github.com/MUNQU/commol.git
cd commol/py-commol
uv sync --extra dev
source .venv/bin/activate
maturin develop --release
pytest
```

Every change ships with its documentation; see
[Contributing](https://munqu.github.io/commol/latest/development/contributing/)
and the [Development Workflow](https://munqu.github.io/commol/latest/development/workflow/).

## License

Commol is licensed under the MIT License. See [LICENSE](https://github.com/MUNQU/commol/blob/main/LICENSE).

## Authors

- Rafael J. Villanueva Micó - [rjvillan@imm.upv.es](mailto:rjvillan@imm.upv.es)
- Carlos Andreu Vilarroig - [caranvi1@upv.es](mailto:caranvi1@upv.es)
- David Martínez Rodríguez - [damarro3@upv.es](mailto:damarro3@upv.es)

## Citation

If you use Commol in your research, please cite:

```bibtex
@software{commol2025,
  title = {Commol: A High-Performance Compartment Modelling Library},
  author = {
    Villanueva Micó, Rafael J.
    and Andreu Vilarroig, Carlos
    and Martínez Rodríguez, David
  },
  year = {2025},
  url = {https://github.com/MUNQU/commol}
}
```

## Support

- Documentation: <https://munqu.github.io/commol/latest/>
- Questions and bugs: <https://github.com/MUNQU/commol/issues>
