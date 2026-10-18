# Installation

[Back to Home](https://munqu.github.io/commol/latest/index.md)

Install Commol from PyPI, or build it from source to work on it.

## Quick path

```bash
pip install commol
```

| Requirement                      | Version                                              |
| -------------------------------- | ---------------------------------------------------- |
| Python                           | 3.12 or later                                        |
| Platforms with ready-made wheels | Linux and macOS (x86_64 and arm64), Windows (x86_64) |

On other platforms, `pip` builds Commol from the source distribution, which needs a Rust toolchain.

## Check the installation

```python
import importlib.metadata

from commol import ModelBuilder, Simulation

print(importlib.metadata.version("commol"))
```

## Build from source

To work on Commol, or to use unreleased code from `main`:

1. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.
1. Install [uv](https://docs.astral.sh/uv/).
1. Clone, install the dependencies, and build the extension:

```bash
git clone https://github.com/MUNQU/commol.git
cd commol/py-commol
uv sync --extra dev --extra docs
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
maturin develop --release
pytest
```

| Step                               | Why                                                                                                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync --extra dev --extra docs` | Creates `.venv` with the test, lint and documentation tools. It does not build Commol (`package = false` in `pyproject.toml`), and a later `uv sync` removes the install of Commol. |
| `source .venv/bin/activate`        | `maturin develop` installs into the active environment.                                                                                                                             |
| `maturin develop --release`        | Compiles the Rust extension and installs Commol in editable mode. Run it again after every Rust change and every `uv sync`.                                                         |

Paths with spaces

Maturin may fail when the project path contains spaces. Clone into a path without them.

See [Development Workflow](https://munqu.github.io/commol/latest/development/workflow/index.md) for branches, checks and pull requests.

## Next steps

| To learn         | Read                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Your first model | [Quick Start](https://munqu.github.io/commol/latest/getting-started/quickstart/index.md) |
| The concepts     | [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md)      |
