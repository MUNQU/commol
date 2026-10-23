# Installation

[Back to Home](https://munqu.github.io/commol/latest/index.md)

Install Commol from PyPI, or build it from source to work on it.

## Quick path

```bash
pip install commol
```

The figures need matplotlib, the optional extra `commol[plot]`:

```bash
pip install 'commol[plot]'
```

| Requirement                      | Version                                              |
| -------------------------------- | ---------------------------------------------------- |
| Python                           | 3.12 or later                                        |
| Platforms with ready-made wheels | Linux and macOS (x86_64 and arm64), Windows (x86_64) |

On other platforms, `pip` builds Commol from the source distribution, which needs a Rust toolchain.

## Check the installation

```python
import commol

print(commol.__version__)
print(commol.build_info())
```

`commol.__version__` is the version you installed. `commol.build_info()` gives it with the full commit of the source; save it with a result to name the code that made it.

| Installed from                               | `__version__`                               | `build_info().commit`                  |
| -------------------------------------------- | ------------------------------------------- | -------------------------------------- |
| A wheel of a release on PyPI                 | The release, such as `0.9.0`                | The commit of the release              |
| The source distribution of a release         | The release                                 | `None`: the source has no git metadata |
| A git commit, or a clone (`maturin develop`) | `0.1.0+g<first 7 characters of the commit>` | The full commit                        |

A build that is not a release has the placeholder version `0.1.0` of the source, so its local version (`+g…`) is what names its commit. The metadata of the package (`pip show commol`) gives the placeholder alone.

## Build from source

To work on Commol, or to use unreleased code from `main`:

1. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.
1. Install [uv](https://docs.astral.sh/uv/), at the version of `required-version` in `py-commol/pyproject.toml`; another version of uv refuses to run in the project. A standalone uv moves to it with `uv self update <version>`.
1. Clone, install the dependencies, and build the extension:

```bash
git clone https://github.com/MUNQU/commol.git
cd commol/py-commol
uv sync --extra dev --extra docs
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
maturin develop --release --features test-kind
pytest
```

| Step                                             | Why                                                                                                                                                                                                                  |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync --extra dev --extra docs`               | Creates `.venv` with the test, lint and documentation tools. It does not build Commol (`package = false` in `pyproject.toml`), and a later `uv sync` removes the install of Commol.                                  |
| `source .venv/bin/activate`                      | `maturin develop` installs into the active environment.                                                                                                                                                              |
| `maturin develop --release --features test-kind` | Compiles the Rust extension and installs Commol in editable mode. The `test-kind` feature adds the model kind that the tests need; a release never has it. Run it again after every Rust change and every `uv sync`. |

Paths with spaces

Maturin may fail when the project path contains spaces. Clone into a path without them.

See [Development Workflow](https://munqu.github.io/commol/latest/development/workflow/index.md) for branches, checks and pull requests.

## Next steps

| To learn         | Read                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Your first model | [Quick Start](https://munqu.github.io/commol/latest/getting-started/quickstart/index.md) |
| The concepts     | [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md)      |
