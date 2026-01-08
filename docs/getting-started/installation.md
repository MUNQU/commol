# Installation

## Requirements

- Python 3.11 or higher
- Rust toolchain (for building from source)

## Install from PyPI

Once published, you can install Commol directly from PyPI:

```bash
pip install commol
```

## Install from Source

To install the latest development version:

### 1. Install Prerequisites

#### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### Install Poetry (optional, for development)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone the Repository

```bash
git clone https://github.com/MUNQU/commol.git
cd commol
```

### 3. Install with pip

```bash
pip install maturin
cd py-commol
maturin develop --release
```

### 4. Install with Poetry (for development)

```bash
cd py-commol
poetry install
maturin develop --release
```

## Verify Installation

Test that Commol is correctly installed:

```python
from commol import ModelBuilder


print("Commol installed successfully!")
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Build your first model
- [Core Concepts](../guide/core-concepts.md) - Understand Commol fundamentals
