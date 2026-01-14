# Installation

## Requirements

- Python 3.11 or higher
- Rust toolchain (for building from source)

!!! warning "Path Requirements"
The project directory path **must not contain tildes (`´`, `~`) or spaces**. Maturin (the Rust-Python build tool) may fail with these characters in the path.

    **Good paths:**

    - `/home/username/projects/commol`
    - `/Users/username/Documents/commol`
    - `/opt/projects/commol`

    **Bad paths:**

    - `~/projects/commol` (contains tilde)
    - `/home/my projects/commol` (contains space)
    - `/Users/User Name/commol` (contains space)

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

### 2. Clone the Repository

```bash
git clone https://github.com/MUNQU/commol.git
cd commol
```

### 3. Create and Activate Virtual Environment

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate   # On Windows
```

### 4. Install Dependencies and Build

```bash
cd py-commol

# Install maturin (build tool)
pip install maturin

# Build and install the package
maturin develop --release
```

!!! important "Virtual Environment"
Make sure your virtual environment is **activated before running `maturin develop`**. This ensures the extension is built into the correct Python environment.

## Verify Installation

Test that Commol is correctly installed:

```python
from commol import ModelBuilder


print("Commol installed successfully!")
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Build your first model
- [Core Concepts](../guide/core-concepts.md) - Understand Commol fundamentals
