# Development Workflow

## Development Setup

Commol is built with Rust and Python. To set up your development environment:

!!! warning "Path Requirements"
The project directory path **must not contain tildes (`~`) or spaces**. Maturin (the Rust-Python build tool) may fail with these characters in the path.

    **Good paths:**

    - `/home/username/projects/commol`
    - `/Users/username/Documents/commol`
    - `/opt/projects/commol`

    **Bad paths:**

    - `~/projects/commol` (contains tilde)
    - `/home/my projects/commol` (contains space)
    - `/Users/User Name/commol` (contains space)

### 1. Install Prerequisites

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python 3.11 or higher
# (use your system's package manager)
```

### 2. Clone and Setup

```bash
git clone https://github.com/MUNQU/commol.git
cd commol

# Create and activate virtual environment
python -m venv venv

source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate   # On Windows

# Install Python dependencies
cd py-commol
pip install -e ".[dev,docs]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 3. Build the Project

```bash
# Build Rust workspace
cargo build --workspace

# Build Python extension (with virtual environment activated)
cd py-commol
maturin develop --release
```

!!! important "Virtual Environment and Maturin"
Make sure your virtual environment is **activated before running `maturin develop`**. This ensures the Rust extension is built into the correct Python environment.

## Branching Strategy

We use a simplified Git Flow approach:

- **`main`**: Production-ready code. Protected branch.
- **`develop`**: Integration branch for features. Used for staging releases.
- **`feature/*`**: New features, refactoring, or significant changes (e.g., `feature/add-stochastic-model`)
- **`fix/*`**: Bug fixes (e.g., `fix/calculation-error`)
- **`docs/*`**: Documentation-only changes (e.g., `docs/update-api-reference`)
- **`test/*`**: Test additions or improvements (e.g., `test/add-sir-model-tests`)
- **`chore/*`**: Maintenance tasks, dependencies (e.g., `chore/update-dependencies`)

## Development Process

### 1. Starting New Work

```bash
# Always start from the latest develop branch
git checkout develop
git pull origin develop

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Making Changes

```bash
# Make your changes to the code

# Run quality checks before committing (with virtual environment activated)
cd py-commol
ruff check . --fix         # Auto-fix Python linting issues
ruff format .              # Format Python code
ty check commol            # Type checking

cd ..
cargo fmt --all                       # Format Rust code
cargo clippy --all-targets --fix      # Fix Rust linting issues
```

### 3. Testing Your Changes

```bash
# Run Python tests (with virtual environment activated)
cd py-commol
pytest -v

# Run Rust tests
cd ..
cargo test --workspace
```

### 4. Committing Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```bash
# Commit format: <type>(<scope>): <description>

git add .
git commit -m "feat(transitions): add support for time-varying rates"
git commit -m "fix(simulation): correct population calculation"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(models): add tests for stratified models"
```

**Commit Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependencies
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `build`: Build system changes

**Branch vs Commit Type**:

The branch name indicates the _primary purpose_ of the work, while commit types describe _individual changes_:

- A `feature/refactor-api` branch may have `refactor`, `test`, and `docs` commits
- A `test/*` branch primarily adds tests but may have `fix` commits too
- A `docs/*` branch is documentation-only, so all commits should use `docs` type

### 5. Pushing and Creating Pull Requests

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
# - Target the 'develop' branch
# - Fill in the PR template with:
#   - Description of changes
#   - Related issues
#   - Testing performed
#   - Screenshots (if UI changes)
```

### 6. Code Review and CI

When you create a PR, automated checks run:

- **Code Quality Pipeline**: Linting, formatting, type checking for both Python and Rust
- **Build Pipeline**: Builds on Ubuntu, Windows, and macOS with Python versions
- **Test Coverage**: Runs all tests and reports coverage

**PR Requirements**:

- All CI checks must pass
- At least one approving review
- No merge conflicts with target branch
- Code coverage should be >95%

### 7. Merging to Develop

Once approved:

```bash
# Merge using "Squash and merge" or "Rebase and merge" on GitHub
# Delete the feature branch after merging
```

## Continuous Integration Pipelines

The project uses three main CI/CD pipelines:

### 1. Code Quality (on every push/PR)

- Python: Ruff (linting/formatting), Ty (type checking), Pytest (testing)
- Rust: rustfmt (formatting), Clippy (linting), cargo test
- Runs on: Ubuntu
- Purpose: Ensure code quality standards

### 2. Build (on every push/PR)

- Builds on: Ubuntu, Windows, macOS
- Python versions: 3.11, 3.12
- Tests: Full test suite on all platforms
- Artifacts: Wheel files for each platform
- Purpose: Ensure cross-platform compatibility

### 3. Release (on GitHub release)

- Builds wheels for all platforms
- Creates source distribution
- Publishes to PyPI using trusted publishing
- Purpose: Automated deployment

## Local Development Tips

### Quick Commands

```bash
# Run all quality checks (Python) - make sure venv is activated
cd py-commol && ruff check . && ruff format . && ty check commol && pytest

# Run all quality checks (Rust)
cargo fmt --all && cargo clippy --all-targets && cargo test --workspace

# Build and test the Python package locally - make sure venv is activated
cd py-commol
maturin develop --release
pytest
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code before committing:

```bash
cd py-commol
pre-commit install
```

This runs:

- Ruff linting and formatting
- Ty type checking
- Trailing whitespace removal
- End-of-file fixing

### Debugging Rust from Python

```bash
# Build with debug symbols (with virtual environment activated)
cd py-commol
maturin develop

# Run Python
python -m your_test_script.py
```

## Building Documentation

The project uses MkDocs Material for documentation with multi-version support via Mike:

```bash
# Install documentation dependencies (with virtual environment activated)
cd py-commol
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

Visit http://127.0.0.1:8000 to view the documentation locally.

### Documentation Versioning

Commol maintains two documentation versions:

- **`latest`**: Documentation for the current stable release from the `main` branch
- **`dev`**: Development documentation from the `develop` branch

#### Automatic Deployment

Documentation is automatically deployed when pushing to tracked branches:

- **Push to `main`**: Deploys to `latest` version
- **Push to `develop`**: Deploys to `dev` version
- **Pull requests**: Build and validate documentation without deploying

Users can switch between versions using the version selector in the documentation site.

#### Manual Deployment (if needed)

If you need to manually deploy documentation:

```bash
# Deploy latest version (from main branch) - with venv activated
cd py-commol
mike deploy --push --update-aliases latest
mike set-default --push latest

# Deploy dev version (from develop branch)
cd py-commol
mike deploy --push dev

# List all deployed versions
mike list

# Delete a version
mike delete <version-name> --push
```

#### Version Management Tips

- The version selector appears in the top-right of the documentation site
- Each version is completely isolated with its own search index
- Old versions remain accessible even after new deployments
- Use `mike list` to see all deployed versions and aliases

## Next Steps

- [Contributing Guidelines](contributing.md) - Best practices for contributions
- [Release Process](release.md) - How releases are managed
