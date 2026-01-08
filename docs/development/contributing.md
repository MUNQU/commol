# Contributing Guidelines

Thank you for considering contributing to Commol! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

Before creating a bug report:

- Check the [issue tracker](https://github.com/MUNQU/commol/issues) for existing reports
- Verify the bug exists in the latest version

When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, Commol version)
- **Code samples** or test cases that demonstrate the problem

### Suggesting Features

Feature suggestions are welcome! Please:

- Check if the feature has already been requested
- Provide a clear use case and rationale
- Describe the desired API or interface
- Consider implementation complexity vs benefit

### Pull Requests

1. **Keep PRs focused**: One feature/fix per PR
2. **Write tests**: All new code should have tests
3. **Update docs**: Update README or documentation as needed
4. **Follow conventions**: Use the project's code style
5. **Be descriptive**: Write clear commit messages and PR descriptions

## Development Guidelines

### Code Style

#### Python

- Follow [PEP 8](https://pep8.org/)
- Use type hints for all functions
- Maximum line length: 88 characters
- Use Ruff for linting and formatting

```bash
cd py-commol
poetry run ruff check .
poetry run ruff format .
poetry run ty check commol
```

#### Rust

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Document public APIs with rustdoc

```bash
cargo fmt --all
cargo clippy --all-targets --all-features
```

### Testing

#### Python Tests

- Write tests using pytest
- Aim for >95% code coverage
- Test both success and error cases
- Use descriptive test names

```python
def test_model_builder_creates_valid_sir_model():
    """Test that ModelBuilder creates a valid SIR model."""
    model = (
        ModelBuilder(name="Test SIR")
        .add_bin(id="S", name="Susceptible")
        # ... rest of model
        .build("DifferenceEquations")
    )
    assert model.name == "Test SIR"
```

Run tests:

```bash
cd py-commol
poetry run pytest -v --cov=epimodel
```

#### Rust Tests

- Write unit tests in the same file
- Write integration tests in `tests/` directory
- Use `#[should_panic]` for error cases

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        // Test implementation
    }
}
```

Run tests:

```bash
cargo test --workspace
```

### Documentation

#### Python Docstrings

Use NumPy-style docstrings:

```python
def run(self, num_steps: int, output_format: str = "dict_of_lists") -> dict:
    """
    Runs the simulation and returns results.

    Parameters
    ----------
    num_steps : int
        Number of simulation steps to run.
    output_format : str, default 'dict_of_lists'
        Format for results ('dict_of_lists' or 'list_of_lists').

    Returns
    -------
    dict
        Simulation results in the specified format.

    Raises
    ------
    ValueError
        If num_steps is negative.

    Examples
    --------
    >>> sim = Simulation(model)
    >>> results = sim.run(num_steps=100)
    """
```

#### Rust Documentation

Use rustdoc:

````rust
/// Runs the simulation for a given number of steps.
///
/// # Arguments
///
/// * `num_steps` - The number of time steps to simulate
///
/// # Returns
///
/// A vector of state vectors, one per time step
///
/// # Examples
///
/// ```
/// let results = engine.run(100);
/// assert_eq!(results.len(), 100);
/// ```
pub fn run(&self, num_steps: usize) -> Vec<Vec<f64>> {
    // Implementation
}
````

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system or dependencies
- `ci`: CI/CD changes
- `chore`: Other changes (releasing, etc.)

**Examples**:

```
feat(transitions): add support for time-dependent rates

Implemented mathematical expression parsing for dynamic
transition rates that change over time.

Closes #123
```

```
fix(simulation): prevent negative population values

Added validation to ensure compartment populations
remain non-negative during simulation.
```

## Code Review Process

### For Contributors

1. **Respond to feedback** promptly and professionally
2. **Make requested changes** or discuss alternatives
3. **Keep the PR updated** with the target branch
4. **Test locally** before requesting re-review

### For Reviewers

1. **Be constructive** and respectful
2. **Explain reasoning** for requested changes
3. **Acknowledge good work** when you see it
4. **Test the changes** when possible

## Release Process

See [Release Process](release.md) for details on versioning and releases.

## Getting Help

- **Documentation**: Check the [docs](https://munqu.github.io/commol)
- **Discussions**: Use [GitHub Discussions](https://github.com/MUNQU/commol/discussions)
- **Issues**: Report bugs via [GitHub Issues](https://github.com/MUNQU/commol/issues)

## Recognition

Contributors will be recognized in:

- The project's AUTHORS file
- Release notes for significant contributions
- GitHub's contributor graph

Thank you for contributing to Commol! 🎉
