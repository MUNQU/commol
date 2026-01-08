# Release Process

Commol follows [Semantic Versioning](https://semver.org/) (SemVer) for version numbering: `MAJOR.MINOR.PATCH`

## Version Guidelines

### MAJOR version (X.0.0)

Incremented for incompatible API changes:

- Breaking changes to public API
- Removal of deprecated features
- Major architectural changes affecting backwards compatibility
- Example: Changing method signatures, removing public methods

### MINOR version (0.X.0)

Incremented for backwards-compatible new features:

- New functionality added
- New disease states, stratifications, or transition types
- Performance improvements without API changes
- Deprecation notices (without removal)
- Example: Adding new mathematical functions, new model types

### PATCH version (0.0.X)

Incremented for backwards-compatible bug fixes:

- Bug fixes that don't change the API
- Documentation improvements
- Internal refactoring
- Security patches
- Example: Fixing calculation errors, typos in error messages

## Release Workflow

### Step 1: Prepare the Release

#### 1.1 Update Version Numbers

Edit the following files:

- `py-commol/pyproject.toml` - Update `version` field
- `Cargo.toml` - Update `[workspace.package]` version

```toml
# py-commol/pyproject.toml
[tool.poetry]
version = "1.2.3"

# Cargo.toml
[workspace.package]
version = "1.2.3"
```

#### 1.2 Update CHANGELOG (if maintained)

Document all changes under the new version:

```markdown
## [1.2.3] - 2024-12-15

### Added

- New mathematical functions for transition rates
- Support for multi-compartment transitions

### Changed

- Improved simulation performance by 20%

### Fixed

- Fixed population conservation bug in stratified models
- Corrected type hints in Simulation class

### Security

- Updated dependencies to patch vulnerabilities
```

#### 1.3 Run Quality Checks

```bash
# Python checks
cd py-commol
poetry run ruff check .
poetry run ruff format .
poetry run ty check commol
poetry run pytest

# Rust checks
cd ..
cargo fmt --all
cargo clippy --all-targets --all-features
cargo test --workspace
```

#### 1.4 Commit Version Changes

```bash
git add py-commol/pyproject.toml Cargo.toml CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
git push origin main
```

### Step 2: Create GitHub Release

#### 2.1 Create Tag

```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

#### 2.2 Create GitHub Release

1. Go to [Releases page](https://github.com/MUNQU/commol/releases)
2. Click "Draft a new release"
3. Select tag: `vX.Y.Z`
4. Set release title: `Version X.Y.Z`
5. Add release notes:

```markdown
## What's New in X.Y.Z

### New Features

- Feature 1 description
- Feature 2 description

### Bug Fixes

- Fix 1 description
- Fix 2 description

### Documentation

- Documentation improvements

### Performance

- Performance improvements

## Breaking Changes (for major versions)

- List breaking changes
- Migration guide

## Installation

pip install epimodel==X.Y.Z

## Full Changelog

See [CHANGELOG.md](https://github.com/MUNQU/commol/blob/main/CHANGELOG.md)
```

6. Click "Publish release"

### Step 3: Automated PyPI Publication

Once the release is published, the GitHub Actions workflow automatically:

1. **Builds wheels** for Linux, Windows, and macOS
2. **Creates source distribution** (sdist)
3. **Publishes to PyPI** using trusted publishing

Monitor progress:

- Go to [Actions tab](https://github.com/MUNQU/commol/actions)
- Watch the "Release to PyPI" workflow

### Step 4: Verify the Release

#### 4.1 Check PyPI

Verify the package appears at: https://pypi.org/project/commol/

#### 4.2 Test Installation

```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install the new version
pip install epimodel==X.Y.Z

# Verify it works
python -c "from commol import ModelBuilder; print('Success!')"
```

#### 4.3 Test Functionality

```python
from commol import ModelBuilder, Simulation


# Run a quick test
model = (
    ModelBuilder(name="Test")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology="DifferenceEquations")
)

sim = Simulation(model)
results = sim.run(num_steps=100)
print(f"Final infected: {results['I'][-1]:.0f}")
```

### Step 5: Sync Branches

```bash
# Merge main back to develop
git checkout develop
git merge main
git push origin develop
```

## Pre-release Versions

For testing before official release:

### Alpha Releases

```bash
# Version: X.Y.Z-alpha.N
git tag -a v1.3.0-alpha.1 -m "Alpha release for testing"
git push origin v1.3.0-alpha.1
```

### Beta Releases

```bash
# Version: X.Y.Z-beta.N
git tag -a v1.3.0-beta.1 -m "Beta release for testing"
git push origin v1.3.0-beta.1
```

### Release Candidates

```bash
# Version: X.Y.Z-rc.N
git tag -a v1.3.0-rc.1 -m "Release candidate"
git push origin v1.3.0-rc.1
```

**Important**: Mark GitHub releases as "pre-release" to prevent automatic PyPI publication.

## Version Support Policy

- **Latest version**: Fully supported with new features and bug fixes
- **Older versions**: No support, users encouraged to upgrade

## PyPI Trusted Publishing Setup

To enable automated PyPI publishing, configure trusted publishing:

1. Go to [PyPI](https://pypi.org) and log in
2. Go to your project settings
3. Navigate to "Publishing" section
4. Add GitHub as a trusted publisher:
   - **Owner**: MUNQU
   - **Repository**: epimodel
   - **Workflow**: release.yml
   - **Environment**: pypi

This allows GitHub Actions to publish without API tokens.

## Release Checklist

- [ ] Version updated in `pyproject.toml`
- [ ] Version updated in `Cargo.toml`
- [ ] CHANGELOG updated with all changes
- [ ] All tests passing locally
- [ ] Version committed and pushed to main
- [ ] Git tag created and pushed
- [ ] GitHub release created with notes
- [ ] CI/CD pipeline completed successfully
- [ ] Package verified on PyPI
- [ ] Installation tested in clean environment
- [ ] Main merged back to develop
- [ ] Release announced (if applicable)

## Troubleshooting

### PyPI Upload Fails

1. Check GitHub Actions logs
2. Verify trusted publishing is configured
3. Ensure version number is unique (not already on PyPI)
4. Check for artifact build errors

### Wheel Build Fails

1. Check that Rust code compiles on all platforms
2. Verify maturin configuration is correct
3. Check for platform-specific dependencies

### Version Conflicts

1. Ensure version is updated in all required files
2. Use `git tag -d vX.Y.Z` to delete incorrect tags locally
3. Use `git push origin :refs/tags/vX.Y.Z` to delete remote tags
