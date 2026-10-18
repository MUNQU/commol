# Release Process

[Back to Codebase Guide](../codebase/index.md)

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
- New compartment states, stratifications, or transition types
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

!!! note "Versions before 1.0.0"
    While Commol is at `0.x`, a breaking change increments the MINOR version
    (`0.1.3` → `0.2.0`) and everything else increments the PATCH version.

## Where the Version Comes From

The version is not stored in the repository. `Cargo.toml` keeps a placeholder
(`0.1.0`), and `pyproject.toml` declares the version as dynamic. When a release
is built, the `set-version` action (`.github/actions/set-version`) reads the
git tag, writes it into `py-commol/Cargo.toml`, and maturin turns it into the
Python package version.

Do not commit version bumps. The tag is the version.

## Release Workflow

A release is always made from `main`. Publishing a GitHub release starts the
**Release to PyPI** workflow (`.github/workflows/release.yaml`), which
publishes to PyPI immediately: the `pypi` environment has no approval step.
The release issue (label `release`) holds the checklist of this page, and the
milestone of the release lists the issues it ships.

!!! warning "A published version is final"
    PyPI never accepts the same version number twice, even after you delete
    it. Check the tag and the commit before you publish.

### Step 1: Choose the Version

An issue of the milestone that will not ship in this release moves to another
milestone, with a comment that says why
([Milestones](workflow.md#milestones)). When every issue left in the milestone
is closed, list the changes since the last release:

```bash
git fetch --tags origin
git log --oneline --first-parent "$(git describe --tags --abbrev=0 origin/main)"..origin/main
```

Use the Version Guidelines above. The `[Unreleased]` section of
`docs/about/changelog.md` lists the changes, and the breaking ones are marked
**Breaking**.

### Step 2: Date the Changelog

On a branch from `main`, in `docs/about/changelog.md`:

1. Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, the day you
   publish, and add an empty `## [Unreleased]` above it.
2. Start the section with a short paragraph that says what the release is
   about and what a user must read before upgrading. It opens the release
   notes.
3. Add the version to the Version History list.

Check the notes the release will have (Step 4):

```bash
python .github/scripts/release_notes.py X.Y.Z
```

Open the pull request to `main` (`docs(changelog): release X.Y.Z`) and merge
it through the merge queue.

### Step 3: Create the Tag

Tag the commit at the top of `main` with an annotated tag:

```bash
git checkout main
git pull --ff-only origin main
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

The tag must start with `v`. Pushing a tag does not publish anything.

### Step 4: Publish the GitHub Release

The notes come from the changelog of `main`, never by hand: its section
`[X.Y.Z]`, with the links turned into links to the site, the install command
and the comparison with the previous release
(`.github/scripts/release_notes.py`). Then publish the release, from the
GitHub [Releases page](https://github.com/MUNQU/commol/releases) ("Draft a new
release", tag `vX.Y.Z`, title `Version X.Y.Z`) or with the GitHub CLI:

```bash
python .github/scripts/release_notes.py X.Y.Z > notes.md
gh release create vX.Y.Z --verify-tag --title "Version X.Y.Z" --notes-file notes.md --latest
```

### Step 5: Watch the PyPI Publication

The **Release to PyPI** workflow:

1. Builds the wheels for Linux and macOS (x86_64 and aarch64) and Windows
   (x86_64), for Python 3.12, 3.13 and 3.14
2. Builds the source distribution (sdist)
3. Runs the coverage studies (`coverage.yaml`), about 10 min
4. Publishes all of them to PyPI with trusted publishing, only when every
   step before passed

If a coverage study fails, nothing reaches PyPI: delete the GitHub release
and the tag, fix the method on `main`, and tag again. To learn the answer
before you tag, run the studies on `main` by hand:

```bash
gh workflow run "Coverage Studies" --ref main
```

Watch it in the [Actions tab](https://github.com/MUNQU/commol/actions) or with:

```bash
gh run watch "$(gh run list --workflow release.yaml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

### Step 6: Verify the Release

Check that the version appears at <https://pypi.org/project/commol/>. Then
install it in a clean environment and run a model:

```bash
uv venv /tmp/commol-check
uv pip install --python /tmp/commol-check/bin/python --refresh commol==X.Y.Z
/tmp/commol-check/bin/python -I check.py
```

```python
# check.py
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Release check")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="a_to_b", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="b_to_c", source=["B"], target=["C"], rate="k2")
    .build(
        Aggregate(
            total=1000,
            bin_fractions={"A": 0.99, "B": 0.01, "C": 0.0},
        ),
        engine="DifferenceEquations",
    )
)

results = Simulation(model).run(num_steps=100)
print(f"Final C: {results['C'][-1]:.0f}")
```

Then close the release issue and the milestone of the release.

## Pre-release Versions

Use a beta tag in the PEP 440 form, as for `v0.1.0b3`:

```bash
git tag -a vX.Y.ZbN -m "Beta release X.Y.ZbN"
git push origin vX.Y.ZbN
```

The `set-version` action turns `X.Y.ZbN` into the Cargo version `X.Y.Z-beta.N`.

Mark the GitHub release as a **pre-release**. The workflow still publishes it
to PyPI, but `pip` installs a pre-release only when it is asked for by version
or with `--pre`.

## Version Support Policy

- **Latest version**: Fully supported with new features and bug fixes
- **Older versions**: No support, users encouraged to upgrade

## PyPI Trusted Publishing Setup

PyPI accepts the uploads of the release workflow without an API token. The
trusted publisher of the `commol` project on PyPI is:

- **Owner**: MUNQU
- **Repository**: commol
- **Workflow**: `release.yaml`
- **Environment**: `pypi`

If you rename the workflow file or the environment, update the trusted
publisher on PyPI first, or the upload fails.

## Release Checklist

The release issue repeats this list.

- [ ] Every issue of the milestone closed, or moved to another milestone with a comment that says why
- [ ] Version chosen
- [ ] Changelog dated on `main`, with its opening paragraph
- [ ] Annotated tag `vX.Y.Z` created on `main` and pushed
- [ ] GitHub release published with the notes of `release_notes.py`
- [ ] **Release to PyPI** workflow passed
- [ ] Version visible on PyPI and tested in a clean environment
- [ ] Release issue and milestone closed

## Troubleshooting

### PyPI Upload Fails

1. Check the logs of the **Release to PyPI** workflow
2. Check that the trusted publisher on PyPI matches the workflow file and
   environment names
3. Check that the version is not already on PyPI. If it is, publish a new
   version; you cannot upload the same version again

### Wheel Build Fails

Nothing reaches PyPI when a build job fails, because the upload job needs all
of them.

1. Fix the problem on `main` (a pull request through the merge queue)
2. Delete the GitHub release and the tag:

    ```bash
    gh release delete vX.Y.Z --yes
    git push origin :refs/tags/vX.Y.Z
    git tag -d vX.Y.Z
    ```

3. Tag the new `main` commit and publish the release again
