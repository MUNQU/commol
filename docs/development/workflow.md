# Development Workflow

[Back to Codebase Guide](../codebase/index.md)

## Development Setup

Follow [Build from source](../getting-started/installation.md#build-from-source),
then install the pre-commit hooks:

```bash
git clone https://github.com/MUNQU/commol.git
cd commol/py-commol
uv sync --extra dev --extra docs
source .venv/bin/activate
maturin develop --release --features test-kind
pre-commit install
```

The `test-kind` feature adds the model kind of the tests
(`py-commol/src/test_kind.rs`); the tests of `tests/model_kinds/` need it, and
a release build never has it. Rebuild after every Rust change (see
[Quick commands](#quick-commands)).

## Branching Strategy

Commol has one branch, `main` (trunk-based development): every pull request
merges into it, through the merge queue, and a release is a tag on it.

- **`main`**: The integration branch, always tested and releasable. Protected
  by its ruleset: pull requests only, squash merges through the merge queue,
  and the required checks. A review is welcome and not required.
- **`release/X.Y`**: Only to fix a version that `main` has left behind. It is
  cut from the tag `vX.Y.Z`, receives fixes copied from `main`
  (`git cherry-pick`), and releases `vX.Y.(Z+1)`. A fix lands on `main` first.
- **Short branches** for one piece of work each, from `main`, merged within days:
- **`feature/*`**: New features, refactoring, or significant changes (e.g., `feature/add-stochastic-model`)
- **`fix/*`**: Bug fixes (e.g., `fix/calculation-error`)
- **`docs/*`**: Documentation-only changes (e.g., `docs/update-api-reference`)
- **`test/*`**: Test additions or improvements (e.g., `test/add-conditional-stratification-tests`)
- **`chore/*`**: Maintenance tasks, dependencies (e.g., `chore/update-dependencies`)

## Planning

Planned work lives in GitHub issues, written from the templates of
`.github/ISSUE_TEMPLATE/`:

| Template | Label | For |
|---|---|---|
| Proposal (`proposal.md`) | `proposal` | A feature, a refactor or a breaking change: goal, current state, requirements with the tests that check them, decisions, tasks |
| Bug or small change (`change.md`) | `change` | A defect or a change that needs no proposal: the behavior, a reproduction and the test that will show it |
| Release (`release.md`) | `release` | The checklist of one release, blocked by the issues it ships |

| What | Where |
|---|---|
| The specification of a change | The body of its issue |
| Decisions and findings during the work | Comments of the issue (`L<n> (decision|implementation): …`), and the Decisions table of the body |
| What a change does to other work | The bodies and comments of the open issues it affects; work it leaves undone, in an open issue |
| Priority and dependencies | The field `Priority` of the project (`Now`, `Next`, `Later`); "Blocked by" links between issues; "Why here" in the Delivery of a proposal |
| Releases | A milestone per release, and its release issue |
| Status | The [commol project](https://github.com/orgs/MUNQU/projects/1): "Board" by status, "By release" by milestone, "By precedence" by the field `Level` |
| Evidence of the delivery | The body of the pull request: each requirement and the tests that pass |
| What lasts after the delivery | The guide, [Design decisions](../codebase/design-decisions.md) and the changelog, as the [definition of done](../codebase/conventions.md#documentation-definition-of-done) requires |

To find the next piece of work, in one call and one line:

```bash
.github/scripts/next.sh        # the next issue; .github/scripts/next.sh 3 for three
```

It runs `.github/scripts/parallel.py --next` and lists the open proposals and
changes that no open issue blocks and nobody has taken, that can run at the
same time as every issue in progress and as each other. It names each issue
it skips and the issue in progress it would touch. A change has no section
Touches: the paths of its section Where count as what it changes. The issues
come by:

1. Effective priority: the highest of the field `Priority` of the issue in the
   project (`Now`, `Next`, `Later`; no value is `Next`) and of every open issue
   it blocks, through chains of "blocked by" links. So an issue that another
   one needs inherits the priority of that issue.
2. Milestone: an open milestone before none; the nearest due date, then the
   oldest. The milestones of commol have no due date, and their numbers follow
   the versions, so the next release comes first.
3. Number.

The dependencies set what must come first; `Priority` is set only to move an
issue ahead of (`Now`) or behind (`Later`) its milestone.

Before it lists them, `next.sh` links each open issue to the release before
its milestone, and each open release to the issues of its milestone
([Milestones](#milestones)), and then writes the precedence level of each open
proposal, change and release into the number field `Level` of the project:
0 when no open issue blocks it, else one more than the highest level of its
open blockers (an open blocker of another repository counts as level 0). The
view "By precedence" groups the open issues by `Level`, so each group can
start when the groups above it close. GitHub Projects cannot group by
"blocked by" links, and its hierarchy view follows sub-issues only, so the
script writes the field. First it adds to the project each open proposal,
change and release that is not an item of it, so that every one gets its
level. It writes only the levels that change, through REST,
with your own `gh` login: no workflow and no stored token. If it cannot write
them (no write access to the project, a rate limit), it says so in one line
and lists the next issues all the same. `start.sh` writes
them too, and so does, at any time:

```bash
python .github/scripts/parallel.py --levels
```

A level is as fresh as the last of these runs. To see the whole plan:

```bash
gh issue list --label proposal --state open
gh issue list --milestone 0.5.0
```

An issue is ready when every issue that blocks it is closed.

The project adds a new issue of the repository (its workflow "Auto-add to
project", with the filter `is:issue`), sets it to `Todo` (workflow "Item added
to project") and moves it to `Done` when it closes. An open issue that the
workflow misses gets its item from `parallel.py --levels`, which `next.sh` and
`start.sh` run.
A pull request that says `Closes #N` links to its issue and closes it when it
merges. `In Progress` is set when the work starts, by `.github/scripts/start.sh`
(see below), or by hand from the project or a terminal:

```bash
gh project item-list 1 --owner MUNQU                    # the items and their status
item=$(gh project item-list 1 --owner MUNQU --format json \
  --jq '.items[] | select(.content.number == <number>) | .id')
gh project item-edit --project-id PVT_kwDODcfAw84BnXyo --id "$item" \
  --field-id PVTSSF_lADODcfAw84BnXyozhl8usw --single-select-option-id 47fc9ee4  # In Progress
```

### Lifecycle of a proposal

1. Open the issue from the template: a scoped goal, the current state with
   paths and symbols, one requirement per behavior (`The <component> SHALL …`,
   with `WHEN … THEN …` scenarios), each scenario with the test that checks
   it. Add `breaking` when it breaks the public API, the milestone of its
   release, its blockers and, when it is not `Next`, its `Priority` in the
   project. A large proposal
   splits into sub-issues.
2. Settle its decisions when its implementation starts, not before. A
   decision that changes what an earlier proposal built, or what a later one
   relies on, updates those issues too, with a comment in each.
3. Start it with one command, which takes it, sets it to `In Progress` and
   creates its branch from `main`, linked to the issue, in a sibling worktree:

   ```bash
   .github/scripts/start.sh <number> <type>/<name>   # --here: in this checkout
   ```

   It refuses an issue that is closed, blocked by an open issue, or taken by
   someone else, and `--dry-run` shows what it would do. Never start by hand:
   a started issue that is not `In Progress` and assigned looks free to the
   next one who runs `next.sh`. Work one task after the other, the tests of a
   requirement before its code. Tick the tasks of the body as they land.
4. Update every open issue the change affects, before the pull request
   merges. To find them, run:

   ```bash
   python .github/scripts/affected.py --closes <number>
   ```

   It lists the open issues whose body names a path that the branch changes,
   in its section Touches or anywhere else, with the items that match. Edit
   the body of each one whose current state, paths or assumptions the change
   alters, and add a comment that says what changed. The script finds
   candidates by path; an issue that relies on a behavior without naming its
   path needs your own check.
   Work the change leaves undone goes to an open issue (the one that will do
   it, or a new one from a template), never only to a comment of the issue
   that closes: a closed issue is not read again.
5. Open the pull request with `Closes #N` and the evidence. The merge closes
   the issue.

### Milestones

Every issue that ships in a release has the milestone of that release, so the
release checklist asks for it:

- **A new bug** gets, when it is opened, the milestone of the nearest release
  it fits: the next one, unless its fix is breaking and that release must not
  break. A bug with no milestone is in no checklist, and `next.sh` lists it
  after every issue that has one. To find the bugs that break this rule:

  ```bash
  gh issue list --state open --label change --search "no:milestone"
  ```

- **At the close of a release**, every issue still open in its milestone moves
  to another milestone, with a comment that says why it did not ship. The
  milestone closes with no open issue.
- **An issue waits for the release before its milestone.** Each open
  proposal, change and release is "blocked by" the open release issue of the
  nearest milestone before its own, by the versions that the titles of the
  milestones name: an issue of 0.10.0 waits for release 0.9.0, and release
  0.10.0 waits for it too. When a new milestone comes in between (0.9.1), its
  release takes the place of the old blocker.
- **A release waits for the issues of its milestone.** Each open release is
  "blocked by" every open proposal and change of its own milestone, so its
  field `Level` and the view "By precedence" count all of them. An issue moved
  to another milestone leaves the release it no longer ships, and blocks the
  release of its new milestone.

`next.sh` and `start.sh` write these links before they read the blockers, and
so does, at any time:

```bash
python .github/scripts/parallel.py --releases
```

The script owns the links between a release and an issue whose milestone
names a version: it adds the one that is missing and removes the others. A
link from a release to an issue with no version in its milestone, or of
another repository, stays, and so do the links between releases other than
the release before. The script refuses a link that closes a cycle, and names
the issue to move: an issue of 0.10.0 that release 0.9.0 waits for moves to
the earlier milestone, and an issue of 0.9.0 that waits, through its blockers,
for release 0.9.0 moves to a later one; or remove the link that makes the
cycle. When the link of a release to an issue it ships and a link to the
release before close a cycle together, the first one stays: a release needs
its issues, and the release before only orders the work. So the issue that
the release needs through a chain of links is not held back by the release,
and the script names it to move to that milestone. A failure to write the
links stops `next.sh` and `start.sh`, since the blockers they read would be
wrong.

An experiment, such as a benchmark, is not a bug and may have no milestone.

### Parallel work

Two proposals can run at the same time when no chain of "blocked by" links
joins them and neither changes what the other changes or adds. Each proposal
says what it touches (section Touches of the template):

| Kind | What it is | Conflicts with |
|---|---|---|
| Changes | An edit to what exists: a behavior, a signature, a contract other code implements (a required method of `SimulationEngine`) | Any Changes or Adds of the same item |
| Adds | New code next to what exists: a module, a method of a shared class, a variant, a registry entry | Only a Changes of the same item: two Adds to one class merge by keeping both |

An item is a path or `path::Symbol`; a path that ends in `/` is a directory.
`.github/scripts/parallel.py` reads the Touches and the links of the open
issues, prints the proposals ready now and the ones each can run with, and
with `--write` keeps the line "Parallel with" of each Delivery up to date; run
it after a proposal is added or its Touches change. The `area:` labels filter
the same zones (`gh issue list --label area:inference -s open`).

To take a proposal while others run:

1. Start it with `.github/scripts/start.sh <number> <type>/<name>`: it assigns
   the issue (an assigned issue is taken), sets it to `In Progress` and
   creates its worktree, a sibling of the repository, which needs its own
   CodeGraph index.
2. Open the pull request as usual. The merge queue of `main` tests it on
   top of the pull requests queued before it, so a conflict in meaning fails
   there and never reaches `main`; a conflict in the lines of a shared
   index (the changelog, `AGENTS.md`) is solved by keeping both.

## Development Process

### 1. Starting New Work

```bash
# Always start from the latest main
git checkout main
git pull origin main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Making Changes

```bash
# Make your changes to the code

# Run quality checks before committing (with virtual environment activated)
cd py-commol
ruff check . ../.github/scripts --fix   # Auto-fix Python linting issues
ruff format . ../.github/scripts        # Format Python code
ty check commol                        # Type checking

cd ..
cargo fmt --all                       # Format Rust code
cargo clippy --all-targets --fix      # Fix Rust linting issues
```

### 3. Testing Your Changes

Run the Python and the Rust tests of [Quick commands](#quick-commands).

### 4. Committing Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages. The scope is required, and a message is one short phrase with no body; see [Commits](../codebase/conventions.md#commits):

```bash
# Commit format: <type>(<scope>): <description>

git add .
git commit -m "feat(transitions): add rate patterns"
git commit -m "fix(simulation): correct the total of a subgroup"
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
# - Target 'main'
# - Fill in the PR template; "Closes #N" names the issue it delivers
```

### 6. Code Review and CI

When you create a PR, automated checks run:

- **Code Quality Pipeline**: Linting, formatting, type checking for both Python and Rust
- **Build Pipeline**: Builds on Ubuntu, Windows, and macOS with Python versions
- **Test Coverage**: Python Coverage (`pytest --cov --cov-fail-under=100`) and
  Rust Code Quality (`.github/scripts/rust_coverage.sh`, the Rust tests and
  `pytest` on an instrumented extension) measure the coverage, fail on a line
  that no test runs and that says no reason, and upload it to Codecov, which feeds the coverage badges of the
  README. See [Test coverage](../codebase/coverage.md#test-coverage)

**PR Requirements**:

- The required checks must pass: `Python Code Quality`, `Python Coverage`,
  `Rust Code Quality`, `Build Documentation` and `Build Passed` (the whole test
  matrix)
- No merge conflicts with target branch
- The merge queue: it merges each pull request on top of the ones queued
  before it, after the required checks pass on that result
- Every Python statement and every Rust line runs in a test, or carries its
  reason: `# pragma: no cover: <reason>` or `// coverage: <reason>`

### 7. Merging

Once its checks pass, add the pull request to the merge queue ("Merge when ready", or
`gh pr merge <number> --squash`): the queue squash-merges it, the only method
`main` allows, once the required checks pass on top of the pull requests before
it. GitHub deletes the branch after the merge (`delete_branch_on_merge`);
delete your local copy with `git branch -D <branch>` after `git fetch --prune`.

A release needs no merge between branches: it is a tag on `main` ([Release
Process](release.md)).

!!! note "Dependabot reads its configuration from `main`"
    Changes to `.github/dependabot.yaml` take effect once they reach `main`.
    Updating the file there also triggers a version-update check right away.

## Continuous Integration Pipelines

| Workflow | Runs on | Checks | Required check |
|---|---|---|---|
| Code Quality (`code-quality.yaml`) | Pull requests, the merge queue and on demand | Python Code Quality: Ruff (lint, format, docstrings `D1`/`D417`) of `py-commol/` and `.github/scripts/`, Ty and Vulture, on the source and the stubs, so it builds nothing. Python Coverage: every test, the documentation checks (`pytest -m docs`) too, under coverage, for every change, the documentation included; it fails under 100 % of the statements. Rust: rustfmt, Clippy (`-D warnings`, so `missing_docs` too), rustdoc without broken links, then the Rust tests and `pytest` on an extension built with the instrumentation of `cargo llvm-cov` (`.github/scripts/rust_coverage.sh`), so the bindings are measured where they run; it fails on a line that no test runs and that has no marker `// coverage: <reason>` (`.github/scripts/rust_uncovered.py`). Both upload their coverage to Codecov (flags `python` and `rust`) with the OIDC token of the run, so no secret is needed; `.github/codecov.yml` makes its checks informational: the thresholds live in the repository ([Test coverage](../codebase/coverage.md#test-coverage)). | `Python Code Quality`, `Python Coverage`, `Rust Code Quality` |
| Build (`build.yaml`) | Pull requests, the merge queue, every Monday on `main` and on demand | Tests, the documentation checks included, without coverage: the job on Ubuntu installs LaTeX and times every test against its budget ([Time budget](../codebase/tests.md#time-budget)). A pull request that changes what the coverage studies run also runs them (`coverage.yaml`): the crates `commol-core`, `commol-model`, `commol-difference`, `commol-calibration`, `commol-random` and `commol-inference`, the bindings (`py-commol/src/`), the Python package, `tests/support/`, the studies, the locks, the manifests and the toolchain. The merge queue does not run them again. A pull request runs one job, Ubuntu with Python 3.12, for a fast answer; the merge queue, the weekly and the manual runs test Ubuntu, Windows and macOS with Python 3.12, 3.13 and 3.14. A failure that only one operating system shows appears in the queue, which then removes the pull request. A change of only documentation or planning files (`docs/`, Markdown at the root, `mkdocs.yaml`, `.github/ISSUE_TEMPLATE/`, `.github/scripts/`) skips the tests. The wheels of every architecture are built by the release (`release.yaml`). | `Build Passed` |
| Documentation (`docs.yaml`) | Pull requests, the merge queue, pushes to `main` and release tags | `mkdocs build --strict`: no broken internal link or anchor. Deploys the site with mike: a release tag as `latest`, a push to `main` as `dev`. | `Build Documentation` |
| Coverage Studies (`coverage.yaml`) | Called by Build for a pull request that changes what the studies run and by the release before it publishes; on demand | `pytest -m coverage_study`: the intervals of each uncertainty method cover the truth at their level ([Design decisions](../codebase/decisions/uncertainty.md#methods-plug-into-one-entry-point)). One job per group of studies, so the slowest study, about 10 min, sets the wait. | `Build Passed`, through Build |
| Links (`links.yaml`) | Pull requests that change `docs/`, `README.md` or `AGENTS.md`; every Monday; on demand | External links, with lychee. | None: a site that is down for a while must not block a pull request |
| Merge queue (`merge_group` in Code Quality, Build and Documentation) | A pull request queued for `main` | The same required checks, on the pull request merged on top of the ones queued before it, at most 2 groups at a time and up to 5 pull requests per group. The tree that reaches `main` is the tree the queue tested, so a push to `main` runs no tests again. | The same |
| Release (`release.yaml`) | A published GitHub release | Builds wheels and the sdist, runs the coverage studies, and publishes to PyPI only when all of them pass. See [Release Process](release.md). | — |

## Local Development Tips

### Quick Commands

--8<-- "AGENTS.md:commands"

### Pre-commit Hooks

Install pre-commit hooks to automatically check code before committing:

```bash
cd py-commol
pre-commit install
```

On every commit, the hooks in `.pre-commit-config.yaml` run:

| Hook | Checks |
|---|---|
| `cargo-fmt` | Rust formatting |
| `cargo-clippy` | Rust lints with `-D warnings`, including `missing_docs` |
| `ty-check` | Python types of `commol` |
| `vulture` | Dead Python code |
| `ruff-check` | Python lints, including docstrings (`D1`, `D417`) |
| `ruff-format` | Python formatting |
| `uv-lock` | `uv.lock` matches `pyproject.toml` |

The Python hooks run their tool with `uv run` from `py-commol/`, so they use
the versions of `uv.lock`, the same as CI. Dependabot updates `uv.lock`, not the
`rev` of a pre-commit repository, so a hook pinned there would drift from CI.

### The Rust toolchain

`rust-toolchain.toml` pins one Rust version, with `clippy` and `rustfmt`.
`rustup` installs and uses it in the repository, so your builds, the
pre-commit hooks and CI (`rustup toolchain install`, and `maturin-action`,
which reads the file) run the same compiler and the same lints. A floating
`stable` would let a lint of a newer clippy pass on your machine and fail in
CI.

Once a month, Dependabot (`.github/dependabot.yaml`, ecosystem
`rust-toolchain`) checks for a newer stable release, and opens a pull
request to `main` that changes `channel`. Its CI runs the new clippy, so
the new lints fail there, not in a feature branch. To move to a newer Rust
by hand, or to fix the lints of that pull request:

1. Change `channel` in `rust-toolchain.toml`.
2. Run `cargo clippy --workspace --all-targets --all-features -- -D warnings`,
   the Rust tests and `cargo doc` (see [Quick Commands](#quick-commands)),
   and fix what the new lints find.
3. Rebuild the extension and run `pytest`.

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

- **`latest`**: Documentation of the last release, built from its tag
- **`dev`**: Development documentation, built from `main`

#### Automatic Deployment

Documentation is deployed automatically:

- **A release tag `vX.Y.Z`**: deploys the `latest` version
- **A push to `main`**: deploys the `dev` version
- **Pull requests**: Build and validate documentation without deploying

Users can switch between versions using the version selector in the documentation site.

#### Manual Deployment (if needed)

If you need to manually deploy documentation:

```bash
# Deploy latest version (from a checkout of the release tag) - with venv activated
cd py-commol
mike deploy --push --update-aliases latest
mike set-default --push latest

# Deploy dev version (from main)
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
