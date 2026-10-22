//! The version and the commit of a build of the extension.
//!
//! The build script (`py-commol/build.rs`) includes this file and writes what
//! it finds into the extension: `COMMOL_VERSION`, the PEP 440 version of
//! `commol.__version__`, and `COMMOL_COMMIT`, the full commit of the source,
//! empty when the source has no git metadata. The crate compiles the file
//! only in its tests.
//!
//! The commit comes from the git metadata itself, not from the `git` program,
//! so a build in a container without git, or in a checkout that another user
//! owns, reads it too.

use std::fs;
use std::path::{Path, PathBuf};

/// The version that `py-commol/Cargo.toml` holds outside a release.
///
/// A release writes the version of its tag over it
/// (`.github/actions/set-version`), so a build with this version is not a
/// release.
pub const PLACEHOLDER: &str = "0.1.0";

/// The PEP 440 version of a build.
///
/// A release gives its version, with a SemVer pre-release written as PEP 440
/// writes it (`0.9.0-beta.1` gives `0.9.0b1`, as maturin writes the version of
/// the package). A build of the placeholder version with a commit adds the
/// commit as a local version: `0.1.0+g<first 7 characters of the commit>`.
pub fn version(cargo_version: &str, commit: Option<&str>) -> String {
    let version = pep440(cargo_version);
    match commit {
        Some(commit) if cargo_version == PLACEHOLDER => format!("{version}+g{}", &commit[..7]),
        _ => version,
    }
}

/// A SemVer version written as PEP 440 writes it: `-alpha.N`, `-beta.N` and
/// `-rc.N` become `aN`, `bN` and `rcN`; any other version stays as it is.
fn pep440(cargo_version: &str) -> String {
    let Some((release, pre)) = cargo_version.split_once('-') else {
        return cargo_version.to_owned();
    };
    let Some((kind, number)) = pre.split_once('.') else {
        return cargo_version.to_owned();
    };
    let letter = match kind {
        "alpha" => "a",
        "beta" => "b",
        "rc" => "rc",
        _ => return cargo_version.to_owned(),
    };
    format!("{release}{letter}{number}")
}

/// The git metadata of a source tree.
#[derive(Debug, PartialEq)]
pub struct GitDir {
    /// The directory of `HEAD`: `.git`, or `.git/worktrees/<name>` for a
    /// linked worktree.
    head_dir: PathBuf,
    /// The directory of the branches and `packed-refs`, which every worktree
    /// of a repository shares.
    common_dir: PathBuf,
}

impl GitDir {
    /// The git metadata of the closest directory, from `start` up, that holds
    /// a `.git`: a directory, or a file `gitdir: <path>` (a linked worktree
    /// or a submodule). None when no directory holds one.
    pub fn find(start: &Path) -> Option<Self> {
        let dir = start.ancestors().find(|dir| dir.join(".git").exists())?;
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return Some(Self::at(dot_git));
        }
        let link = fs::read_to_string(&dot_git).ok()?;
        let path = link.strip_prefix("gitdir:")?.trim();
        Some(Self::at(dir.join(path)))
    }

    /// The metadata whose `HEAD` is in `head_dir`; its file `commondir`, if
    /// any, names the directory of the branches.
    fn at(head_dir: PathBuf) -> Self {
        let common_dir = fs::read_to_string(head_dir.join("commondir"))
            .map_or_else(|_| head_dir.clone(), |common| head_dir.join(common.trim()));
        Self {
            head_dir,
            common_dir,
        }
    }

    /// The full commit of `HEAD`, through its branch when `HEAD` names one.
    /// None when the commit cannot be read.
    pub fn commit(&self) -> Option<String> {
        let commit = match self.head()?.strip_prefix("ref:") {
            Some(branch) => self.resolve(branch.trim())?,
            None => self.head()?,
        };
        is_commit(&commit).then_some(commit)
    }

    /// The content of `HEAD`: a commit, or `ref: <branch>`.
    fn head(&self) -> Option<String> {
        let head = fs::read_to_string(self.head_dir.join("HEAD")).ok()?;
        Some(head.trim().to_owned())
    }

    /// The commit of a branch: its own file, or its line in `packed-refs`.
    fn resolve(&self, branch: &str) -> Option<String> {
        if let Ok(commit) = fs::read_to_string(self.common_dir.join(branch)) {
            return Some(commit.trim().to_owned());
        }
        let packed = fs::read_to_string(self.common_dir.join("packed-refs")).ok()?;
        packed.lines().find_map(|line| {
            let (commit, name) = line.split_once(' ')?;
            (name == branch).then(|| commit.to_owned())
        })
    }

    /// The files that change when `HEAD` moves to another commit, those that
    /// exist: `HEAD`, its log, the file of its branch and `packed-refs`. The
    /// build script runs again when one of them changes.
    pub fn watched(&self) -> Vec<PathBuf> {
        let mut files = vec![
            self.head_dir.join("HEAD"),
            self.head_dir.join("logs").join("HEAD"),
            self.common_dir.join("packed-refs"),
        ];
        if let Some(branch) = self.head().as_deref().and_then(|h| h.strip_prefix("ref:")) {
            files.push(self.common_dir.join(branch.trim()));
        }
        files.retain(|file| file.is_file());
        files
    }
}

/// Whether `text` is a full commit: 40 hexadecimal digits (SHA-1), or 64
/// (SHA-256).
fn is_commit(text: &str) -> bool {
    matches!(text.len(), 40 | 64) && text.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const OTHER: &str = "89abcdef0123456789abcdef0123456789abcdef";

    /// An empty directory of its own under the temporary directory.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("commol-build-info-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("a scratch directory");
        dir
    }

    fn write(path: &Path, text: &str) {
        fs::create_dir_all(path.parent().expect("a parent")).expect("its parent");
        fs::write(path, text).expect("a file");
    }

    #[test]
    fn a_release_gives_its_version_and_the_placeholder_names_its_commit() {
        assert_eq!(version("0.9.0", Some(COMMIT)), "0.9.0");
        assert_eq!(version("0.9.0-beta.1", None), "0.9.0b1");
        assert_eq!(version(PLACEHOLDER, Some(COMMIT)), "0.1.0+g0123456");
        assert_eq!(version(PLACEHOLDER, None), "0.1.0");
    }

    #[test]
    fn a_pre_release_is_written_as_pep_440_writes_it() {
        assert_eq!(pep440("1.2.3-alpha.4"), "1.2.3a4");
        assert_eq!(pep440("1.2.3-beta.4"), "1.2.3b4");
        assert_eq!(pep440("1.2.3-rc.4"), "1.2.3rc4");
        assert_eq!(pep440("1.2.3-dev.4"), "1.2.3-dev.4");
        assert_eq!(pep440("1.2.3-beta"), "1.2.3-beta");
    }

    #[test]
    fn the_commit_of_a_branch_comes_from_its_file_or_from_packed_refs() {
        let root = scratch("branch");
        let git = root.join(".git");
        write(&git.join("HEAD"), "ref: refs/heads/main\n");
        write(&git.join("refs/heads/main"), &format!("{COMMIT}\n"));
        write(
            &git.join("packed-refs"),
            &format!("# pack-refs with: peeled\n{OTHER} refs/heads/other\n"),
        );
        let source = root.join("py-commol");
        fs::create_dir_all(&source).expect("a source directory");

        let found = GitDir::find(&source).expect("the git metadata");
        assert_eq!(found.commit().as_deref(), Some(COMMIT));
        let mut watched = found.watched();
        watched.sort();
        assert_eq!(
            watched,
            vec![
                git.join("HEAD"),
                git.join("packed-refs"),
                git.join("refs/heads/main")
            ]
        );

        write(&git.join("HEAD"), "ref: refs/heads/other\n");
        assert_eq!(found.commit().as_deref(), Some(OTHER));
        write(&git.join("HEAD"), "ref: refs/heads/none\n");
        assert_eq!(found.commit(), None);
        fs::remove_dir_all(&root).expect("removed");
    }

    #[test]
    fn a_linked_worktree_reads_its_head_and_the_shared_branches() {
        let root = scratch("worktree");
        let common = root.join("main/.git");
        let head_dir = common.join("worktrees/linked");
        write(&head_dir.join("HEAD"), "ref: refs/heads/feature\n");
        write(&head_dir.join("commondir"), "../..\n");
        write(&head_dir.join("logs/HEAD"), "a log\n");
        write(&common.join("refs/heads/feature"), COMMIT);
        let linked = root.join("linked");
        write(
            &linked.join(".git"),
            &format!("gitdir: {}\n", head_dir.display()),
        );

        let found = GitDir::find(&linked).expect("the git metadata");
        assert_eq!(found.commit().as_deref(), Some(COMMIT));
        assert_eq!(found.watched().len(), 3);
        fs::remove_dir_all(&root).expect("removed");
    }

    #[test]
    fn a_detached_head_is_its_commit_and_anything_else_is_none() {
        let root = scratch("detached");
        let git = root.join(".git");
        write(&git.join("HEAD"), &format!("{COMMIT}\n"));
        let found = GitDir::find(&root).expect("the git metadata");
        assert_eq!(found.commit().as_deref(), Some(COMMIT));
        assert_eq!(found.watched(), vec![git.join("HEAD")]);

        write(&git.join("HEAD"), "not a commit\n");
        assert_eq!(found.commit(), None);
        fs::remove_file(git.join("HEAD")).expect("removed");
        assert_eq!(found.commit(), None);
        assert!(found.watched().is_empty());

        let file = scratch("not-a-link");
        write(&file.join(".git"), "something else\n");
        assert_eq!(GitDir::find(&file), None);
        fs::remove_dir_all(&root).expect("removed");
        fs::remove_dir_all(&file).expect("removed");
    }
}
