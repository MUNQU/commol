---
name: Release
about: The checklist of one release, blocked by the issues it ships
title: "Release X.Y.Z"
labels: release
---

Follow [the release process](https://github.com/MUNQU/commol/blob/develop/docs/development/release.md) once every issue that blocks this one is closed. Put this issue in the milestone of the release, and mark the issues it ships as blocking it.

- [ ] Every issue of the milestone closed
- [ ] Version chosen
- [ ] Changelog dated in `develop`, with its opening paragraph
- [ ] `develop` merged into `main` with a merge commit
- [ ] **Build** run on `main` passed
- [ ] Annotated tag `vX.Y.Z` created on `main` and pushed
- [ ] GitHub release published with the notes of `python .github/scripts/release_notes.py X.Y.Z`
- [ ] **Release to PyPI** workflow passed
- [ ] Version visible on PyPI and tested in a clean environment
- [ ] Release issue and milestone closed
