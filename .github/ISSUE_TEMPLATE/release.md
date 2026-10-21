---
name: Release
about: The checklist of one release, blocked by the issues it ships
title: "Release X.Y.Z"
labels: release
---

Follow [the release process](https://github.com/MUNQU/commol/blob/main/docs/development/release.md) once every issue that blocks this one is closed. Put this issue in the milestone of the release, and name the milestone by its version (`X.Y.Z`): this issue waits for the proposals and changes of the milestone, and the issues of the later milestones wait for this one (`python .github/scripts/parallel.py --releases` writes both links).

- [ ] Every issue of the milestone closed, or moved to another milestone with a comment that says why
- [ ] Version chosen
- [ ] Changelog dated on `main`, with its opening paragraph
- [ ] Annotated tag `vX.Y.Z` created on `main` and pushed
- [ ] GitHub release published with the notes of `python .github/scripts/release_notes.py X.Y.Z`
- [ ] **Release to PyPI** workflow passed
- [ ] Version visible on PyPI and tested in a clean environment
- [ ] Release issue and milestone closed
