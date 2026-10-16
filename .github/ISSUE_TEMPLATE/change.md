---
name: Bug or small change
about: A defect, or a change small enough to need no proposal
title: "<type>(<scope>): <outcome>"
labels: change
---

## Behavior

- **Now:** <what happens, with the error message if any>
- **Expected:** The <component> SHALL <behavior>.

## Reproduce

```python
# The smallest model that shows it, with abstract names (A, B, k1).
```

Test that will reproduce it: `py-commol/tests/<package>/test_<topic>.py::test_<name>`

## Where

<!-- The paths and symbols you suspect, if you know them. -->

## Done when

- [ ] The test fails before the change and passes after it
- [ ] The documentation the change needs, and its changelog entry
