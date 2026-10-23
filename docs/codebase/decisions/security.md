# Security

[Back to Design decisions](../design-decisions.md)

How user expressions are checked before they reach the engine.

## Every user expression is validated

Rate expressions (`context/transition.py`), axis patterns (`api/patterns.py`)
and calibration constraints (`context/calibration.py`) pass through
`validate_expression_security` (`utils/security.py`) before they reach the
engine. The limits are in
`SecurityConfig`.

**Rule:** any new field that accepts an expression must call it.

**Guard:** `tests/rates/test_security.py` (the limits and the rejected
expressions); `tests/axis_patterns/test_patterns.py` (a pattern calls it).
