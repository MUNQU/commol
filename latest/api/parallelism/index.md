# Parallelism API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The threads of every parallel method and the levels that run in parallel. Guide: [Threads and Parallelism](https://munqu.github.io/commol/latest/guide/parallelism/index.md).

## set_threads

```python
set_threads(n: int, levels: Levels | None = None) -> None
```

Set the number of threads of every parallel method, in this context.

The setting holds in the current Python thread (a `contextvars` context), not in other threads. Without it, commol uses one thread per core and every level in parallel.

Parameters:

| Name     | Type           | Description                                                                           | Default    |
| -------- | -------------- | ------------------------------------------------------------------------------------- | ---------- |
| `n`      | `int`          | The number of threads, at least 1.                                                    | *required* |
| `levels` | `(all, outer)` | Which levels run in parallel when a method nests them; None keeps the current levels. | `"all"`    |

Raises:

| Type         | Description                                |
| ------------ | ------------------------------------------ |
| `ValueError` | If n is below 1 or levels is neither name. |

Source code in `commol/api/parallelism.py`

```python
def set_threads(n: int, levels: Levels | None = None) -> None:
    """
    Set the number of threads of every parallel method, in this context.

    The setting holds in the current Python thread (a `contextvars`
    context), not in other threads. Without it, commol uses one thread per
    core and every level in parallel.

    Parameters
    ----------
    n : int
        The number of threads, at least 1.
    levels : {"all", "outer"} | None
        Which levels run in parallel when a method nests them; `None` keeps
        the current levels.

    Raises
    ------
    ValueError
        If `n` is below 1 or `levels` is neither name.
    """
    commol_rs.parallelism.setting().set(_choice(n, levels))
```

## threads

```python
threads(n: int | None = None, levels: Levels | None = None) -> Generator[None]
```

Limit the threads of every parallel method within a block.

The previous setting comes back when the block ends, also on an error.

Parameters:

| Name     | Type           | Description                                                                           | Default                                                           |
| -------- | -------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `n`      | \`int          | None\`                                                                                | The number of threads, at least 1; None keeps the current number. |
| `levels` | `(all, outer)` | Which levels run in parallel when a method nests them; None keeps the current levels. | `"all"`                                                           |

Yields:

| Type   | Description                      |
| ------ | -------------------------------- |
| `None` | The block runs with the setting. |

Raises:

| Type         | Description                                |
| ------------ | ------------------------------------------ |
| `ValueError` | If n is below 1 or levels is neither name. |

Source code in `commol/api/parallelism.py`

```python
@contextmanager
def threads(n: int | None = None, levels: Levels | None = None) -> Generator[None]:
    """
    Limit the threads of every parallel method within a block.

    The previous setting comes back when the block ends, also on an error.

    Parameters
    ----------
    n : int | None
        The number of threads, at least 1; `None` keeps the current number.
    levels : {"all", "outer"} | None
        Which levels run in parallel when a method nests them; `None` keeps
        the current levels.

    Yields
    ------
    None
        The block runs with the setting.

    Raises
    ------
    ValueError
        If `n` is below 1 or `levels` is neither name.
    """
    variable = commol_rs.parallelism.setting()
    token = variable.set(_choice(n, levels))
    try:
        yield
    finally:
        variable.reset(token)
```

## get_threads

```python
get_threads() -> int
```

The number of threads of the parallel methods in this context.

Returns:

| Type  | Description                                              |
| ----- | -------------------------------------------------------- |
| `int` | The number set, or the number of cores when none is set. |

Source code in `commol/api/parallelism.py`

```python
def get_threads() -> int:
    """
    The number of threads of the parallel methods in this context.

    Returns
    -------
    int
        The number set, or the number of cores when none is set.
    """
    return commol_rs.parallelism.current().threads
```

## get_levels

```python
get_levels() -> Levels
```

The levels that run in parallel in this context.

Returns:

| Type           | Description                                |
| -------------- | ------------------------------------------ |
| `{all, outer}` | The levels set, or "all" when none is set. |

Source code in `commol/api/parallelism.py`

```python
def get_levels() -> Levels:
    """
    The levels that run in parallel in this context.

    Returns
    -------
    {"all", "outer"}
        The levels set, or `"all"` when none is set.
    """
    return cast(Levels, commol_rs.parallelism.current().levels)
```
