# Package API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The version of the installed commol and the commit it was built from. Save them with a result to name the code that made it. Guide: [Check the installation](https://munqu.github.io/commol/latest/getting-started/installation/#check-the-installation).

`commol.__version__` is the PEP 440 version, the same as `build_info().version`: the version of a release, or the placeholder version with the commit as a local version (`0.1.0+g<first 7 characters of the commit>`) for a build that is not a release.

Commol logs to the logger `commol` and its children, and adds only a `logging.NullHandler`. Your application chooses where the messages go. Guide: [Log messages](https://munqu.github.io/commol/latest/getting-started/quickstart/#log-messages).

## build_info

```python
build_info() -> BuildInfo
```

Return the version of the installed commol and the commit it was built from.

Save it with a result to name the code that made it.

Returns:

| Type        | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| `BuildInfo` | The version and the full commit, read when the extension was built. |

Source code in `commol/api/build_info.py`

```python
def build_info() -> BuildInfo:
    """
    Return the version of the installed commol and the commit it was built from.

    Save it with a result to name the code that made it.

    Returns
    -------
    BuildInfo
        The version and the full commit, read when the extension was built.
    """
    return BuildInfo(version=_commol_rs.__version__, commit=_commol_rs.__commit__)
```

## BuildInfo

Bases: `BaseModel`

The version of the installed commol and the commit it was built from.

Attributes:

| Name      | Type  | Description                                                                                                                                                                                                                                                                                                |
| --------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `version` | `str` | The PEP 440 version, the one commol.__version__ gives. A release gives its version, such as "0.9.0". A build that is not a release, such as an install from a git commit or an editable install, gives the placeholder version of the source with the commit as a local version, such as "0.1.0+g0123abc". |
| `commit`  | \`str | None\`                                                                                                                                                                                                                                                                                                     |

Source code in `commol/api/build_info.py`

```python
class BuildInfo(BaseModel):
    """
    The version of the installed commol and the commit it was built from.

    Attributes
    ----------
    version : str
        The PEP 440 version, the one `commol.__version__` gives. A release
        gives its version, such as ``"0.9.0"``. A build that is not a release,
        such as an install from a git commit or an editable install, gives the
        placeholder version of the source with the commit as a local version,
        such as ``"0.1.0+g0123abc"``.
    commit : str | None
        The full hash of the git commit of the source, or None when the source
        has no git metadata, such as a source distribution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    commit: str | None
```
