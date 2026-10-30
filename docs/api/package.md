# Package API

[Back to API Reference](index.md)

The version of the installed commol and the commit it was built from. Save
them with a result to name the code that made it. Guide:
[Check the installation](../getting-started/installation.md#check-the-installation).

`commol.__version__` is the PEP 440 version, the same as
`build_info().version`: the version of a release, or the placeholder
version with the commit as a local version (`0.1.0+g<first 7 characters of
the commit>`) for a build that is not a release.

Commol logs to the logger `commol` and to one child per module, such as
`commol.resampling`, and adds only a `logging.NullHandler`. Your application chooses where the messages go. Guide:
[Log messages](../getting-started/quickstart.md#log-messages).

::: commol.build_info

::: commol.BuildInfo
