"""The version of the installed commol and the commit it was built from.

The build script of the extension (`py-commol/build.rs`) reads both when the
extension is compiled; this module only passes them on.
"""

from pydantic import BaseModel, ConfigDict

from commol.commol_rs import _commol_rs


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
