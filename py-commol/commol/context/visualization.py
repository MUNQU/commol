"""Plot configuration."""

from typing import Self

from pydantic import ConfigDict, BaseModel, Field


class PlotConfig(BaseModel):
    """
    Configuration for plot layout and styling.

    Attributes
    ----------
    figsize : tuple[float, float]
        Figure size in inches (width, height), of a figure of running totals;
        the other figures take their size from their panels.
    dpi : int
        Dots per inch for figure resolution.
    layout : tuple[int, int] | None
        Panel layout as (rows, cols). If None, computed from the number of
        panels.
    style : str | list[str] | None
        A matplotlib style, or a list of styles applied in order: a name of
        ``matplotlib.style.available`` (such as ``"ggplot"`` or
        ``"seaborn-v0_8-whitegrid"``) or the path of a style file.
        See: https://matplotlib.org/stable/users/explain/customizing.html

    Notes
    -----
    The style applies to the figure of one call only: after the call,
    ``matplotlib.rcParams`` is as it was before it, so the figures that you
    draw with matplotlib keep your style. Given ``axes=``, a figure draws on
    the panels of an earlier figure, and `figsize`, `dpi` and `layout` are
    those of that figure.

    Examples
    --------
    >>> # Direct configuration
    >>> config = PlotConfig(style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-talk"])

    >>> # Using presets
    >>> config = PlotConfig.presentation()
    >>> config = PlotConfig.publication()
    """

    model_config = ConfigDict(extra="forbid")

    figsize: tuple[float, float] = Field(
        default=(12, 8),
        description="Figure size in inches (width, height)",
    )
    dpi: int = Field(
        default=100,
        description="Dots per inch for figure resolution",
    )
    layout: tuple[int, int] | None = Field(
        default=None,
        description="Panel layout as (rows, cols). If None, computed.",
    )
    style: str | list[str] | None = Field(
        default=None,
        description="A matplotlib style, or styles applied in order",
    )

    @classmethod
    def presentation(cls) -> Self:
        """
        Create a configuration optimized for presentations.

        Large figure size, high DPI, a white grid, the large text of the
        "talk" style and a palette for colour-blind readers.

        Returns
        -------
        PlotConfig
            Configuration with presentation-friendly settings.
        """
        return cls(
            figsize=(16, 10),
            dpi=150,
            style=[
                "seaborn-v0_8-whitegrid",
                "seaborn-v0_8-talk",
                "seaborn-v0_8-colorblind",
            ],
        )

    @classmethod
    def publication(cls) -> Self:
        """
        Create a configuration optimized for academic publications.

        Smaller figure size suitable for journal columns, high DPI for print
        quality, ticks and the small text of the "paper" style.

        Returns
        -------
        PlotConfig
            Configuration with publication-friendly settings.
        """
        return cls(
            figsize=(8, 6),
            dpi=300,
            style=["seaborn-v0_8-ticks", "seaborn-v0_8-paper"],
        )

    @classmethod
    def notebook(cls) -> Self:
        """
        Create a configuration optimized for Jupyter notebooks.

        Balanced defaults for interactive exploration.

        Returns
        -------
        PlotConfig
            Configuration with notebook-friendly settings.
        """
        return cls(
            figsize=(12, 8),
            dpi=100,
            style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-notebook"],
        )
