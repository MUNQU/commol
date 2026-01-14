from typing import Self

from pydantic import BaseModel, Field


class PlotConfig(BaseModel):
    """
    Configuration for plot layout and styling.

    Attributes
    ----------
    figsize : tuple[float, float]
        Figure size in inches (width, height).
    dpi : int
        Dots per inch for figure resolution.
    layout : tuple[int, int] | None
        Subplot layout as (rows, cols). If None, automatically calculated based on
        number of bins.
    style : str | None
        Seaborn style preset. Must be a valid seaborn style value:
        "darkgrid", "whitegrid", "dark", "white", or "ticks".
        See: https://seaborn.pydata.org/generated/seaborn.set_style.html
    palette : str | None
        Seaborn color palette name. Must be a valid seaborn palette value:
        "deep", "muted", "bright", "pastel", "dark", "colorblind",
        or any matplotlib colormap like "Set1", "Set2", "tab10", etc.
        See: https://seaborn.pydata.org/tutorial/color_palettes.html
    context : str | None
        Seaborn context for scaling plot elements. Must be a valid seaborn context:
        "paper", "notebook", "talk", or "poster".
        See: https://seaborn.pydata.org/generated/seaborn.set_context.html

    Examples
    --------
    >>> # Direct configuration
    >>> config = PlotConfig(style="whitegrid", palette="Set2", context="talk")

    >>> # Using presets
    >>> config = PlotConfig.presentation()
    >>> config = PlotConfig.publication()
    """

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
        description="Subplot layout as (rows, cols). If None, auto-calculated.",
    )
    style: str | None = Field(
        default=None,
        description=(
            "Seaborn style preset: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'"
        ),
    )
    palette: str | None = Field(
        default=None,
        description="Seaborn color palette name (e.g., 'deep', 'Set2', 'colorblind')",
    )
    context: str | None = Field(
        default=None,
        description="Seaborn context: 'paper', 'notebook', 'talk', 'poster'",
    )

    @classmethod
    def presentation(cls) -> Self:
        """
        Create a configuration optimized for presentations.

        Large figure size, high DPI, and "talk" context for readable text on slides.

        Returns
        -------
        PlotConfig
            Configuration with presentation-friendly settings.
        """
        return cls(
            figsize=(16, 10),
            dpi=150,
            style="whitegrid",
            context="talk",
            palette="colorblind",
        )

    @classmethod
    def publication(cls) -> Self:
        """
        Create a configuration optimized for academic publications.

        Smaller figure size suitable for journal columns, high DPI for print quality,
        and minimal styling.

        Returns
        -------
        PlotConfig
            Configuration with publication-friendly settings.
        """
        return cls(
            figsize=(8, 6),
            dpi=300,
            style="ticks",
            context="paper",
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
            style="whitegrid",
            context="notebook",
        )
