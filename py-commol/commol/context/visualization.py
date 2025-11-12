from typing import Literal

from pydantic import BaseModel, Field

# Type aliases for Seaborn parameters
SeabornStyle = Literal["darkgrid", "whitegrid", "dark", "white", "ticks"]
SeabornContext = Literal["paper", "notebook", "talk", "poster"]


class SeabornStyleConfig(BaseModel):
    """
    Configuration for Seaborn styling options.

    Attributes
    ----------
    style : str | None
        Seaborn style preset. Options: "darkgrid", "whitegrid", "dark", "white",
        "ticks".
    palette : str | None
        Color palette name. Examples: "deep", "muted", "bright", "pastel", "dark",
        "colorblind", "Set1", "Set2", "Set3", "tab10", etc.
    context : str | None
        Seaborn context for scaling plot elements. Options: "paper", "notebook",
        "talk", "poster".
    """

    style: SeabornStyle | None = Field(
        default=None,
        description="Seaborn style preset",
    )
    palette: str | None = Field(
        default=None,
        description="Color palette name",
    )
    context: SeabornContext | None = Field(
        default=None,
        description="Seaborn context for scaling plot elements",
    )


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
    seaborn : SeabornStyleConfig
        Seaborn-specific styling configuration.
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
    seaborn: SeabornStyleConfig = Field(
        default_factory=SeabornStyleConfig,
        description="Seaborn styling configuration",
    )
