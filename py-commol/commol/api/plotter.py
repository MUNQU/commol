import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from commol.api.simulation import Simulation
from commol.context.calibration import ObservedDataPoint
from commol.context.visualization import (
    PlotConfig,
    SeabornContext,
    SeabornStyle,
    SeabornStyleConfig,
)

logger = logging.getLogger(__name__)


class SimulationPlotter:
    """
    A facade for plotting simulation results using Seaborn.

    This class provides methods to visualize simulation results with automatic
    subplot organization, Seaborn styling, and support for overlaying observed data.

    Attributes
    ----------
    simulation : Simulation
        The simulation instance that generated the results.
    results : dict[str, list[float]]
        Simulation results in dict_of_lists format (bin_id -> values).
    """

    def __init__(
        self,
        simulation: Simulation,
        results: dict[str, list[float]],
    ):
        """
        Initialize the SimulationPlotter.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance that generated the results.
        results : dict[str, list[float]]
            Simulation results in dict_of_lists format.
            Keys are bin IDs, values are lists of population values over time.
        """
        self.simulation = simulation
        self.results = results
        self.bins = list(results.keys())
        self.num_steps = len(next(iter(results.values()))) - 1 if results else 0

        logger.info(
            f"SimulationPlotter initialized with {len(self.bins)} bins "
            f"and {self.num_steps} steps"
        )

    def plot_series(
        self,
        output_file: str | None = None,
        observed_data: list[ObservedDataPoint] | None = None,
        config: PlotConfig | None = None,
        bins: list[str] | None = None,
        seaborn_style: SeabornStyle | None = None,
        palette: str | None = None,
        context: SeabornContext | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Plot simulation results as time series with one subplot per bin.

        Creates a figure with subplots arranged in a grid, where each subplot shows
        the evolution of one bin over time. Optionally overlays observed data points.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure. If None, figure is not saved (only returned).
        observed_data : list[ObservedDataPoint] | None
            Optional observed data points to overlay on corresponding bin subplots.
        config : PlotConfig | None
            Configuration for plot layout and styling. If None, uses defaults.
        bins : list[str] | None
            List of bin IDs to plot. If None, plots all bins.
        seaborn_style : SeabornStyle | None
            Seaborn style preset: "darkgrid", "whitegrid", "dark", "white", "ticks".
            Overrides config if provided.
        palette : str | None
            Color palette name (overrides config if provided).
        context : SeabornContext | None
            Seaborn context: "paper", "notebook", "talk", "poster".
            Overrides config if provided.
        **kwargs : str | int | float | bool | None
            Additional keyword arguments passed to seaborn.lineplot().
            Common parameters: linewidth, alpha, linestyle, marker, etc.

        Returns
        -------
        Figure
            The matplotlib Figure object.

        Examples
        --------
        >>> plotter = SimulationPlotter(simulation, results)
        >>> plotter.plot_series("output.png", seaborn_style="darkgrid")
        """
        logger.info("Starting plot_series")

        # Use provided config or create default
        if config is None:
            config = PlotConfig()

        # Override config with direct parameters if provided
        seaborn_config = self._build_seaborn_config(
            config.seaborn, seaborn_style, palette, context
        )

        # Apply Seaborn styling
        self._apply_seaborn_style(seaborn_config)

        # Select bins to plot
        bins_to_plot = bins if bins is not None else self.bins

        # Calculate layout
        layout = config.layout or self._calculate_layout(len(bins_to_plot))

        # Group observed data by compartment
        observed_by_bin = self._group_observed_data(observed_data)

        # Create figure and subplots
        fig, axes = plt.subplots(
            layout[0], layout[1], figsize=config.figsize, dpi=config.dpi
        )

        # Ensure axes is always a flat array
        if layout[0] == 1 and layout[1] == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Plot each bin
        for idx, bin_id in enumerate(bins_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]
            self._plot_bin_series(
                ax,
                bin_id,
                observed_by_bin.get(bin_id, []),
                dict(kwargs),
            )

        # Hide unused subplots
        for idx in range(len(bins_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
            logger.info(f"Series plot saved to {output_file}")

        plt.show()

        return fig

    def plot_cumulative(
        self,
        output_file: str | None = None,
        observed_data: list[ObservedDataPoint] | None = None,
        config: PlotConfig | None = None,
        bins: list[str] | None = None,
        seaborn_style: SeabornStyle | None = None,
        palette: str | None = None,
        context: SeabornContext | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Plot cumulative (accumulated) simulation results with one subplot per bin.

        Creates a figure showing the running sum of each bin's values over time.
        Useful for tracking total infections, deaths, or other accumulated quantities.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure. If None, figure is not saved (only returned).
        observed_data : list[ObservedDataPoint] | None
            Optional observed data points to overlay (also shown as cumulative).
        config : PlotConfig | None
            Configuration for plot layout and styling. If None, uses defaults.
        bins : list[str] | None
            List of bin IDs to plot. If None, plots all bins.
        seaborn_style : SeabornStyle | None
            Seaborn style preset: "darkgrid", "whitegrid", "dark", "white", "ticks".
            Overrides config if provided.
        palette : str | None
            Color palette name (overrides config if provided).
        context : SeabornContext | None
            Seaborn context: "paper", "notebook", "talk", "poster".
            Overrides config if provided.
        **kwargs : str | int | float | bool | None
            Additional keyword arguments passed to seaborn.lineplot().
            Common parameters: linewidth, alpha, linestyle, marker, etc.

        Returns
        -------
        Figure
            The matplotlib Figure object.

        Examples
        --------
        >>> plotter = SimulationPlotter(simulation, results)
        >>> plotter.plot_cumulative("cumulative.png", bins=["I", "R"])
        """
        logger.info("Starting plot_cumulative")

        # Use provided config or create default
        if config is None:
            config = PlotConfig()

        # Override config with direct parameters if provided
        seaborn_config = self._build_seaborn_config(
            config.seaborn, seaborn_style, palette, context
        )

        # Apply Seaborn styling
        self._apply_seaborn_style(seaborn_config)

        # Select bins to plot
        bins_to_plot = bins if bins is not None else self.bins

        # Calculate cumulative results
        cumulative_results = self._calculate_cumulative(bins_to_plot)

        # Calculate layout
        layout = config.layout or self._calculate_layout(len(bins_to_plot))

        # Group and accumulate observed data
        observed_by_bin = self._group_observed_data(observed_data)
        cumulative_observed = self._calculate_cumulative_observed(observed_by_bin)

        # Create figure and subplots
        fig, axes = plt.subplots(
            layout[0], layout[1], figsize=config.figsize, dpi=config.dpi
        )

        # Ensure axes is always a flat array
        if layout[0] == 1 and layout[1] == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Plot each bin
        for idx, bin_id in enumerate(bins_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]
            self._plot_bin_cumulative(
                ax,
                bin_id,
                cumulative_results[bin_id],
                cumulative_observed.get(bin_id, []),
                dict(kwargs),
            )

        # Hide unused subplots
        for idx in range(len(bins_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
            logger.info(f"Cumulative plot saved to {output_file}")

        plt.show()

        return fig

    def _build_seaborn_config(
        self,
        base_config: SeabornStyleConfig,
        style: SeabornStyle | None,
        palette: str | None,
        context: SeabornContext | None,
    ) -> SeabornStyleConfig:
        """
        Build Seaborn configuration, with direct parameters overriding config.
        """
        return SeabornStyleConfig(
            style=style if style is not None else base_config.style,
            palette=palette if palette is not None else base_config.palette,
            context=context if context is not None else base_config.context,
        )

    def _apply_seaborn_style(self, config: SeabornStyleConfig) -> None:
        """
        Apply Seaborn styling configuration.
        """
        if config.style:
            sns.set_style(config.style)
            logger.debug(f"Applied Seaborn style: {config.style}")

        if config.palette:
            sns.set_palette(config.palette)
            logger.debug(f"Applied Seaborn palette: {config.palette}")

        if config.context:
            sns.set_context(config.context)
            logger.debug(f"Applied Seaborn context: {config.context}")

    def _calculate_layout(self, num_bins: int) -> tuple[int, int]:
        """
        Calculate optimal subplot layout (rows, cols) for given number of bins.
        """
        if num_bins == 1:
            return (1, 1)
        elif num_bins == 2:
            return (1, 2)
        elif num_bins <= 4:
            return (2, 2)
        else:
            # For more bins, try to make a roughly square grid
            cols = math.ceil(math.sqrt(num_bins))
            rows = math.ceil(num_bins / cols)
            return (rows, cols)

    def _group_observed_data(
        self, observed_data: list[ObservedDataPoint] | None
    ) -> dict[str, list[ObservedDataPoint]]:
        """
        Group observed data points by compartment (bin) ID.
        """
        if not observed_data:
            return {}

        grouped: dict[str, list[ObservedDataPoint]] = defaultdict(list)
        for point in observed_data:
            grouped[point.compartment].append(point)

        # Sort each group by step
        for compartment in grouped:
            grouped[compartment].sort(key=lambda p: p.step)

        logger.debug(
            f"Grouped {len(observed_data)} observed data points into "
            f"{len(grouped)} compartments"
        )

        return dict(grouped)

    def _plot_bin_series(
        self,
        ax: "Axes",
        bin_id: str,
        observed: list[ObservedDataPoint],
        plot_kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot time series for a single bin on given axes.
        """
        time_steps = list(range(len(self.results[bin_id])))
        values = self.results[bin_id]

        # Build parameters for lineplot
        params = {
            "x": time_steps,
            "y": values,
            "ax": ax,
            "label": "Simulation",
        }
        params.update(plot_kwargs)

        # Plot simulation results as line
        sns.lineplot(**params)

        # Overlay observed data if available
        if observed:
            obs_steps = [p.step for p in observed]
            obs_values = [p.value for p in observed]
            sns.scatterplot(
                x=obs_steps,
                y=obs_values,
                ax=ax,
                label="Observed",
                color="red",
                s=30,
                alpha=0.7,
                zorder=5,
            )

        # Get bin unit from model for label
        bin_obj = next(
            (
                b
                for b in self.simulation.model_definition.population.bins
                if b.id == bin_id
            ),
            None,
        )
        unit_str = f"{bin_obj.unit}" if bin_obj and bin_obj.unit else ""
        bin_name = bin_obj.name if bin_obj and bin_obj.name else bin_id

        ax.set_xlabel("Step")
        ax.set_ylabel(f"{unit_str}")
        ax.set_title(f"{bin_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _calculate_cumulative(self, bins_to_plot: list[str]) -> dict[str, list[float]]:
        """
        Calculate cumulative (running sum) for specified bins.
        """
        cumulative: dict[str, list[float]] = {}

        for bin_id in bins_to_plot:
            values = self.results[bin_id]
            cumsum = []
            running_total = 0.0

            for value in values:
                running_total += value
                cumsum.append(running_total)

            cumulative[bin_id] = cumsum

        return cumulative

    def _calculate_cumulative_observed(
        self, observed_by_bin: dict[str, list[ObservedDataPoint]]
    ) -> dict[str, list[tuple[int, float]]]:
        """
        Calculate cumulative observed data (step, cumulative_value).
        """
        cumulative: dict[str, list[tuple[int, float]]] = {}

        for bin_id, points in observed_by_bin.items():
            cumsum = []
            running_total = 0.0

            for point in points:
                running_total += point.value
                cumsum.append((point.step, running_total))

            cumulative[bin_id] = cumsum

        return cumulative

    def _plot_bin_cumulative(
        self,
        ax: "Axes",
        bin_id: str,
        cumulative_values: list[float],
        cumulative_observed: list[tuple[int, float]],
        plot_kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot cumulative data for a single bin on given axes.
        """
        time_steps = list(range(len(cumulative_values)))

        # Build parameters for lineplot
        params = {
            "x": time_steps,
            "y": cumulative_values,
            "ax": ax,
            "label": "Simulation (Cumulative)",
        }
        params.update(plot_kwargs)

        # Plot cumulative simulation results
        sns.lineplot(**params)

        # Overlay cumulative observed data if available
        if cumulative_observed:
            obs_steps = [step for step, _ in cumulative_observed]
            obs_values = [value for _, value in cumulative_observed]
            sns.scatterplot(
                x=obs_steps,
                y=obs_values,
                ax=ax,
                label="Observed (Cumulative)",
                color="red",
                s=30,
                alpha=0.7,
                zorder=5,
            )

        # Get bin unit from model for label
        bin_obj = next(
            (
                b
                for b in self.simulation.model_definition.population.bins
                if b.id == bin_id
            ),
            None,
        )
        unit_str = f"{bin_obj.unit}" if bin_obj and bin_obj.unit else ""
        bin_name = bin_obj.name if bin_obj and bin_obj.name else bin_id

        ax.set_xlabel("Step")
        ax.set_ylabel(f"{unit_str}")
        ax.set_title(f"{bin_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
