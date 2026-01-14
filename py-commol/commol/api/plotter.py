import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from commol.api.simulation import Simulation
from commol.context.calibration import CalibrationResult, ObservedDataPoint
from commol.context.probabilistic_calibration import ProbabilisticCalibrationResult
from commol.context.visualization import PlotConfig

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
        scale_values: dict[str, float] | None = None,
        calibration_result: CalibrationResult
        | ProbabilisticCalibrationResult
        | None = None,
        config: PlotConfig | None = None,
        bins: list[str] | None = None,
        seaborn_style: str | None = None,
        palette: str | None = None,
        context: str | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Plot simulation results as time series with one subplot per bin.

        Creates a figure with subplots arranged in a grid, where each subplot shows
        the evolution of one bin over time. Optionally overlays observed data points.
        If a ProbabilisticCalibrationResult is provided, plots confidence intervals.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure. If None, figure is not saved (only returned).
        observed_data : list[ObservedDataPoint] | None
            Optional observed data points to overlay on corresponding bin subplots.
        scale_values : dict[str, float] | None
            Optional calibrated scale values
            (e.g., from CalibrationResult.best_parameters).
            Maps scale_id to scale value. Observed data points with a scale_id will be
            unscaled for plotting: unscaled_value = observed_value / scale.
            This allows observed data to be comparable with model predictions.
        calibration_result : CalibrationResult | ProbabilisticCalibrationResult | None
            Optional calibration result. If ProbabilisticCalibrationResult is provided,
            plots the median prediction with confidence interval bands.
            If CalibrationResult is provided, uses best_parameters as scale_values.
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
        """
        logger.info("Starting plot_series")

        config = config or PlotConfig()
        bins_to_plot = bins if bins is not None else self.bins

        self._setup_series_style(config, seaborn_style, palette, context)
        scale_values = self._extract_series_scale_values(
            calibration_result, scale_values
        )

        observed_by_bin = self._group_observed_data(observed_data)

        fig, axes = self._create_series_figure(config, bins_to_plot)
        self._plot_all_series_bins(
            axes,
            bins_to_plot,
            observed_by_bin,
            scale_values,
            calibration_result,
            kwargs,
        )
        self._finalize_series_plot(axes, bins_to_plot, output_file, config)

        return fig

    def plot_cumulative(
        self,
        output_file: str | None = None,
        observed_data: list[ObservedDataPoint] | None = None,
        scale_values: dict[str, float] | None = None,
        calibration_result: CalibrationResult
        | ProbabilisticCalibrationResult
        | None = None,
        config: PlotConfig | None = None,
        bins: list[str] | None = None,
        seaborn_style: str | None = None,
        palette: str | None = None,
        context: str | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Plot cumulative (accumulated) simulation results with one subplot per bin.

        Creates a figure showing the running sum of each bin's values over time.
        Useful for tracking total infections, deaths, or other accumulated quantities.
        If a ProbabilisticCalibrationResult is provided, plots confidence intervals.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure. If None, figure is not saved (only returned).
        observed_data : list[ObservedDataPoint] | None
            Optional observed data points to overlay (also shown as cumulative).
        scale_values : dict[str, float] | None
            Optional calibrated scale values
            (e.g., from CalibrationResult.best_parameters).
            Maps scale_id to scale value. Observed data points with a scale_id will be
            unscaled for plotting: unscaled_value = observed_value / scale.
        calibration_result : CalibrationResult | ProbabilisticCalibrationResult | None
            Optional calibration result. If ProbabilisticCalibrationResult is provided,
            plots the median prediction with confidence interval bands.
            If CalibrationResult is provided, uses best_parameters as scale_values.
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
        """
        logger.info("Starting plot_cumulative")

        config = config or PlotConfig()
        bins_to_plot = bins if bins is not None else self.bins

        self._setup_cumulative_style(config, seaborn_style, palette, context)
        scale_values = self._extract_cumulative_scale_values(
            calibration_result, scale_values
        )

        observed_by_bin = self._group_observed_data(observed_data)
        cumulative_observed = self._calculate_cumulative_observed(
            observed_by_bin, scale_values or {}
        )

        fig, axes = self._create_cumulative_figure(config, bins_to_plot)
        self._plot_all_cumulative_bins(
            axes, bins_to_plot, cumulative_observed, calibration_result, kwargs
        )
        self._finalize_cumulative_plot(axes, bins_to_plot, output_file, config)

        return fig

    def _apply_seaborn_style(
        self,
        config: PlotConfig,
        style: str | None = None,
        palette: str | None = None,
        context: str | None = None,
    ) -> None:
        """
        Apply Seaborn styling configuration.

        Direct parameters override config values.
        """
        effective_style = style if style is not None else config.style
        effective_palette = palette if palette is not None else config.palette
        effective_context = context if context is not None else config.context

        if effective_style:
            sns.set_style(effective_style)
            logger.debug(f"Applied Seaborn style: {effective_style}")

        if effective_palette:
            sns.set_palette(effective_palette)
            logger.debug(f"Applied Seaborn palette: {effective_palette}")

        if effective_context:
            sns.set_context(effective_context)
            logger.debug(f"Applied Seaborn context: {effective_context}")

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

    def _setup_series_style(
        self,
        config: PlotConfig,
        seaborn_style: str | None,
        palette: str | None,
        context: str | None,
    ) -> None:
        """
        Set up plot styling with Seaborn configuration for series plots.
        """
        self._apply_seaborn_style(config, seaborn_style, palette, context)

    def _extract_series_scale_values(
        self,
        calib_result: CalibrationResult | ProbabilisticCalibrationResult | None,
        scale_values: dict[str, float] | None,
    ) -> dict[str, float] | None:
        """
        Extract scale values from calibration result for series plotting.
        """
        if calib_result is None:
            return scale_values

        if isinstance(calib_result, CalibrationResult):
            return scale_values or calib_result.best_parameters

        if isinstance(calib_result, ProbabilisticCalibrationResult):
            if scale_values is None:
                return {
                    param_name: stats.median
                    for (
                        param_name,
                        stats,
                    ) in calib_result.selected_ensemble.parameter_statistics.items()
                    if param_name.startswith("scale_")
                }

        return scale_values

    def _create_series_figure(
        self, config: PlotConfig, bins_to_plot: list[str]
    ) -> tuple["Figure", list["Axes"]]:
        """
        Create figure and axes array for series plotting.
        """
        layout = config.layout or self._calculate_layout(len(bins_to_plot))
        fig, axes = plt.subplots(
            layout[0], layout[1], figsize=config.figsize, dpi=config.dpi
        )

        # Ensure axes is always a flat array
        if layout[0] == 1 and layout[1] == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        return fig, axes

    def _plot_all_series_bins(
        self,
        axes: list["Axes"],
        bins_to_plot: list[str],
        observed_by_bin: dict[str, list[ObservedDataPoint]],
        scale_values: dict[str, float] | None,
        calibration_result: CalibrationResult | ProbabilisticCalibrationResult | None,
        kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot series data for all bins across subplots.
        """
        for idx, bin_id in enumerate(bins_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]

            if isinstance(calibration_result, ProbabilisticCalibrationResult):
                self._plot_bin_series_probabilistic(
                    ax,
                    bin_id,
                    observed_by_bin.get(bin_id, []),
                    scale_values or {},
                    calibration_result,
                    dict(kwargs),
                )
            else:
                self._plot_bin_series(
                    ax,
                    bin_id,
                    observed_by_bin.get(bin_id, []),
                    scale_values or {},
                    dict(kwargs),
                )

    def _finalize_series_plot(
        self,
        axes: list["Axes"],
        bins_to_plot: list[str],
        output_file: str | None,
        config: PlotConfig,
    ) -> None:
        """
        Finalize series plot by hiding unused subplots and saving.
        """
        for idx in range(len(bins_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
            logger.info(f"Series plot saved to {output_file}")

    def _setup_cumulative_style(
        self,
        config: PlotConfig,
        seaborn_style: str | None,
        palette: str | None,
        context: str | None,
    ) -> None:
        """
        Set up plot styling with Seaborn configuration for cumulative plots.
        """
        self._apply_seaborn_style(config, seaborn_style, palette, context)

    def _extract_cumulative_scale_values(
        self,
        calib_result: CalibrationResult | ProbabilisticCalibrationResult | None,
        scale_values: dict[str, float] | None,
    ) -> dict[str, float] | None:
        """
        Extract scale values from calibration result for cumulative plotting.
        """
        if calib_result is None:
            return scale_values

        if isinstance(calib_result, CalibrationResult):
            return scale_values or calib_result.best_parameters

        if isinstance(calib_result, ProbabilisticCalibrationResult):
            if scale_values is None:
                return {
                    param_name: stats.median
                    for (
                        param_name,
                        stats,
                    ) in calib_result.selected_ensemble.parameter_statistics.items()
                    if param_name.startswith("scale_")
                }

        return scale_values

    def _create_cumulative_figure(
        self, config: PlotConfig, bins_to_plot: list[str]
    ) -> tuple["Figure", list["Axes"]]:
        """
        Create figure and axes array for cumulative plotting.
        """
        layout = config.layout or self._calculate_layout(len(bins_to_plot))
        fig, axes = plt.subplots(
            layout[0], layout[1], figsize=config.figsize, dpi=config.dpi
        )

        # Ensure axes is always a flat array
        if layout[0] == 1 and layout[1] == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        return fig, axes

    def _plot_all_cumulative_bins(
        self,
        axes: list["Axes"],
        bins_to_plot: list[str],
        cumulative_observed: dict[str, list[tuple[int, float]]],
        calibration_result: CalibrationResult | ProbabilisticCalibrationResult | None,
        kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot cumulative data for all bins across subplots.
        """
        for idx, bin_id in enumerate(bins_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]

            if isinstance(calibration_result, ProbabilisticCalibrationResult):
                self._plot_bin_cumulative_probabilistic(
                    ax,
                    bin_id,
                    cumulative_observed.get(bin_id, []),
                    calibration_result,
                    dict(kwargs),
                )
            else:
                cumulative_results = self._calculate_cumulative(bins_to_plot)
                self._plot_bin_cumulative(
                    ax,
                    bin_id,
                    cumulative_results[bin_id],
                    cumulative_observed.get(bin_id, []),
                    dict(kwargs),
                )

    def _finalize_cumulative_plot(
        self,
        axes: list["Axes"],
        bins_to_plot: list[str],
        output_file: str | None,
        config: PlotConfig,
    ) -> None:
        """
        Finalize cumulative plot by hiding unused subplots and saving.
        """
        for idx in range(len(bins_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
            logger.info(f"Cumulative plot saved to {output_file}")

    def _plot_bin_series(
        self,
        ax: "Axes",
        bin_id: str,
        observed: list[ObservedDataPoint],
        scale_values: dict[str, float],
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
            # Apply scale if observation has a scale_id
            obs_values = [
                p.value / scale_values[p.scale_id]
                if p.scale_id and p.scale_id in scale_values
                else p.value
                for p in observed
            ]
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

    def _plot_bin_series_probabilistic(
        self,
        ax: "Axes",
        bin_id: str,
        observed: list[ObservedDataPoint],
        scale_values: dict[str, float],
        prob_result: ProbabilisticCalibrationResult,
        plot_kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot time series for a single bin with probabilistic confidence intervals.
        """
        if bin_id not in prob_result.selected_ensemble.prediction_median:
            logger.warning(
                f"Bin '{bin_id}' not found in probabilistic result predictions"
            )
            return

        time_steps = list(
            range(len(prob_result.selected_ensemble.prediction_median[bin_id]))
        )
        median_values = prob_result.selected_ensemble.prediction_median[bin_id]
        ci_lower = prob_result.selected_ensemble.prediction_ci_lower[bin_id]
        ci_upper = prob_result.selected_ensemble.prediction_ci_upper[bin_id]

        # Build parameters for lineplot (median)
        params = {
            "x": time_steps,
            "y": median_values,
            "ax": ax,
            "label": "Median Prediction",
        }
        params.update(plot_kwargs)

        # Plot median prediction
        sns.lineplot(**params)

        # Plot confidence interval as filled area
        ax.fill_between(
            time_steps,
            ci_lower,
            ci_upper,
            alpha=0.3,
            label="95% CI",
        )

        # Overlay observed data if available
        if observed:
            obs_steps = [p.step for p in observed]
            # Apply scale if observation has a scale_id
            obs_values = [
                p.value / scale_values[p.scale_id]
                if p.scale_id and p.scale_id in scale_values
                else p.value
                for p in observed
            ]
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

    def _plot_bin_cumulative_probabilistic(
        self,
        ax: "Axes",
        bin_id: str,
        cumulative_observed: list[tuple[int, float]],
        prob_result: ProbabilisticCalibrationResult,
        plot_kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot cumulative data for a single bin with probabilistic confidence intervals.
        """
        if bin_id not in prob_result.selected_ensemble.prediction_median:
            logger.warning(
                f"Bin '{bin_id}' not found in probabilistic result predictions"
            )
            return

        # Calculate cumulative values from the probabilistic predictions
        median_values = prob_result.selected_ensemble.prediction_median[bin_id]
        ci_lower = prob_result.selected_ensemble.prediction_ci_lower[bin_id]
        ci_upper = prob_result.selected_ensemble.prediction_ci_upper[bin_id]

        # Calculate cumulative sums
        cumulative_median = np.cumsum(median_values).tolist()
        cumulative_ci_lower = np.cumsum(ci_lower).tolist()
        cumulative_ci_upper = np.cumsum(ci_upper).tolist()

        time_steps = list(range(len(cumulative_median)))

        # Build parameters for lineplot (median)
        params = {
            "x": time_steps,
            "y": cumulative_median,
            "ax": ax,
            "label": "Median Prediction (Cumulative)",
        }
        params.update(plot_kwargs)

        # Plot cumulative median prediction
        sns.lineplot(**params)

        # Plot cumulative confidence interval as filled area
        ax.fill_between(
            time_steps,
            cumulative_ci_lower,
            cumulative_ci_upper,
            alpha=0.3,
            label="95% CI",
        )

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
        self,
        observed_by_bin: dict[str, list[ObservedDataPoint]],
        scale_values: dict[str, float],
    ) -> dict[str, list[tuple[int, float]]]:
        """
        Calculate cumulative observed data (step, cumulative_value).
        Applies scale if observation has a scale_id.
        """
        cumulative: dict[str, list[tuple[int, float]]] = {}

        for bin_id, points in observed_by_bin.items():
            cumsum = []
            running_total = 0.0

            for point in points:
                # Apply scale if observation has a scale_id
                value = (
                    point.value / scale_values[point.scale_id]
                    if point.scale_id and point.scale_id in scale_values
                    else point.value
                )
                running_total += value
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
