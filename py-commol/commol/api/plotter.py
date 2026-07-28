import logging
import math
from collections import defaultdict
from collections.abc import Callable
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
        calibration_result: CalibrationResult
        | ProbabilisticCalibrationResult
        | None = None,
        config: PlotConfig | None = None,
        bins: list[str] | None = None,
        show_legend: bool = True,
        step_to_label: Callable[[int], str] | None = None,
        tick_every: int | None = None,
        x_label: str = "Step",
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
            Model predictions with a scale_id will be scaled for plotting using
            scale values from the calibration_result, so observed values remain in
            their original units.
        calibration_result : CalibrationResult | ProbabilisticCalibrationResult | None
            Optional calibration result. If ProbabilisticCalibrationResult is provided,
            plots the median prediction with confidence interval bands.
            Scale values are extracted from best_parameters (CalibrationResult) or
            parameter_statistics (ProbabilisticCalibrationResult) to scale model
            predictions for comparison with observed data.
        config : PlotConfig | None
            Configuration for plot layout and styling (figsize, dpi, layout,
            style, palette, context). If None, uses defaults.
        bins : list[str] | None
            List of bin IDs to plot. If None, plots all bins.
        show_legend : bool
            Whether to show the legend on each subplot. Default True.
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

        self._apply_seaborn_style(config)
        scale_values = self._extract_scale_values(calibration_result)

        observed_by_bin = self._group_observed_data(observed_data)

        fig, axes = self._create_series_figure(config, bins_to_plot)
        self._plot_all_series_bins(
            axes,
            bins_to_plot,
            observed_by_bin,
            scale_values,
            calibration_result,
            kwargs,
            show_legend,
            step_to_label,
            tick_every,
            x_label,
        )
        self._finalize_series_plot(axes, bins_to_plot, output_file, config)

        return fig

    def plot_cumulative(
        self,
        output_file: str | None = None,
        observed_data: list[ObservedDataPoint] | None = None,
        calibration_result: CalibrationResult
        | ProbabilisticCalibrationResult
        | None = None,
        config: PlotConfig | None = None,
        bins: list[str] | None = None,
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
            Observed data points with a scale_id will be unscaled for plotting using
            scale values from the calibration_result.
        calibration_result : CalibrationResult | ProbabilisticCalibrationResult | None
            Optional calibration result. If ProbabilisticCalibrationResult is provided,
            plots the median prediction with confidence interval bands.
            Scale values are extracted from best_parameters (CalibrationResult) or
            parameter_statistics (ProbabilisticCalibrationResult) to unscale observed
            data for comparison with model predictions.
        config : PlotConfig | None
            Configuration for plot layout and styling (figsize, dpi, layout,
            style, palette, context). If None, uses defaults.
        bins : list[str] | None
            List of bin IDs to plot. If None, plots all bins.
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

        self._apply_seaborn_style(config)
        scale_values = self._extract_scale_values(calibration_result)

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

    def _apply_seaborn_style(self, config: PlotConfig) -> None:
        """Apply Seaborn styling configuration from PlotConfig."""
        if config.style:
            sns.set_style(config.style)
            logger.debug(f"Applied Seaborn style: {config.style}")

        if config.palette:
            sns.set_palette(config.palette)
            logger.debug(f"Applied Seaborn palette: {config.palette}")

        if config.context:
            sns.set_context(config.context)
            logger.debug(f"Applied Seaborn context: {config.context}")

    @staticmethod
    def _apply_x_labels(
        ax: "Axes",
        time_steps: list[int],
        step_to_label: Callable[[int], str],
        tick_every: int | None,
    ) -> None:
        """
        Label the x axis at every `tick_every` steps.

        The tick grid is phased to the first plotted step rather than to the
        absolute step 0: for a windowed series the plotted steps are window
        ends, so anchoring at 0 would put ticks on steps that carry no plotted
        point and hand `step_to_label` a step outside the series' convention.
        """
        if tick_every is not None and time_steps:
            anchor = time_steps[0]
            ticks = [t for t in time_steps if (t - anchor) % tick_every == 0]
        else:
            ticks = time_steps
        ax.set_xticks(ticks)
        ax.set_xticklabels([step_to_label(t) for t in ticks], rotation=45, ha="right")

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

    @staticmethod
    def _apply_windows(
        series: list[float],
        observed: list[ObservedDataPoint],
    ) -> tuple[list[int], list[float]] | None:
        """
        Return (steps, values) where each value is
        series[step] - series[step - window_steps].

        The simulation line covers every multiple of window_steps across the full
        series, so the plot starts at step window_steps rather than the first
        observation step.

        Returns None if no observed point has window_steps set.
        """
        window_steps = next((p.window_steps for p in observed if p.window_steps), None)
        if window_steps is None:
            return None
        steps = list(range(window_steps, len(series), window_steps))
        values = [series[t] - series[t - window_steps] for t in steps]
        return steps, values

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

    def _observed_component_ids(
        self,
        bin_id: str,
        observed: list[ObservedDataPoint],
    ) -> list[str]:
        for point in observed:
            if point.compartments:
                return point.compartments
        return [bin_id]

    @staticmethod
    def _sum_series(
        results: dict[str, list[float]],
        component_ids: list[str],
    ) -> list[float]:
        missing = [
            component_id
            for component_id in component_ids
            if component_id not in results
        ]
        if missing:
            raise KeyError(f"Missing result series for aggregate components: {missing}")
        return [
            sum(results[component_id][idx] for component_id in component_ids)
            for idx in range(len(results[component_ids[0]]))
        ]

    def _extract_scale_values(
        self,
        calib_result: CalibrationResult | ProbabilisticCalibrationResult | None,
    ) -> dict[str, float] | None:
        """
        Extract scale values from calibration result.

        For CalibrationResult, returns best_parameters.
        For ProbabilisticCalibrationResult, returns median values for all parameters.
        The observed data scale_id decides which entries are used as scale values.
        """
        if calib_result is None:
            return None

        if isinstance(calib_result, CalibrationResult):
            return calib_result.best_parameters

        if isinstance(calib_result, ProbabilisticCalibrationResult):
            return {
                param_name: stats.median
                for (
                    param_name,
                    stats,
                ) in calib_result.selected_ensemble.parameter_statistics.items()
            }

        return None

    @staticmethod
    def _model_scale_for_observed(
        observed: list[ObservedDataPoint],
        scale_values: dict[str, float],
    ) -> float:
        scale_ids = {point.scale_id for point in observed if point.scale_id}
        if not scale_ids:
            return 1.0
        if len(scale_ids) > 1:
            logger.warning(
                "Multiple scale_id values found for one plotted series; leaving "
                "model predictions unscaled."
            )
            return 1.0

        scale_id = next(iter(scale_ids))
        if scale_id not in scale_values:
            logger.warning(
                f"Scale parameter '{scale_id}' not found in calibration result; "
                "leaving model predictions unscaled."
            )
            return 1.0
        return scale_values[scale_id]

    @staticmethod
    def _scale_series(values: list[float], scale: float) -> list[float]:
        if scale == 1.0:
            return values
        return [value * scale for value in values]

    def _create_series_figure(
        self, config: PlotConfig, bins_to_plot: list[str]
    ) -> tuple["Figure", list["Axes"]]:
        """
        Create figure and axes array for series plotting.
        """
        layout = config.layout or self._calculate_layout(len(bins_to_plot))
        rows, cols = layout
        figsize = (cols * 4.5, rows * 3.5 + 1.0)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=config.dpi)

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
        show_legend: bool = True,
        step_to_label: Callable[[int], str] | None = None,
        tick_every: int | None = None,
        x_label: str = "Step",
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
                    show_legend,
                    step_to_label,
                    tick_every,
                    x_label,
                )
            else:
                self._plot_bin_series(
                    ax,
                    bin_id,
                    observed_by_bin.get(bin_id, []),
                    scale_values or {},
                    dict(kwargs),
                    show_legend,
                    step_to_label,
                    tick_every,
                    x_label,
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
        show_legend: bool = True,
        step_to_label: Callable[[int], str] | None = None,
        tick_every: int | None = None,
        x_label: str = "Step",
    ) -> None:
        """
        Plot time series for a single bin on given axes.
        """
        component_ids = self._observed_component_ids(bin_id, observed)
        series = self._sum_series(self.results, component_ids)
        windowed = self._apply_windows(series, observed)
        if windowed is not None:
            time_steps, values = windowed
        else:
            time_steps = list(range(len(series)))
            values = series
        values = self._scale_series(
            values,
            self._model_scale_for_observed(observed, scale_values),
        )

        # Build parameters for lineplot
        params = {
            "x": time_steps,
            "y": values,
            "ax": ax,
            "label": "Simulation",
            "legend": show_legend,
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
                legend=show_legend,
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

        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{unit_str}")
        ax.set_title(f"{bin_name}")
        if step_to_label:
            self._apply_x_labels(ax, time_steps, step_to_label, tick_every)
        if show_legend:
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
        show_legend: bool = True,
        step_to_label: Callable[[int], str] | None = None,
        tick_every: int | None = None,
        x_label: str = "Step",
    ) -> None:
        """
        Plot time series for a single bin with probabilistic confidence intervals.
        """
        component_ids = self._observed_component_ids(bin_id, observed)
        missing = [
            component_id
            for component_id in component_ids
            if component_id not in prob_result.selected_ensemble.prediction_median
        ]
        if missing:
            logger.warning(
                f"Bin '{bin_id}' aggregate components not found in probabilistic "
                f"result predictions: {missing}"
            )
            return

        median_values = self._sum_series(
            prob_result.selected_ensemble.prediction_median,
            component_ids,
        )
        ci_lower = self._sum_series(
            prob_result.selected_ensemble.prediction_ci_lower,
            component_ids,
        )
        ci_upper = self._sum_series(
            prob_result.selected_ensemble.prediction_ci_upper,
            component_ids,
        )

        windowed_median = self._apply_windows(median_values, observed)
        if windowed_median is not None:
            time_steps, _ = windowed_median
            win_med = prob_result.selected_ensemble.windowed_prediction_median
            if win_med and bin_id in win_med:
                plot_median = win_med[bin_id]
                plot_ci_lower = (
                    prob_result.selected_ensemble.windowed_prediction_ci_lower[bin_id]
                )
                plot_ci_upper = (
                    prob_result.selected_ensemble.windowed_prediction_ci_upper[bin_id]
                )
            elif win_med and all(
                component_id in win_med for component_id in component_ids
            ):
                plot_median = self._sum_series(win_med, component_ids)
                plot_ci_lower = self._sum_series(
                    prob_result.selected_ensemble.windowed_prediction_ci_lower,
                    component_ids,
                )
                plot_ci_upper = self._sum_series(
                    prob_result.selected_ensemble.windowed_prediction_ci_upper,
                    component_ids,
                )
            else:
                _, plot_median = windowed_median
                _, plot_ci_lower = self._apply_windows(ci_lower, observed)  # type: ignore[misc]
                _, plot_ci_upper = self._apply_windows(ci_upper, observed)  # type: ignore[misc]
        else:
            time_steps = list(range(len(median_values)))
            plot_median = median_values
            plot_ci_lower = ci_lower
            plot_ci_upper = ci_upper

        model_scale = self._model_scale_for_observed(observed, scale_values)
        plot_median = self._scale_series(plot_median, model_scale)
        plot_ci_lower = self._scale_series(plot_ci_lower, model_scale)
        plot_ci_upper = self._scale_series(plot_ci_upper, model_scale)

        # Build parameters for lineplot (median)
        params = {
            "x": time_steps,
            "y": plot_median,
            "ax": ax,
            "label": "Median Prediction",
            "legend": show_legend,
        }
        params.update(plot_kwargs)

        # Plot median prediction
        sns.lineplot(**params)

        # Plot confidence interval as filled area
        ax.fill_between(
            time_steps,
            plot_ci_lower,
            plot_ci_upper,
            alpha=0.3,
            label="95% CI" if show_legend else "_nolegend_",
        )

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
                legend=show_legend,
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

        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{unit_str}")
        ax.set_title(f"{bin_name}")
        if step_to_label:
            self._apply_x_labels(ax, time_steps, step_to_label, tick_every)
        if show_legend:
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
