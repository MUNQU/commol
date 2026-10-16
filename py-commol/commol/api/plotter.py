"""Plots of simulation results, observations and ensemble bands."""

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from commol.api.results import SimulationResults, same_position
from commol.api.simulation import Simulation
from commol.api.windows import window_ends, windowed_totals
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
    results : SimulationResults
        Simulation results, with their positions on the axis.
    """

    def __init__(
        self,
        simulation: Simulation,
        results: SimulationResults,
    ):
        """
        Initialize the SimulationPlotter.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance that generated the results.
        results : SimulationResults
            Simulation results, such as those of ``Simulation.run`` or
            ``Simulation.run_at``. The x axis shows their positions.
        """
        self.simulation = simulation
        self.results = results
        self.bins = list(results.keys())

        logger.info(
            f"SimulationPlotter initialized with {len(self.bins)} bins "
            f"and {len(results.positions)} positions"
        )

    def plot_series(
        self,
        output_file: str | None = None,
        observed_data: list[ObservedDataPoint] | None = None,
        calibration_result: CalibrationResult
        | ProbabilisticCalibrationResult
        | None = None,
        config: PlotConfig | None = None,
        series: list[str] | None = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Plot simulation results with one subplot per series.

        Creates a figure with subplots arranged in a grid, where each subplot shows
        the evolution of one series along the axis. Optionally overlays observed data
        points.
        If a ProbabilisticCalibrationResult is provided, plots confidence intervals.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure. If None, figure is not saved (only returned).
        observed_data : list[ObservedDataPoint] | None
            Optional observed data points to overlay on corresponding bin subplots.
            Model predictions with a factor_id will be scaled for plotting using
            factor values from the calibration_result, so observed values remain in
            their original units.
        calibration_result : CalibrationResult | ProbabilisticCalibrationResult | None
            Optional calibration result. If ProbabilisticCalibrationResult is provided,
            plots the median prediction with confidence interval bands.
            Factor values are extracted from best_parameters (CalibrationResult) or
            the lowest-loss selected member (ProbabilisticCalibrationResult) to
            multiply model predictions for comparison with observed data.
        config : PlotConfig | None
            Configuration for plot layout and styling (figsize, dpi, layout,
            style, palette, context). If None, uses defaults.
        series : list[str] | None
            What to plot, one subplot each. An entry is an output name (an
            expanded compartment or accumulator, such as ``A_g1``) or a bin or
            accumulator id, which plots the sum of its outputs. An exact output
            name wins over an id. If None, plots every output.
        show_legend : bool
            Whether to show the legend on each subplot. Default True.
        position_label : Callable[[float], str] | None
            Turns a position on the axis into an x-axis tick label, such as a
            date. If None, the ticks show the position.
        tick_every : float | None
            Labels the plotted positions `tick_every` apart on the axis,
            counted from the first plotted position. If None, labels every
            plotted position.
        x_label : str
            The x-axis label. Default "Step".
        **kwargs : str | int | float | bool | None
            Additional keyword arguments passed to seaborn.lineplot().
            Common parameters: linewidth, alpha, linestyle, marker, etc.

        Returns
        -------
        Figure
            The matplotlib Figure object.

        Raises
        ------
        ValueError
            If an entry of ``series`` is neither an output name nor a bin or
            accumulator id.
        """
        logger.info("Starting plot_series")

        config = config or PlotConfig()
        bins_to_plot = self._series_to_plot(series)

        self._apply_seaborn_style(config)
        factor_values = self._extract_factor_values(calibration_result)

        observed_by_bin = self._group_observed_data(observed_data)

        fig, axes = self._create_series_figure(config, bins_to_plot)
        self._plot_all_series_bins(
            axes,
            bins_to_plot,
            observed_by_bin,
            factor_values,
            calibration_result,
            kwargs,
            show_legend,
            position_label,
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
        series: list[str] | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Plot cumulative (accumulated) simulation results with one subplot per series.

        Creates a figure showing the running sum of each series along the axis.
        Useful for tracking totals, losses, or other accumulated quantities.
        If a ProbabilisticCalibrationResult is provided, plots confidence intervals.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure. If None, figure is not saved (only returned).
        observed_data : list[ObservedDataPoint] | None
            Optional observed data points to overlay (also shown as cumulative).
            Observed data points with a factor_id will be unscaled for plotting using
            factor values from the calibration_result.
        calibration_result : CalibrationResult | ProbabilisticCalibrationResult | None
            Optional calibration result. If ProbabilisticCalibrationResult is provided,
            plots the median prediction with confidence interval bands.
            Factor values are extracted from best_parameters (CalibrationResult) or
            the lowest-loss selected member (ProbabilisticCalibrationResult) to
            unscale observed data for comparison with model predictions.
        config : PlotConfig | None
            Configuration for plot layout and styling (figsize, dpi, layout,
            style, palette, context). If None, uses defaults.
        series : list[str] | None
            What to plot, one subplot each: an output name or a bin or
            accumulator id, as in :meth:`plot_series`. If None, plots every
            output.
        **kwargs : str | int | float | bool | None
            Additional keyword arguments passed to seaborn.lineplot().
            Common parameters: linewidth, alpha, linestyle, marker, etc.

        Returns
        -------
        Figure
            The matplotlib Figure object.

        Raises
        ------
        ValueError
            If an entry of ``series`` is neither an output name nor a bin or
            accumulator id.
        """
        logger.info("Starting plot_cumulative")

        config = config or PlotConfig()
        bins_to_plot = self._series_to_plot(series)

        self._apply_seaborn_style(config)
        factor_values = self._extract_factor_values(calibration_result)

        observed_by_bin = self._group_observed_data(observed_data)
        cumulative_observed = self._calculate_cumulative_observed(
            observed_by_bin, factor_values or {}
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
        positions: Sequence[float],
        position_label: Callable[[float], str],
        tick_every: float | None,
    ) -> None:
        """
        Label the x axis at the plotted positions `tick_every` apart.

        The tick grid is phased to the first plotted position rather than to
        position 0: for a windowed series the plotted positions are window
        ends, so anchoring at 0 would put ticks on positions that carry no
        plotted point and hand `position_label` a position outside the
        series' convention.
        """
        if tick_every is not None and positions:
            anchor = positions[0]
            ticks = [
                position
                for position in positions
                if same_position(
                    (position - anchor) / tick_every,
                    round((position - anchor) / tick_every),
                )
            ]
        else:
            ticks = list(positions)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [position_label(position) for position in ticks], rotation=45, ha="right"
        )

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
        positions: Sequence[float],
        observed: list[ObservedDataPoint],
    ) -> tuple[list[float], list[float]] | None:
        """
        Return (ends, values) where each value is the increment of `series`
        over the window that closes at its end (:func:`commol.windowed_totals`).

        The simulation line covers every window end across the full series
        (:func:`commol.window_ends`), so the plot starts at the first window
        end rather than at the first observation.

        Returns None if no observed point has a window set.

        Raises
        ------
        ValueError
            If the series has no value at a window end or start, such as a
            window of 2.5 on the integer positions of ``Simulation.run``.
        """
        window = next((p.window for p in observed if p.window), None)
        if window is None:
            return None
        ends = window_ends(window, positions[-1] if positions else 0.0)
        return ends, windowed_totals(series, positions, window, ends)

    def _group_observed_data(
        self, observed_data: list[ObservedDataPoint] | None
    ) -> dict[str, list[ObservedDataPoint]]:
        """
        Group observed data points by the label of their series.
        """
        if not observed_data:
            return {}

        grouped: dict[str, list[ObservedDataPoint]] = defaultdict(list)
        for point in observed_data:
            grouped[point.label].append(point)

        # Sort each group by position
        for compartment in grouped:
            grouped[compartment].sort(key=lambda p: p.at)

        logger.debug(
            f"Grouped {len(observed_data)} observed data points into "
            f"{len(grouped)} compartments"
        )

        return dict(grouped)

    def _series_to_plot(self, series: list[str] | None) -> list[str]:
        """The series to plot, each checked to resolve to outputs."""
        if series is None:
            return list(self.bins)
        for series_id in series:
            self._series_components(series_id)
        return list(series)

    def _series_components(self, series_id: str) -> list[str]:
        """Output names whose sum is the series: itself, or a bin's outputs."""
        if series_id in self.results:
            return [series_id]
        try:
            return self.simulation.outputs_for(series_id)
        except KeyError:
            sources = sorted(
                self.simulation.model_definition.structure.outputs_by_source(
                    self.simulation.model_definition
                )
            )
            raise ValueError(
                f"'{series_id}' is neither an output nor a bin or accumulator id. "
                f"Outputs: {self.bins}. Bins and accumulators: {sources}"
            ) from None

    def _observed_component_ids(
        self,
        bin_id: str,
        observed: list[ObservedDataPoint],
    ) -> list[str]:
        for point in observed:
            if point.outputs:
                return point.outputs
        return self._series_components(bin_id)

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

    def _extract_factor_values(
        self,
        calib_result: CalibrationResult | ProbabilisticCalibrationResult | None,
    ) -> dict[str, float] | None:
        """
        Extract factor values from calibration result.

        For CalibrationResult, returns best_parameters.
        For ProbabilisticCalibrationResult, returns median values for all parameters.
        The observed data factor_id decides which entries are used as factor values.
        """
        if calib_result is None:
            return None

        if isinstance(calib_result, CalibrationResult):
            return calib_result.best_parameters

        if isinstance(calib_result, ProbabilisticCalibrationResult):
            return calib_result.selected_ensemble.point_parameters

        return None

    @staticmethod
    def _model_factor_for_observed(
        observed: list[ObservedDataPoint],
        factor_values: dict[str, float],
    ) -> float:
        factor_ids = {point.factor_id for point in observed if point.factor_id}
        if not factor_ids:
            return 1.0
        if len(factor_ids) > 1:
            logger.warning(
                "Multiple factor_id values found for one plotted series; leaving "
                "model predictions unscaled."
            )
            return 1.0

        factor_id = next(iter(factor_ids))
        if factor_id not in factor_values:
            logger.warning(
                f"Factor parameter '{factor_id}' not found in calibration result; "
                "leaving model predictions unscaled."
            )
            return 1.0
        return factor_values[factor_id]

    @staticmethod
    def _apply_factor_to_series(values: list[float], factor: float) -> list[float]:
        if factor == 1.0:
            return values
        return [value * factor for value in values]

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
        factor_values: dict[str, float] | None,
        calibration_result: CalibrationResult | ProbabilisticCalibrationResult | None,
        kwargs: dict[str, str | int | float | bool | None],
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
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
                    factor_values or {},
                    calibration_result,
                    dict(kwargs),
                    show_legend,
                    position_label,
                    tick_every,
                    x_label,
                )
            else:
                self._plot_bin_series(
                    ax,
                    bin_id,
                    observed_by_bin.get(bin_id, []),
                    factor_values or {},
                    dict(kwargs),
                    show_legend,
                    position_label,
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
        cumulative_observed: dict[str, list[tuple[int | float, float]]],
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
        factor_values: dict[str, float],
        plot_kwargs: dict[str, str | int | float | bool | None],
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
    ) -> None:
        """
        Plot the series of a single bin on given axes.
        """
        component_ids = self._observed_component_ids(bin_id, observed)
        series = self._sum_series(self.results, component_ids)
        windowed = self._apply_windows(series, self.results.positions, observed)
        if windowed is not None:
            positions, values = windowed
        else:
            positions = self.results.positions
            values = series
        values = self._apply_factor_to_series(
            values,
            self._model_factor_for_observed(observed, factor_values),
        )

        # Build parameters for lineplot
        params = {
            "x": positions,
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
            obs_positions = [p.at for p in observed]
            obs_values = [p.value for p in observed]
            sns.scatterplot(
                x=obs_positions,
                y=obs_values,
                ax=ax,
                label="Observed",
                color="red",
                s=30,
                alpha=0.7,
                zorder=5,
                legend=show_legend,
            )

        model = self.simulation.model_definition
        label = model.structure.series_label(model, bin_id)
        unit_str = label.unit or ""
        bin_name = label.name

        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{unit_str}")
        ax.set_title(f"{bin_name}")
        if position_label:
            self._apply_x_labels(ax, positions, position_label, tick_every)
        if show_legend:
            ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_bin_series_probabilistic(
        self,
        ax: "Axes",
        bin_id: str,
        observed: list[ObservedDataPoint],
        factor_values: dict[str, float],
        prob_result: ProbabilisticCalibrationResult,
        plot_kwargs: dict[str, str | int | float | bool | None],
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
    ) -> None:
        """
        Plot the series of a single bin with probabilistic confidence intervals.
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

        prediction_positions = prob_result.selected_ensemble.prediction_positions
        windowed_median = self._apply_windows(
            median_values, prediction_positions, observed
        )
        uses_memberwise_windowed_prediction = False
        if windowed_median is not None:
            positions, _ = windowed_median
            win_med = prob_result.selected_ensemble.windowed_prediction_median
            win_positions = prob_result.selected_ensemble.windowed_prediction_positions
            if win_med and bin_id in win_med:
                uses_memberwise_windowed_prediction = True
                plot_median = win_med[bin_id]
                positions = win_positions.get(bin_id, positions)
                plot_ci_lower = (
                    prob_result.selected_ensemble.windowed_prediction_ci_lower[bin_id]
                )
                plot_ci_upper = (
                    prob_result.selected_ensemble.windowed_prediction_ci_upper[bin_id]
                )
            elif win_med and all(
                component_id in win_med for component_id in component_ids
            ):
                uses_memberwise_windowed_prediction = True
                plot_median = self._sum_series(win_med, component_ids)
                positions = win_positions.get(component_ids[0], positions)
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
                # `observed` already yielded `windowed_median`, so it carries
                # a window and neither call returns None. The fallbacks keep
                # the unwindowed series, matching the outer non-windowed branch.
                windowed_ci_lower = self._apply_windows(
                    ci_lower, prediction_positions, observed
                )
                windowed_ci_upper = self._apply_windows(
                    ci_upper, prediction_positions, observed
                )
                plot_ci_lower = windowed_ci_lower[1] if windowed_ci_lower else ci_lower
                plot_ci_upper = windowed_ci_upper[1] if windowed_ci_upper else ci_upper
        else:
            positions = prediction_positions
            plot_median = median_values
            plot_ci_lower = ci_lower
            plot_ci_upper = ci_upper

        model_factor = (
            1.0
            if uses_memberwise_windowed_prediction
            else self._model_factor_for_observed(observed, factor_values)
        )
        plot_median = self._apply_factor_to_series(plot_median, model_factor)
        plot_ci_lower = self._apply_factor_to_series(plot_ci_lower, model_factor)
        plot_ci_upper = self._apply_factor_to_series(plot_ci_upper, model_factor)

        # Build parameters for lineplot (median)
        params = {
            "x": positions,
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
            positions,
            plot_ci_lower,
            plot_ci_upper,
            alpha=0.3,
            label="95% CI" if show_legend else "_nolegend_",
        )

        # Overlay observed data if available
        if observed:
            obs_positions = [p.at for p in observed]
            obs_values = [p.value for p in observed]
            sns.scatterplot(
                x=obs_positions,
                y=obs_values,
                ax=ax,
                label="Observed",
                color="red",
                s=30,
                alpha=0.7,
                zorder=5,
                legend=show_legend,
            )

        model = self.simulation.model_definition
        label = model.structure.series_label(model, bin_id)
        unit_str = label.unit or ""
        bin_name = label.name

        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{unit_str}")
        ax.set_title(f"{bin_name}")
        if position_label:
            self._apply_x_labels(ax, positions, position_label, tick_every)
        if show_legend:
            ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_bin_cumulative_probabilistic(
        self,
        ax: "Axes",
        bin_id: str,
        cumulative_observed: list[tuple[int | float, float]],
        prob_result: ProbabilisticCalibrationResult,
        plot_kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot cumulative data for a single bin with probabilistic confidence intervals.
        """
        component_ids = self._series_components(bin_id)
        ensemble = prob_result.selected_ensemble
        missing = [
            component_id
            for component_id in component_ids
            if component_id not in ensemble.prediction_median
        ]
        if missing:
            logger.warning(
                f"Series '{bin_id}' components not found in probabilistic result "
                f"predictions: {missing}"
            )
            return

        # Calculate cumulative values from the probabilistic predictions
        median_values = self._sum_series(ensemble.prediction_median, component_ids)
        ci_lower = self._sum_series(ensemble.prediction_ci_lower, component_ids)
        ci_upper = self._sum_series(ensemble.prediction_ci_upper, component_ids)

        # Calculate cumulative sums
        cumulative_median = np.cumsum(median_values).tolist()
        cumulative_ci_lower = np.cumsum(ci_lower).tolist()
        cumulative_ci_upper = np.cumsum(ci_upper).tolist()

        positions = ensemble.prediction_positions

        # Build parameters for lineplot (median)
        params = {
            "x": positions,
            "y": cumulative_median,
            "ax": ax,
            "label": "Median Prediction (Cumulative)",
        }
        params.update(plot_kwargs)

        # Plot cumulative median prediction
        sns.lineplot(**params)

        # Plot cumulative confidence interval as filled area
        ax.fill_between(
            positions,
            cumulative_ci_lower,
            cumulative_ci_upper,
            alpha=0.3,
            label="95% CI",
        )

        # Overlay cumulative observed data if available
        if cumulative_observed:
            obs_positions = [at for at, _ in cumulative_observed]
            obs_values = [value for _, value in cumulative_observed]
            sns.scatterplot(
                x=obs_positions,
                y=obs_values,
                ax=ax,
                label="Observed (Cumulative)",
                color="red",
                s=30,
                alpha=0.7,
                zorder=5,
            )

        model = self.simulation.model_definition
        label = model.structure.series_label(model, bin_id)
        unit_str = label.unit or ""
        bin_name = label.name

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
            values = self._sum_series(self.results, self._series_components(bin_id))
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
        factor_values: dict[str, float],
    ) -> dict[str, list[tuple[int | float, float]]]:
        """
        Calculate cumulative observed data (position, cumulative_value).
        Divides by the factor if the observation has a factor_id.
        """
        cumulative: dict[str, list[tuple[int | float, float]]] = {}

        for bin_id, points in observed_by_bin.items():
            cumsum = []
            running_total = 0.0

            for point in points:
                # Divide by the factor if the observation has a factor_id
                value = (
                    point.value / factor_values[point.factor_id]
                    if point.factor_id and point.factor_id in factor_values
                    else point.value
                )
                running_total += value
                cumsum.append((point.at, running_total))

            cumulative[bin_id] = cumsum

        return cumulative

    def _plot_bin_cumulative(
        self,
        ax: "Axes",
        bin_id: str,
        cumulative_values: list[float],
        cumulative_observed: list[tuple[int | float, float]],
        plot_kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot cumulative data for a single bin on given axes.
        """
        positions = self.results.positions

        # Build parameters for lineplot
        params = {
            "x": positions,
            "y": cumulative_values,
            "ax": ax,
            "label": "Simulation (Cumulative)",
        }
        params.update(plot_kwargs)

        # Plot cumulative simulation results
        sns.lineplot(**params)

        # Overlay cumulative observed data if available
        if cumulative_observed:
            obs_positions = [at for at, _ in cumulative_observed]
            obs_values = [value for _, value in cumulative_observed]
            sns.scatterplot(
                x=obs_positions,
                y=obs_values,
                ax=ax,
                label="Observed (Cumulative)",
                color="red",
                s=30,
                alpha=0.7,
                zorder=5,
            )

        model = self.simulation.model_definition
        label = model.structure.series_label(model, bin_id)
        unit_str = label.unit or ""
        bin_name = label.name

        ax.set_xlabel("Step")
        ax.set_ylabel(f"{unit_str}")
        ax.set_title(f"{bin_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
