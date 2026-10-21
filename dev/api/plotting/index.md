# Plotting API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Plots simulation results with seaborn. Guide: [Plotting](https://munqu.github.io/commol/latest/guide/simulations/#plotting).

## SimulationPlotter

A facade for plotting simulation results using Seaborn.

This class provides methods to visualize simulation results with automatic subplot organization, Seaborn styling, and support for overlaying observed data.

Attributes:

| Name               | Type                   | Description                                                                       |
| ------------------ | ---------------------- | --------------------------------------------------------------------------------- |
| `simulation`       | `Simulation`           | The simulation instance that generated the results.                               |
| `results`          | `SimulationResults`    | Simulation results, with their positions on the axis; for replicates, their mean. |
| `replicates`       | \`SimulationReplicates | None\`                                                                            |
| `confidence_level` | `float`                | Confidence level of the band of the replicates.                                   |

Source code in `commol/api/plotter.py`

```python
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
        Simulation results, with their positions on the axis; for replicates,
        their mean.
    replicates : SimulationReplicates | None
        The replicates of a stochastic run, or None for one run.
    confidence_level : float
        Confidence level of the band of the replicates.
    """

    def __init__(
        self,
        simulation: Simulation,
        results: SimulationResults | SimulationReplicates,
        confidence_level: float = 0.95,
    ):
        """
        Initialize the SimulationPlotter.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance that generated the results.
        results : SimulationResults | SimulationReplicates
            Simulation results, such as those of ``Simulation.run`` or
            ``Simulation.run_at``. The x axis shows their positions. For the
            replicates of a stochastic run (``Simulation.replicates``),
            ``plot_series`` draws their mean and a band of
            ``confidence_level``, reduced replicate by replicate
            (`member_statistics`).
        confidence_level : float, default=0.95
            Confidence level of the band of the replicates or of a
            `Predictive`, strictly between 0 and 1.
        """
        self.simulation = simulation
        self.confidence_level = confidence_level
        if isinstance(results, SimulationReplicates):
            self.replicates: SimulationReplicates | None = results
            self.results = results.mean()
        else:
            self.replicates = None
            self.results = results
        self.bins = list(self.results.keys())

        logger.info(
            f"SimulationPlotter initialized with {len(self.bins)} bins "
            f"and {len(results.positions)} positions"
        )

    @_scoped_style
    def plot_series(
        self,
        output_file: str | None = None,
        observed_data: list[ObservedDataPoint] | None = None,
        calibration_result: CalibrationResult | Predictive | None = None,
        config: PlotConfig | None = None,
        series: list[str] | None = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        prediction_intervals: PredictionIntervals | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Plot simulation results with one subplot per series.

        Creates a figure with subplots arranged in a grid, where each subplot shows
        the evolution of one series along the axis. Optionally overlays observed data
        points.
        With a `Predictive`, plots the weighted median of its draws and their
        band of ``confidence_level``.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure. If None, figure is not saved (only returned).
        observed_data : list[ObservedDataPoint] | None
            Optional observed data points to overlay on corresponding bin subplots.
            Model predictions with a factor_id will be scaled for plotting using
            factor values from the calibration_result, so observed values remain in
            their original units. When the points of a series share one sum of
            outputs and one factor and have no window, the model line covers
            every position of the results. Otherwise the model of each point is
            read through its own operator (outputs, window, factor) at its own
            position, as the loss reads it, and the line joins these values.
        calibration_result : CalibrationResult | Predictive | None
            Optional calibration result. A `CalibrationResult` gives the factor
            values (``best_parameters``) that multiply model predictions. A
            `Predictive` plots the band of its draws instead of the simulation
            line; each draw's series is read through the observation operator
            of the problem (sums, windows and factors of the observations of
            the series, each at its own position when they do not share one
            operator with no window), then the draws are reduced with their
            weights.
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
            Turns a position on the axis into an x-axis tick label. If None,
            the ticks show the date of their position when the model has an
            origin (``ModelBuilder(origin=...)``): ``2020-01-02``, with the
            time when it is not midnight. Otherwise they show the position.
        tick_every : float | None
            Labels the plotted positions `tick_every` apart on the axis,
            counted from the first plotted position. If None, labels every
            plotted position.
        x_label : str
            The x-axis label. Default "Step".
        prediction_intervals : PredictionIntervals | None
            Prediction intervals of a conformal method
            (``ConformalResult.intervals``). The subplot of each of their
            series, by label, shows the band between their lower and upper
            ends and their point prediction; an unbounded interval is left
            out of the band.
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
            accumulator id, or the results have no value at the position of
            an observation read on its own or at the start of its window.
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
        if prediction_intervals is not None:
            for ax, bin_id in zip(axes, bins_to_plot, strict=False):
                self._plot_prediction_intervals(
                    ax, bin_id, prediction_intervals, show_legend
                )
        self._finalize_plot(axes, bins_to_plot, output_file, config, "Series")

        return fig

    @_scoped_style
    def plot_cumulative(
        self,
        output_file: str | None = None,
        observed_data: list[ObservedDataPoint] | None = None,
        calibration_result: CalibrationResult | Predictive | None = None,
        config: PlotConfig | None = None,
        series: list[str] | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Plot cumulative (accumulated) simulation results with one subplot per series.

        Creates a figure showing the running sum of each series along the axis.
        Useful for tracking totals, losses, or other accumulated quantities.
        With a `Predictive`, plots the band of the running totals of its draws.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure. If None, figure is not saved (only returned).
        observed_data : list[ObservedDataPoint] | None
            Optional observed data points to overlay (also shown as cumulative).
            Observed data points with a factor_id will be unscaled for plotting using
            factor values from the calibration_result.
        calibration_result : CalibrationResult | Predictive | None
            Optional calibration result. A `CalibrationResult` gives the factor
            values (``best_parameters``) that unscale observed data. A
            `Predictive` plots the band of the running total of each draw,
            and unscales observed data by the weighted median of each factor.
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
        self._finalize_plot(axes, bins_to_plot, output_file, config, "Cumulative")

        return fig

    @staticmethod
    @_scoped_style
    def plot_prcc(
        result: PRCCResult,
        output_file: str | None = None,
        config: PlotConfig | None = None,
        x_label: str = "Step",
    ) -> "Figure":
        """
        Plot the PRCC of each input: one subplot per series.

        A series of a `Trajectory` draws one line per input over its
        positions; any other quantity draws one bar per input.

        Parameters
        ----------
        result : PRCCResult
            The result of `PRCC`.
        output_file : str | None
            Path to save the figure. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        x_label : str
            The x-axis label of a series over positions. Default "Step".

        Returns
        -------
        Figure
            The matplotlib Figure object.
        """
        config = config or PlotConfig()
        SimulationPlotter._apply_seaborn_style(config)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, series in enumerate(result.series):
            groups[series].append(index)
        names = list(groups)
        fig, axes = SimulationPlotter._create_grid(config, len(names))
        for ax, name in zip(axes, names, strict=False):
            indices = groups[name]
            positions = [
                position
                for position in (result.positions[i] for i in indices)
                if position is not None
            ]
            if len(positions) == len(indices):
                for j, input_name in enumerate(result.inputs):
                    ax.plot(
                        positions,
                        [result.coefficients[i][j] for i in indices],
                        marker="o",
                        label=input_name,
                    )
                ax.set_xlabel(x_label)
                ax.legend()
            else:
                ax.bar(result.inputs, result.coefficients[indices[0]])
            ax.axhline(0.0, color="grey", linewidth=0.8)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("PRCC")
            ax.set_title(name)
        SimulationPlotter._finalize_plot(axes, names, output_file, config, "PRCC")
        return fig

    @staticmethod
    @_scoped_style
    def plot_morris(
        result: MorrisResult,
        output_file: str | None = None,
        config: PlotConfig | None = None,
    ) -> "Figure":
        """
        Plot ``σ`` against ``μ*`` of each input: one subplot per quantity,
        one labelled point per input.

        An input far to the right matters; one high up acts through
        interactions or a non-linear effect.

        Parameters
        ----------
        result : MorrisResult
            The result of `Morris`.
        output_file : str | None
            Path to save the figure. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.

        Returns
        -------
        Figure
            The matplotlib Figure object.
        """
        config = config or PlotConfig()
        SimulationPlotter._apply_seaborn_style(config)
        fig, axes = SimulationPlotter._create_grid(config, len(result.labels))
        for ax, q in zip(axes, range(len(result.labels)), strict=False):
            for j, input_name in enumerate(result.inputs):
                mu_star, sigma = result.mu_star[q][j], result.sigma[q][j]
                ax.scatter([mu_star], [sigma], label=input_name)
                ax.annotate(input_name, (mu_star, sigma))
            ax.set_xlabel("μ*")
            ax.set_ylabel("σ")
            ax.set_title(result.labels[q])
            ax.legend()
        SimulationPlotter._finalize_plot(
            axes, result.labels, output_file, config, "Morris"
        )
        return fig

    @staticmethod
    @_scoped_style
    def plot_sobol(
        result: SobolResult | ChaosResult,
        output_file: str | None = None,
        config: PlotConfig | None = None,
    ) -> "Figure":
        """
        Plot the first-order and total Sobol indices: one subplot per
        quantity, one pair of bars per input, with the intervals of a
        `SobolResult`.

        Parameters
        ----------
        result : SobolResult | ChaosResult
            The result of `Sobol` or `PolynomialChaos`.
        output_file : str | None
            Path to save the figure. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.

        Returns
        -------
        Figure
            The matplotlib Figure object.
        """
        config = config or PlotConfig()
        SimulationPlotter._apply_seaborn_style(config)
        inputs = result.inputs
        fig, axes = SimulationPlotter._create_grid(config, len(result.labels))
        width = 0.4
        for ax, q in zip(axes, range(len(result.labels)), strict=False):
            places = list(range(len(inputs)))
            for offset, name, values in (
                (-width / 2, "First order", result.first[q]),
                (width / 2, "Total", result.total[q]),
            ):
                errors = None
                if isinstance(result, SobolResult):
                    lower, upper = (
                        (result.first_lower[q], result.first_upper[q])
                        if name == "First order"
                        else (result.total_lower[q], result.total_upper[q])
                    )
                    errors = [
                        [
                            max(v - low, 0.0)
                            for v, low in zip(values, lower, strict=True)
                        ],
                        [
                            max(high - v, 0.0)
                            for v, high in zip(values, upper, strict=True)
                        ],
                    ]
                ax.bar(
                    [place + offset for place in places],
                    values,
                    width,
                    yerr=errors,
                    capsize=3,
                    label=name,
                )
            ax.set_xticks(places)
            ax.set_xticklabels(inputs)
            ax.set_ylabel("Sobol index")
            ax.set_title(result.labels[q])
            ax.legend()
        SimulationPlotter._finalize_plot(
            axes, result.labels, output_file, config, "Sobol"
        )
        return fig

    @staticmethod
    @_scoped_style
    def plot_profiles(
        result: ProfileResult,
        output_file: str | None = None,
        config: PlotConfig | None = None,
    ) -> "Figure":
        """
        Plot each profile: one subplot per parameter or prediction, with the
        statistic of each point, the threshold, and the ends of the interval.

        A profile that stays under the threshold up to a bound is open on that
        side; a flat one stays near 0 over its whole range.

        Parameters
        ----------
        result : ProfileResult
            The result of `ProfileLikelihood` or `PredictionProfile`.
        output_file : str | None
            Path to save the figure. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.

        Returns
        -------
        Figure
            The matplotlib Figure object.
        """
        config = config or PlotConfig()
        SimulationPlotter._apply_seaborn_style(config)
        names = list(result.profiles)
        fig, axes = SimulationPlotter._create_grid(config, len(names))
        for ax, name in zip(axes, names, strict=False):
            profile = result.profiles[name]
            ax.plot(profile.values, profile.statistics, marker="o", markersize=3)
            ax.axhline(profile.threshold, color="grey", linestyle="--", linewidth=1)
            interval = result.intervals[name]
            for end in (interval.lower, interval.upper):
                if end is not None:
                    ax.axvline(end, color="grey", linestyle=":", linewidth=1)
            ax.set_xlabel(name)
            ax.set_ylabel("Profile statistic")
            ax.set_title(name)
        SimulationPlotter._finalize_plot(axes, names, output_file, config, "Profiles")
        return fig

    @staticmethod
    def _create_grid(config: PlotConfig, count: int) -> tuple["Figure", list["Axes"]]:
        """A figure of `count` subplots in the layout of `config`."""
        rows, cols = config.layout or SimulationPlotter._calculate_layout(count)
        return SimulationPlotter._create_figure(
            rows, cols, (cols * 4.5, rows * 3.5 + 1.0), config.dpi
        )

    @staticmethod
    def _apply_seaborn_style(config: PlotConfig) -> None:
        """
        Apply the seaborn style, palette and context of `config`.

        It sets `matplotlib.rcParams`, so only a method under `_scoped_style`
        calls it: the style stays in the figure of that method.
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

    def _label_positions(
        self,
        ax: "Axes",
        positions: Sequence[float],
        position_label: Callable[[float], str] | None,
        tick_every: float | None,
    ) -> None:
        """
        Label the x axis with `position_label`, or with dates on a dated axis.

        With no `position_label` and a model with an origin, a tick shows the
        date of its position: at the plotted positions `tick_every` apart, or
        at the ticks matplotlib places when `tick_every` is None. With
        neither, the ticks show the position.
        """
        if position_label is None:
            model = self.simulation.model_definition
            if model.origin is None:
                return
            axis = model.axis_units()
            if tick_every is None:
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda at, _: _date_label(axis, at))
                )
                ax.tick_params(axis="x", labelrotation=45)
                return
            position_label = functools.partial(_date_label, axis)
        self._apply_x_labels(ax, positions, position_label, tick_every)

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
        position 0: a series read at its observations is plotted at their
        positions, so anchoring at 0 would put ticks on positions that carry
        no plotted point.
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

    @staticmethod
    def _calculate_layout(num_bins: int) -> tuple[int, int]:
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
    def _reads_each_observation(observed: list[ObservedDataPoint]) -> bool:
        """
        Whether the model of a series is read at each observation: when one
        has a window, or the observations differ in outputs or factor.

        Otherwise one operator reads every observation, and the model line
        covers every position of the results.
        """
        operators = {
            (tuple(point.outputs or ()), point.factor_id) for point in observed
        }
        return any(point.window for point in observed) or len(operators) > 1

    def _observation_values(
        self,
        results: SimulationResults,
        bin_id: str,
        observed: list[ObservedDataPoint],
        factor_values: dict[str, float],
    ) -> list[float]:
        """
        The model value of each observation, read through its own operator as
        the loss reads it: the sum of its outputs at its position, its
        increment over its window (:func:`commol.windowed_totals`), times its
        factor.

        Raises
        ------
        ValueError
            If the results have no value at the position of an observation or
            at the start of its window, such as a window of 2.5 on the integer
            positions of ``Simulation.run``.
        """
        values = []
        for point in observed:
            series = self._sum_series(
                results, point.outputs or self._series_components(bin_id)
            )
            value = (
                windowed_totals(series, results.positions, point.window, [point.at])[0]
                if point.window
                else series[index_at(results.positions, point.at)]
            )
            values.append(value * self._factor_value(point.factor_id, factor_values))
        return values

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
        """The outputs of the one operator of the observations of a series."""
        if observed and observed[0].outputs:
            return observed[0].outputs
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
        calib_result: CalibrationResult | Predictive | None,
    ) -> dict[str, float] | None:
        """
        Extract factor values from calibration result.

        For a CalibrationResult, returns its best parameters. For a
        Predictive, returns the weighted median of each parameter of its
        sample. The observed data factor_id decides which entries are used as
        factor values.
        """
        if calib_result is None:
            return None

        if isinstance(calib_result, CalibrationResult):
            return calib_result.best_parameters

        return {
            name: values[0]
            for name, values in calib_result.sample.quantiles([0.5]).items()
        }

    @staticmethod
    def _factor_value(factor_id: str | None, factor_values: dict[str, float]) -> float:
        """The value of a factor, or 1 for none or one the result lacks."""
        if factor_id is None:
            return 1.0
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
        A figure of the layout, sized by its subplots, and its axes in order.
        """
        rows, cols = config.layout or self._calculate_layout(len(bins_to_plot))
        return self._create_figure(
            rows, cols, (cols * 4.5, rows * 3.5 + 1.0), config.dpi
        )

    @staticmethod
    def _create_figure(
        rows: int, cols: int, figsize: tuple[float, float], dpi: int
    ) -> tuple["Figure", list["Axes"]]:
        """A figure of `rows` by `cols` subplots, and its axes row by row."""
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, squeeze=False)
        return fig, list(axes.flat)

    def _plot_all_series_bins(
        self,
        axes: list["Axes"],
        bins_to_plot: list[str],
        observed_by_bin: dict[str, list[ObservedDataPoint]],
        factor_values: dict[str, float] | None,
        calibration_result: CalibrationResult | Predictive | None,
        kwargs: dict[str, str | int | float | bool | None],
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
    ) -> None:
        """
        Plot series data for all bins across subplots.
        """
        for ax, bin_id in zip(axes, bins_to_plot, strict=False):
            if isinstance(calibration_result, Predictive):
                self._plot_bin_series_predictive(
                    ax,
                    bin_id,
                    observed_by_bin.get(bin_id, []),
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

    @staticmethod
    def _plot_prediction_intervals(
        ax: "Axes",
        series_id: str,
        intervals: PredictionIntervals,
        show_legend: bool,
    ) -> None:
        """
        The band of the prediction intervals of one series and its point
        prediction, when the intervals have the series.
        """
        series = intervals.series.get(series_id)
        if series is None:
            return
        bounded = [
            (at, lower, upper)
            for at, lower, upper in zip(
                series.positions, series.lower, series.upper, strict=True
            )
            if math.isfinite(lower) and math.isfinite(upper)
        ]
        ax.fill_between(
            [at for at, _, _ in bounded],
            [lower for _, lower, _ in bounded],
            [upper for _, _, upper in bounded],
            alpha=0.3,
            label=f"{intervals.level:.0%} prediction interval",
        )
        ax.plot(series.positions, series.prediction, linestyle="--", label="Prediction")
        if show_legend:
            ax.legend()

    @staticmethod
    def _finalize_plot(
        axes: list["Axes"],
        bins_to_plot: list[str],
        output_file: str | None,
        config: PlotConfig,
        name: str,
    ) -> None:
        """
        Hide the subplots no series fills, and save the figure to
        `output_file` when there is one.
        """
        for ax in axes[len(bins_to_plot) :]:
            ax.set_visible(False)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
            logger.info(f"{name} plot saved to {output_file}")

    def _create_cumulative_figure(
        self, config: PlotConfig, bins_to_plot: list[str]
    ) -> tuple["Figure", list["Axes"]]:
        """A figure of the layout, of the size of the configuration."""
        rows, cols = config.layout or self._calculate_layout(len(bins_to_plot))
        return self._create_figure(rows, cols, config.figsize, config.dpi)

    def _plot_all_cumulative_bins(
        self,
        axes: list["Axes"],
        bins_to_plot: list[str],
        cumulative_observed: dict[str, list[tuple[int | float, float]]],
        calibration_result: CalibrationResult | Predictive | None,
        kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot cumulative data for all bins across subplots.
        """
        if not isinstance(calibration_result, Predictive):
            cumulative_results = self._calculate_cumulative(bins_to_plot)
        for ax, bin_id in zip(axes, bins_to_plot, strict=False):
            if isinstance(calibration_result, Predictive):
                self._plot_bin_cumulative_predictive(
                    ax,
                    bin_id,
                    cumulative_observed.get(bin_id, []),
                    calibration_result,
                    dict(kwargs),
                )
            else:
                self._plot_bin_cumulative(
                    ax,
                    bin_id,
                    cumulative_results[bin_id],
                    cumulative_observed.get(bin_id, []),
                    dict(kwargs),
                )

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
        positions, values = self._model_line(
            self.results, bin_id, observed, factor_values
        )

        # Build parameters for lineplot
        params = {
            "x": positions,
            "y": values,
            # Each point is drawn as it is: observations at one position with
            # different windows are never averaged.
            "estimator": None,
            "ax": ax,
            "label": "Simulation"
            if self.replicates is None
            else f"Mean of {len(self.replicates)} replicates",
            "legend": show_legend,
        }
        params.update(plot_kwargs)

        # Plot simulation results as line
        sns.lineplot(**params)
        if self.replicates is not None:
            self._plot_replicate_band(ax, bin_id, observed, factor_values)

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
        self._label_positions(ax, positions, position_label, tick_every)
        if show_legend:
            ax.legend()
        ax.grid(True, alpha=0.3)

    def _model_line(
        self,
        results: SimulationResults,
        bin_id: str,
        observed: list[ObservedDataPoint],
        factor_values: dict[str, float],
    ) -> tuple[list[float], list[float]]:
        """
        The positions and values of the model of one series in one run.

        When the observations of the series do not share one operator with no
        window, the model value of each observation at its position
        (`_observation_values`). Otherwise the sum of the outputs of their
        operator at every position of the results, times its factor.
        """
        if self._reads_each_observation(observed):
            return [point.at for point in observed], self._observation_values(
                results, bin_id, observed, factor_values
            )
        series = self._sum_series(
            results, self._observed_component_ids(bin_id, observed)
        )
        factor = self._factor_value(
            observed[0].factor_id if observed else None, factor_values
        )
        return results.positions, self._apply_factor_to_series(series, factor)

    def _plot_replicate_band(
        self,
        ax: "Axes",
        bin_id: str,
        observed: list[ObservedDataPoint],
        factor_values: dict[str, float],
    ) -> None:
        """
        The band of the replicates: each replicate reduced to the plotted
        series first, then the interval across them (`member_statistics`).
        """
        assert self.replicates is not None
        per_replicate = []
        positions: list[float] = self.results.positions
        for replicate in self.replicates:
            positions, series = self._model_line(
                replicate, bin_id, observed, factor_values
            )
            per_replicate.append(series)
        statistics = member_statistics(per_replicate, self.confidence_level)
        ax.fill_between(
            positions,
            statistics["ci_lower"],
            statistics["ci_upper"],
            alpha=0.3,
            label=f"{self.confidence_level:.0%} of the replicates",
        )

    def _predictive_series_band(
        self,
        bin_id: str,
        observed: list[ObservedDataPoint],
        predictive: Predictive,
    ) -> SeriesBand:
        """
        The band of one series of a predictive.

        When the observations of the series do not share one operator with no
        window, each draw is read through the operator of each observation at
        its position, as the loss reads it. With one operator and a factor,
        each draw is read through it at every position of the predictive.
        Otherwise, the band of the sum of the outputs.
        """
        if self._reads_each_observation(observed):
            points = [
                ObservedDataPoint(
                    at=point.at,
                    outputs=point.outputs or self._series_components(bin_id),
                    label=bin_id,
                    value=0.0,
                    window=point.window,
                    factor_id=point.factor_id,
                )
                for point in observed
            ]
            return predictive.at_observations(points, NoNoise()).band(
                self.confidence_level
            )
        component_ids = self._observed_component_ids(bin_id, observed)
        factor_id = observed[0].factor_id if observed else None
        if factor_id is None:
            return predictive.series_band(component_ids, self.confidence_level)
        points = [
            ObservedDataPoint(
                at=at,
                outputs=component_ids,
                label=bin_id,
                value=0.0,
                factor_id=factor_id,
            )
            for at in predictive.positions
        ]
        return predictive.at_observations(points, NoNoise()).band(self.confidence_level)

    def _plot_bin_series_predictive(
        self,
        ax: "Axes",
        bin_id: str,
        observed: list[ObservedDataPoint],
        predictive: Predictive,
        plot_kwargs: dict[str, str | int | float | bool | None],
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
    ) -> None:
        """
        Plot the weighted median and band of one series of a predictive.
        """
        band = self._predictive_series_band(bin_id, observed, predictive)
        positions = band.positions
        params = {
            "x": positions,
            "y": band.median,
            "ax": ax,
            "label": "Median of the draws",
            "legend": show_legend,
        }
        params.update(plot_kwargs)
        sns.lineplot(**params)
        ax.fill_between(
            positions,
            band.lower,
            band.upper,
            alpha=0.3,
            label=f"{self.confidence_level:.0%} of the draws"
            if show_legend
            else "_nolegend_",
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
        self._label_positions(ax, positions, position_label, tick_every)
        if show_legend:
            ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_bin_cumulative_predictive(
        self,
        ax: "Axes",
        bin_id: str,
        cumulative_observed: list[tuple[int | float, float]],
        predictive: Predictive,
        plot_kwargs: dict[str, str | int | float | bool | None],
    ) -> None:
        """
        Plot the band of the running total of one series of a predictive.
        """
        band = predictive.series_band(
            self._series_components(bin_id), self.confidence_level, cumulative=True
        )
        positions = band.positions
        params = {
            "x": positions,
            "y": band.median,
            "ax": ax,
            "label": "Median of the draws (cumulative)",
        }
        params.update(plot_kwargs)
        sns.lineplot(**params)
        ax.fill_between(
            positions,
            band.lower,
            band.upper,
            alpha=0.3,
            label=f"{self.confidence_level:.0%} of the draws",
        )

        self._finish_cumulative_axes(ax, bin_id, cumulative_observed)

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

        self._finish_cumulative_axes(ax, bin_id, cumulative_observed)

    def _finish_cumulative_axes(
        self,
        ax: "Axes",
        bin_id: str,
        cumulative_observed: list[tuple[int | float, float]],
    ) -> None:
        """Overlay the running total of the observations, and label the axes."""
        if cumulative_observed:
            sns.scatterplot(
                x=[at for at, _ in cumulative_observed],
                y=[value for _, value in cumulative_observed],
                ax=ax,
                label="Observed (Cumulative)",
                color="red",
                s=30,
                alpha=0.7,
                zorder=5,
            )

        model = self.simulation.model_definition
        label = model.structure.series_label(model, bin_id)
        ax.set_xlabel("Step")
        ax.set_ylabel(label.unit or "")
        ax.set_title(label.name)
        ax.legend()
        ax.grid(True, alpha=0.3)
```

### Methods:

#### __init__

```python
__init__(simulation: Simulation, results: SimulationResults | SimulationReplicates, confidence_level: float = 0.95)
```

Initialize the SimulationPlotter.

Parameters:

| Name               | Type                | Description                                                                                  | Default                                                                                                                                                                                                                                                                                      |
| ------------------ | ------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `simulation`       | `Simulation`        | The simulation instance that generated the results.                                          | *required*                                                                                                                                                                                                                                                                                   |
| `results`          | \`SimulationResults | SimulationReplicates\`                                                                       | Simulation results, such as those of Simulation.run or Simulation.run_at. The x axis shows their positions. For the replicates of a stochastic run (Simulation.replicates), plot_series draws their mean and a band of confidence_level, reduced replicate by replicate (member_statistics). |
| `confidence_level` | `float`             | Confidence level of the band of the replicates or of a Predictive, strictly between 0 and 1. | `0.95`                                                                                                                                                                                                                                                                                       |

Source code in `commol/api/plotter.py`

```python
def __init__(
    self,
    simulation: Simulation,
    results: SimulationResults | SimulationReplicates,
    confidence_level: float = 0.95,
):
    """
    Initialize the SimulationPlotter.

    Parameters
    ----------
    simulation : Simulation
        The simulation instance that generated the results.
    results : SimulationResults | SimulationReplicates
        Simulation results, such as those of ``Simulation.run`` or
        ``Simulation.run_at``. The x axis shows their positions. For the
        replicates of a stochastic run (``Simulation.replicates``),
        ``plot_series`` draws their mean and a band of
        ``confidence_level``, reduced replicate by replicate
        (`member_statistics`).
    confidence_level : float, default=0.95
        Confidence level of the band of the replicates or of a
        `Predictive`, strictly between 0 and 1.
    """
    self.simulation = simulation
    self.confidence_level = confidence_level
    if isinstance(results, SimulationReplicates):
        self.replicates: SimulationReplicates | None = results
        self.results = results.mean()
    else:
        self.replicates = None
        self.results = results
    self.bins = list(self.results.keys())

    logger.info(
        f"SimulationPlotter initialized with {len(self.bins)} bins "
        f"and {len(results.positions)} positions"
    )
```

#### plot_series

```python
plot_series(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | Predictive | None = None, config: PlotConfig | None = None, series: list[str] | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', prediction_intervals: PredictionIntervals | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Plot simulation results with one subplot per series.

Creates a figure with subplots arranged in a grid, where each subplot shows the evolution of one series along the axis. Optionally overlays observed data points. With a `Predictive`, plots the weighted median of its draws and their band of `confidence_level`.

Parameters:

| Name                   | Type                       | Description                                               | Default                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------- | -------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_file`          | \`str                      | None\`                                                    | Path to save the figure. If None, figure is not saved (only returned).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `observed_data`        | \`list[ObservedDataPoint]  | None\`                                                    | Optional observed data points to overlay on corresponding bin subplots. Model predictions with a factor_id will be scaled for plotting using factor values from the calibration_result, so observed values remain in their original units. When the points of a series share one sum of outputs and one factor and have no window, the model line covers every position of the results. Otherwise the model of each point is read through its own operator (outputs, window, factor) at its own position, as the loss reads it, and the line joins these values. |
| `calibration_result`   | \`CalibrationResult        | Predictive                                                | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `config`               | \`PlotConfig               | None\`                                                    | Configuration for plot layout and styling (figsize, dpi, layout, style, palette, context). If None, uses defaults.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `series`               | \`list[str]                | None\`                                                    | What to plot, one subplot each. An entry is an output name (an expanded compartment or accumulator, such as A_g1) or a bin or accumulator id, which plots the sum of its outputs. An exact output name wins over an id. If None, plots every output.                                                                                                                                                                                                                                                                                                             |
| `show_legend`          | `bool`                     | Whether to show the legend on each subplot. Default True. | `True`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `position_label`       | \`Callable\[[float], str\] | None\`                                                    | Turns a position on the axis into an x-axis tick label. If None, the ticks show the date of their position when the model has an origin (ModelBuilder(origin=...)): 2020-01-02, with the time when it is not midnight. Otherwise they show the position.                                                                                                                                                                                                                                                                                                         |
| `tick_every`           | \`float                    | None\`                                                    | Labels the plotted positions tick_every apart on the axis, counted from the first plotted position. If None, labels every plotted position.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `x_label`              | `str`                      | The x-axis label. Default "Step".                         | `'Step'`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `prediction_intervals` | \`PredictionIntervals      | None\`                                                    | Prediction intervals of a conformal method (ConformalResult.intervals). The subplot of each of their series, by label, shows the band between their lower and upper ends and their point prediction; an unbounded interval is left out of the band.                                                                                                                                                                                                                                                                                                              |
| `**kwargs`             | \`str                      | int                                                       | float                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Figure` | The matplotlib Figure object. |

Raises:

| Type         | Description                                                                                                                                                                                |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ValueError` | If an entry of series is neither an output name nor a bin or accumulator id, or the results have no value at the position of an observation read on its own or at the start of its window. |

Source code in `commol/api/plotter.py`

```python
@_scoped_style
def plot_series(
    self,
    output_file: str | None = None,
    observed_data: list[ObservedDataPoint] | None = None,
    calibration_result: CalibrationResult | Predictive | None = None,
    config: PlotConfig | None = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: PredictionIntervals | None = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Plot simulation results with one subplot per series.

    Creates a figure with subplots arranged in a grid, where each subplot shows
    the evolution of one series along the axis. Optionally overlays observed data
    points.
    With a `Predictive`, plots the weighted median of its draws and their
    band of ``confidence_level``.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure. If None, figure is not saved (only returned).
    observed_data : list[ObservedDataPoint] | None
        Optional observed data points to overlay on corresponding bin subplots.
        Model predictions with a factor_id will be scaled for plotting using
        factor values from the calibration_result, so observed values remain in
        their original units. When the points of a series share one sum of
        outputs and one factor and have no window, the model line covers
        every position of the results. Otherwise the model of each point is
        read through its own operator (outputs, window, factor) at its own
        position, as the loss reads it, and the line joins these values.
    calibration_result : CalibrationResult | Predictive | None
        Optional calibration result. A `CalibrationResult` gives the factor
        values (``best_parameters``) that multiply model predictions. A
        `Predictive` plots the band of its draws instead of the simulation
        line; each draw's series is read through the observation operator
        of the problem (sums, windows and factors of the observations of
        the series, each at its own position when they do not share one
        operator with no window), then the draws are reduced with their
        weights.
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
        Turns a position on the axis into an x-axis tick label. If None,
        the ticks show the date of their position when the model has an
        origin (``ModelBuilder(origin=...)``): ``2020-01-02``, with the
        time when it is not midnight. Otherwise they show the position.
    tick_every : float | None
        Labels the plotted positions `tick_every` apart on the axis,
        counted from the first plotted position. If None, labels every
        plotted position.
    x_label : str
        The x-axis label. Default "Step".
    prediction_intervals : PredictionIntervals | None
        Prediction intervals of a conformal method
        (``ConformalResult.intervals``). The subplot of each of their
        series, by label, shows the band between their lower and upper
        ends and their point prediction; an unbounded interval is left
        out of the band.
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
        accumulator id, or the results have no value at the position of
        an observation read on its own or at the start of its window.
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
    if prediction_intervals is not None:
        for ax, bin_id in zip(axes, bins_to_plot, strict=False):
            self._plot_prediction_intervals(
                ax, bin_id, prediction_intervals, show_legend
            )
    self._finalize_plot(axes, bins_to_plot, output_file, config, "Series")

    return fig
```

#### plot_cumulative

```python
plot_cumulative(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | Predictive | None = None, config: PlotConfig | None = None, series: list[str] | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Plot cumulative (accumulated) simulation results with one subplot per series.

Creates a figure showing the running sum of each series along the axis. Useful for tracking totals, losses, or other accumulated quantities. With a `Predictive`, plots the band of the running totals of its draws.

Parameters:

| Name                 | Type                      | Description | Default                                                                                                                                                                                   |
| -------------------- | ------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_file`        | \`str                     | None\`      | Path to save the figure. If None, figure is not saved (only returned).                                                                                                                    |
| `observed_data`      | \`list[ObservedDataPoint] | None\`      | Optional observed data points to overlay (also shown as cumulative). Observed data points with a factor_id will be unscaled for plotting using factor values from the calibration_result. |
| `calibration_result` | \`CalibrationResult       | Predictive  | None\`                                                                                                                                                                                    |
| `config`             | \`PlotConfig              | None\`      | Configuration for plot layout and styling (figsize, dpi, layout, style, palette, context). If None, uses defaults.                                                                        |
| `series`             | \`list[str]               | None\`      | What to plot, one subplot each: an output name or a bin or accumulator id, as in :meth:plot_series. If None, plots every output.                                                          |
| `**kwargs`           | \`str                     | int         | float                                                                                                                                                                                     |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Figure` | The matplotlib Figure object. |

Raises:

| Type         | Description                                                                  |
| ------------ | ---------------------------------------------------------------------------- |
| `ValueError` | If an entry of series is neither an output name nor a bin or accumulator id. |

Source code in `commol/api/plotter.py`

```python
@_scoped_style
def plot_cumulative(
    self,
    output_file: str | None = None,
    observed_data: list[ObservedDataPoint] | None = None,
    calibration_result: CalibrationResult | Predictive | None = None,
    config: PlotConfig | None = None,
    series: list[str] | None = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Plot cumulative (accumulated) simulation results with one subplot per series.

    Creates a figure showing the running sum of each series along the axis.
    Useful for tracking totals, losses, or other accumulated quantities.
    With a `Predictive`, plots the band of the running totals of its draws.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure. If None, figure is not saved (only returned).
    observed_data : list[ObservedDataPoint] | None
        Optional observed data points to overlay (also shown as cumulative).
        Observed data points with a factor_id will be unscaled for plotting using
        factor values from the calibration_result.
    calibration_result : CalibrationResult | Predictive | None
        Optional calibration result. A `CalibrationResult` gives the factor
        values (``best_parameters``) that unscale observed data. A
        `Predictive` plots the band of the running total of each draw,
        and unscales observed data by the weighted median of each factor.
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
    self._finalize_plot(axes, bins_to_plot, output_file, config, "Cumulative")

    return fig
```

#### plot_prcc

```python
plot_prcc(result: PRCCResult, output_file: str | None = None, config: PlotConfig | None = None, x_label: str = 'Step') -> Figure
```

Plot the PRCC of each input: one subplot per series.

A series of a `Trajectory` draws one line per input over its positions; any other quantity draws one bar per input.

Parameters:

| Name          | Type         | Description                                                  | Default                                                        |
| ------------- | ------------ | ------------------------------------------------------------ | -------------------------------------------------------------- |
| `result`      | `PRCCResult` | The result of PRCC.                                          | *required*                                                     |
| `output_file` | \`str        | None\`                                                       | Path to save the figure. If None, the figure is only returned. |
| `config`      | \`PlotConfig | None\`                                                       | Layout and style. If None, the defaults.                       |
| `x_label`     | `str`        | The x-axis label of a series over positions. Default "Step". | `'Step'`                                                       |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Figure` | The matplotlib Figure object. |

Source code in `commol/api/plotter.py`

```python
@staticmethod
@_scoped_style
def plot_prcc(
    result: PRCCResult,
    output_file: str | None = None,
    config: PlotConfig | None = None,
    x_label: str = "Step",
) -> "Figure":
    """
    Plot the PRCC of each input: one subplot per series.

    A series of a `Trajectory` draws one line per input over its
    positions; any other quantity draws one bar per input.

    Parameters
    ----------
    result : PRCCResult
        The result of `PRCC`.
    output_file : str | None
        Path to save the figure. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    x_label : str
        The x-axis label of a series over positions. Default "Step".

    Returns
    -------
    Figure
        The matplotlib Figure object.
    """
    config = config or PlotConfig()
    SimulationPlotter._apply_seaborn_style(config)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, series in enumerate(result.series):
        groups[series].append(index)
    names = list(groups)
    fig, axes = SimulationPlotter._create_grid(config, len(names))
    for ax, name in zip(axes, names, strict=False):
        indices = groups[name]
        positions = [
            position
            for position in (result.positions[i] for i in indices)
            if position is not None
        ]
        if len(positions) == len(indices):
            for j, input_name in enumerate(result.inputs):
                ax.plot(
                    positions,
                    [result.coefficients[i][j] for i in indices],
                    marker="o",
                    label=input_name,
                )
            ax.set_xlabel(x_label)
            ax.legend()
        else:
            ax.bar(result.inputs, result.coefficients[indices[0]])
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("PRCC")
        ax.set_title(name)
    SimulationPlotter._finalize_plot(axes, names, output_file, config, "PRCC")
    return fig
```

#### plot_morris

```python
plot_morris(result: MorrisResult, output_file: str | None = None, config: PlotConfig | None = None) -> Figure
```

Plot `σ` against `μ*` of each input: one subplot per quantity, one labelled point per input.

An input far to the right matters; one high up acts through interactions or a non-linear effect.

Parameters:

| Name          | Type           | Description           | Default                                                        |
| ------------- | -------------- | --------------------- | -------------------------------------------------------------- |
| `result`      | `MorrisResult` | The result of Morris. | *required*                                                     |
| `output_file` | \`str          | None\`                | Path to save the figure. If None, the figure is only returned. |
| `config`      | \`PlotConfig   | None\`                | Layout and style. If None, the defaults.                       |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Figure` | The matplotlib Figure object. |

Source code in `commol/api/plotter.py`

```python
@staticmethod
@_scoped_style
def plot_morris(
    result: MorrisResult,
    output_file: str | None = None,
    config: PlotConfig | None = None,
) -> "Figure":
    """
    Plot ``σ`` against ``μ*`` of each input: one subplot per quantity,
    one labelled point per input.

    An input far to the right matters; one high up acts through
    interactions or a non-linear effect.

    Parameters
    ----------
    result : MorrisResult
        The result of `Morris`.
    output_file : str | None
        Path to save the figure. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.

    Returns
    -------
    Figure
        The matplotlib Figure object.
    """
    config = config or PlotConfig()
    SimulationPlotter._apply_seaborn_style(config)
    fig, axes = SimulationPlotter._create_grid(config, len(result.labels))
    for ax, q in zip(axes, range(len(result.labels)), strict=False):
        for j, input_name in enumerate(result.inputs):
            mu_star, sigma = result.mu_star[q][j], result.sigma[q][j]
            ax.scatter([mu_star], [sigma], label=input_name)
            ax.annotate(input_name, (mu_star, sigma))
        ax.set_xlabel("μ*")
        ax.set_ylabel("σ")
        ax.set_title(result.labels[q])
        ax.legend()
    SimulationPlotter._finalize_plot(
        axes, result.labels, output_file, config, "Morris"
    )
    return fig
```

#### plot_sobol

```python
plot_sobol(result: SobolResult | ChaosResult, output_file: str | None = None, config: PlotConfig | None = None) -> Figure
```

Plot the first-order and total Sobol indices: one subplot per quantity, one pair of bars per input, with the intervals of a `SobolResult`.

Parameters:

| Name          | Type          | Description   | Default                                                        |
| ------------- | ------------- | ------------- | -------------------------------------------------------------- |
| `result`      | \`SobolResult | ChaosResult\` | The result of Sobol or PolynomialChaos.                        |
| `output_file` | \`str         | None\`        | Path to save the figure. If None, the figure is only returned. |
| `config`      | \`PlotConfig  | None\`        | Layout and style. If None, the defaults.                       |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Figure` | The matplotlib Figure object. |

Source code in `commol/api/plotter.py`

```python
@staticmethod
@_scoped_style
def plot_sobol(
    result: SobolResult | ChaosResult,
    output_file: str | None = None,
    config: PlotConfig | None = None,
) -> "Figure":
    """
    Plot the first-order and total Sobol indices: one subplot per
    quantity, one pair of bars per input, with the intervals of a
    `SobolResult`.

    Parameters
    ----------
    result : SobolResult | ChaosResult
        The result of `Sobol` or `PolynomialChaos`.
    output_file : str | None
        Path to save the figure. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.

    Returns
    -------
    Figure
        The matplotlib Figure object.
    """
    config = config or PlotConfig()
    SimulationPlotter._apply_seaborn_style(config)
    inputs = result.inputs
    fig, axes = SimulationPlotter._create_grid(config, len(result.labels))
    width = 0.4
    for ax, q in zip(axes, range(len(result.labels)), strict=False):
        places = list(range(len(inputs)))
        for offset, name, values in (
            (-width / 2, "First order", result.first[q]),
            (width / 2, "Total", result.total[q]),
        ):
            errors = None
            if isinstance(result, SobolResult):
                lower, upper = (
                    (result.first_lower[q], result.first_upper[q])
                    if name == "First order"
                    else (result.total_lower[q], result.total_upper[q])
                )
                errors = [
                    [
                        max(v - low, 0.0)
                        for v, low in zip(values, lower, strict=True)
                    ],
                    [
                        max(high - v, 0.0)
                        for v, high in zip(values, upper, strict=True)
                    ],
                ]
            ax.bar(
                [place + offset for place in places],
                values,
                width,
                yerr=errors,
                capsize=3,
                label=name,
            )
        ax.set_xticks(places)
        ax.set_xticklabels(inputs)
        ax.set_ylabel("Sobol index")
        ax.set_title(result.labels[q])
        ax.legend()
    SimulationPlotter._finalize_plot(
        axes, result.labels, output_file, config, "Sobol"
    )
    return fig
```

#### plot_profiles

```python
plot_profiles(result: ProfileResult, output_file: str | None = None, config: PlotConfig | None = None) -> Figure
```

Plot each profile: one subplot per parameter or prediction, with the statistic of each point, the threshold, and the ends of the interval.

A profile that stays under the threshold up to a bound is open on that side; a flat one stays near 0 over its whole range.

Parameters:

| Name          | Type            | Description                                           | Default                                                        |
| ------------- | --------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| `result`      | `ProfileResult` | The result of ProfileLikelihood or PredictionProfile. | *required*                                                     |
| `output_file` | \`str           | None\`                                                | Path to save the figure. If None, the figure is only returned. |
| `config`      | \`PlotConfig    | None\`                                                | Layout and style. If None, the defaults.                       |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Figure` | The matplotlib Figure object. |

Source code in `commol/api/plotter.py`

```python
@staticmethod
@_scoped_style
def plot_profiles(
    result: ProfileResult,
    output_file: str | None = None,
    config: PlotConfig | None = None,
) -> "Figure":
    """
    Plot each profile: one subplot per parameter or prediction, with the
    statistic of each point, the threshold, and the ends of the interval.

    A profile that stays under the threshold up to a bound is open on that
    side; a flat one stays near 0 over its whole range.

    Parameters
    ----------
    result : ProfileResult
        The result of `ProfileLikelihood` or `PredictionProfile`.
    output_file : str | None
        Path to save the figure. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.

    Returns
    -------
    Figure
        The matplotlib Figure object.
    """
    config = config or PlotConfig()
    SimulationPlotter._apply_seaborn_style(config)
    names = list(result.profiles)
    fig, axes = SimulationPlotter._create_grid(config, len(names))
    for ax, name in zip(axes, names, strict=False):
        profile = result.profiles[name]
        ax.plot(profile.values, profile.statistics, marker="o", markersize=3)
        ax.axhline(profile.threshold, color="grey", linestyle="--", linewidth=1)
        interval = result.intervals[name]
        for end in (interval.lower, interval.upper):
            if end is not None:
                ax.axvline(end, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel(name)
        ax.set_ylabel("Profile statistic")
        ax.set_title(name)
    SimulationPlotter._finalize_plot(axes, names, output_file, config, "Profiles")
    return fig
```

## PlotConfig

Bases: `BaseModel`

Configuration for plot layout and styling.

Attributes:

| Name      | Type                  | Description                            |
| --------- | --------------------- | -------------------------------------- |
| `figsize` | `tuple[float, float]` | Figure size in inches (width, height). |
| `dpi`     | `int`                 | Dots per inch for figure resolution.   |
| `layout`  | \`tuple[int, int]     | None\`                                 |
| `style`   | \`str                 | None\`                                 |
| `palette` | \`str                 | None\`                                 |
| `context` | \`str                 | None\`                                 |

Notes

`style`, `palette` and `context` apply to the figure of one call of `SimulationPlotter` only: after the call, `matplotlib.rcParams` is as it was before it, so the figures that you draw with matplotlib keep your style.

Examples:

```pycon
>>> # Direct configuration
>>> config = PlotConfig(style="whitegrid", palette="Set2", context="talk")
```

```pycon
>>> # Using presets
>>> config = PlotConfig.presentation()
>>> config = PlotConfig.publication()
```

Source code in `commol/context/visualization.py`

```python
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

    Notes
    -----
    ``style``, ``palette`` and ``context`` apply to the figure of one call of
    `SimulationPlotter` only: after the call, ``matplotlib.rcParams`` is as it
    was before it, so the figures that you draw with matplotlib keep your style.

    Examples
    --------
    >>> # Direct configuration
    >>> config = PlotConfig(style="whitegrid", palette="Set2", context="talk")

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
```

### Methods:

#### presentation

```python
presentation() -> Self
```

Create a configuration optimized for presentations.

Large figure size, high DPI, and "talk" context for readable text on slides.

Returns:

| Type         | Description                                        |
| ------------ | -------------------------------------------------- |
| `PlotConfig` | Configuration with presentation-friendly settings. |

Source code in `commol/context/visualization.py`

```python
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
```

#### publication

```python
publication() -> Self
```

Create a configuration optimized for academic publications.

Smaller figure size suitable for journal columns, high DPI for print quality, and minimal styling.

Returns:

| Type         | Description                                       |
| ------------ | ------------------------------------------------- |
| `PlotConfig` | Configuration with publication-friendly settings. |

Source code in `commol/context/visualization.py`

```python
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
```

#### notebook

```python
notebook() -> Self
```

Create a configuration optimized for Jupyter notebooks.

Balanced defaults for interactive exploration.

Returns:

| Type         | Description                                    |
| ------------ | ---------------------------------------------- |
| `PlotConfig` | Configuration with notebook-friendly settings. |

Source code in `commol/context/visualization.py`

```python
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
```
