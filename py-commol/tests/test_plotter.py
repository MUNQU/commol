import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Use non-interactive backend to prevent figures from displaying during tests
matplotlib.use("Agg")

from commol import (
    Model,
    ModelBuilder,
    ObservedDataPoint,
    PlotConfig,
    Simulation,
)
from commol.api.plotter import SimulationPlotter
from commol.constants import ModelTypes


class TestSimulationPlotter:
    @pytest.fixture(scope="class")
    def sir_model(self) -> Model:
        """Create a basic SIR model for testing."""
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )
        return builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

    @pytest.fixture
    def simulation_results(self, sir_model: Model):
        """Run simulation and return results."""
        simulation = Simulation(sir_model)
        results = simulation.run(100, output_format="dict_of_lists")
        return simulation, results

    def test_plotter_initialization(self, simulation_results):
        """Test that SimulationPlotter initializes correctly."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        assert plotter.simulation == simulation
        assert plotter.results == results
        assert len(plotter.bins) == 3
        assert plotter.num_steps == 100

    def test_plot_series_basic(self, simulation_results):
        """Test basic series plotting without saving."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        fig = plotter.plot_series()

        assert fig is not None
        plt.close(fig)

    def test_plot_series_saves_to_file(self, simulation_results):
        """Test that plot_series saves to a file."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "series_plot.png"

            fig = plotter.plot_series(output_file=str(output_path))

            assert output_path.exists()
            assert output_path.stat().st_size > 0
            plt.close(fig)

    def test_plot_series_with_observed_data(self, simulation_results):
        """Test series plotting with observed data overlay."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        # Create some observed data points
        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(0, 100, 10)
        ]

        fig = plotter.plot_series(observed_data=observed_data)

        assert fig is not None
        plt.close(fig)

    def test_plot_series_with_scale_values(self, simulation_results):
        """Test series plotting with scale values for observed data."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        scale_factor = 0.5
        observed_data = [
            ObservedDataPoint(
                step=i,
                compartment="I",
                value=results["I"][i] * scale_factor,
                scale_id="detection_rate",
            )
            for i in range(0, 100, 10)
        ]

        scale_values = {"detection_rate": scale_factor}

        fig = plotter.plot_series(
            observed_data=observed_data, scale_values=scale_values
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_series_subset_of_bins(self, simulation_results):
        """Test plotting only a subset of bins."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        fig = plotter.plot_series(bins=["I", "R"])

        assert fig is not None
        plt.close(fig)

    def test_plot_series_with_custom_config(self, simulation_results):
        """Test series plotting with custom PlotConfig."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        config = PlotConfig(
            figsize=(12, 8),
            dpi=150,
            layout=(1, 3),
            style="whitegrid",
            palette="Set2",
            context="notebook",
        )

        fig = plotter.plot_series(config=config)

        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_series_with_seaborn_overrides(self, simulation_results):
        """Test that direct seaborn parameters override config."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        fig = plotter.plot_series(
            seaborn_style="dark",
            palette="husl",
            context="talk",
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_cumulative_basic(self, simulation_results):
        """Test basic cumulative plotting."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        fig = plotter.plot_cumulative()

        assert fig is not None
        plt.close(fig)

    def test_plot_cumulative_saves_to_file(self, simulation_results):
        """Test that plot_cumulative saves to a file."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cumulative_plot.png"

            fig = plotter.plot_cumulative(output_file=str(output_path))

            assert output_path.exists()
            assert output_path.stat().st_size > 0
            plt.close(fig)

    def test_plot_cumulative_with_observed_data(self, simulation_results):
        """Test cumulative plotting with observed data."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(0, 100, 10)
        ]

        fig = plotter.plot_cumulative(observed_data=observed_data)

        assert fig is not None
        plt.close(fig)

    def test_calculate_layout(self, simulation_results):
        """Test layout calculation for different bin counts."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        assert plotter._calculate_layout(1) == (1, 1)
        assert plotter._calculate_layout(2) == (1, 2)
        assert plotter._calculate_layout(3) == (2, 2)
        assert plotter._calculate_layout(4) == (2, 2)
        assert plotter._calculate_layout(5) == (2, 3)
        assert plotter._calculate_layout(9) == (3, 3)

    def test_group_observed_data(self, simulation_results):
        """Test grouping observed data by compartment."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        observed_data = [
            ObservedDataPoint(step=10, compartment="I", value=50.0),
            ObservedDataPoint(step=20, compartment="I", value=60.0),
            ObservedDataPoint(step=10, compartment="R", value=10.0),
        ]

        grouped = plotter._group_observed_data(observed_data)

        assert len(grouped) == 2
        assert len(grouped["I"]) == 2
        assert len(grouped["R"]) == 1
        assert grouped["I"][0].step == 10
        assert grouped["I"][1].step == 20

    def test_plot_series_with_kwargs(self, simulation_results):
        """Test that additional kwargs are passed to seaborn.lineplot."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        fig = plotter.plot_series(
            linewidth=3,
            alpha=0.7,
            linestyle="--",
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_cumulative_with_kwargs(self, simulation_results):
        """Test that additional kwargs work with cumulative plot."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        fig = plotter.plot_cumulative(
            linewidth=2,
            alpha=0.8,
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_config_presets(self, simulation_results):
        """Test PlotConfig preset class methods."""
        simulation, results = simulation_results
        plotter = SimulationPlotter(simulation, results)

        # Test presentation preset
        config = PlotConfig.presentation()
        assert config.figsize == (16, 10)
        assert config.dpi == 150
        assert config.context == "talk"

        fig = plotter.plot_series(config=config)
        assert fig is not None
        plt.close(fig)

        # Test publication preset
        config = PlotConfig.publication()
        assert config.figsize == (8, 6)
        assert config.dpi == 300
        assert config.context == "paper"

        fig = plotter.plot_series(config=config)
        assert fig is not None
        plt.close(fig)

        # Test notebook preset
        config = PlotConfig.notebook()
        assert config.figsize == (12, 8)
        assert config.context == "notebook"

        fig = plotter.plot_series(config=config)
        assert fig is not None
        plt.close(fig)
