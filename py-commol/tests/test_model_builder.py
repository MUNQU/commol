from commol.api.model_builder import ModelBuilder
from commol.context.model import Model
from commol.constants import ModelTypes


class TestModelBuilder:
    def test_build_simple_model(self):
        """
        Test that a simple model can be built using the ModelBuilder.
        """
        builder = (
            ModelBuilder(name="TestModel", version="0.1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.1)
            .add_parameter(id="gamma", value=0.05)
            .add_transition(
                id="infection",
                source=["S", "I"],
                target=["I", "I"],
                rate="beta * S * I",
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

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

        assert isinstance(model, Model)
        assert model.name == "TestModel"
        assert model.version == "0.1.0"
        assert len(model.population.disease_states) == 3
        assert len(model.parameters) == 2
        assert len(model.dynamics.transitions) == 2
        assert model.population.initial_conditions.population_size == 1000
