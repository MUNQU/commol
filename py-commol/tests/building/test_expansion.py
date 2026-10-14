"""Python helpers expand a model exactly as the engine does."""

from commol.constants import PrintEquationsOutputFormat
from commol.context._model.engine_state import expanded_outputs
from commol.context._model.helpers import ModelCompartmentHelper
from commol.context.stratification import Stratification
from commol.context.stratification_condition import StratificationCondition
from tests.support import a_to_b_builder, build, conditional_pathway_builder


class TestEngineExpansion:
    def test_equation_helper_uses_the_engine_expansion(self) -> None:
        model = build(conditional_pathway_builder())
        compartments, accumulators = expanded_outputs(model)
        helper = ModelCompartmentHelper(model)

        assert helper.generate_compartments() == [
            (output.source, output.categories) for output in compartments
        ]
        assert helper.generate_accumulator_outputs() == [
            (output.source, output.categories) for output in accumulators
        ]

    def test_category_less_condition_expands_like_the_engine(self) -> None:
        model = build(
            a_to_b_builder(
                rate="k * A",
                per_compartment=True,
                stratifications={"group": {"g1": 1.0, "g2": 0.0}},
            )
        )
        # A category-less condition is permitted by the engine but rejected by
        # the Pydantic validator, so the stratification is built without it.
        model.population.stratifications.append(
            Stratification.model_construct(
                id="care",
                categories=["home", "hosp"],
                description=None,
                conditions=[
                    StratificationCondition.model_construct(
                        stratification="group", category=None, to=None
                    )
                ],
            )
        )
        helper = ModelCompartmentHelper(model)

        helper_names = [
            helper.compartment_to_string(compartment, PrintEquationsOutputFormat.TEXT)
            for compartment in helper.generate_compartments()
        ]

        assert helper_names == [c.name for c in model.initial_state()]
        assert "A_g2_hosp" in helper_names
