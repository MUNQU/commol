# Examples

Complete examples demonstrating different modeling scenarios using abstract, domain-agnostic compartment models.

## Example 1: Basic Three-State Model

A simple model with three states and two transitions:

```python
from commol import ModelBuilder, Simulation

model = (
    ModelBuilder(name="Three-State Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="flow_AB", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k2")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ]
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 2: Four-State Model with Intermediate States

Adding an intermediate state between A and B:

```python
model = (
    ModelBuilder(name="Four-State Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="M", name="Intermediate")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.4, description="A to M rate")
    .add_parameter(id="k2", value=0.2, description="M to B rate")
    .add_parameter(id="k3", value=0.1, description="B to C rate")
    .add_transition(id="flow_AM", source=["A"], target=["M"], rate="k1 * A * B / N")
    .add_transition(id="flow_MB", source=["M"], target=["B"], rate="k2")
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k3")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.999},
            {"bin": "M", "fraction": 0.0},
            {"bin": "B", "fraction": 0.001},
            {"bin": "C", "fraction": 0.0},
        ]
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 3: Periodic (Time-Dependent) Rates

Modelling periodic variation in a transition rate:

```python
model = (
    ModelBuilder(name="Periodic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k_mean", value=0.3)
    .add_parameter(id="k_amp", value=0.2)
    .add_parameter(id="k2", value=0.1)
    .add_transition(
        id="periodic_flow",
        source=["A"],
        target=["B"],
        rate="k_mean * (1 + k_amp * sin(2 * pi * step / 365)) * A * B / N"
    )
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k2")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ]
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=365 * 3)
```

## Example 4: Stratified Model with Group-Specific Rates

Different rates per category group:

```python
model = (
    ModelBuilder(name="Stratified Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_stratification(id="group", categories=["g1", "g2", "g3"])
    .add_parameter(id="k1", value=0.0003)
    .add_parameter(id="k2_g1", value=0.15)
    .add_parameter(id="k2_g2", value=0.12)
    .add_parameter(id="k2_g3", value=0.08)
    .add_transition(
        id="flow_AB",
        source=["A"],
        target=["B"],
        rate="k1"
    )
    .add_transition(
        id="flow_BC",
        source=["B"],
        target=["C"],
        stratified_rates=[
            {
                "conditions": [{"stratification": "group", "category": "g1"}],
                "rate": "k2_g1"
            },
            {
                "conditions": [{"stratification": "group", "category": "g2"}],
                "rate": "k2_g2"
            },
            {
                "conditions": [{"stratification": "group", "category": "g3"}],
                "rate": "k2_g3"
            },
        ]
    )
    .set_initial_conditions(
        population_size=10000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.25},
                    {"category": "g2", "fraction": 0.55},
                    {"category": "g3", "fraction": 0.20},
                ]
            }
        ]
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)

print(f"B_g1: {results['B_g1'][-1]:.2f}")
print(f"B_g2: {results['B_g2'][-1]:.2f}")
print(f"B_g3: {results['B_g3'][-1]:.2f}")
```

## Example 5: Waning Effect (Cyclic Transition)

Entities in the final state gradually return to the initial state:

```python
model = (
    ModelBuilder(name="Cyclic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_parameter(id="k3", value=0.01)  # Return rate
    .add_transition(id="flow_AB", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k2")
    .add_transition(id="return_CA", source=["C"], target=["A"], rate="k3")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ]
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=1000)
```

## Example 6: Capacity-Constrained Rate

The flow rate slows as a downstream compartment approaches a capacity limit:

```python
model = (
    ModelBuilder(name="Capacity Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.5)
    .add_parameter(id="k2_max", value=0.15)
    .add_parameter(id="capacity", value=100.0)
    .add_transition(id="flow_AB", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(
        id="flow_BC",
        source=["B"],
        target=["C"],
        rate="k2_max * (1 - max(0, (B - capacity) / capacity))"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.95},
            {"bin": "B", "fraction": 0.05},
            {"bin": "C", "fraction": 0.0},
        ]
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 7: Multi-Stratified Model with Intersection Rates

A model with two stratifications and a rate specific to a particular intersection of categories:

```python
model = (
    ModelBuilder(name="Multi-Stratified Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_stratification(id="type", categories=["t1", "t2"])
    .add_parameter(id="k1_low", value=0.3)
    .add_parameter(id="k1_high", value=0.6)
    .add_parameter(id="k2", value=0.1)
    .add_transition(
        id="flow_AB",
        source=["A"],
        target=["B"],
        rate="k1_low * A * B / N",
        stratified_rates=[
            {
                "conditions": [{"stratification": "type", "category": "t2"}],
                "rate": "k1_high * A * B / N"
            }
        ]
    )
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k2")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.6},
                    {"category": "g2", "fraction": 0.4},
                ]
            },
            {
                "stratification": "type",
                "fractions": [
                    {"category": "t1", "fraction": 0.8},
                    {"category": "t2", "fraction": 0.2},
                ]
            }
        ]
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 8: Subpopulation-Normalized Rate

Using the automatically computed `N_{category}` variable to normalize a rate by subpopulation size instead of total population:

```python
model = (
    ModelBuilder(name="Subpopulation-Normalized Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k1", value=0.4)
    .add_parameter(id="k2", value=0.1)
    .add_transition(
        id="flow_AB",
        source=["A"],
        target=["B"],
        rate="k1 * A * B / N",
        stratified_rates=[
            {
                "conditions": [{"stratification": "group", "category": "g1"}],
                "rate": "k1 * A * B / N_g1"  # Normalized by subpopulation
            }
        ]
    )
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k2")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.5},
                    {"category": "g2", "fraction": 0.5},
                ]
            }
        ]
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 9: Conditional Stratification

A second stratification that only applies to a subset of the first. Here, the `subtype` stratification only expands `g2` compartments — `g1` compartments are not split further.

```python
model = (
    ModelBuilder(name="Conditional Stratification Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    # group applies to all compartments
    .add_stratification(id="group", categories=["g1", "g2"])
    # subtype only applies to g2 compartments
    .add_stratification(
        id="subtype",
        categories=["s1", "s2"],
        conditions=[{"stratification": "group", "category": "g2"}]
    )
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2_default", value=0.1)
    .add_parameter(id="k2_s1", value=0.15)
    .add_parameter(id="k2_s2", value=0.05)
    .add_transition(
        id="flow_AB",
        source=["A"],
        target=["B"],
        rate="k1 * A * B / N"
    )
    .add_transition(
        id="flow_BC",
        source=["B"],
        target=["C"],
        rate="k2_default",
        stratified_rates=[
            {
                "conditions": [{"stratification": "subtype", "category": "s1"}],
                "rate": "k2_s1"
            },
            {
                "conditions": [{"stratification": "subtype", "category": "s2"}],
                "rate": "k2_s2"
            },
        ]
    )
    .set_initial_conditions(
        population_size=10000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.6},
                    {"category": "g2", "fraction": 0.4},
                ]
            },
            {
                "stratification": "subtype",
                "fractions": [
                    {"category": "s1", "fraction": 0.7},
                    {"category": "s2", "fraction": 0.3},
                ]
            }
        ]
    )
    .build(typology="DifferenceEquations")
)

# Generated compartments:
# A_g1, B_g1, C_g1          (group=g1, no subtype)
# A_g2_s1, B_g2_s1, C_g2_s1 (group=g2, subtype=s1)
# A_g2_s2, B_g2_s2, C_g2_s2 (group=g2, subtype=s2)

simulation = Simulation(model)
results = simulation.run(num_steps=100)

print(f"B_g1:    {results['B_g1'][-1]:.2f}")
print(f"B_g2_s1: {results['B_g2_s1'][-1]:.2f}")
print(f"B_g2_s2: {results['B_g2_s2'][-1]:.2f}")
```

## Example 10: Parameter Calibration

Calibrate model parameters to match observed data:

```python
from commol import (
    ModelBuilder,
    Simulation,
    Calibrator,
    CalibrationProblem,
    CalibrationParameter,
    ObservedDataPoint,
    ParticleSwarmConfig,
)

model = (
    ModelBuilder(name="Model for Calibration", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=None)   # To be calibrated
    .add_parameter(id="k2", value=None)   # To be calibrated
    .add_transition(
        id="flow_AB",
        source=["A"],
        target=["B"],
        rate="k1 * A * B / N"
    )
    .add_transition(
        id="flow_BC",
        source=["B"],
        target=["C"],
        rate="k2 * B"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ]
    )
    .build(typology="DifferenceEquations")
)

observed_data = [
    ObservedDataPoint(step=0,  compartment="B", value=10.0),
    ObservedDataPoint(step=10, compartment="B", value=45.2),
    ObservedDataPoint(step=20, compartment="B", value=78.5),
    ObservedDataPoint(step=30, compartment="B", value=62.3),
    ObservedDataPoint(step=40, compartment="B", value=38.1),
    ObservedDataPoint(step=50, compartment="B", value=18.7),
    ObservedDataPoint(step=60, compartment="B", value=8.2),
]

simulation = Simulation(model)

parameters = [
    CalibrationParameter(
        id="k1",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3
    ),
    CalibrationParameter(
        id="k2",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.1
    ),
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_function="sse",
    optimization_config=ParticleSwarmConfig(
        num_particles=30,
        max_iterations=500,
        verbose=True
    ),
)

calibrator = Calibrator(simulation, problem)
result = calibrator.run()

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final loss: {result.final_loss:.6f}")
print(f"k1: {result.best_parameters['k1']:.6f}")
print(f"k2: {result.best_parameters['k2']:.6f}")

model.update_parameters(result.best_parameters)
calibrated_sim = Simulation(model)
results = calibrated_sim.run(num_steps=100)
```

## Next Steps

- [API Reference](../api/model-builder.md) - Complete API documentation
- [Calibration Guide](calibration.md) - Comprehensive calibration documentation
- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Contributing](../development/contributing.md) - Build your own examples
