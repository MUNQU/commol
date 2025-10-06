# Examples

Complete examples demonstrating different modeling scenarios.

## Example 1: Basic SIR Model

Classic Susceptible-Infected-Recovered model:

```python
from epimodel import ModelBuilder, Simulation
from epimodel.constants import ModelTypes
import matplotlib.pyplot as plt

# Build model
model = (
    ModelBuilder(name="Basic SIR", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0}
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Simulate
simulation = Simulation(model)
results = simulation.run(num_steps=100)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(results["S"], label="Susceptible", color="blue")
plt.plot(results["I"], label="Infected", color="red")
plt.plot(results["R"], label="Recovered", color="green")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.title("SIR Model")
plt.legend()
plt.grid(True)
plt.show()
```

## Example 2: SEIR Model

Adding an exposed (incubation) period:

```python
model = (
    ModelBuilder(name="SEIR Model", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="E", name="Exposed")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.4, description="Transmission rate")
    .add_parameter(id="sigma", value=0.2, description="Incubation rate")
    .add_parameter(id="gamma", value=0.1, description="Recovery rate")
    .add_transition(id="exposure", source=["S"], target=["E"], rate="beta * S * I / N")
    .add_transition(id="infection", source=["E"], target=["I"], rate="sigma")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions={"S": 0.999, "E": 0.0, "I": 0.001, "R": 0.0}
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

simulation = Simulation(model)
results = simulation.run(num_steps=200)
```

## Example 3: Seasonal Transmission

Modeling seasonal variation in transmission:

```python
model = (
    ModelBuilder(name="Seasonal SIR", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta_mean", value=0.3)
    .add_parameter(id="beta_amp", value=0.2)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(
        id="seasonal_infection",
        source=["S"],
        target=["I"],
        # Seasonal forcing: peaks in winter (day 0, 365, ...)
        rate="beta_mean * (1 + beta_amp * sin(2 * pi * step / 365)) * S * I / N"
    )
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0}
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Run for 3 years to see seasonal pattern
simulation = Simulation(model)
results = simulation.run(num_steps=365 * 3)

plt.figure(figsize=(12, 6))
plt.plot(results["I"], color="red")
plt.xlabel("Time (days)")
plt.ylabel("Infected Population")
plt.title("Seasonal Transmission Pattern")
plt.grid(True)
plt.show()
```

## Example 4: Age-Stratified Model

Different age groups with varying susceptibility:

```python
model = (
    ModelBuilder(name="Age-Stratified SIR", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_stratification(id="age", categories=["child", "adult", "elderly"])
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma_child", value=0.15)
    .add_parameter(id="gamma_adult", value=0.12)
    .add_parameter(id="gamma_elderly", value=0.08)
    # Note: In a real age-stratified model, you'd add transitions for each age group
    # This is a simplified example
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma_adult")
    .set_initial_conditions(
        population_size=10000,
        disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0},
        stratification_fractions={
            "age": {
                "child": 0.25,
                "adult": 0.55,
                "elderly": 0.20
            }
        }
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

simulation = Simulation(model)
results = simulation.run(num_steps=200)

# Plot stratified results
plt.figure(figsize=(12, 8))

# Plot infected by age group
for age in ["child", "adult", "elderly"]:
    key = f"I_{age}"
    if key in results:
        plt.plot(results[key], label=f"Infected ({age})")

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.title("Age-Stratified Infections")
plt.legend()
plt.grid(True)
plt.show()
```

## Example 5: Vaccination Campaign

Adding vaccination to an SIR model:

```python
model = (
    ModelBuilder(name="SIR with Vaccination", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.4)
    .add_parameter(id="gamma", value=0.1)
    .add_parameter(id="vax_rate", value=0.01)  # 1% per day
    .add_parameter(id="vax_eff", value=0.9)     # 90% effective
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .add_transition(
        id="vaccination",
        source=["S"],
        target=["R"],
        rate="vax_rate * vax_eff"  # Effective vaccination rate
    )
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0}
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Compare with and without vaccination
simulation = Simulation(model)
results_vax = simulation.run(num_steps=200)

# Model without vaccination
model_no_vax = (
    ModelBuilder(name="SIR No Vaccination", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.4)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0}
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

simulation_no_vax = Simulation(model_no_vax)
results_no_vax = simulation_no_vax.run(num_steps=200)

# Compare
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results_no_vax["I"], label="No Vaccination", color="red")
plt.plot(results_vax["I"], label="With Vaccination", color="orange")
plt.xlabel("Time (days)")
plt.ylabel("Infected")
plt.title("Impact of Vaccination on Infections")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(results_no_vax["R"], label="No Vaccination", color="green")
plt.plot(results_vax["R"], label="With Vaccination", color="lightgreen")
plt.xlabel("Time (days)")
plt.ylabel("Recovered/Immune")
plt.title("Immunity Over Time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Example 6: Healthcare Capacity

Modeling reduced recovery when hospitals are overwhelmed:

```python
model = (
    ModelBuilder(name="SIR with Healthcare Capacity", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.5)
    .add_parameter(id="gamma_max", value=0.15)
    .add_parameter(id="hospital_cap", value=100.0)
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(
        id="recovery_saturated",
        source=["I"],
        target=["R"],
        # Recovery slows as infections approach hospital capacity
        rate="gamma_max * (1 - max(0, (I - hospital_cap) / hospital_cap))"
    )
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions={"S": 0.95, "I": 0.05, "R": 0.0}
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

simulation = Simulation(model)
results = simulation.run(num_steps=150)

plt.figure(figsize=(10, 6))
plt.plot(results["I"], label="Infected", color="red")
plt.axhline(y=100, color="black", linestyle="--", label="Hospital Capacity")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.title("Healthcare Capacity Saturation")
plt.legend()
plt.grid(True)
plt.show()
```

## Example 7: Waning Immunity

Recovered individuals gradually become susceptible again:

```python
model = (
    ModelBuilder(name="SIRS Model", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma", value=0.1)
    .add_parameter(id="omega", value=0.01)  # Waning immunity rate
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .add_transition(id="waning", source=["R"], target=["S"], rate="omega")
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0}
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Run for longer to see endemic equilibrium
simulation = Simulation(model)
results = simulation.run(num_steps=1000)

plt.figure(figsize=(12, 6))
plt.plot(results["S"], label="Susceptible", color="blue")
plt.plot(results["I"], label="Infected", color="red")
plt.plot(results["R"], label="Recovered", color="green")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.title("SIRS Model with Waning Immunity")
plt.legend()
plt.grid(True)
plt.show()
```

## Next Steps

- [API Reference](../api/model-builder.md) - Complete API documentation
- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Contributing](../development/contributing.md) - Build your own examples
