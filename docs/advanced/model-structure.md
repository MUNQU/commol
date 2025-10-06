# Model Structure (Advanced)

!!! info "For Advanced Users"
This section documents the internal structure of the `Model` class and its components.

    **Most users don't need this** - use `ModelBuilder` instead to create models.

    This is useful for:
    - Understanding the internal model representation
    - Working with `ModelLoader.from_json()`
    - Contributing to the library
    - Debugging complex models

## Model

::: epimodel.context.model.Model
options:
show_root_heading: true
show_source: true
heading_level: 3

## Population

::: epimodel.context.population.Population
options:
show_root_heading: true
show_source: true
heading_level: 3

## Disease States

::: epimodel.context.disease_state.DiseaseState
options:
show_root_heading: true
show_source: true
heading_level: 3

## Stratifications

::: epimodel.context.stratification.Stratification
options:
show_root_heading: true
show_source: true
heading_level: 3

## Parameters

::: epimodel.context.parameter.Parameter
options:
show_root_heading: true
show_source: true
heading_level: 3

## Transitions

::: epimodel.context.transition.Transition
options:
show_root_heading: true
show_source: true
heading_level: 3

## Initial Conditions

::: epimodel.context.initial_conditions.InitialConditions
options:
show_root_heading: true
show_source: true
heading_level: 3

## Dynamics

::: epimodel.context.dynamics.Dynamics
options:
show_root_heading: true
show_source: true
heading_level: 3
