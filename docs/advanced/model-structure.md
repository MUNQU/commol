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

::: commol.context.model.Model
options:
show_root_heading: true
show_source: true
heading_level: 3

## Population

::: commol.context.population.Population
options:
show_root_heading: true
show_source: true
heading_level: 3

## Bins

::: commol.context.bin.Bin
options:
show_root_heading: true
show_source: true
heading_level: 3

## Stratifications

::: commol.context.stratification.Stratification
options:
show_root_heading: true
show_source: true
heading_level: 3

## Parameters

::: commol.context.parameter.Parameter
options:
show_root_heading: true
show_source: true
heading_level: 3

## Transitions

::: commol.context.dynamics.Transition
options:
show_root_heading: true
show_source: true
heading_level: 3

## Initial Conditions

::: commol.context.initial_conditions.InitialConditions
options:
show_root_heading: true
show_source: true
heading_level: 3

## Dynamics

::: commol.context.dynamics.Dynamics
options:
show_root_heading: true
show_source: true
heading_level: 3
