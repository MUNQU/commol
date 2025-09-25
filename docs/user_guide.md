# User Guide

## JSON Schema Documentation

This document defines a **standard JSON schema** for describing epidemiological models that can be implemented as **ODEs, agent-based models (ABM)**, or **network-based models**.
The schema is designed to be **flexible and extensible**, so the same structure can represent different modeling approaches while keeping consistency across model types.

---

### Top-Level Structure

```json
{
  "model": { ... },
  "population": { ... },
  "parameters": [ ... ],
  "dynamics": { ... },
  "network": { ... }
}
```

### Sections:
1. **`model`** → General information about the model.
2. **`population`** → Compartments, stratifications, and initialization.
3. **`parameters`** → Global model parameters.
4. **`dynamics`** → Transitions, interactions, and update rules.
5. **`contact_structures`** → (optional) Explicit graph structure for abm and network models.

---

#### 1. `model`

Describes the model itself.

| Field | Type | Description | Optional |
|-------|------|-------------|----------|
| `name` | string | Model name (unique identifier). | - |
| `description` | string | Human-readable description. | Optional |
| `version` | string | Version number of the schema/model. | Optional |

**Example:**
```json
"model": {
  "name": "SIRV_example",
  "description": "A basic SIRV model with vaccination and waning immunity.",
  "version": "1.0"
}
```

---

#### 2. `population`

Defines **compartments, stratifications, and initial population setup** for the model.

| Field | Type | Applies to | Description | Optional |
|-------|------|------------|-------------|----------|
| `subpopulations` | list of objects | All | Defines distinct compartments or states (e.g., S, I, R, V). | - |
| `stratifications` | list of objects | All | Defines categorical subdivisions of the population (e.g., age, risk, vaccination status). | - |
| `initial_conditions` | object | ODE | Initial counts per compartment and stratification. | - |
| `initialization` | object | ABM / Network | How agents/nodes are generated and assigned compartments and attributes. | - |

##### 2.1 `subpopulations`

Each object represents a **state or compartment**:

- **`symbol`** (string): Short identifier for formulas/transitions.
- **`name`** (string): Human-readable description of the compartment.

**Example:**
```json
[
  { "symbol": "N", "name": "Total population" },
  { "symbol": "S", "name": "Susceptible" },
  { "symbol": "I", "name": "Infected" },
  { "symbol": "R", "name": "Recovered" }
]
```
---
**NOTE**
N (Total population) subpopulation will be always added intenally by default as the sum of all the subpopulations which form the model.
---

##### 2.2 `stratifications`

Each object defines a **categorical partition**:

- **`name`** (string): Stratification identifier used in conditions, contact matrices, etc.
- **`categories`** (list(strings)): Possible values of this stratification.

**Example:**
```json
[
  { "name": "age", "categories": ["young", "adult", "elderly"] },
  { "name": "vaccination_status", "categories": ["unvaccinated", "vaccinated"] }
]
```

##### 2.3 `initial_conditions` (ODE only)

Specifies the **initial number of individuals** in each subpopulation, optionally stratified.

- **Keys** (string): Subpopulation symbols.
- **Values** (integer | object): Number (unstratified) or object mapping stratification categories (string) to counts (integer).

**Example:**
```json
"initial_conditions": {
  "S": { "young": 4000, "adult": 5000 },
  "I": { "young": 20, "adult": 10 }
}
```

##### 2.4 `initialization` (ABM / Network only)

Specifies **how agents/nodes are generated** and assigned attributes/compartments.

- **`population_size`** (integer): Total agents/nodes to generate.
- **`distribution`** (object): Probabilities for attributes and initial compartments:
  - Keys (string): `"initial_compartment_fraction"` and stratification names.
  - Values (integer): objects mapping category to fraction/probability.

**Example:**
```json
"initialization": {
  "population_size": 2000,
  "distribution": {
    "initial_compartment_fraction": { "S": 0.99, "I": 0.01 }
    "age": { "young": 0.5, "adult": 0.5 },
  }
}
```

---

#### 3. `parameters`

Global model parameters.

| Field | Type | Description | Optional |
|-------|------|-------------|----------|
| `name` | string | Identifier used in formulas. | - |
| `description` | string | Human-readable description. | Optional |
| `value` | float | Default numerical value. | - |

**Example:**
```json
"parameters": [
  { "name": "beta", "description": "Transmission rate", "value": 0.25 },
  { "name": "gamma", "description": "Recovery rate", "value": 0.1 }
]
```

---

#### 4. `dynamics`

Defines **how the system evolves**.

| Field | Type | Applies to | Description | Optional |
|-------|------|------------|-------------|----------|
| `type` | enum[string] | All | `"ODE"`, `"ABM"`, `"network"`. | - |
| `agent_attributes` | list of objects | ABM / Network | Additional properties or internal state variables for agents or nodes (e.g., days since vaccination). | - |
| `transitions` | list of objects | All | Rules describing state changes. | - |

##### 4.1 `agent_attributes`

Defines additional properties or internal state variables for agents or nodes in ABM and network models. These attributes describe characteristics, counters, or dynamic features of individual agents, beyond their compartment membership.

- **`name`** (string): Unique identifier for the attribute.
- **`type`** enum(string): Data type (`integer`, `float`, `boolean`, `string`).
- **`initial_value`** enum(string): Default value assigned to new agents/nodes.
- **`description`** (string, optional): Human-readable description of the attribute.

```json
{ "name": "days_since_vaccine", "type": "integer", "initial_value": 0 }
```

##### 4.2 `transitions`

Defines **how agents or compartments move between states** in the model. Each transition is represented as an object. The structure is designed to accommodate **ODE flows**, **agent-based interactions**, and **network-based transmission** in a unified way.

- **`name`** (string): Transmission name.
- **`source`** (string): Subpopulations/compartments that act as origin states.
- **`target`** (string): Subpopulations/compartments that act as destination states.
- **`rate`** (string): Represents the mathematical flow between compartments (ODE only).
- **`mechanism`** enum(string): Defines how the transition occurs at the agent level (ABM / Network only):
  - `"pairwise_random_contact"` → Random pairwise contacts per timestep.
  - `"edge_based"` → Contact through network edges.
  - `"probability"` → Fixed probability per agent per timestep.
- **`"params"`** (object): Holds parameters that configure how the mechanism operates. This may include numeric values, symbolic references to global parameters, or formulas. Parameters can control aspects such as contact rates, probabilities, or scaling factors. They provide a flexible way to make the same mechanism behave differently across scenarios without redefining its structure.(ABM / Network only).
- **`"conditions"`** (object, optional): Defines logical restrictions that determine when the transition can occur. A condition is made up of one or more rules combined with logical operators. Each rule specifies a variable (which can be a stratification or an agent attribute), an operator (eq, neq, gt, get, lt, let), and a value. Conditions are essential for capturing context-dependent behaviors, such as age-specific policies, time-dependent immunity, or individual health states.
- **`"trigger"`** (list of strings, optional): Subpopulations that can cause the transition (e.g., infection triggered by I). Include trigger when the event is caused by other agents (infection, contact-based events). Omit trigger for purely internal/autonomous transitions (recovery, waning immunity) unless your engine requires explicit triggers for scheduling. (ABM / Network only).
- **`"contact_matrix_stratifications"`** (list of strings, optional): Specifies which stratifications should be used to build contact matrices for a given transition. Ensures that contact probabilities between agents or compartments are not uniform but instead follow a structured matrix. (ABM / Network only).

**ODE Transition**
```json
{
  "name": "infection",
  "source": ["S"],
  "target": ["I"],
  "rate": "beta * S * I / N",
  "conditions": {
    "logic": "and",
    "rules": [
      { "variable": "stratifications:age", "operator": "eq", "value": "adult" }
    ]
  }
}
```

**ABM / Network Transition**
```json
{
  "name": "recovery",
  "source": ["I"],
  "target": ["R"],
  "mechanism": "probability",
  "params": {
    "probability_per_timestep": "gamma"
  },
  "conditions": {
    "logic": "or",
    "rules": [
      { "variable": "attributes:days_infected", "operator": "gt", "value": 10 },
      { "variable": "stratifications:age", "operator": "eq", "value": "adult" }
    ]
  },
  "effects": {
    "days_infected": 0
  }
}
```

---

#### 5. `contact_structures` (ABM / Network only)

Defines how agents interact based on stratified contact patterns or network connectivity. Allows to encode mixing matrices, layered contact networks, and time-varying contact structures.

| Field | Type | Description | Optional |
|-------|------|-------------|----------|
| `structure` | object | Constact structure of the population. | Optional |

##### 5.1 `contact_structures`

Defines how agents interact based on stratified contact patterns or network connectivity.

- **`name`** (string): Miximg matrix name.
- **`stratification`** (string): Stratification group the mixing matrix applies.
- **`structures`** (list of objects):
  - **`initial_time_instant`** (integer): Initial time instant when the structure is applied.
  - **`final_time_instant`** (integer): Initial time instant when the structure is applied.
  - **`matrix`** (2D array of floats): Square matrix where entry i,j is the average number of contacts per unit time an individual in stratum i has with individuals in stratum j. The order of the matrix is the same given in stratification categories.
- **`notes`** (string, optional): Free-text description or reference source of the matrix.


**Basic contact matrix**
```json
"contact_structures": [
  {
    "name": "age_mixing"
    "stratification": "age",
    "structures": [
      {
      "initial_time_instant": 0,
      "final_time_instant": 180,
      "matrix":  [
          [8.0, 1.5, 0.5],
          [1.5, 7.0, 1.0],
          [0.5, 1.0, 3.0]
        ]
      }
    ],
    "notes": "Synthetic age-stratified contact matrix"
  }
]
```
