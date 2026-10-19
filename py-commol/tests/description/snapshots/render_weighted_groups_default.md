# Weighted

## Overview

| Field | Value |
|---|---|
| Name | Weighted |
| Kind | aggregate |
| Engine | DifferenceEquations |
| Engine settings | defaults |
| Dynamics | difference equation |
| Step unit | step |

## Entities and state variables

### Bins

| Bin | Name | Unit |
|---|---|---|
| A | State "A" \ start | item |
| B | State B | item |

### Accumulators

| Accumulator | Name |
|---|---|
| moved | Moved |

### Stratifications

| Stratification | Categories | Splits |
|---|---|---|
| group | g1, g2 | every compartment |
| tier | t1, t2 | where group = g1 |

### Compartments

| Compartment | Bin | Categories |
|---|---|---|
| A_g1_t1 | A | group=g1, tier=t1 |
| A_g1_t2 | A | group=g1, tier=t2 |
| A_g2 | A | group=g2 |
| B_g1_t1 | B | group=g1, tier=t1 |
| B_g1_t2 | B | group=g1, tier=t2 |
| B_g2 | B | group=g2 |

## Dynamics

### States and transitions

```mermaid
flowchart LR
    n0["State #quot;A#quot; \ start"]
    n1["State B"]
    n0 -->|"ab: s * k * neighbors_B / neighbors"| n1
```

### Expanded compartments

The graph of the expanded compartments has 6 nodes and 3 edges; it is not drawn here. `description.save("model.dot")` writes it as Graphviz DOT.

### Compact form

**Bin Transitions**

`ab (A -> B)`

$$
\begin{aligned}
A_{g1,t1} \to B_{g1,t1} &: \frac{s \cdot k \cdot ((B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot B_{g2})}{((A_{g1,t1} + A_{g1,t2} + B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot (A_{g2} + B_{g2})) \cdot A_{g1,t1}} \\
A_{g1,t2} \to B_{g1,t2} &: \frac{s \cdot k \cdot ((B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot B_{g2})}{((A_{g1,t1} + A_{g1,t2} + B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot (A_{g2} + B_{g2})) \cdot A_{g1,t2}} \\
A_{g2} \to B_{g2} &: \frac{s \cdot k \cdot (0.5 \cdot (B_{g1,t1} + B_{g1,t2}) + B_{g2})}{(0.5 \cdot (A_{g1,t1} + A_{g1,t2} + B_{g1,t1} + B_{g1,t2}) + (A_{g2} + B_{g2})) \cdot A_{g2}}
\end{aligned}
$$

Total System: 6 coupled equations (2 bins × 2 group × 2 tier)

### Expanded form

$$
\begin{aligned}
\frac{A_{g1,t1}(t+\Delta t) - A_{g1,t1}(t)}{\Delta t} &= - (\frac{s \cdot k \cdot ((B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot B_{g2})}{((A_{g1,t1} + A_{g1,t2} + B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot (A_{g2} + B_{g2})) \cdot A_{g1,t1}}) \\
\frac{A_{g1,t2}(t+\Delta t) - A_{g1,t2}(t)}{\Delta t} &= - (\frac{s \cdot k \cdot ((B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot B_{g2})}{((A_{g1,t1} + A_{g1,t2} + B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot (A_{g2} + B_{g2})) \cdot A_{g1,t2}}) \\
\frac{A_{g2}(t+\Delta t) - A_{g2}(t)}{\Delta t} &= - (\frac{s \cdot k \cdot (0.5 \cdot (B_{g1,t1} + B_{g1,t2}) + B_{g2})}{(0.5 \cdot (A_{g1,t1} + A_{g1,t2} + B_{g1,t1} + B_{g1,t2}) + (A_{g2} + B_{g2})) \cdot A_{g2}}) \\
\frac{B_{g1,t1}(t+\Delta t) - B_{g1,t1}(t)}{\Delta t} &= (\frac{s \cdot k \cdot ((B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot B_{g2})}{((A_{g1,t1} + A_{g1,t2} + B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot (A_{g2} + B_{g2})) \cdot A_{g1,t1}}) \\
\frac{B_{g1,t2}(t+\Delta t) - B_{g1,t2}(t)}{\Delta t} &= (\frac{s \cdot k \cdot ((B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot B_{g2})}{((A_{g1,t1} + A_{g1,t2} + B_{g1,t1} + B_{g1,t2}) + 0.5 \cdot (A_{g2} + B_{g2})) \cdot A_{g1,t2}}) \\
\frac{B_{g2}(t+\Delta t) - B_{g2}(t)}{\Delta t} &= (\frac{s \cdot k \cdot (0.5 \cdot (B_{g1,t1} + B_{g1,t2}) + B_{g2})}{(0.5 \cdot (A_{g1,t1} + A_{g1,t2} + B_{g1,t1} + B_{g1,t2}) + (A_{g2} + B_{g2})) \cdot A_{g2}})
\end{aligned}
$$

### Neighbors

Units are neighbors by the categories of 'group' (Groups): in a compartment of category g, neighbors_<bin> is the sum over g' of w[g][g'] times the amount of the bin in g', and neighbors the same sum over every bin. The weights w[g][g'] are:

|  | g1 | g2 |
|---|---|---|
| g1 | 1 | 0.5 |
| g2 | 0.5 | 1 |

## Schedule

Each step applies every flow once, computed from the state at the start of the step: X(n + 1) = X(n) + sum of the flows. The impulses of a step act first, and its flows run from the new state.

## Initialization

### Bin amounts

| Bin | Amount | Unit |
|---|---|---|
| A | set by calibration | item |
| B | 400 | item |

### Stratification fractions

| Stratification | Category | Fraction | Of |
|---|---|---|---|
| group | g1 | 0.25 | each bin |
| group | g2 | 0.75 | each bin |
| tier | t1 | 0.5 | g1 |
| tier | t2 | 0.5 | g1 |

### Compartment overrides

| Compartment | Amount | Unit |
|---|---|---|
| B_g2 | 200 | item |

### Initial state

| Compartment | Amount | Unit |
|---|---|---|
| A_g1_t1 | 0 | item |
| A_g1_t2 | 0 | item |
| A_g2 | 0 | item |
| B_g1_t1 | 50 | item |
| B_g1_t2 | 50 | item |
| B_g2 | 200 | item |

## Parameters

| Parameter | Value | Unit | Set by calibration | Description |
|---|---|---|---|---|
| k | 0.1 |  | no | rate \| per step |
| s |  |  | yes |  |

## Outputs

| Source | Outputs |
|---|---|
| A | A_g1_t1, A_g1_t2, A_g2 |
| B | B_g1_t1, B_g1_t2, B_g2 |
| moved | moved_g1_t1, moved_g1_t2, moved_g2 |
