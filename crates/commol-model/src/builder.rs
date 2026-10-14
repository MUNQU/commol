//! Expansion of a model and its compilation into a [`CompiledModel`].

use crate::flows::{PlannedFlow, plan_transition_flows};
use crate::helpers::rate_to_string;
use crate::labels::OutputLabels;
use crate::layout::CompartmentLayout;
use crate::types::{
    AccumulatorOutputRef, CompiledModel, FastRateExpression, FastRateOp,
    GeneratedAccumulatorOutput, ImpulseFlow, ImpulseSchedule, SeriesParameter, SubgroupTotalLayout,
    SubgroupTotalMapping, TransitionFlow, VarSlot,
};
use commol_core::math_expression::jit::ast::{BinaryOperator, Expr, UnaryOperator};
use commol_core::math_expression::jit::parser::parse_expression;
use commol_core::math_expression::preprocessing::{
    SPECIAL_VAR_E, SPECIAL_VAR_PI, SPECIAL_VAR_STEP, SPECIAL_VAR_T, SPECIAL_VAR_TOTAL,
    SUBGROUP_TOTAL_PREFIX,
};
use commol_core::{
    CompartmentalStructure, MathExpression, MathExpressionContext, Model, RateMathExpression,
};
use std::collections::{HashMap, HashSet};

/// Series parameter payload collected before parameter slots are
/// reserved, so the final [`SeriesParameter`] can be constructed with the
/// correct index up front.
struct SeriesPayload {
    name: String,
    data: Vec<(f64, f64)>,
    mode: commol_core::SeriesMode,
}

impl CompiledModel {
    /// Compile a model, whatever its engine.
    ///
    /// This constructor performs several pre-computation steps for efficiency:
    /// 1. Generates all stratified compartment combinations
    /// 2. Distributes the total across the compartments
    /// 3. Pre-computes transition flows with parsed rate expressions
    /// 4. Pre-computes the subgroup total mappings for stratifications
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not compartmental, when a name of
    /// the model conflicts with a name the engine provides ([`check_names`]),
    /// when a transition cannot be planned, or when an initial-condition
    /// override is not valid for the expansion.
    pub fn try_from_model(model: &Model) -> Result<Self, String> {
        let structure = model.compartmental()?;
        check_names(model)?;

        // Generate all compartment and accumulator output combinations
        let layout = CompartmentLayout::new(structure);
        let compartments: Vec<String> = layout.names().to_vec();
        let compartment_map: HashMap<String, usize> = compartments
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();
        let accumulator_outputs = generate_accumulators(structure);
        let accumulator_names: Vec<String> = accumulator_outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();
        let accumulator_map: HashMap<AccumulatorOutputRef, usize> = accumulator_outputs
            .iter()
            .enumerate()
            .map(|(idx, output)| (output.output_ref.clone(), idx))
            .collect();
        let output_names: Vec<String> = compartments
            .iter()
            .chain(&accumulator_names)
            .cloned()
            .collect();

        // Initial amount of each compartment, in the same order as
        // `compartments`
        let state = initial_state(structure)?;
        debug_assert!(
            state
                .iter()
                .map(|compartment| &compartment.name)
                .eq(compartments.iter())
        );
        let amounts: Vec<f64> = state.iter().map(|c| c.amount).collect();

        // Labels of every output: compartments first, then accumulators
        let output_labels: Vec<OutputLabels> = layout
            .labels()
            .iter()
            .cloned()
            .chain(
                expand_accumulators(structure)
                    .into_iter()
                    .map(|output| OutputLabels {
                        source: output.source,
                        categories: output.categories,
                    }),
            )
            .collect();

        // Store constant parameters for quick lookup
        // Formula parameters will be evaluated on each step
        let mut constant_parameters: HashMap<String, f64> = HashMap::new();
        // Constant parameters and parameters that calibration sets: their
        // values do not change during a run.
        let mut fixed_parameters: HashSet<String> = HashSet::new();
        let mut formula_parameters: Vec<(String, RateMathExpression)> = Vec::new();
        // Collect series payloads first; final struct construction happens
        // after parameter slots are reserved so indices can be filled in once,
        // up front, instead of being patched in afterwards.
        let mut series_payloads: Vec<SeriesPayload> = Vec::new();

        for p in &model.parameters {
            match &p.value {
                Some(commol_core::ParameterValue::Constant(val)) => {
                    constant_parameters.insert(p.id.clone(), *val);
                    fixed_parameters.insert(p.id.clone());
                }
                Some(commol_core::ParameterValue::Formula(formula)) => {
                    let rate_expr = RateMathExpression::from_string(formula.clone());
                    formula_parameters.push((p.id.clone(), rate_expr));
                }
                Some(commol_core::ParameterValue::Series { data, mode }) => {
                    series_payloads.push(SeriesPayload {
                        name: p.id.clone(),
                        data: data.clone(),
                        mode: mode.clone(),
                    });
                }
                None => {
                    // Parameter needs calibration - skip it for now
                    // During calibration, set_parameter() will provide the value
                    fixed_parameters.insert(p.id.clone());
                }
            }
        }

        // Initialize expression context with constant parameters
        let mut expression_context = MathExpressionContext::new();
        expression_context.set_parameters(constant_parameters);
        expression_context.init_compartments(compartments.clone());

        // Pre-compute the subgroup total mappings for stratifications
        // (must be done before transition flows, as absolute-flow inference
        // needs to know about subgroup total names)
        let subgroup_total_layout = build_subgroup_total_mappings(&layout, structure);

        let subgroup_total_param_names: HashSet<String> = subgroup_total_layout
            .iter()
            .map(|m| m.parameter_name.clone())
            .collect();

        // Pre-allocate every parameter slot the per-step loop will touch so
        // `set_parameter_str` can update in place and JIT input resolution can
        // use direct indexed parameter reads where possible.
        expression_context.reserve_parameters(
            std::iter::once(SPECIAL_VAR_TOTAL.to_string())
                .chain(std::iter::once("t".to_string()))
                .chain(model.parameters.iter().map(|p| p.id.clone()))
                .chain(
                    subgroup_total_layout
                        .iter()
                        .map(|m| m.parameter_name.clone()),
                )
                .chain(series_payloads.iter().map(|payload| payload.name.clone()))
                .chain(formula_parameters.iter().map(|(name, _)| name.clone())),
        );
        let total_parameter_index = expression_context
            .parameter_index(SPECIAL_VAR_TOTAL)
            .expect("total parameter slot must be reserved");
        let t_parameter_index = expression_context
            .parameter_index("t")
            .expect("t parameter slot must be reserved");

        let mut subgroup_total_mappings: Vec<SubgroupTotalMapping> = subgroup_total_layout
            .into_iter()
            .map(|layout| {
                let parameter_index = expression_context
                    .parameter_index(&layout.parameter_name)
                    .expect("subgroup total parameter slot must be reserved");
                SubgroupTotalMapping {
                    contributing_compartment_indices: layout.contributing_compartment_indices,
                    parameter_name: layout.parameter_name,
                    parameter_index,
                }
            })
            .collect();

        let series_parameters: Vec<SeriesParameter> = series_payloads
            .into_iter()
            .map(|payload| {
                let parameter_index = expression_context
                    .parameter_index(&payload.name)
                    .expect("series parameter slot must be reserved");
                SeriesParameter::new(payload.name, parameter_index, payload.data, payload.mode)
            })
            .collect::<Result<_, _>>()?;

        // Plan the transition flows and compile them
        let planned_flows = plan_transition_flows(
            structure,
            &layout,
            &accumulator_map,
            &subgroup_total_param_names,
        )?;
        let schedules = impulse_schedules(structure)?;
        let (impulse_plans, flow_plans): (Vec<PlannedFlow>, Vec<PlannedFlow>) = planned_flows
            .into_iter()
            .partition(|flow| schedules.contains_key(&flow.transition_id));
        let impulse_schedules: Vec<(String, ImpulseSchedule)> = impulse_plans
            .iter()
            .map(|plan| {
                (
                    plan.transition_id.clone(),
                    schedules[&plan.transition_id].clone(),
                )
            })
            .collect();
        let impulse_flows: Vec<ImpulseFlow> = impulse_schedules
            .into_iter()
            .zip(compile_transition_flows(
                impulse_plans,
                &compartment_map,
                &expression_context,
            ))
            .map(|((transition_id, schedule), flow)| ImpulseFlow {
                transition_id,
                flow,
                schedule,
            })
            .collect();
        let transition_flows =
            compile_transition_flows(flow_plans, &compartment_map, &expression_context);

        let used_context_variables = collect_used_context_variables(
            transition_flows
                .iter()
                .chain(impulse_flows.iter().map(|impulse| &impulse.flow)),
            &formula_parameters,
        );
        let depends_on_step = series_parameters.iter().any(SeriesParameter::reads_step)
            || reads_step(
                transition_flows
                    .iter()
                    .chain(impulse_flows.iter().map(|impulse| &impulse.flow)),
                &formula_parameters,
            );
        let time_thresholds = time_thresholds(
            transition_flows
                .iter()
                .chain(impulse_flows.iter().map(|impulse| &impulse.flow)),
            &formula_parameters,
            &fixed_parameters,
        );
        subgroup_total_mappings
            .retain(|mapping| used_context_variables.contains(&mapping.parameter_name));

        // The HashMap-backed compartment context is only needed when an
        // expression must fall back to evalexpr. Both the JIT path (via
        // `resolved_slots`) and the fast path (via `fast_rate_expression`)
        // read compartment values directly from the state, so they do
        // not require the HashMap to be kept up to date.
        let requires_compartment_context = !formula_parameters.is_empty()
            || transition_flows
                .iter()
                .chain(impulse_flows.iter().map(|impulse| &impulse.flow))
                .any(|flow| {
                    matches!(flow.rate_expression, RateMathExpression::Formula(_))
                        && flow.resolved_slots.is_none()
                        && flow.fast_rate_expression.is_none()
                });

        let initial_state: Vec<f64> = amounts
            .into_iter()
            .chain(std::iter::repeat_n(0.0, accumulator_map.len()))
            .collect();

        Ok(Self {
            compartments,
            total: structure.initial_conditions.total,
            output_labels,
            output_names,
            initial_state,
            expression_context,
            transition_flows,
            impulse_flows,
            depends_on_step,
            time_thresholds,
            subgroup_total_mappings,
            formula_parameters,
            series_parameters,
            total_parameter_index,
            t_parameter_index,
            requires_compartment_context,
        })
    }
}

/// The impulse schedule of each impulse transition, by transition id.
///
/// # Errors
///
/// Returns a message that names the transition when it sets both
/// `impulse_at` and `impulse_every`, when a position is not finite or is
/// negative, or when the period is not positive.
fn impulse_schedules(
    structure: &CompartmentalStructure,
) -> Result<HashMap<String, ImpulseSchedule>, String> {
    let mut schedules = HashMap::new();
    for transition in &structure.transitions {
        let schedule = match (&transition.impulse_at, transition.impulse_every) {
            (None, None) => continue,
            (Some(_), Some(_)) => {
                return Err(format!(
                    "Transition '{}' sets both impulse_at and impulse_every; set one",
                    transition.id
                ));
            }
            (Some(positions), None) => {
                if let Some(bad) = positions.iter().find(|p| !p.is_finite() || **p < 0.0) {
                    return Err(format!(
                        "Transition '{}' has the impulse position {bad}; positions must \
                         be finite and not negative",
                        transition.id
                    ));
                }
                let mut positions = positions.clone();
                positions.sort_by(f64::total_cmp);
                positions.dedup();
                ImpulseSchedule::At(positions)
            }
            (None, Some((period, offset))) => {
                if !(period.is_finite() && period > 0.0) {
                    return Err(format!(
                        "Transition '{}' has the impulse period {period}; it must be \
                         positive",
                        transition.id
                    ));
                }
                if !(offset.is_finite() && offset >= 0.0) {
                    return Err(format!(
                        "Transition '{}' has the impulse offset {offset}; it must be \
                         finite and not negative",
                        transition.id
                    ));
                }
                ImpulseSchedule::Every { period, offset }
            }
        };
        schedules.insert(transition.id.clone(), schedule);
    }
    Ok(schedules)
}

/// Check whether a stratification's conditions are all satisfied by the
/// already-applied categories of the current compartment being built.
fn stratification_conditions_met(
    conditions: &Option<Vec<commol_core::StratificationCondition>>,
    applied: &HashMap<String, String>,
) -> bool {
    match conditions {
        None => true,
        Some(conds) => conds.iter().all(|c| match &c.category {
            Some(cat) => applied.get(&c.stratification) == Some(cat),
            None => true,
        }),
    }
}

/// A bin or accumulator expanded over the stratifications.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpandedOutput {
    /// Expanded name, for example `IH_y60` or `S_oe60_home`.
    pub name: String,
    /// Bin or accumulator the output expands from.
    pub source: String,
    /// `(stratification id, category)` pairs applied to the output, in
    /// stratification declaration order. A conditional stratification whose
    /// conditions the output does not meet is absent.
    pub categories: Vec<(String, String)>,
}

/// Expand source ids over the stratifications, in declaration order.
///
/// This is the single source of truth for the expansion of bins and
/// accumulators. When a stratification has `conditions`, it only expands the
/// outputs whose already-applied categories satisfy all of them; the others are
/// kept as-is, so the result is a non-uniform Cartesian product.
fn expand<'a>(
    structure: &CompartmentalStructure,
    source_ids: impl IntoIterator<Item = &'a str>,
) -> Vec<(ExpandedOutput, HashMap<String, String>)> {
    // Each element: (source, name, applied categories by stratification id)
    let mut partials: Vec<(String, String, HashMap<String, String>)> = source_ids
        .into_iter()
        .map(|id| (id.to_string(), id.to_string(), HashMap::new()))
        .collect();

    for stratification in &structure.stratifications {
        let mut new_partials = Vec::with_capacity(partials.len());
        for (source, name, applied) in partials {
            if stratification_conditions_met(&stratification.conditions, &applied) {
                for cat in &stratification.categories {
                    let mut new_applied = applied.clone();
                    new_applied.insert(stratification.id.clone(), cat.clone());
                    new_partials.push((source.clone(), format!("{}_{}", name, cat), new_applied));
                }
            } else {
                new_partials.push((source, name, applied));
            }
        }
        partials = new_partials;
    }

    partials
        .into_iter()
        .map(|(source, name, applied)| {
            let categories = structure
                .stratifications
                .iter()
                .filter_map(|s| applied.get(&s.id).map(|c| (s.id.clone(), c.clone())))
                .collect();
            (
                ExpandedOutput {
                    name,
                    source,
                    categories,
                },
                applied,
            )
        })
        .collect()
}

/// Expanded compartments of a structure, in engine order.
pub fn expand_compartments(structure: &CompartmentalStructure) -> Vec<ExpandedOutput> {
    expand(structure, structure.bins.iter().map(|b| b.id.as_str()))
        .into_iter()
        .map(|(output, _)| output)
        .collect()
}

/// Expanded accumulator outputs of a structure, in engine order.
pub fn expand_accumulators(structure: &CompartmentalStructure) -> Vec<ExpandedOutput> {
    expand(
        structure,
        structure.accumulators.iter().map(|a| a.id.as_str()),
    )
    .into_iter()
    .map(|(output, _)| output)
    .collect()
}

pub(crate) fn accumulator_output_ref(
    accumulator_id: String,
    applied: &HashMap<String, String>,
    stratifications: &[commol_core::Stratification],
) -> AccumulatorOutputRef {
    AccumulatorOutputRef {
        accumulator_id,
        categories: stratifications
            .iter()
            .map(|stratification| applied.get(&stratification.id).cloned())
            .collect(),
    }
}

pub(crate) fn generate_accumulators(
    structure: &CompartmentalStructure,
) -> Vec<GeneratedAccumulatorOutput> {
    expand(
        structure,
        structure.accumulators.iter().map(|a| a.id.as_str()),
    )
    .into_iter()
    .map(|(output, applied)| GeneratedAccumulatorOutput {
        name: output.name,
        output_ref: accumulator_output_ref(output.source, &applied, &structure.stratifications),
    })
    .collect()
}

/// One expanded compartment of a model and the amount it starts with.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialCompartment {
    /// Expanded compartment name, for example `IH_y60` or `S_oe60_home`.
    pub name: String,
    /// Bin the compartment expands from.
    pub bin: String,
    /// `(stratification id, category)` pairs applied to the compartment, in
    /// stratification declaration order. A conditional stratification whose
    /// conditions the compartment does not meet is absent.
    pub categories: Vec<(String, String)>,
    /// Initial amount, including `compartment_fractions` overrides.
    pub amount: f64,
}

/// Relative tolerance of the check that overrides keep the sum of the initial
/// amounts within the total.
const TOTAL_TOLERANCE: f64 = 1e-9;

/// Expand a structure into its compartments and their initial amounts.
///
/// This is the single source of truth for the initial state of a
/// compartmental model. Bins
/// are expanded over the stratifications in declaration order; when a
/// stratification has conditions that a compartment's already-applied
/// categories do not meet, that compartment is kept as-is and its amount
/// is not split by that stratification's fractions. An uncalibrated bin
/// fraction counts as zero.
///
/// After the uniform distribution is computed, `compartment_fractions` overrides
/// replace the value of the named compartments with `total * fraction`.
///
/// # Errors
///
/// Returns a message when the total is not finite or not greater than 0, or
/// when an override names a compartment that is not part of the expansion,
/// names a compartment twice, holds a fraction outside `[0.0, 1.0]`, or makes
/// the sum of the initial amounts exceed the total.
pub fn initial_state(
    structure: &CompartmentalStructure,
) -> Result<Vec<InitialCompartment>, String> {
    let initial_conditions = &structure.initial_conditions;
    let total = initial_conditions.total;
    if !total.is_finite() || total <= 0.0 {
        return Err(format!(
            "The total of the initial conditions must be finite and greater than 0, got {total}."
        ));
    }

    let bin_fraction_map: HashMap<&str, Option<f64>> = initial_conditions
        .bin_fractions
        .iter()
        .map(|bf| (bf.bin.as_str(), bf.fraction))
        .collect();
    let category_fraction_map: HashMap<(&str, &str), f64> = structure
        .stratifications
        .iter()
        .filter_map(|stratification| {
            initial_conditions
                .stratification_fractions
                .iter()
                .find(|sf| sf.stratification == stratification.id)
        })
        .flat_map(|item| {
            item.fractions.iter().map(|fraction| {
                (
                    (item.stratification.as_str(), fraction.category.as_str()),
                    fraction.fraction,
                )
            })
        })
        .collect();

    // Each compartment holds its bin's share of the total, split by the
    // fraction of every category applied to it. An uncalibrated bin or an
    // undeclared category counts as zero.
    let mut state: Vec<InitialCompartment> = expand_compartments(structure)
        .into_iter()
        .map(|output| {
            let bin_fraction = bin_fraction_map
                .get(output.source.as_str())
                .and_then(|f| *f)
                .unwrap_or(0.0);
            let amount =
                output
                    .categories
                    .iter()
                    .fold(total * bin_fraction, |amount, (id, cat)| {
                        amount
                            * category_fraction_map
                                .get(&(id.as_str(), cat.as_str()))
                                .copied()
                                .unwrap_or(0.0)
                    });
            InitialCompartment {
                name: output.name,
                bin: output.source,
                categories: output.categories,
                amount,
            }
        })
        .collect();

    let overrides = &initial_conditions.compartment_fractions;
    if overrides.is_empty() {
        return Ok(state);
    }

    let compartment_index: HashMap<String, usize> = state
        .iter()
        .enumerate()
        .map(|(index, compartment)| (compartment.name.clone(), index))
        .collect();

    // Validate every override before applying any of them, so a rejected
    // configuration names the first offending entry.
    let mut seen_compartments: HashSet<&str> = HashSet::new();
    for compartment_fraction in overrides {
        if !compartment_index.contains_key(&compartment_fraction.compartment) {
            let available: Vec<&str> = state.iter().map(|c| c.name.as_str()).collect();
            return Err(format!(
                "Unknown compartment '{}' in initial conditions. Available compartments: {:?}",
                compartment_fraction.compartment, available
            ));
        }
        if !seen_compartments.insert(compartment_fraction.compartment.as_str()) {
            return Err(format!(
                "Duplicate compartment '{}' in initial conditions.",
                compartment_fraction.compartment
            ));
        }
        if !(0.0..=1.0).contains(&compartment_fraction.fraction) {
            return Err(format!(
                "Fraction for compartment '{}' must be between 0.0 and 1.0, got {}.",
                compartment_fraction.compartment, compartment_fraction.fraction
            ));
        }
    }

    for compartment_fraction in overrides {
        let index = compartment_index[&compartment_fraction.compartment];
        state[index].amount = total * compartment_fraction.fraction;
    }

    let sum: f64 = state.iter().map(|c| c.amount).sum();
    if sum > total * (1.0 + TOTAL_TOLERANCE) {
        return Err(format!(
            "Per-compartment overrides would make the initial amounts exceed \
             the total ({sum} > {total})."
        ));
    }

    Ok(state)
}

/// Compile planned transition flows into runtime flows.
fn compile_transition_flows(
    planned_flows: Vec<PlannedFlow>,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Vec<TransitionFlow> {
    planned_flows
        .into_iter()
        .map(|flow| {
            let rate_expression = RateMathExpression::from_string(flow.rate);
            let resolved_slots =
                resolve_jit_slots(&rate_expression, compartment_map, expression_context);
            let fast_rate_expression =
                build_fast_rate_expression(&rate_expression, compartment_map, expression_context);
            let (source_index, target_index) = flow.pieces[0];
            TransitionFlow {
                source_index,
                target_index,
                extra_pieces: flow.pieces[1..].to_vec(),
                accumulator_indices: flow.accumulator_indices,
                rate_expression,
                fast_rate_expression,
                is_absolute_flow: flow.absolute,
                resolved_slots,
            }
        })
        .collect()
}

/// Build pre-computed subgroup total mappings for stratifications.
///
/// This function creates mappings for:
/// 1. Subgroup totals over categories (`total_young`, `total_young_urban`, ...)
/// 2. Bin totals (`S`, `I`, `R`): the sum of all stratified versions
/// 3. Partial bin totals (`S_young` when a second stratification splits it)
///
/// Every mapping is built from the compartment labels.
fn build_subgroup_total_mappings(
    layout: &CompartmentLayout,
    structure: &CompartmentalStructure,
) -> Vec<SubgroupTotalLayout> {
    if structure.stratifications.is_empty() {
        return Vec::new();
    }

    let mut totals: HashMap<String, Vec<usize>> = HashMap::new();
    let partial_bin_totals = structure.stratifications.len() >= 2;

    for (compartment_index, label) in layout.labels().iter().enumerate() {
        let categories: Vec<&str> = label
            .categories
            .iter()
            .map(|(_, category)| category.as_str())
            .collect();
        if categories.is_empty() {
            continue;
        }

        // Bin total
        totals
            .entry(label.source.clone())
            .or_default()
            .push(compartment_index);

        let full_mask: u32 = (1 << categories.len()) - 1;
        for subset_mask in 1..=full_mask {
            let subset: Vec<&str> = categories
                .iter()
                .enumerate()
                .filter(|(k, _)| (subset_mask >> k) & 1 == 1)
                .map(|(_, category)| *category)
                .collect();
            let combination = subset.join("_");

            // Subgroup total over every bin
            totals
                .entry(format!("{SUBGROUP_TOTAL_PREFIX}{combination}"))
                .or_default()
                .push(compartment_index);

            // Partial bin total. With one stratification a partial sum would
            // duplicate either the compartment or the bin total.
            if partial_bin_totals && subset_mask != full_mask {
                totals
                    .entry(format!("{}_{}", label.source, combination))
                    .or_default()
                    .push(compartment_index);
            }
        }
    }

    totals
        .into_iter()
        .map(
            |(parameter_name, contributing_compartment_indices)| SubgroupTotalLayout {
                contributing_compartment_indices,
                parameter_name,
            },
        )
        .collect()
}

/// Names of the subgroup totals a rate expression may reference.
pub(crate) fn subgroup_total_name_set(
    layout: &CompartmentLayout,
    structure: &CompartmentalStructure,
) -> HashSet<String> {
    build_subgroup_total_mappings(layout, structure)
        .into_iter()
        .map(|mapping| mapping.parameter_name)
        .collect()
}

/// Names of the subgroup totals a rate expression of a structure may read,
/// sorted.
///
/// They are the `total_<categories>` names, the bin totals and the partial bin
/// totals. Category suffixes follow the declaration order of the
/// stratifications, and a conditional stratification adds its categories only
/// to the compartments it expands. A structure without stratifications has
/// none.
pub fn subgroup_total_names(structure: &CompartmentalStructure) -> Vec<String> {
    let mut names: Vec<String> =
        subgroup_total_name_set(&CompartmentLayout::new(structure), structure)
            .into_iter()
            .collect();
    names.sort_unstable();
    names
}

/// Check the names a model uses against the names the engine provides.
///
/// The engine provides one variable per expanded compartment and one per
/// subgroup total: `total_<categories>`, bin totals and partial bin totals.
/// This function returns an error when:
///
/// - a parameter id is one of those names, because the parameter and the
///   engine variable would share one slot;
/// - a transition rate or a formula parameter uses a `total_…` name that is
///   not a subgroup total of the model. Category suffixes follow the
///   declaration order of the stratifications, so `total_t1_g1` is not a name
///   when `group` is declared before `type`. A parameter or compartment whose
///   id starts with `total_` is not such a name.
///
/// # Errors
///
/// Returns a message when the model is not compartmental, or a message that
/// names each conflict and lists the valid subgroup total names.
pub fn check_names(model: &Model) -> Result<(), String> {
    let structure = model.compartmental()?;
    let layout = CompartmentLayout::new(structure);
    let compartments: HashSet<&str> = layout.names().iter().map(String::as_str).collect();
    let subgroup_totals = subgroup_total_name_set(&layout, structure);
    let parameter_ids: HashSet<&str> = model.parameters.iter().map(|p| p.id.as_str()).collect();

    let mut reserved: Vec<String> = model
        .parameters
        .iter()
        .filter_map(|parameter| {
            let id = parameter.id.as_str();
            if compartments.contains(id) {
                Some(format!("'{id}' (a compartment)"))
            } else if subgroup_totals.contains(id) {
                Some(format!("'{id}' (a subgroup total)"))
            } else {
                None
            }
        })
        .collect();
    if !reserved.is_empty() {
        reserved.sort();
        return Err(format!(
            "Parameter ids cannot be names the engine provides: {}. Rename the \
             parameter(s).",
            reserved.join(", ")
        ));
    }

    let mut used: HashSet<String> = HashSet::new();
    for transition in &structure.transitions {
        if let Some(rate) = &transition.rate {
            // A deserialized rate is a `Parameter` until it is parsed again.
            used.extend(RateMathExpression::from_string(rate_to_string(rate)).get_variables());
        }
        for stratified in transition.stratified_rates.iter().flatten() {
            used.extend(RateMathExpression::from_string(stratified.rate.clone()).get_variables());
        }
    }
    for parameter in &model.parameters {
        if let Some(commol_core::ParameterValue::Formula(formula)) = &parameter.value {
            used.extend(RateMathExpression::from_string(formula.clone()).get_variables());
        }
    }

    let mut unknown: Vec<&String> = used
        .iter()
        .filter(|name| name.starts_with(SUBGROUP_TOTAL_PREFIX))
        .filter(|name| !subgroup_totals.contains(*name))
        .filter(|name| !compartments.contains(name.as_str()))
        .filter(|name| !parameter_ids.contains(name.as_str()))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort();
    let mut valid: Vec<&str> = subgroup_totals
        .iter()
        .map(String::as_str)
        .filter(|name| name.starts_with(SUBGROUP_TOTAL_PREFIX))
        .collect();
    valid.sort_unstable();
    let valid = if valid.is_empty() {
        "none (the model has no stratifications)".to_string()
    } else {
        valid.join(", ")
    };
    Err(format!(
        "Unknown subgroup total(s) {}. Category suffixes follow the \
         declaration order of the stratifications. Valid names: {valid}",
        unknown
            .iter()
            .map(|name| format!("'{name}'"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Whether a rate or a formula parameter reads `step`.
fn reads_step<'a>(
    transition_flows: impl IntoIterator<Item = &'a TransitionFlow>,
    formula_parameters: &[(String, RateMathExpression)],
) -> bool {
    transition_flows
        .into_iter()
        .map(|flow| &flow.rate_expression)
        .chain(formula_parameters.iter().map(|(_, expression)| expression))
        .any(|expression| expression.reads_variable(SPECIAL_VAR_STEP))
}

/// The values that a rate or a formula parameter compares `t` with (`t < c`,
/// `c <= t`, ...), when `c` reads only numbers and `fixed_parameters`.
///
/// A rate that reads `t` can jump where `t` reaches such a value; a
/// differential engine restarts its integration there, so it cannot step over
/// a short window. A comparison of `t` with the state is not a fixed position
/// and is not collected.
fn time_thresholds<'a>(
    transition_flows: impl IntoIterator<Item = &'a TransitionFlow>,
    formula_parameters: &[(String, RateMathExpression)],
    fixed_parameters: &HashSet<String>,
) -> Vec<MathExpression> {
    let mut thresholds = Vec::new();
    let expressions = transition_flows
        .into_iter()
        .map(|flow| &flow.rate_expression)
        .chain(formula_parameters.iter().map(|(_, expression)| expression));
    for expression in expressions {
        if let RateMathExpression::Formula(formula) = expression
            && let Ok(ast) = parse_expression(formula.preprocessed())
        {
            collect_time_thresholds(&ast, fixed_parameters, &mut thresholds);
        }
    }
    let mut seen = HashSet::new();
    thresholds
        .into_iter()
        .map(|threshold| threshold.to_string())
        .filter(|threshold| seen.insert(threshold.clone()))
        .map(MathExpression::new)
        .collect()
}

fn collect_time_thresholds(expr: &Expr, fixed_parameters: &HashSet<String>, out: &mut Vec<Expr>) {
    match expr {
        Expr::Constant(_) | Expr::Variable(_) => {}
        Expr::BinaryOp { op, left, right } => {
            if matches!(
                op,
                BinaryOperator::Lt | BinaryOperator::Gt | BinaryOperator::Le | BinaryOperator::Ge
            ) {
                let is_t =
                    |side: &Expr| matches!(side, Expr::Variable(name) if name == SPECIAL_VAR_T);
                if is_t(left) && is_fixed(right, fixed_parameters) {
                    out.push((**right).clone());
                } else if is_t(right) && is_fixed(left, fixed_parameters) {
                    out.push((**left).clone());
                }
            }
            collect_time_thresholds(left, fixed_parameters, out);
            collect_time_thresholds(right, fixed_parameters, out);
        }
        Expr::UnaryOp { operand, .. } => collect_time_thresholds(operand, fixed_parameters, out),
        Expr::FunctionCall { args, .. } => {
            for arg in args {
                collect_time_thresholds(arg, fixed_parameters, out);
            }
        }
        Expr::Conditional {
            condition,
            true_expr,
            false_expr,
        } => {
            for part in [condition, true_expr, false_expr] {
                collect_time_thresholds(part, fixed_parameters, out);
            }
        }
    }
}

/// Whether an expression reads only numbers, `pi`, `e` and `fixed_parameters`.
fn is_fixed(expr: &Expr, fixed_parameters: &HashSet<String>) -> bool {
    match expr {
        Expr::Constant(_) => true,
        Expr::Variable(name) => {
            name == SPECIAL_VAR_PI || name == SPECIAL_VAR_E || fixed_parameters.contains(name)
        }
        Expr::BinaryOp { left, right, .. } => {
            is_fixed(left, fixed_parameters) && is_fixed(right, fixed_parameters)
        }
        Expr::UnaryOp { operand, .. } => is_fixed(operand, fixed_parameters),
        Expr::FunctionCall { args, .. } => args.iter().all(|arg| is_fixed(arg, fixed_parameters)),
        Expr::Conditional {
            condition,
            true_expr,
            false_expr,
        } => [condition, true_expr, false_expr]
            .into_iter()
            .all(|part| is_fixed(part, fixed_parameters)),
    }
}

fn collect_used_context_variables<'a>(
    transition_flows: impl IntoIterator<Item = &'a TransitionFlow>,
    formula_parameters: &[(String, RateMathExpression)],
) -> HashSet<String> {
    let mut variables = HashSet::new();

    for flow in transition_flows {
        variables.extend(flow.rate_expression.get_variables());
    }

    for (_, expression) in formula_parameters {
        variables.extend(expression.get_variables());
    }

    variables
}

/// Resolve the input variables of a JIT-compiled rate expression to direct
/// storage slots (compartment index, parameter name, or step alias) so the
/// per-step inner loop can fill the JIT input buffer without HashMap probes
/// for compartments. Returns `None` for non-JIT rates (constants, single
/// parameter references, or evalexpr fallbacks) — those have a fast path
/// of their own.
fn resolve_jit_slots(
    rate_expression: &RateMathExpression,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<Vec<VarSlot>> {
    let expr = match rate_expression {
        RateMathExpression::Formula(expr) => expr,
        _ => return None,
    };
    let names = expr.jit_variable_names()?;
    Some(
        names
            .iter()
            .map(|name| resolve_var_slot(name, compartment_map, expression_context))
            .collect(),
    )
}

fn build_fast_rate_expression(
    rate_expression: &RateMathExpression,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<FastRateExpression> {
    let mut ops = Vec::new();
    match rate_expression {
        RateMathExpression::Constant(value) => return Some(FastRateExpression::Constant(*value)),
        RateMathExpression::Parameter(name) => {
            return Some(FastRateExpression::Slot(resolve_var_slot(
                name,
                compartment_map,
                expression_context,
            )));
        }
        RateMathExpression::Formula(expr) => {
            let ast = parse_expression(expr.preprocessed()).ok()?;
            if let Some(specialized) =
                specialized_fast_expr(&ast, compartment_map, expression_context)
            {
                return Some(specialized);
            }
            push_fast_ops_from_ast(&ast, compartment_map, expression_context, &mut ops)?;
        }
    }
    Some(FastRateExpression::Program(ops))
}

fn specialized_fast_expr(
    expr: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<FastRateExpression> {
    match expr {
        Expr::BinaryOp {
            op: BinaryOperator::Mul,
            left,
            right,
        } => {
            if let Some((a, b, c)) = match_mul3(left, right, compartment_map, expression_context) {
                return Some(FastRateExpression::Mul3(a, b, c));
            }
            if let Some((subtract, factor, value)) =
                match_one_minus_mul3(left, right, compartment_map, expression_context)
            {
                return Some(FastRateExpression::OneMinusMul3 {
                    subtract,
                    factor,
                    value,
                });
            }
            let left = slot_from_ast(left, compartment_map, expression_context)?;
            let right = slot_from_ast(right, compartment_map, expression_context)?;
            Some(FastRateExpression::Mul2(left, right))
        }
        Expr::BinaryOp {
            op: BinaryOperator::Div,
            left,
            right,
        } => match_mul_add_div(left, right, compartment_map, expression_context).map(
            |(first, second, add_left, add_right, denominator)| FastRateExpression::MulAddDiv {
                first,
                second,
                add_left,
                add_right,
                denominator,
            },
        ),
        Expr::Variable(name) => Some(FastRateExpression::Slot(resolve_var_slot(
            name,
            compartment_map,
            expression_context,
        ))),
        Expr::Constant(value) => Some(FastRateExpression::Constant(*value)),
        _ => None,
    }
}

fn match_mul3(
    left: &Expr,
    right: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<(VarSlot, VarSlot, VarSlot)> {
    if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = left
    {
        return Some((
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
            slot_from_ast(right, compartment_map, expression_context)?,
        ));
    }

    if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = right
    {
        return Some((
            slot_from_ast(left, compartment_map, expression_context)?,
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
        ));
    }

    None
}

fn match_one_minus_mul3(
    left: &Expr,
    right: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<(VarSlot, VarSlot, VarSlot)> {
    if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = left
    {
        if let Some(subtract) = match_one_minus(inner_left, compartment_map, expression_context) {
            return Some((
                subtract,
                slot_from_ast(inner_right, compartment_map, expression_context)?,
                slot_from_ast(right, compartment_map, expression_context)?,
            ));
        }
        if let Some(subtract) = match_one_minus(inner_right, compartment_map, expression_context) {
            return Some((
                subtract,
                slot_from_ast(inner_left, compartment_map, expression_context)?,
                slot_from_ast(right, compartment_map, expression_context)?,
            ));
        }
    }

    if let Some(subtract) = match_one_minus(left, compartment_map, expression_context)
        && let Expr::BinaryOp {
            op: BinaryOperator::Mul,
            left: inner_left,
            right: inner_right,
        } = right
    {
        return Some((
            subtract,
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
        ));
    }

    None
}

fn match_one_minus(
    expr: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<VarSlot> {
    if let Expr::BinaryOp {
        op: BinaryOperator::Sub,
        left,
        right,
    } = expr
        && matches!(left.as_ref(), Expr::Constant(value) if *value == 1.0)
    {
        return slot_from_ast(right, compartment_map, expression_context);
    }
    None
}

fn match_mul_add_div(
    numerator: &Expr,
    denominator: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<(VarSlot, VarSlot, VarSlot, VarSlot, VarSlot)> {
    let denominator = slot_from_ast(denominator, compartment_map, expression_context)?;
    let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left,
        right,
    } = numerator
    else {
        return None;
    };

    let (first, second, add_expr) = if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = left.as_ref()
    {
        (
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
            right.as_ref(),
        )
    } else if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = right.as_ref()
    {
        (
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
            left.as_ref(),
        )
    } else {
        return None;
    };

    let Expr::BinaryOp {
        op: BinaryOperator::Add,
        left: add_left,
        right: add_right,
    } = add_expr
    else {
        return None;
    };

    Some((
        first,
        second,
        slot_from_ast(add_left, compartment_map, expression_context)?,
        slot_from_ast(add_right, compartment_map, expression_context)?,
        denominator,
    ))
}

fn slot_from_ast(
    expr: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<VarSlot> {
    match expr {
        Expr::Variable(name) => Some(resolve_var_slot(name, compartment_map, expression_context)),
        _ => None,
    }
}

fn push_fast_ops_from_ast(
    expr: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
    ops: &mut Vec<FastRateOp>,
) -> Option<()> {
    match expr {
        Expr::Constant(value) => ops.push(FastRateOp::Constant(*value)),
        Expr::Variable(name) => ops.push(FastRateOp::Slot(resolve_var_slot(
            name,
            compartment_map,
            expression_context,
        ))),
        Expr::UnaryOp { op, operand } => match op {
            UnaryOperator::Neg => {
                push_fast_ops_from_ast(operand, compartment_map, expression_context, ops)?;
                ops.push(FastRateOp::Neg);
            }
            UnaryOperator::Not => return None,
        },
        Expr::BinaryOp { op, left, right } => {
            push_fast_ops_from_ast(left, compartment_map, expression_context, ops)?;
            push_fast_ops_from_ast(right, compartment_map, expression_context, ops)?;
            match op {
                BinaryOperator::Add => ops.push(FastRateOp::Add),
                BinaryOperator::Sub => ops.push(FastRateOp::Sub),
                BinaryOperator::Mul => ops.push(FastRateOp::Mul),
                BinaryOperator::Div => ops.push(FastRateOp::Div),
                BinaryOperator::Mod
                | BinaryOperator::Pow
                | BinaryOperator::Lt
                | BinaryOperator::Gt
                | BinaryOperator::Le
                | BinaryOperator::Ge
                | BinaryOperator::Eq
                | BinaryOperator::Ne
                | BinaryOperator::And
                | BinaryOperator::Or => return None,
            }
        }
        Expr::FunctionCall { .. } | Expr::Conditional { .. } => return None,
    }
    Some(())
}

fn resolve_var_slot(
    name: &str,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> VarSlot {
    if let Some(&idx) = compartment_map.get(name) {
        VarSlot::Compartment(idx)
    } else if name == "step" {
        VarSlot::Step
    } else if name == "t" {
        VarSlot::Time
    } else if name == SPECIAL_VAR_PI {
        VarSlot::Constant(std::f64::consts::PI)
    } else if name == SPECIAL_VAR_E {
        VarSlot::Constant(std::f64::consts::E)
    } else if let Some(idx) = expression_context.parameter_index(name) {
        VarSlot::ParameterIndex(idx)
    } else {
        VarSlot::Parameter(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate all stratified compartment combinations.
    ///
    /// Returns a tuple of (compartments vector, compartment_map for lookups).
    fn generate_compartments(model: &Model) -> (Vec<String>, HashMap<String, usize>) {
        let compartments: Vec<String> = expand_compartments(model.compartmental().unwrap())
            .into_iter()
            .map(|output| output.name)
            .collect();

        let compartment_map: HashMap<String, usize> = compartments
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();

        (compartments, compartment_map)
    }
    use commol_core::{
        Bin, BinFraction, CompartmentFraction, EngineType, InitialConditions, ModelEngine,
        ModelStructure, Stratification, StratificationCondition, StratificationFraction,
        StratificationFractions,
    };

    /// The compartmental structure of a test model.
    fn structure(model: &Model) -> &CompartmentalStructure {
        model.compartmental().unwrap()
    }

    fn structure_mut(model: &mut Model) -> &mut CompartmentalStructure {
        model.compartmental_mut().unwrap()
    }

    /// Initial amount of each compartment, in the order of `compartments`.
    fn initial_amounts(model: &Model, compartments: &[String]) -> Result<Vec<f64>, String> {
        let state = initial_state(structure(model))?;
        let amounts: HashMap<&str, f64> = state
            .iter()
            .map(|compartment| (compartment.name.as_str(), compartment.amount))
            .collect();
        Ok(compartments
            .iter()
            .map(|name| *amounts.get(name.as_str()).unwrap_or(&0.0))
            .collect())
    }

    fn bin_fraction(bin: &str, fraction: f64) -> BinFraction {
        BinFraction {
            bin: bin.to_string(),
            fraction: Some(fraction),
        }
    }

    fn group_stratification() -> Stratification {
        Stratification {
            id: "group".to_string(),
            categories: vec!["g1".to_string(), "g2".to_string()],
            conditions: None,
        }
    }

    fn group_fractions(g1: f64, g2: f64) -> StratificationFractions {
        StratificationFractions {
            stratification: "group".to_string(),
            fractions: vec![
                StratificationFraction {
                    category: "g1".to_string(),
                    fraction: g1,
                },
                StratificationFraction {
                    category: "g2".to_string(),
                    fraction: g2,
                },
            ],
        }
    }

    fn sub_fractions(s1: f64, s2: f64) -> StratificationFractions {
        StratificationFractions {
            stratification: "sub".to_string(),
            fractions: vec![
                StratificationFraction {
                    category: "s1".to_string(),
                    fraction: s1,
                },
                StratificationFraction {
                    category: "s2".to_string(),
                    fraction: s2,
                },
            ],
        }
    }

    fn make_model(
        stratifications: Vec<Stratification>,
        stratification_fractions: Vec<StratificationFractions>,
        compartment_fractions: Vec<CompartmentFraction>,
    ) -> Model {
        Model {
            name: "test".to_string(),
            description: None,
            version: None,
            parameters: vec![],
            engine: ModelEngine::new(EngineType::DifferenceEquations),
            structure: ModelStructure::Compartmental(CompartmentalStructure {
                bins: vec![
                    Bin {
                        id: "A".to_string(),
                        name: "A".to_string(),
                    },
                    Bin {
                        id: "B".to_string(),
                        name: "B".to_string(),
                    },
                ],
                accumulators: vec![],
                stratifications,
                transitions: vec![],
                initial_conditions: InitialConditions {
                    total: 1000.0,
                    bin_fractions: vec![bin_fraction("A", 1.0), bin_fraction("B", 0.0)],
                    stratification_fractions,
                    compartment_fractions,
                },
            }),
        }
    }

    /// The thresholds `collect_time_thresholds` finds in `formula`, as text.
    fn thresholds_of(formula: &str) -> Vec<String> {
        let fixed: HashSet<String> = ["t_on", "k"].map(String::from).into();
        let expression = MathExpression::new(formula.to_string());
        let ast = parse_expression(expression.preprocessed()).unwrap();
        let mut thresholds = Vec::new();
        collect_time_thresholds(&ast, &fixed, &mut thresholds);
        thresholds.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn a_comparison_of_t_with_a_fixed_value_is_a_threshold() {
        assert_eq!(thresholds_of("if(t >= 10, k, 0)"), ["10"]);
        assert_eq!(thresholds_of("if(t_on + 1.5 > t, k, 0)"), ["(t_on + 1.5)"]);
        assert_eq!(
            thresholds_of("k * max(if(t < t_on, 1, 0), if(t >= 2 * pi, 1, 0))"),
            ["t_on", "(2 * pi)"]
        );
    }

    #[test]
    fn a_comparison_of_t_with_the_state_or_a_formula_is_not_a_threshold() {
        assert!(thresholds_of("if(t >= A, k, 0)").is_empty());
        assert!(thresholds_of("if(t >= r, k, 0)").is_empty());
        assert!(thresholds_of("if(t - 10 >= 0, k, 0)").is_empty());
        assert!(thresholds_of("if(step >= 10, k, 0)").is_empty());
    }

    #[test]
    fn override_is_applied_after_uniform_expansion() {
        let model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![CompartmentFraction {
                compartment: "A_g2".to_string(),
                fraction: 0.25,
            }],
        );
        let (compartments, _) = generate_compartments(&model);
        let amounts = initial_amounts(&model, &compartments).unwrap();

        assert_eq!(compartments, vec!["A_g1", "A_g2", "B_g1", "B_g2"]);
        assert_eq!(amounts, vec![600.0, 250.0, 0.0, 0.0]);
    }

    #[test]
    fn override_on_truncated_conditional_compartment() {
        let sub = Stratification {
            id: "sub".to_string(),
            categories: vec!["s1".to_string(), "s2".to_string()],
            conditions: Some(vec![StratificationCondition {
                stratification: "group".to_string(),
                category: Some("g1".to_string()),
                to: None,
            }]),
        };
        let model = make_model(
            vec![group_stratification(), sub],
            vec![group_fractions(0.7, 0.3), sub_fractions(0.5, 0.5)],
            vec![CompartmentFraction {
                compartment: "A_g2".to_string(),
                fraction: 0.1,
            }],
        );
        let (compartments, _) = generate_compartments(&model);
        let amounts = initial_amounts(&model, &compartments).unwrap();

        assert_eq!(
            compartments,
            vec!["A_g1_s1", "A_g1_s2", "A_g2", "B_g1_s1", "B_g1_s2", "B_g2"]
        );
        assert_eq!(amounts, vec![350.0, 350.0, 100.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn unknown_compartment_override_is_a_build_error() {
        let model = make_model(
            vec![],
            vec![],
            vec![CompartmentFraction {
                compartment: "Missing".to_string(),
                fraction: 0.1,
            }],
        );
        let (compartments, _) = generate_compartments(&model);
        let error = initial_amounts(&model, &compartments).unwrap_err();

        assert!(error.contains("Unknown compartment 'Missing'"));
    }

    #[test]
    fn the_total_is_kept_with_overrides() {
        let model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![
                CompartmentFraction {
                    compartment: "A_g1".to_string(),
                    fraction: 0.4,
                },
                CompartmentFraction {
                    compartment: "A_g2".to_string(),
                    fraction: 0.6,
                },
            ],
        );
        let (compartments, _) = generate_compartments(&model);
        let amounts = initial_amounts(&model, &compartments).unwrap();

        assert_eq!(amounts, vec![400.0, 600.0, 0.0, 0.0]);
        assert!((amounts.iter().sum::<f64>() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn negative_compartment_fraction_is_a_build_error() {
        let model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![CompartmentFraction {
                compartment: "A_g1".to_string(),
                fraction: -0.5,
            }],
        );
        let (compartments, _) = generate_compartments(&model);
        let error = initial_amounts(&model, &compartments).unwrap_err();

        assert!(error.contains("between 0.0 and 1.0"), "{error}");
    }

    #[test]
    fn fraction_above_one_is_a_build_error() {
        let model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![CompartmentFraction {
                compartment: "A_g1".to_string(),
                fraction: 1.5,
            }],
        );
        let (compartments, _) = generate_compartments(&model);
        let error = initial_amounts(&model, &compartments).unwrap_err();

        assert!(error.contains("between 0.0 and 1.0"), "{error}");
    }

    #[test]
    fn duplicate_compartment_override_is_a_build_error() {
        let model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![
                CompartmentFraction {
                    compartment: "A_g1".to_string(),
                    fraction: 0.1,
                },
                CompartmentFraction {
                    compartment: "A_g1".to_string(),
                    fraction: 0.2,
                },
            ],
        );
        let (compartments, _) = generate_compartments(&model);
        let error = initial_amounts(&model, &compartments).unwrap_err();

        assert!(error.contains("Duplicate compartment 'A_g1'"), "{error}");
    }

    #[test]
    fn boundary_compartment_fractions_are_accepted() {
        let model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![
                CompartmentFraction {
                    compartment: "A_g1".to_string(),
                    fraction: 0.0,
                },
                CompartmentFraction {
                    compartment: "A_g2".to_string(),
                    fraction: 1.0,
                },
            ],
        );
        let (compartments, _) = generate_compartments(&model);
        let amounts = initial_amounts(&model, &compartments).unwrap();

        assert_eq!(amounts, vec![0.0, 1000.0, 0.0, 0.0]);
    }

    #[test]
    fn override_exceeding_the_total_is_a_build_error() {
        let model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![CompartmentFraction {
                compartment: "B_g1".to_string(),
                fraction: 0.5,
            }],
        );
        let error = initial_state(structure(&model)).unwrap_err();

        assert!(error.contains("exceed the total"), "{error}");
    }

    #[test]
    fn initial_state_reports_bin_and_applied_categories() {
        let sub = Stratification {
            id: "sub".to_string(),
            categories: vec!["s1".to_string(), "s2".to_string()],
            conditions: Some(vec![StratificationCondition {
                stratification: "group".to_string(),
                category: Some("g1".to_string()),
                to: None,
            }]),
        };
        let model = make_model(
            vec![group_stratification(), sub],
            vec![group_fractions(0.6, 0.4), sub_fractions(0.5, 0.5)],
            vec![],
        );
        let state = initial_state(structure(&model)).unwrap();
        let a_g1_s2 = &state[1];
        let a_g2 = &state[2];

        assert_eq!(a_g1_s2.name, "A_g1_s2");
        assert_eq!(a_g1_s2.bin, "A");
        assert_eq!(
            a_g1_s2.categories,
            vec![
                ("group".to_string(), "g1".to_string()),
                ("sub".to_string(), "s2".to_string())
            ]
        );
        assert_eq!(a_g1_s2.amount, 300.0);
        assert_eq!(a_g2.name, "A_g2");
        assert_eq!(
            a_g2.categories,
            vec![("group".to_string(), "g2".to_string())]
        );
        assert_eq!(a_g2.amount, 400.0);
    }

    fn model_with_rate(rate: &str) -> Model {
        use commol_core::{Parameter, ParameterValue, Transition};
        let sub = Stratification {
            id: "sub".to_string(),
            categories: vec!["s1".to_string(), "s2".to_string()],
            conditions: None,
        };
        let mut model = make_model(
            vec![group_stratification(), sub],
            vec![group_fractions(0.5, 0.5), sub_fractions(0.5, 0.5)],
            vec![],
        );
        model.parameters = vec![Parameter {
            id: "k".to_string(),
            value: Some(ParameterValue::Constant(0.1)),
            description: None,
        }];
        structure_mut(&mut model).transitions = vec![Transition {
            id: "ab".to_string(),
            source: vec!["A".to_string()],
            target: vec!["B".to_string()],
            accumulators: vec![],
            rate: Some(RateMathExpression::from_string(rate.to_string())),
            stratified_rates: None,
            per_compartment: None,
            impulse_at: None,
            impulse_every: None,
        }];
        model
    }

    #[test]
    fn subgroup_total_out_of_declaration_order_is_a_build_error() {
        let error = CompiledModel::try_from_model(&model_with_rate("k * total_s1_g1 / total"))
            .err()
            .expect("total_s1_g1 names no subgroup");
        assert!(error.contains("'total_s1_g1'"), "{error}");
        assert!(error.contains("total_g1_s1"), "{error}");
    }

    #[test]
    fn subgroup_total_names_follow_the_declaration_order() {
        let names = subgroup_total_names(structure(&model_with_rate("k")));

        for name in ["total_g1", "total_s2", "total_g1_s1", "A", "A_g1", "B_s2"] {
            assert!(names.contains(&name.to_string()), "{name}: {names:?}");
        }
        assert!(!names.contains(&"total_s1_g1".to_string()), "{names:?}");
        assert!(names.is_sorted());
    }

    #[test]
    fn unknown_subgroup_total_is_a_build_error() {
        let error = CompiledModel::try_from_model(&model_with_rate("k * total_x"))
            .err()
            .expect("total_x names no subgroup");
        assert!(error.contains("'total_x'"), "{error}");
    }

    #[test]
    fn subgroup_total_in_declaration_order_builds() {
        assert!(CompiledModel::try_from_model(&model_with_rate("k * total_g1_s1 / total")).is_ok());
    }

    #[test]
    fn modulo_is_an_operator_in_a_rate() {
        // `step % 2` is 0 at even steps and 1 at odd steps (per-capita rate).
        let mut model = CompiledModel::try_from_model(&model_with_rate("step % 2")).unwrap();
        let state = model.initial_state().to_vec();
        let mut rates = vec![0.0; state.len()];
        // Compartment 0 is A_g1_s1 with an amount of 250.
        model.derivatives(0.0, 0.0, &state, &mut rates).unwrap();
        assert_eq!(rates[0], 0.0);
        model.derivatives(1.0, 1.0, &state, &mut rates).unwrap();
        assert_eq!(rates[0], -250.0);
    }

    fn model_with_parameter(id: &str) -> Model {
        use commol_core::{Parameter, ParameterValue};
        let mut model = model_with_rate("k");
        model.parameters.push(Parameter {
            id: id.to_string(),
            value: Some(ParameterValue::Constant(1.0)),
            description: None,
        });
        model
    }

    #[test]
    fn parameter_id_cannot_be_a_compartment_or_a_subgroup_total() {
        for (id, kind) in [
            ("A_g1_s1", "a compartment"),
            ("total_g1", "a subgroup total"),
            ("A_g1", "a subgroup total"),
            ("A", "a subgroup total"),
        ] {
            let error =
                check_names(&model_with_parameter(id)).expect_err("a reserved id must be rejected");
            assert!(error.contains(&format!("'{id}' ({kind})")), "{error}");
        }
        assert!(check_names(&model_with_parameter("total_max")).is_ok());
    }

    #[test]
    fn a_real_total_starts_with_amounts_that_sum_to_it() {
        let mut model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![],
        );
        structure_mut(&mut model).initial_conditions.total = 2.5;
        let state = initial_state(structure(&model)).unwrap();

        let sum: f64 = state.iter().map(|compartment| compartment.amount).sum();
        assert!((sum - 2.5).abs() < 1e-12, "{sum}");
        assert!(CompiledModel::try_from_model(&model).is_ok());
    }

    #[test]
    fn a_total_that_is_not_finite_and_positive_is_a_build_error() {
        for total in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut model = make_model(vec![], vec![], vec![]);
            structure_mut(&mut model).initial_conditions.total = total;
            let error = initial_state(structure(&model)).unwrap_err();

            assert!(error.contains("finite and greater than 0"), "{error}");
        }
    }
}
