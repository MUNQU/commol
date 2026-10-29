//! Expansion of a model and its compilation into a [`CompiledModel`].

use crate::flows::{PlannedFlow, plan_transition_flows};
use crate::helpers::stratification_conditions_met;
use crate::labels::OutputLabels;
use crate::layout::CompartmentLayout;
use crate::matrices::{check_matrices, matrix_variables, scaled_matrices};
use crate::types::{
    AccumulatorOutputRef, CompiledModel, FAST_STACK_DEPTH, FastRateExpression, FastRateOp,
    FlowChange, GeneratedAccumulatorOutput, ImpulseFlow, ImpulseSchedule, SeriesParameter,
    SubgroupTotalLayout, SubgroupTotalMapping, TransitionFlow, VarSlot,
};
use commol_core::math_expression::jit::ast::{BinaryOperator, Expr, UnaryOperator};
use commol_core::math_expression::variables::{
    ENGINE_VARIABLES, SPECIAL_VAR_E, SPECIAL_VAR_PI, SPECIAL_VAR_STEP, SPECIAL_VAR_T,
    SPECIAL_VAR_TOTAL, SUBGROUP_TOTAL_PREFIX,
};
use commol_core::{
    Aggregate, MathExpression, MathExpressionContext, Model, NEIGHBORS, NEIGHBORS_PREFIX,
    ParameterValue, Process, RateMathExpression, Rounding, SeriesMode,
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
    /// Compile an aggregate model, whatever its engine.
    ///
    /// This constructor performs several pre-computation steps for efficiency:
    /// 1. Generates all stratified compartment combinations
    /// 2. Splits the amount of each bin among its compartments
    /// 3. Pre-computes transition flows with parsed rate expressions
    /// 4. Pre-computes the subgroup total mappings for stratifications
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not aggregate, when a check of the
    /// model fails ([`check_model`]),
    /// when a transition cannot be planned, or when an initial-condition
    /// override is not valid for the expansion.
    pub fn try_from_model(model: &Model) -> Result<Self, String> {
        let aggregate = model.aggregate()?;
        let process = &model.process;
        check_model(model)?;

        // Generate all compartment and accumulator output combinations
        let layout = CompartmentLayout::new(process);
        let compartments: Vec<String> = layout.names().to_vec();
        let compartment_map: HashMap<String, usize> = compartments
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();
        let accumulator_outputs = generate_accumulators(process);
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
        let state = initial_state(process, aggregate)?;
        debug_assert!(
            state
                .iter()
                .map(|compartment| &compartment.name)
                .eq(compartments.iter())
        );
        let amounts: Vec<f64> = state.iter().map(|c| c.amount).collect();
        let total = amounts.iter().sum();

        // Labels of every output: compartments first, then accumulators
        let output_labels: Vec<OutputLabels> = layout
            .labels()
            .iter()
            .cloned()
            .chain(
                expand_accumulators(process)
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
        // values do not change during a run, and a caller may set them.
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
                    rate_expr
                        .check()
                        .map_err(|error| format!("Parameter '{}': {error}", p.id))?;
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
        // The entries and the spectral radius of each matrix do not change
        // during a run; `set_parameter` writes them again when a caller sets
        // a scale. A matrix with a scale to calibrate has them once it is set.
        let matrices = scaled_matrices(model);
        for matrix in &matrices {
            if let Ok(values) = matrix.variables(|_| None) {
                constant_parameters.extend(matrix.names().map(str::to_string).zip(values));
            }
        }

        // Initialize expression context with constant parameters
        let mut expression_context = MathExpressionContext::new();
        expression_context.set_parameters(constant_parameters);
        expression_context.init_compartments(compartments.clone());

        // Pre-compute the subgroup total mappings for stratifications
        // (must be done before transition flows, as absolute-flow inference
        // needs to know about subgroup total names)
        let subgroup_total_layout = build_subgroup_total_mappings(&layout, process);

        let subgroup_total_param_names: HashSet<String> = subgroup_total_layout
            .iter()
            .map(|m| m.parameter_name.clone())
            .collect();

        // Pre-allocate every parameter slot the per-step loop will touch so
        // `set_parameter_str` can update in place and JIT input resolution can
        // use direct indexed parameter reads where possible.
        expression_context.reserve_parameters(
            std::iter::once(SPECIAL_VAR_TOTAL.to_string())
                .chain(std::iter::once(SPECIAL_VAR_T.to_string()))
                .chain(model.parameters.iter().map(|p| p.id.clone()))
                .chain(
                    subgroup_total_layout
                        .iter()
                        .map(|m| m.parameter_name.clone()),
                )
                .chain(series_payloads.iter().map(|payload| payload.name.clone()))
                .chain(formula_parameters.iter().map(|(name, _)| name.clone()))
                .chain(
                    matrices
                        .iter()
                        .flat_map(|matrix| matrix.names().map(str::to_string)),
                ),
        );
        let total_parameter_index = expression_context
            .parameter_index(SPECIAL_VAR_TOTAL)
            .expect("total parameter slot must be reserved");
        let t_parameter_index = expression_context
            .parameter_index(SPECIAL_VAR_T)
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
            process,
            aggregate.neighbors.as_ref(),
            &layout,
            &accumulator_map,
            &subgroup_total_param_names,
        )?;
        let schedules = impulse_schedules(process)?;
        let (impulse_plans, flow_plans): (Vec<PlannedFlow>, Vec<PlannedFlow>) = planned_flows
            .into_iter()
            .partition(|flow| schedules.contains_key(&flow.transition_id));
        let impulse_flows: Vec<ImpulseFlow> =
            compile_transition_flows(impulse_plans, &compartment_map, &expression_context)?
                .into_iter()
                .map(|flow| ImpulseFlow {
                    schedule: schedules[&flow.transition_id].clone(),
                    flow,
                })
                .collect();
        let transition_flows =
            compile_transition_flows(flow_plans, &compartment_map, &expression_context)?;

        let used_context_variables = collect_used_context_variables(
            transition_flows
                .iter()
                .chain(impulse_flows.iter().map(|impulse| &impulse.flow)),
            &formula_parameters,
        );
        let depends_on_step = series_parameters.iter().any(SeriesParameter::reads_step)
            || reads_variable(
                transition_flows
                    .iter()
                    .chain(impulse_flows.iter().map(|impulse| &impulse.flow)),
                &formula_parameters,
                SPECIAL_VAR_STEP,
            );
        let reads_time = series_parameters
            .iter()
            .any(SeriesParameter::changes_along_t)
            || reads_variable(
                transition_flows
                    .iter()
                    .chain(impulse_flows.iter().map(|impulse| &impulse.flow)),
                &formula_parameters,
                SPECIAL_VAR_T,
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

        // The HashMap-backed compartment context is only needed by the
        // formula parameters, which read it through their compiled function.
        // The rates of the flows read the state directly, through
        // `resolved_slots` or `fast_rate_expression`.
        let requires_compartment_context = !formula_parameters.is_empty();

        let initial_state: Vec<f64> = amounts
            .into_iter()
            .chain(std::iter::repeat_n(0.0, accumulator_map.len()))
            .collect();

        let num_compartments = compartments.len();
        let flow_changes = transition_flows
            .iter()
            .map(|flow| FlowChange::of(flow, num_compartments))
            .collect();
        let impulse_changes = impulse_flows
            .iter()
            .map(|impulse| FlowChange::of(&impulse.flow, num_compartments))
            .collect();
        let flow_buffer = vec![0.0; transition_flows.len()];

        Ok(Self {
            compartments,
            total,
            output_labels,
            output_names,
            initial_state,
            expression_context,
            transition_flows,
            flow_changes,
            impulse_changes,
            flow_buffer,
            impulse_flows,
            depends_on_step,
            reads_time,
            time_thresholds,
            subgroup_total_mappings,
            formula_parameters,
            series_parameters,
            matrices,
            settable_parameters: fixed_parameters,
            total_parameter_index,
            t_parameter_index,
            requires_compartment_context,
        })
    }
}

/// Relative tolerance with which an amount is a whole number: the rounding
/// of an amount times a fraction.
const WHOLE_TOLERANCE: f64 = 1e-9;

/// The whole number `value` is, within the rounding of an amount times a
/// fraction, or `None` when it is not one.
pub fn as_whole(value: f64) -> Option<f64> {
    let whole = value.round();
    ((value - whole).abs() <= WHOLE_TOLERANCE * whole.abs().max(1.0)).then_some(whole)
}

/// The first five of `items`, joined, and how many more there are.
fn first_five(items: &[String]) -> String {
    let shown = items.len().min(5);
    let more = if items.len() > shown {
        format!(" and {} more", items.len() - shown)
    } else {
        String::new()
    };
    format!("{}{more}", items[..shown].join(", "))
}

/// Check that a stochastic engine can start the model from whole units.
///
/// A stochastic engine moves whole units, and it makes the compartments of
/// each bin whole without moving a unit to another bin, so the amount of
/// every bin must be a whole number. When the amount of a compartment is not,
/// the settings of the engine must name how to make it whole
/// ([`commol_core::Rounding`]); there is no default.
fn check_whole_units(model: &Model, aggregate: &Aggregate) -> Result<(), String> {
    let engine = model.engine.engine_type();
    if !engine.is_stochastic() {
        return Ok(());
    }
    let state = initial_state(&model.process, aggregate)?;
    let fractional_bins: Vec<String> = model
        .process
        .bins
        .iter()
        .filter_map(|bin| {
            let amount: f64 = state
                .iter()
                .filter(|compartment| compartment.bin == bin.id)
                .map(|compartment| compartment.amount)
                .sum();
            as_whole(amount)
                .is_none()
                .then(|| format!("'{}' ({amount})", bin.id))
        })
        .collect();
    if !fractional_bins.is_empty() {
        return Err(format!(
            "The {} engine moves whole units, so the amount of each bin must be a whole \
             number, and the amounts of {} are not whole",
            engine.name(),
            first_five(&fractional_bins),
        ));
    }
    if model.engine.settings().rounding().is_some() {
        return Ok(());
    }
    let fractional: Vec<String> = state
        .into_iter()
        .filter(|compartment| as_whole(compartment.amount).is_none())
        .map(|compartment| format!("'{}' ({})", compartment.name, compartment.amount))
        .collect();
    if fractional.is_empty() {
        return Ok(());
    }
    Err(format!(
        "The {} engine moves whole units, and the initial amounts of {} are not whole. \
         Name a rounding in the settings of the engine: \"{}\" (the same whole amounts \
         in every run, which sum to the amount of each bin) or \"{}\" (each run draws \
         the amounts of each bin from a multinomial with the fractions of the model)",
        engine.name(),
        first_five(&fractional),
        Rounding::LargestRemainder.name(),
        Rounding::Random.name(),
    ))
}

/// Whether `position` is a valid position that is not an integer. A
/// position that is not finite or is negative is not one: the compilation of
/// the model rejects it with its own message.
fn between_integers(position: f64) -> bool {
    position.is_finite() && position >= 0.0 && position.fract() != 0.0
}

/// The smallest position of `positions` that is between two integers.
fn first_between_integers(positions: impl Iterator<Item = f64>) -> Option<f64> {
    positions
        .filter(|&position| between_integers(position))
        .min_by(f64::total_cmp)
}

/// Check the series points and the impulse positions against the engine.
///
/// A `"pulse"` series acts on one whole step, so each of its points must be
/// an integer, whatever the engine. An engine that reports only at integer
/// steps ([`commol_core::EngineType::reports_at_integer_steps`]) cannot run a series point
/// or an impulse between two steps either.
fn check_positions_of_engine(model: &Model) -> Result<(), String> {
    let series = || {
        model
            .parameters
            .iter()
            .filter_map(|parameter| match &parameter.value {
                Some(ParameterValue::Series { data, mode }) => {
                    Some((parameter.id.as_str(), data, mode))
                }
                _ => None,
            })
    };
    for (id, data, mode) in series() {
        if *mode == SeriesMode::Pulse
            && let Some(position) = first_between_integers(data.iter().map(|&(p, _)| p))
        {
            return Err(format!(
                "Series parameter '{id}' is a pulse at {position}, which is not an \
                 integer step; a pulse acts on one whole step"
            ));
        }
    }
    let engine = model.engine.engine_type();
    if !engine.reports_at_integer_steps() {
        return Ok(());
    }
    for transition in &model.process.transitions {
        let position = match (&transition.impulse_at, transition.impulse_every) {
            (Some(positions), None) => first_between_integers(positions.iter().copied()),
            (None, Some((period, offset))) => [offset, period]
                .into_iter()
                .find(|&value| between_integers(value)),
            _ => None,
        };
        if let Some(position) = position {
            return Err(format!(
                "Impulse transition '{}' acts at {position}, which is not an integer \
                 step; the {} engine steps only at integers",
                transition.id,
                engine.name()
            ));
        }
    }
    for (id, data, _) in series() {
        if let Some(position) = first_between_integers(data.iter().map(|&(p, _)| p)) {
            return Err(format!(
                "Series parameter '{id}' has a point at {position}, which is not an \
                 integer step; the {} engine steps only at integers",
                engine.name()
            ));
        }
    }
    Ok(())
}

/// The impulse schedule of each impulse transition, by transition id.
///
/// # Errors
///
/// Returns the message of [`crate::rules::check_impulse`] for the first
/// impulse outside its rule.
fn impulse_schedules(process: &Process) -> Result<HashMap<String, ImpulseSchedule>, String> {
    let mut schedules = HashMap::new();
    for transition in &process.transitions {
        crate::rules::check_impulse(
            &transition.id,
            transition.impulse_at.as_deref(),
            transition.impulse_every,
        )?;
        let schedule = match (&transition.impulse_at, transition.impulse_every) {
            (Some(positions), None) => {
                let mut positions = positions.clone();
                positions.sort_by(f64::total_cmp);
                positions.dedup();
                ImpulseSchedule::At(positions)
            }
            (None, Some((period, offset))) => ImpulseSchedule::Every { period, offset },
            _ => continue,
        };
        schedules.insert(transition.id.clone(), schedule);
    }
    Ok(schedules)
}

/// A bin or accumulator expanded over the stratifications.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpandedOutput {
    /// Expanded name, for example `A_g1` or `B_g2_t1`.
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
    process: &Process,
    source_ids: impl IntoIterator<Item = &'a str>,
) -> Vec<(ExpandedOutput, HashMap<String, String>)> {
    // Each element: (source, name, applied categories by stratification id)
    let mut partials: Vec<(String, String, HashMap<String, String>)> = source_ids
        .into_iter()
        .map(|id| (id.to_string(), id.to_string(), HashMap::new()))
        .collect();

    for stratification in &process.stratifications {
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
            let categories = process
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

/// Expanded compartments of a process, in engine order.
pub fn expand_compartments(process: &Process) -> Vec<ExpandedOutput> {
    expand(process, process.bins.iter().map(|b| b.id.as_str()))
        .into_iter()
        .map(|(output, _)| output)
        .collect()
}

/// Expanded accumulator outputs of a process, in engine order.
pub fn expand_accumulators(process: &Process) -> Vec<ExpandedOutput> {
    expand(process, process.accumulators.iter().map(|a| a.id.as_str()))
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

pub(crate) fn generate_accumulators(process: &Process) -> Vec<GeneratedAccumulatorOutput> {
    expand(process, process.accumulators.iter().map(|a| a.id.as_str()))
        .into_iter()
        .map(|(output, applied)| GeneratedAccumulatorOutput {
            name: output.name,
            output_ref: accumulator_output_ref(output.source, &applied, &process.stratifications),
        })
        .collect()
}

/// One expanded compartment of a model and the amount it starts with.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialCompartment {
    /// Expanded compartment name, for example `A_g1` or `B_g2_t1`.
    pub name: String,
    /// Bin the compartment expands from.
    pub bin: String,
    /// `(stratification id, category)` pairs applied to the compartment, in
    /// stratification declaration order. A conditional stratification whose
    /// conditions the compartment does not meet is absent.
    pub categories: Vec<(String, String)>,
    /// Initial amount, in the unit of the bin, including
    /// `compartment_amounts` overrides.
    pub amount: f64,
}

/// Expand a process into its compartments and their initial amounts.
///
/// This is the single source of truth for the initial state of an aggregate
/// model. Bins are expanded over the stratifications in declaration order;
/// when a stratification has conditions that a compartment's already-applied
/// categories do not meet, that compartment is kept as-is and its amount is
/// not split by that stratification's fractions. Each compartment holds the
/// amount of its bin, in the unit of the bin, times the fraction of each
/// category applied to it. An uncalibrated bin amount counts as zero.
///
/// After the expansion, `compartment_amounts` overrides replace the amount
/// of the named compartments.
///
/// # Errors
///
/// Returns the message of the rule of a bin amount, the fractions of a
/// stratification or an override ([`crate::rules::check_bin_amount`],
/// [`crate::rules::check_fractions`],
/// [`crate::rules::check_compartment_amount`]), or a message when an
/// override names a compartment that is not part of the expansion.
pub fn initial_state(
    process: &Process,
    aggregate: &Aggregate,
) -> Result<Vec<InitialCompartment>, String> {
    let initial_conditions = &aggregate.initial_conditions;
    let bin_amounts = &initial_conditions.bin_amounts;
    for (bin, amount) in bin_amounts {
        if let Some(amount) = amount {
            crate::rules::check_bin_amount(bin, *amount)?;
        }
    }
    let stratification_fractions = &initial_conditions.stratification_fractions;
    for (stratification, fractions) in stratification_fractions {
        crate::rules::check_fractions(
            stratification,
            fractions
                .iter()
                .map(|(category, &fraction)| (category.as_str(), fraction)),
        )?;
    }

    let mut state: Vec<InitialCompartment> = expand_compartments(process)
        .into_iter()
        .map(|output| {
            let bin_amount = bin_amounts
                .get(output.source.as_str())
                .and_then(|amount| *amount)
                .unwrap_or(0.0);
            let amount = output
                .categories
                .iter()
                .fold(bin_amount, |amount, (id, cat)| {
                    amount
                        * stratification_fractions
                            .get(id.as_str())
                            .and_then(|fractions| fractions.get(cat.as_str()))
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

    let overrides = &initial_conditions.compartment_amounts;
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
    for (compartment, &amount) in overrides {
        if !compartment_index.contains_key(compartment) {
            let available: Vec<&str> = state.iter().map(|c| c.name.as_str()).collect();
            return Err(format!(
                "Unknown compartment '{compartment}' in initial conditions. Available \
                 compartments: {available:?}"
            ));
        }
        crate::rules::check_compartment_amount(compartment, amount)?;
    }

    for (compartment, amount) in overrides {
        state[compartment_index[compartment]].amount = *amount;
    }

    Ok(state)
}

/// Compile planned transition flows into runtime flows.
///
/// # Errors
///
/// Returns a message that names the transition when the rate of a flow is
/// not in the language of formulas.
fn compile_transition_flows(
    planned_flows: Vec<PlannedFlow>,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Result<Vec<TransitionFlow>, String> {
    planned_flows
        .into_iter()
        .map(|flow| {
            let rate_expression = RateMathExpression::from_string(flow.rate);
            rate_expression
                .check()
                .map_err(|error| format!("Transition '{}': {error}", flow.transition_id))?;
            let resolved_slots =
                resolve_jit_slots(&rate_expression, compartment_map, expression_context);
            let fast_rate_expression =
                build_fast_rate_expression(&rate_expression, compartment_map, expression_context);
            let (source_index, target_index) = flow.pieces[0];
            Ok(TransitionFlow {
                transition_id: flow.transition_id,
                source_index,
                target_index,
                extra_pieces: flow.pieces[1..].to_vec(),
                accumulator_indices: flow.accumulator_indices,
                rate_expression,
                fast_rate_expression,
                is_absolute_flow: flow.absolute,
                resolved_slots,
            })
        })
        .collect()
}

/// Where the name of a subgroup total comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TotalSource {
    /// `total_<categories>`: the compartments of every bin with these categories.
    Categories,
    /// A bin total or a partial bin total: the bin id, then some categories.
    Bin,
}

/// Every subgroup total of a process, one entry per contributing
/// compartment: its name, where the name comes from and the compartment index.
///
/// The entries are:
/// 1. Subgroup totals over categories (`total_g1`, `total_g1_t1`, ...)
/// 2. Bin totals (`A`, `B`, `C`): the sum of all stratified versions
/// 3. Partial bin totals (`A_g1` when a second stratification splits it)
///
/// Every entry is built from the compartment labels.
fn subgroup_total_entries(
    layout: &CompartmentLayout,
    process: &Process,
) -> Vec<(String, TotalSource, usize)> {
    let mut entries = Vec::new();
    if process.stratifications.is_empty() {
        return entries;
    }
    let partial_bin_totals = process.stratifications.len() >= 2;

    for (compartment_index, label) in layout.labels().iter().enumerate() {
        let categories: Vec<&str> = label
            .categories
            .iter()
            .map(|(_, category)| category.as_str())
            .collect();
        if categories.is_empty() {
            continue;
        }

        entries.push((label.source.clone(), TotalSource::Bin, compartment_index));

        let full_mask: u32 = (1 << categories.len()) - 1;
        for subset_mask in 1..=full_mask {
            let subset: Vec<&str> = categories
                .iter()
                .enumerate()
                .filter(|(k, _)| (subset_mask >> k) & 1 == 1)
                .map(|(_, category)| *category)
                .collect();
            let combination = subset.join("_");

            entries.push((
                format!("{SUBGROUP_TOTAL_PREFIX}{combination}"),
                TotalSource::Categories,
                compartment_index,
            ));

            // Partial bin total. With one stratification a partial sum would
            // duplicate either the compartment or the bin total.
            if partial_bin_totals && subset_mask != full_mask {
                entries.push((
                    format!("{}_{}", label.source, combination),
                    TotalSource::Bin,
                    compartment_index,
                ));
            }
        }
    }
    entries
}

/// Build pre-computed subgroup total mappings for stratifications, one per
/// name of [`subgroup_total_entries`].
fn build_subgroup_total_mappings(
    layout: &CompartmentLayout,
    process: &Process,
) -> Vec<SubgroupTotalLayout> {
    let mut totals: HashMap<String, Vec<usize>> = HashMap::new();
    for (name, _, compartment_index) in subgroup_total_entries(layout, process) {
        totals.entry(name).or_default().push(compartment_index);
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

/// The subgroup total that sums each set of compartments, by the sorted
/// indices of the set. When two names sum one set, the first in sorted order
/// is kept, so the choice does not depend on the order of a map.
pub(crate) fn subgroup_totals_by_compartments(
    layout: &CompartmentLayout,
    process: &Process,
) -> HashMap<Vec<usize>, String> {
    let mut mappings = build_subgroup_total_mappings(layout, process);
    mappings.sort_unstable_by(|a, b| a.parameter_name.cmp(&b.parameter_name));
    let mut totals: HashMap<Vec<usize>, String> = HashMap::new();
    for mapping in mappings {
        let mut compartments = mapping.contributing_compartment_indices;
        compartments.sort_unstable();
        compartments.dedup();
        totals.entry(compartments).or_insert(mapping.parameter_name);
    }
    totals
}

/// Names of the subgroup totals a rate expression may reference.
pub(crate) fn subgroup_total_name_set(
    layout: &CompartmentLayout,
    process: &Process,
) -> HashSet<String> {
    build_subgroup_total_mappings(layout, process)
        .into_iter()
        .map(|mapping| mapping.parameter_name)
        .collect()
}

/// Names of the subgroup totals a rate expression of a process may read,
/// sorted.
///
/// They are the `total_<categories>` names, the bin totals and the partial bin
/// totals. Category suffixes follow the declaration order of the
/// stratifications, and a conditional stratification adds its categories only
/// to the compartments it expands. A process without stratifications has
/// none.
pub fn subgroup_total_names(process: &Process) -> Vec<String> {
    let mut names: Vec<String> = subgroup_total_name_set(&CompartmentLayout::new(process), process)
        .into_iter()
        .collect();
    names.sort_unstable();
    names
}

/// Check an aggregate model: the checks of its structure
/// ([`Aggregate::check`]), the variables its rates read, and the names it
/// uses against the names the engine provides.
///
/// The engine provides one variable per expanded compartment and one per
/// subgroup total: `total_<categories>`, bin totals and partial bin totals.
/// A transition rate may also read the neighbors of a unit, `neighbors_<bin>`
/// and `neighbors`, which the structure resolves ([`commol_core::Neighbors`]).
/// This function returns an error when:
///
/// - the variant or the neighbors of the structure do not fit the process;
/// - a rate, a stratified rate or a formula parameter is not in the language
///   of formulas ([`RateMathExpression::check`]);
/// - a `"pulse"` series has a point that is not an integer, or the engine
///   steps only at integers (`DifferenceEquations`, `StochasticSteps`) and a
///   series point or an impulse position is not an integer;
/// - a parameter, bin or accumulator id is a variable the engine provides in
///   every expression ([`ENGINE_VARIABLES`]: `total`, `step`, `t`, `pi`, `e`),
///   because the engine variable would hide it, or it is `neighbors` or
///   starts with `neighbors_`;
/// - a transition rate reads `neighbors_<bin>` for an id that is not a bin,
///   or reads the neighbors and the structure does not say who they are;
///   a formula parameter reads the neighbors, which only a transition rate
///   has;
/// - a name built from a bin (a compartment, a bin total or a partial bin
///   total) is also a `total_<categories>` subgroup total, as for a bin
///   `total_y` stratified by `group = [y, o]` and `type = [y, o]`: `total_y_y`
///   would name two sums;
/// - a parameter id is one of those names, because the parameter and the
///   engine variable would share one slot;
/// - a transition rate or a formula parameter uses a `total_…` name that is
///   not a subgroup total of the model. Category suffixes follow the
///   declaration order of the stratifications, so `total_t1_g1` is not a name
///   when `group` is declared before `type`. A parameter or compartment whose
///   id starts with `total_` is not such a name;
/// - two matrices share an id, a matrix does not fit the process, the id or a
///   variable (`<M>_<i>_<j>`, `<M>_radius`) of a matrix is the name of
///   another thing, two matrices give one variable, a rate reads a matrix
///   the process does not have or the neighbors inside a matrix sum, or a
///   formula parameter reads a matrix sum (`check_matrices`);
/// - a rate or a formula parameter reads a variable that no parameter, bin,
///   compartment, engine variable, subgroup total, variable of a matrix or
///   name of the neighbors defines. A category id is not a variable.
///
/// # Errors
///
/// Returns a message when the model is not aggregate, or a message that
/// names each conflict and lists the valid names.
pub fn check_model(model: &Model) -> Result<(), String> {
    let aggregate = model.aggregate()?;
    let process = &model.process;
    aggregate.check(process)?;
    check_formulas(model)?;
    check_engine_variable_ids(model, process)?;
    check_neighbor_terms(model, aggregate)?;
    check_whole_units(model, aggregate)?;
    check_positions_of_engine(model)?;
    let layout = CompartmentLayout::new(process);
    check_bin_names_against_subgroup_totals(&layout, process)?;
    let compartments: HashSet<&str> = layout.names().iter().map(String::as_str).collect();
    let subgroup_totals = subgroup_total_name_set(&layout, process);
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

    let formulas = model_formulas(model);
    let used: HashSet<String> = formulas
        .iter()
        .flat_map(|formula| {
            RateMathExpression::from_string(formula.text.to_string()).get_variables()
        })
        .collect();
    let mut unknown: Vec<&String> = used
        .iter()
        .filter(|name| name.starts_with(SUBGROUP_TOTAL_PREFIX))
        .filter(|name| !subgroup_totals.contains(*name))
        .filter(|name| !compartments.contains(name.as_str()))
        .filter(|name| !parameter_ids.contains(name.as_str()))
        .collect();
    if !unknown.is_empty() {
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
        return Err(format!(
            "Unknown subgroup total(s) {}. Category suffixes follow the \
             declaration order of the stratifications. Valid names: {valid}",
            unknown
                .iter()
                .map(|name| format!("'{name}'"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    // A later kind names a name that is also of an earlier kind: a bin is
    // also its bin total, and a compartment of no category is its bin.
    let taken: HashMap<&str, &str> = subgroup_totals
        .iter()
        .map(|name| (name.as_str(), "a subgroup total"))
        .chain(compartments.iter().map(|&name| (name, "a compartment")))
        .chain(process.bins.iter().map(|bin| (bin.id.as_str(), "a bin")))
        .chain(
            process
                .accumulators
                .iter()
                .map(|accumulator| (accumulator.id.as_str(), "an accumulator")),
        )
        .chain(parameter_ids.iter().map(|&id| (id, "a parameter")))
        .chain(
            ENGINE_VARIABLES
                .iter()
                .map(|&name| (name, "an engine variable")),
        )
        .collect();
    let of = |owner: &str| -> Vec<(&str, &str)> {
        formulas
            .iter()
            .filter(|formula| formula.owner == owner)
            .map(|formula| (formula.id, formula.text))
            .collect()
    };
    check_matrices(model, &of("transition"), &of("parameter"), &taken)?;
    let matrix_names: Vec<String> = matrix_variables(model)
        .into_iter()
        .map(|variable| variable.name)
        .collect();

    let defined: HashSet<&str> = parameter_ids
        .iter()
        .copied()
        .chain(compartments.iter().copied())
        .chain(subgroup_totals.iter().map(String::as_str))
        .chain(process.bins.iter().map(|bin| bin.id.as_str()))
        .chain(matrix_names.iter().map(String::as_str))
        .chain(ENGINE_VARIABLES)
        .chain(std::iter::once(NEIGHBORS))
        .collect();
    check_formula_variables(model, &formulas, &defined)
}

/// A formula of a model: a rate, a stratified rate or the value of a formula
/// parameter.
struct ModelFormula<'a> {
    /// `"transition"` or `"parameter"`.
    owner: &'static str,
    /// The id of the transition or the parameter.
    id: &'a str,
    /// `"rate"`, `"stratified rate"` or `"formula"`.
    part: &'static str,
    /// The formula.
    text: &'a str,
}

/// Every formula of a model, in the order of its transitions and then of its
/// parameters.
fn model_formulas(model: &Model) -> Vec<ModelFormula<'_>> {
    let mut formulas = Vec::new();
    for transition in &model.process.transitions {
        let rates = transition.rate.iter().map(|rate| ("rate", rate)).chain(
            transition
                .stratified_rates
                .iter()
                .flatten()
                .map(|stratified| ("stratified rate", &stratified.rate)),
        );
        for (part, text) in rates {
            formulas.push(ModelFormula {
                owner: "transition",
                id: &transition.id,
                part,
                text,
            });
        }
    }
    for parameter in &model.parameters {
        if let Some(ParameterValue::Formula(text)) = &parameter.value {
            formulas.push(ModelFormula {
                owner: "parameter",
                id: &parameter.id,
                part: "formula",
                text,
            });
        }
    }
    formulas
}

/// Check that every rate and formula parameter of a model is in the
/// language of formulas ([`RateMathExpression::check`]).
///
/// # Errors
///
/// Returns the message of the check, after the transition or the parameter
/// that holds the formula.
fn check_formulas(model: &Model) -> Result<(), String> {
    for formula in model_formulas(model) {
        RateMathExpression::from_string(formula.text.to_string())
            .check()
            .map_err(|error| {
                let owner = match formula.owner {
                    "transition" => "Transition",
                    _ => "Parameter",
                };
                format!("{owner} '{}': {error}", formula.id)
            })?;
    }
    Ok(())
}

/// Check that every variable a formula of the model reads is `defined`.
///
/// # Errors
///
/// Returns a message that names the transition or the parameter, its
/// formula, the variables it reads that are not defined, and the parameters
/// and the bins of the model.
fn check_formula_variables(
    model: &Model,
    formulas: &[ModelFormula<'_>],
    defined: &HashSet<&str>,
) -> Result<(), String> {
    let list = |mut ids: Vec<&str>| {
        ids.sort_unstable();
        if ids.is_empty() {
            "none".to_string()
        } else {
            ids.join(", ")
        }
    };
    for formula in formulas {
        let undefined: Vec<String> = RateMathExpression::from_string(formula.text.to_string())
            .get_variables()
            .into_iter()
            .filter(|name| !defined.contains(name.as_str()) && !is_neighbor_term(name))
            .collect();
        if !undefined.is_empty() {
            return Err(format!(
                "Undefined variables in {} '{}' {} '{}': {}. Available parameters: {}. \
                 Available bins: {}.",
                formula.owner,
                formula.id,
                formula.part,
                formula.text,
                undefined.join(", "),
                list(model.parameters.iter().map(|p| p.id.as_str()).collect()),
                list(model.process.bins.iter().map(|b| b.id.as_str()).collect()),
            ));
        }
    }
    Ok(())
}

/// The variables a rate reads that name the neighbors of a unit:
/// `neighbors` and `neighbors_<bin>`.
pub(crate) fn neighbor_terms(rate: &str) -> Vec<String> {
    let mut terms: Vec<String> = RateMathExpression::from_string(rate.to_string())
        .get_variables()
        .into_iter()
        .filter(|name| is_neighbor_term(name))
        .collect();
    terms.sort_unstable();
    terms.dedup();
    terms
}

/// Whether a name is `neighbors` or starts with `neighbors_`.
fn is_neighbor_term(name: &str) -> bool {
    name == NEIGHBORS || name.starts_with(NEIGHBORS_PREFIX)
}

/// The message for a rate that reads the neighbors of a unit when the
/// structure does not say who they are.
pub(crate) fn missing_neighbors(transition_id: &str, term: &str) -> String {
    format!(
        "Transition '{transition_id}' reads '{term}', so the structure must say who the \
         neighbors of a unit are: neighbors=Complete() (every unit is a neighbor of every \
         other) or neighbors=Groups(stratification, weights=None) (units are neighbors by \
         the categories of one stratification)"
    )
}

/// Reject a read of the neighbors that the model cannot resolve: a
/// `neighbors_<bin>` that names no bin, a rate that reads the neighbors when
/// the structure has none, and a formula parameter that reads them.
fn check_neighbor_terms(model: &Model, aggregate: &Aggregate) -> Result<(), String> {
    let process = &model.process;
    for transition in &process.transitions {
        let rates = transition.rate.iter().cloned().chain(
            transition
                .stratified_rates
                .iter()
                .flatten()
                .map(|stratified| stratified.rate.clone()),
        );
        for rate in rates {
            for term in neighbor_terms(&rate) {
                if let Some(bin) = term.strip_prefix(NEIGHBORS_PREFIX)
                    && !process.bins.iter().any(|candidate| candidate.id == bin)
                {
                    let bins: Vec<&str> = process.bins.iter().map(|b| b.id.as_str()).collect();
                    return Err(format!(
                        "Transition '{}' reads '{term}', but '{bin}' is not a bin; \
                         `{NEIGHBORS_PREFIX}<bin>` names a bin of {bins:?}",
                        transition.id
                    ));
                }
                if aggregate.neighbors.is_none() {
                    return Err(missing_neighbors(&transition.id, &term));
                }
            }
        }
    }
    for parameter in &model.parameters {
        if let Some(commol_core::ParameterValue::Formula(formula)) = &parameter.value
            && let Some(term) = neighbor_terms(formula).first()
        {
            return Err(format!(
                "Parameter '{}' reads '{term}'; only a transition rate reads the \
                 neighbors of a unit, which depend on the compartment it starts from",
                parameter.id
            ));
        }
    }
    Ok(())
}

/// Reject a parameter, bin or accumulator id that is a variable the engine
/// provides in every expression, or a name of the neighbors of a unit.
fn check_engine_variable_ids(model: &Model, process: &Process) -> Result<(), String> {
    let ids = model
        .parameters
        .iter()
        .map(|parameter| ("parameter", parameter.id.as_str()))
        .chain(process.bins.iter().map(|bin| ("bin", bin.id.as_str())))
        .chain(
            process
                .accumulators
                .iter()
                .map(|accumulator| ("accumulator", accumulator.id.as_str())),
        );
    let (neighbor_names, conflicts): (Vec<_>, Vec<_>) = ids
        .filter(|(_, id)| ENGINE_VARIABLES.contains(id) || is_neighbor_term(id))
        .partition(|(_, id)| is_neighbor_term(id));
    let describe = |ids: Vec<(&str, &str)>| {
        ids.into_iter()
            .map(|(kind, id)| format!("{kind} '{id}'"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if !conflicts.is_empty() {
        return Err(format!(
            "Ids cannot be variables the engine provides ({}): {}. Rename them.",
            ENGINE_VARIABLES.join(", "),
            describe(conflicts)
        ));
    }
    if !neighbor_names.is_empty() {
        return Err(format!(
            "Ids cannot be '{NEIGHBORS}' or start with '{NEIGHBORS_PREFIX}', the names of \
             the neighbors of a unit: {}. Rename them.",
            describe(neighbor_names)
        ));
    }
    Ok(())
}

/// Reject a compartment, bin total or partial bin total whose name is also a
/// `total_<categories>` subgroup total.
fn check_bin_names_against_subgroup_totals(
    layout: &CompartmentLayout,
    process: &Process,
) -> Result<(), String> {
    let entries = subgroup_total_entries(layout, process);
    let by_categories: HashSet<&str> = entries
        .iter()
        .filter(|(_, source, _)| *source == TotalSource::Categories)
        .map(|(name, _, _)| name.as_str())
        .collect();
    let mut conflicts: Vec<&str> = entries
        .iter()
        .filter(|(_, source, _)| *source == TotalSource::Bin)
        .map(|(name, _, _)| name.as_str())
        .chain(layout.names().iter().map(String::as_str))
        .filter(|name| by_categories.contains(name))
        .collect();
    if conflicts.is_empty() {
        return Ok(());
    }
    conflicts.sort_unstable();
    conflicts.dedup();
    Err(format!(
        "Names built from a bin are also subgroup totals: {}. Rename the bin \
         so that its compartments do not read as `{SUBGROUP_TOTAL_PREFIX}<categories>`.",
        conflicts
            .iter()
            .map(|name| format!("'{name}'"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Whether a rate or a formula parameter reads the variable `name`.
fn reads_variable<'a>(
    transition_flows: impl IntoIterator<Item = &'a TransitionFlow>,
    formula_parameters: &[(String, RateMathExpression)],
    name: &str,
) -> bool {
    transition_flows
        .into_iter()
        .map(|flow| &flow.rate_expression)
        .chain(formula_parameters.iter().map(|(_, expression)| expression))
        .any(|expression| expression.reads_variable(name))
}

/// The values that a rate or a formula parameter compares `t` with (`t < c`,
/// `c <= t`, ...), when `c` reads only numbers and `fixed_parameters`.
///
/// A rate that reads `t` can jump where `t` reaches such a value; the engines
/// in continuous time stop there (`DifferentialEquations` restarts its
/// integration, `StochasticEvents` its wait), so neither steps over a short
/// window. A comparison of `t` with the state is not a fixed position
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
            && let Ok(ast) = formula.tree()
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
    // A comparison is a chain of one operator: the language has no chain of
    // comparisons.
    if let Expr::Chain { first: left, rest } = expr
        && let [(op, right)] = rest.as_slice()
        && matches!(
            op,
            BinaryOperator::Lt | BinaryOperator::Gt | BinaryOperator::Le | BinaryOperator::Ge
        )
    {
        let is_t = |side: &Expr| matches!(side, Expr::Variable(name) if name == SPECIAL_VAR_T);
        if is_t(left) && is_fixed(right, fixed_parameters) {
            out.push(right.clone());
        } else if is_t(right) && is_fixed(left, fixed_parameters) {
            out.push(left.as_ref().clone());
        }
    }
    for operand in expr.operands() {
        collect_time_thresholds(operand, fixed_parameters, out);
    }
}

/// Whether an expression reads only numbers, `pi`, `e` and `fixed_parameters`.
fn is_fixed(expr: &Expr, fixed_parameters: &HashSet<String>) -> bool {
    match expr {
        Expr::Constant(_) => true,
        Expr::Variable(name) => {
            name == SPECIAL_VAR_PI || name == SPECIAL_VAR_E || fixed_parameters.contains(name)
        }
        _ => expr
            .operands()
            .into_iter()
            .all(|operand| is_fixed(operand, fixed_parameters)),
    }
}

/// A part of a tree as an operator of a chain reads it: a node, or the
/// start of a chain up to an operator, which applies before that operator.
/// `a * b * c` is `(a * b) * c`: the last `*` multiplies the part `a * b`
/// by `c`.
#[derive(Clone, Copy)]
enum Part<'a> {
    /// A node.
    Node(&'a Expr),
    /// The first operand of a chain and the operators after it, one or more.
    Start(&'a Expr, &'a [(BinaryOperator, Expr)]),
}

impl<'a> Part<'a> {
    /// `first` with the operators of `rest`.
    fn before(first: &'a Expr, rest: &'a [(BinaryOperator, Expr)]) -> Self {
        if rest.is_empty() {
            Part::Node(first)
        } else {
            Part::Start(first, rest)
        }
    }

    /// The node, when the part is one.
    fn node(self) -> Option<&'a Expr> {
        match self {
            Part::Node(expr) => Some(expr),
            Part::Start(..) => None,
        }
    }

    /// The operator the part applies last, what it applies to and its
    /// operand.
    fn binary(self) -> Option<(BinaryOperator, Part<'a>, &'a Expr)> {
        let (first, rest) = match self {
            Part::Node(Expr::Chain { first, rest }) => (first.as_ref(), rest.as_slice()),
            Part::Start(first, rest) => (first, rest),
            Part::Node(_) => return None,
        };
        let ((op, right), earlier) = rest
            .split_last()
            .expect("a chain, and the start of one, hold an operator");
        Some((*op, Part::before(first, earlier), right))
    }

    /// What a product multiplies, when the part is one.
    fn product(self) -> Option<(Part<'a>, Part<'a>)> {
        match self.binary()? {
            (BinaryOperator::Mul, left, right) => Some((left, Part::Node(right))),
            _ => None,
        }
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
/// for compartments. Returns `None` for constants and single parameter
/// references, which have a fast path of their own.
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
            let ast = expr.tree().ok()?;
            if let Some(specialized) =
                specialized_fast_expr(&ast, compartment_map, expression_context)
            {
                return Some(specialized);
            }
            push_fast_ops_from_ast(&ast, compartment_map, expression_context, &mut ops)?;
        }
    }
    (stack_depth(&ops) <= FAST_STACK_DEPTH).then_some(FastRateExpression::Program(ops))
}

/// The deepest stack a program of a fast rate reaches.
fn stack_depth(ops: &[FastRateOp]) -> usize {
    let (mut depth, mut deepest) = (0usize, 0usize);
    for op in ops {
        match op {
            FastRateOp::Constant(_) | FastRateOp::Slot(_) => depth += 1,
            FastRateOp::Add | FastRateOp::Sub | FastRateOp::Mul | FastRateOp::Div => depth -= 1,
            FastRateOp::Neg => {}
        }
        deepest = deepest.max(depth);
    }
    deepest
}

fn specialized_fast_expr(
    expr: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<FastRateExpression> {
    let slot = |part: Part| slot_from(part, compartment_map, expression_context);
    match Part::Node(expr).binary() {
        Some((BinaryOperator::Mul, left, right)) => {
            let right = Part::Node(right);
            if let Some((a, b, c)) = match_mul3(left, right, &slot) {
                return Some(FastRateExpression::Mul3(a, b, c));
            }
            if let Some((subtract, factor, value)) = match_one_minus_mul3(left, right, &slot) {
                return Some(FastRateExpression::OneMinusMul3 {
                    subtract,
                    factor,
                    value,
                });
            }
            Some(FastRateExpression::Mul2(slot(left)?, slot(right)?))
        }
        Some((BinaryOperator::Div, numerator, denominator)) => {
            match_mul_add_div(numerator, Part::Node(denominator), &slot).map(
                |(first, second, add_left, add_right, denominator)| FastRateExpression::MulAddDiv {
                    first,
                    second,
                    add_left,
                    add_right,
                    denominator,
                },
            )
        }
        _ => match expr {
            Expr::Variable(name) => Some(FastRateExpression::Slot(resolve_var_slot(
                name,
                compartment_map,
                expression_context,
            ))),
            _ => None,
        },
    }
}

/// `a * b * c`, read as `(a * b) * c` or `a * (b * c)`, of three slots.
fn match_mul3(
    left: Part,
    right: Part,
    slot: &impl Fn(Part) -> Option<VarSlot>,
) -> Option<(VarSlot, VarSlot, VarSlot)> {
    if let Some((inner_left, inner_right)) = left.product() {
        return Some((slot(inner_left)?, slot(inner_right)?, slot(right)?));
    }
    if let Some((inner_left, inner_right)) = right.product() {
        return Some((slot(left)?, slot(inner_left)?, slot(inner_right)?));
    }
    None
}

/// `(1 - a) * b * c`, with `1 - a` as any factor of the first product, or
/// `(1 - a) * (b * c)`.
fn match_one_minus_mul3(
    left: Part,
    right: Part,
    slot: &impl Fn(Part) -> Option<VarSlot>,
) -> Option<(VarSlot, VarSlot, VarSlot)> {
    if let Some((inner_left, inner_right)) = left.product() {
        if let Some(subtract) = match_one_minus(inner_left, slot) {
            return Some((subtract, slot(inner_right)?, slot(right)?));
        }
        if let Some(subtract) = match_one_minus(inner_right, slot) {
            return Some((subtract, slot(inner_left)?, slot(right)?));
        }
    }

    if let Some(subtract) = match_one_minus(left, slot)
        && let Some((inner_left, inner_right)) = right.product()
    {
        return Some((subtract, slot(inner_left)?, slot(inner_right)?));
    }

    None
}

/// The slot `a` of `1 - a`.
fn match_one_minus(part: Part, slot: &impl Fn(Part) -> Option<VarSlot>) -> Option<VarSlot> {
    if let Some((BinaryOperator::Sub, left, right)) = part.binary()
        && matches!(left.node(), Some(Expr::Constant(value)) if *value == 1.0)
    {
        return slot(Part::Node(right));
    }
    None
}

/// `a * b * (c + d) / e`, the product of two on either side of the sum, of
/// five slots.
fn match_mul_add_div(
    numerator: Part,
    denominator: Part,
    slot: &impl Fn(Part) -> Option<VarSlot>,
) -> Option<(VarSlot, VarSlot, VarSlot, VarSlot, VarSlot)> {
    let denominator = slot(denominator)?;
    let (left, right) = numerator.product()?;

    let (first, second, add) = if let Some((inner_left, inner_right)) = left.product() {
        (slot(inner_left)?, slot(inner_right)?, right)
    } else if let Some((inner_left, inner_right)) = right.product() {
        (slot(inner_left)?, slot(inner_right)?, left)
    } else {
        return None;
    };

    let Some((BinaryOperator::Add, add_left, add_right)) = add.binary() else {
        return None;
    };

    Some((
        first,
        second,
        slot(add_left)?,
        slot(Part::Node(add_right))?,
        denominator,
    ))
}

/// The slot of a part that is a variable.
fn slot_from(
    part: Part,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<VarSlot> {
    match part.node()? {
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
            // coverage: the check of the language rejects '!' outside the condition of an if
            UnaryOperator::Not => return None,
        },
        Expr::Chain { first, rest } => {
            push_fast_ops_from_ast(first, compartment_map, expression_context, ops)?;
            for (op, operand) in rest {
                push_fast_ops_from_ast(operand, compartment_map, expression_context, ops)?;
                ops.push(match op {
                    BinaryOperator::Add => FastRateOp::Add,
                    BinaryOperator::Sub => FastRateOp::Sub,
                    BinaryOperator::Mul => FastRateOp::Mul,
                    BinaryOperator::Div => FastRateOp::Div,
                    BinaryOperator::Mod
                    | BinaryOperator::Lt
                    | BinaryOperator::Gt
                    | BinaryOperator::Le
                    | BinaryOperator::Ge
                    | BinaryOperator::Eq
                    | BinaryOperator::Ne
                    | BinaryOperator::And
                    | BinaryOperator::Or => return None,
                });
            }
        }
        Expr::Power { .. }
        | Expr::FunctionCall { .. }
        | Expr::Conditional { .. }
        | Expr::MatrixSum { .. } => return None,
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
    } else if name == SPECIAL_VAR_STEP {
        VarSlot::Step
    } else if name == SPECIAL_VAR_T {
        VarSlot::Time
    } else if name == SPECIAL_VAR_PI {
        VarSlot::Constant(std::f64::consts::PI)
    } else if name == SPECIAL_VAR_E {
        VarSlot::Constant(std::f64::consts::E)
    } else {
        VarSlot::ParameterIndex(expression_context.parameter_index(name).expect(
            "check_model rejects a variable that is not a compartment, an engine variable or a \
             parameter of the context",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate all stratified compartment combinations.
    ///
    /// Returns a tuple of (compartments vector, compartment_map for lookups).
    fn generate_compartments(model: &Model) -> (Vec<String>, HashMap<String, usize>) {
        let compartments: Vec<String> = expand_compartments(&model.process)
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
        Bin, EngineType, InitialConditions, ModelEngine, ModelStructure, Stratification,
        StratificationCondition,
    };

    /// The aggregate structure of a test model.
    fn aggregate(model: &Model) -> &Aggregate {
        model.aggregate().unwrap()
    }

    fn aggregate_mut(model: &mut Model) -> &mut Aggregate {
        model.aggregate_mut().unwrap()
    }

    /// The initial state of a test model.
    fn model_initial_state(model: &Model) -> Result<Vec<InitialCompartment>, String> {
        initial_state(&model.process, aggregate(model))
    }

    /// Initial amount of each compartment, in the order of `compartments`.
    fn initial_amounts(model: &Model, compartments: &[String]) -> Result<Vec<f64>, String> {
        let state = model_initial_state(model)?;
        let amounts: HashMap<&str, f64> = state
            .iter()
            .map(|compartment| (compartment.name.as_str(), compartment.amount))
            .collect();
        Ok(compartments
            .iter()
            .map(|name| *amounts.get(name.as_str()).unwrap_or(&0.0))
            .collect())
    }

    fn group_stratification() -> Stratification {
        Stratification {
            id: "group".to_string(),
            categories: vec!["g1".to_string(), "g2".to_string()],
            conditions: None,
        }
    }

    /// The fractions of a stratification, as `make_model` takes them.
    type Fractions = (&'static str, [(&'static str, f64); 2]);

    fn group_fractions(g1: f64, g2: f64) -> Fractions {
        ("group", [("g1", g1), ("g2", g2)])
    }

    fn sub_fractions(s1: f64, s2: f64) -> Fractions {
        ("sub", [("s1", s1), ("s2", s2)])
    }

    fn make_model(
        stratifications: Vec<Stratification>,
        stratification_fractions: Vec<Fractions>,
        compartment_amounts: Vec<(&str, f64)>,
    ) -> Model {
        Model {
            name: "test".to_string(),
            description: None,
            version: None,
            parameters: vec![],
            engine: ModelEngine::new(EngineType::DifferenceEquations),
            process: Process {
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
                matrices: vec![],
                transitions: vec![],
            },
            structure: ModelStructure::Aggregate(Aggregate {
                variant: None,
                initial_conditions: InitialConditions {
                    bin_amounts: [
                        ("A".to_string(), Some(1000.0)),
                        ("B".to_string(), Some(0.0)),
                    ]
                    .into_iter()
                    .collect(),
                    stratification_fractions: stratification_fractions
                        .into_iter()
                        .map(|(id, fractions)| {
                            let fractions = fractions
                                .into_iter()
                                .map(|(category, fraction)| (category.to_string(), fraction))
                                .collect();
                            (id.to_string(), fractions)
                        })
                        .collect(),
                    compartment_amounts: compartment_amounts
                        .into_iter()
                        .map(|(name, amount)| (name.to_string(), amount))
                        .collect(),
                },
                neighbors: None,
            }),
        }
    }

    /// The thresholds `collect_time_thresholds` finds in `formula`, as text.
    fn thresholds_of(formula: &str) -> Vec<String> {
        let fixed: HashSet<String> = ["t_on", "k"].map(String::from).into();
        let expression = MathExpression::new(formula.to_string());
        let ast = expression.tree().unwrap();
        let mut thresholds = Vec::new();
        collect_time_thresholds(&ast, &fixed, &mut thresholds);
        thresholds.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn a_comparison_of_t_with_a_fixed_value_is_a_threshold() {
        assert_eq!(thresholds_of("if(t >= 10, k, 0)"), ["10"]);
        assert_eq!(thresholds_of("if(t_on + 1.5 > t, k, 0)"), ["t_on + 1.5"]);
        assert_eq!(
            thresholds_of("k * max(if(t < t_on, 1, 0), if(t >= 2 * pi, 1, 0))"),
            ["t_on", "2 * pi"]
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
            vec![("A_g2", 250.0)],
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
            vec![("A_g2", 100.0)],
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
        let model = make_model(vec![], vec![], vec![("Missing", 100.0)]);
        let (compartments, _) = generate_compartments(&model);
        let error = initial_amounts(&model, &compartments).unwrap_err();

        assert!(error.contains("Unknown compartment 'Missing'"));
    }

    #[test]
    fn overrides_set_the_amounts_they_name() {
        let model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![("A_g1", 400.0), ("A_g2", 600.0)],
        );
        let (compartments, _) = generate_compartments(&model);
        let amounts = initial_amounts(&model, &compartments).unwrap();

        assert_eq!(amounts, vec![400.0, 600.0, 0.0, 0.0]);
    }

    #[test]
    fn an_override_above_the_amount_of_its_bin_builds() {
        let model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![("A_g1", 0.0), ("B_g1", 5000.0)],
        );
        let (compartments, _) = generate_compartments(&model);
        let amounts = initial_amounts(&model, &compartments).unwrap();

        assert_eq!(amounts, vec![0.0, 400.0, 5000.0, 0.0]);
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
        let state = model_initial_state(&model).unwrap();
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
        model.process.transitions = vec![Transition {
            id: "ab".to_string(),
            source: vec!["A".to_string()],
            target: vec!["B".to_string()],
            accumulators: vec![],
            rate: Some(rate.to_string()),
            stratified_rates: None,
            per_compartment: None,
            impulse_at: None,
            impulse_every: None,
        }];
        model
    }

    #[test]
    fn a_rate_or_a_formula_parameter_outside_the_language_is_a_build_error() {
        use commol_core::{Parameter, ParameterValue, StratifiedRate};
        let condition = "is only allowed in the condition of if(condition, a, b)";
        let error = CompiledModel::try_from_model(&model_with_rate("k * (step > 1)"))
            .err()
            .unwrap();
        assert_eq!(
            error,
            format!("Transition 'ab': Formula 'k * (step > 1)': the comparison '>' {condition}")
        );

        let mut model = model_with_rate("k");
        model.process.transitions[0].stratified_rates = Some(vec![StratifiedRate {
            conditions: vec![StratificationCondition {
                stratification: "group".to_string(),
                category: Some("g1".to_string()),
                to: None,
            }],
            rate: "!k".to_string(),
            absolute: None,
        }]);
        assert_eq!(
            check_model(&model).unwrap_err(),
            format!("Transition 'ab': Formula '!k': the operator '!' {condition}")
        );

        let mut model = model_with_rate("k");
        model.parameters.push(Parameter {
            id: "f".to_string(),
            value: Some(ParameterValue::Formula("k > 1".to_string())),
            description: None,
        });
        assert_eq!(
            CompiledModel::try_from_model(&model).err().unwrap(),
            format!("Parameter 'f': Formula 'k > 1': the comparison '>' {condition}")
        );
    }

    #[test]
    fn a_formula_that_reads_an_undefined_variable_is_a_build_error() {
        use commol_core::{Parameter, ParameterValue};
        let error = check_model(&model_with_rate("k9 * A + k * k8")).unwrap_err();
        assert_eq!(
            error,
            "Undefined variables in transition 'ab' rate 'k9 * A + k * k8': k9, k8. \
             Available parameters: k. Available bins: A, B."
        );

        let mut model = model_with_rate("f");
        model.parameters.push(Parameter {
            id: "f".to_string(),
            value: Some(ParameterValue::Formula("k * k9".to_string())),
            description: None,
        });
        assert_eq!(
            check_model(&model).unwrap_err(),
            "Undefined variables in parameter 'f' formula 'k * k9': k9. Available \
             parameters: f, k. Available bins: A, B."
        );

        let mut model = model_with_rate("k9");
        model.parameters.clear();
        assert_eq!(
            check_model(&model).unwrap_err(),
            "Undefined variables in transition 'ab' rate 'k9': k9. Available \
             parameters: none. Available bins: A, B."
        );

        let defined = "k * A * A_g1 * A_g1_s1 * total_g1 * total * step * t * pi * e";
        assert_eq!(check_model(&model_with_rate(defined)), Ok(()));
    }

    #[test]
    fn a_category_in_a_rate_is_rejected() {
        assert_eq!(
            check_model(&model_with_rate("k * g1")).unwrap_err(),
            "Undefined variables in transition 'ab' rate 'k * g1': g1. Available \
             parameters: k. Available bins: A, B."
        );
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
        let names = subgroup_total_names(&model_with_rate("k").process);

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
        // `step % 2` is 0 at even steps and 1 at odd steps (proportional rate).
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
                check_model(&model_with_parameter(id)).expect_err("a reserved id must be rejected");
            assert!(error.contains(&format!("'{id}' ({kind})")), "{error}");
        }
        assert!(check_model(&model_with_parameter("total_max")).is_ok());
    }

    #[test]
    fn an_engine_variable_is_not_a_parameter_bin_or_accumulator_id() {
        let parameter = model_with_parameter("step");
        let mut bin = model_with_rate("k");
        bin.process.bins[1].id = "t".to_string();
        let mut accumulator = model_with_rate("k");
        accumulator.process.accumulators = vec![commol_core::Accumulator {
            id: "total".to_string(),
            name: "total".to_string(),
        }];

        for (model, conflict) in [
            (parameter, "parameter 'step'"),
            (bin, "bin 't'"),
            (accumulator, "accumulator 'total'"),
        ] {
            let error = check_model(&model).expect_err("an engine variable must be rejected");
            assert!(error.contains(conflict), "{error}");
        }
    }

    #[test]
    fn a_name_of_the_neighbors_is_not_a_parameter_bin_or_accumulator_id() {
        let parameter = model_with_parameter("neighbors_A");
        let mut bin = model_with_rate("k");
        bin.process.bins[1].id = "neighbors".to_string();

        for (model, conflict) in [
            (parameter, "parameter 'neighbors_A'"),
            (bin, "bin 'neighbors'"),
        ] {
            let error = check_model(&model).expect_err("a name of the neighbors is reserved");
            assert!(error.contains(conflict), "{error}");
        }
    }

    #[test]
    fn a_rate_reads_the_neighbors_of_a_bin_only_when_the_structure_has_them() {
        let unknown = model_with_rate("k * neighbors_C");
        let error = check_model(&unknown).unwrap_err();
        assert!(error.contains("'C' is not a bin"), "{error}");

        let mut model = model_with_rate("k * neighbors_B / neighbors");
        let error = check_model(&model).unwrap_err();
        assert!(
            error.contains("Complete()") && error.contains("Groups("),
            "{error}"
        );

        aggregate_mut(&mut model).neighbors = Some(commol_core::Neighbors::Complete);
        assert_eq!(check_model(&model), Ok(()));
    }

    #[test]
    fn a_formula_parameter_does_not_read_the_neighbors() {
        use commol_core::{Parameter, ParameterValue};
        let mut model = model_with_rate("k");
        aggregate_mut(&mut model).neighbors = Some(commol_core::Neighbors::Complete);
        model.parameters.push(Parameter {
            id: "f".to_string(),
            value: Some(ParameterValue::Formula("k * neighbors".to_string())),
            description: None,
        });

        let error = check_model(&model).unwrap_err();
        assert!(
            error.contains("'f'") && error.contains("transition rate"),
            "{error}"
        );
    }

    #[test]
    fn a_bin_name_that_reads_as_a_subgroup_total_is_rejected() {
        let group = Stratification {
            id: "group".to_string(),
            categories: vec!["y".to_string(), "o".to_string()],
            conditions: None,
        };
        let kind = Stratification {
            id: "type".to_string(),
            categories: vec!["y".to_string(), "o".to_string()],
            conditions: None,
        };
        let mut model = make_model(vec![group, kind], vec![], vec![]);
        model.process.bins[0].id = "total_y".to_string();

        let error = check_model(&model).expect_err("total_y_y names two sums");
        assert!(error.contains("'total_y_y'"), "{error}");
    }

    #[test]
    fn a_bin_that_starts_with_the_prefix_but_names_no_subgroup_builds() {
        let mut model = make_model(vec![group_stratification()], vec![], vec![]);
        model.process.bins[0].id = "total_x".to_string();

        assert!(check_model(&model).is_ok());
    }

    fn set_bin_amount(model: &mut Model, bin: &str, amount: f64) {
        aggregate_mut(model)
            .initial_conditions
            .bin_amounts
            .insert(bin.to_string(), Some(amount));
    }

    #[test]
    fn a_real_bin_amount_splits_into_its_categories() {
        let mut model = make_model(
            vec![group_stratification()],
            vec![group_fractions(0.6, 0.4)],
            vec![],
        );
        set_bin_amount(&mut model, "A", 2.5);
        let state = model_initial_state(&model).unwrap();

        let amounts: Vec<f64> = state.iter().map(|compartment| compartment.amount).collect();
        assert_eq!(amounts, vec![1.5, 1.0, 0.0, 0.0]);
        assert!(CompiledModel::try_from_model(&model).is_ok());
    }

    #[test]
    fn bins_that_all_start_at_zero_build() {
        let mut model = make_model(vec![], vec![], vec![]);
        set_bin_amount(&mut model, "A", 0.0);
        let compiled = CompiledModel::try_from_model(&model).unwrap();

        assert_eq!(compiled.initial_state(), [0.0, 0.0]);
        assert_eq!(compiled.total(), 0.0);
    }

    #[test]
    fn a_bin_that_is_not_whole_is_a_build_error_of_a_stochastic_engine() {
        let mut model = make_model(vec![], vec![], vec![]);
        set_bin_amount(&mut model, "B", 10.5);
        model.engine = ModelEngine::new(EngineType::StochasticEvents);

        let error = check_model(&model).unwrap_err();

        assert!(
            error.contains("the amount of each bin must be a whole number"),
            "{error}"
        );
        assert!(error.contains("'B' (10.5)"), "{error}");
        assert!(!error.contains("'A'"), "{error}");
    }

    #[test]
    fn a_stochastic_engine_names_the_compartments_that_are_not_whole() {
        let mut model = make_model(
            vec![group_stratification()],
            vec![group_fractions(1.0 / 3.0, 2.0 / 3.0)],
            vec![],
        );
        model.engine = ModelEngine::new(EngineType::StochasticEvents);

        let error = check_model(&model).unwrap_err();

        assert!(error.contains("'A_g1' (333.3"), "{error}");
        assert!(error.contains("Name a rounding"), "{error}");
    }
}
