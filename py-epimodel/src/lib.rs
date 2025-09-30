use pyo3::prelude::*;

/// Core data structures for epidemiological models.
fn core_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<epimodel_core::Model>()?;
    m.add_class::<epimodel_core::Population>()?;
    m.add_class::<epimodel_core::DiseaseState>()?;
    m.add_class::<epimodel_core::Stratification>()?;
    m.add_class::<epimodel_core::Transition>()?;
    m.add_class::<epimodel_core::Parameter>()?;
    m.add_class::<epimodel_core::InitialConditions>()?;
    m.add_class::<epimodel_core::Condition>()?;
    m.add_class::<epimodel_core::Rule>()?;
    m.add_class::<epimodel_core::LogicOperator>()?;
    m.add_class::<epimodel_core::ModelTypes>()?;
    m.add_class::<epimodel_core::VariablePrefixes>()?;
    m.add_class::<epimodel_core::Dynamics>()?;
    Ok(())
}

/// Difference equation solver.
fn difference_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<epimodel_difference::DifferenceEquations>()?;
    Ok(())
}

/// High-performance mathematical epidemiology library.
#[pymodule]
fn epimodel_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let core_mod = PyModule::new(m.py(), "core")?;
    core_module(&core_mod)?;
    m.add_submodule(&core_mod)?;

    let difference_mod = PyModule::new(m.py(), "difference")?;
    difference_module(&difference_mod)?;
    m.add_submodule(&difference_mod)?;

    Ok(())
}
