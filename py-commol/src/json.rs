//! The port of the bridge: every data object of the Python API crosses as
//! the JSON of its `model_dump()`, which [`read`] reads with serde into the
//! type the engine owns ([`commol_core::json`]). An error is a `ValueError`
//! that names the path of its field.
//!
//! [`as_read`] reads an object by the name of its Python class and writes
//! back what it read: a test checks with it that each data class and its
//! Rust type agree. [`check`] only reads it: a data object checks with it,
//! when it is built, every rule that joins its fields.

use commol_calibration::{FitRecord, Observation, OptimizerSettings, ProblemSettings};
use commol_inference::settings::{
    PriorsSettings, conformal, identifiability, local, profiles, resampling, sensitivity, smc,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::de::DeserializeOwned;

/// Read `text`, the JSON of a setting, into `T`.
///
/// # Errors
///
/// A `ValueError` with the message of [`commol_core::json::read`]: the path
/// of the field, and the message of the engine.
pub(crate) fn read<T: DeserializeOwned>(text: &str) -> PyResult<T> {
    commol_core::json::read(text).map_err(PyValueError::new_err)
}

/// Read `text` into `T` and write what it read.
fn read_back<T: DeserializeOwned + serde::Serialize>(text: &str) -> PyResult<String> {
    read::<T>(text).map(|value| commol_core::json::write(&value))
}

/// The error of a class whose JSON the engine does not read.
fn unknown(object: &str) -> PyErr {
    PyValueError::new_err(format!(
        "The engine reads no object '{object}'; it reads {OBJECTS:?}"
    ))
}

/// The Python class of each object the engine reads, and its reader.
macro_rules! objects {
    ($($name:literal => $type:ty),* $(,)?) => {
        /// The names of the Python classes whose JSON the engine reads.
        const OBJECTS: &[&str] = &[$($name),*];

        /// Read `text` as the object of the class `object`, and write it back.
        fn read_object(object: &str, text: &str) -> PyResult<String> {
            match object {
                $($name => read_back::<$type>(text),)*
                _ => Err(unknown(object)),
            }
        }

        /// Read `text` as the object of the class `object`.
        fn check_object(object: &str, text: &str) -> PyResult<()> {
            match object {
                $($name => read::<$type>(text).map(drop),)*
                _ => Err(unknown(object)),
            }
        }
    };
}

objects! {
    "CalibrationProblem" => ProblemSettings,
    "ObservedDataPoint" => Observation,
    "CalibrationParameter" => commol_calibration::CalibrationParameter,
    "CalibrationConstraint" => commol_calibration::CalibrationConstraint,
    "CalibrationResult" => FitRecord,
    "NelderMeadConfig" => OptimizerSettings,
    "ParticleSwarmConfig" => OptimizerSettings,
    "Quantile" => commol_calibration::ReplicateSummary,
    "NormalObservations" => commol_calibration::ObservationModel,
    "LogNormalObservations" => commol_calibration::ObservationModel,
    "StudentTObservations" => commol_calibration::ObservationModel,
    "PoissonObservations" => commol_calibration::ObservationModel,
    "NegativeBinomialObservations" => commol_calibration::ObservationModel,
    "BinomialObservations" => commol_calibration::ObservationModel,
    "Uniform" => commol_inference::Family,
    "LogUniform" => commol_inference::Family,
    "Beta" => commol_inference::Family,
    "Normal" => commol_inference::Family,
    "LogNormal" => commol_inference::Family,
    "Gamma" => commol_inference::Family,
    "Priors" => PriorsSettings,
    "ValueAt" => commol_inference::quantity::Quantity,
    "Gain" => commol_inference::quantity::Quantity,
    "Peak" => commol_inference::quantity::Quantity,
    "PeakPosition" => commol_inference::quantity::Quantity,
    "Trajectory" => commol_inference::quantity::Quantity,
    "Ratio" => commol_inference::quantity::Quantity,
    "BayesianBootstrap" => resampling::Bootstrap,
    "ParametricBootstrap" => resampling::Bootstrap,
    "ResidualBootstrap" => resampling::Bootstrap,
    "WildBootstrap" => resampling::Bootstrap,
    "BlockJackknife" => resampling::BlockJackknifeSettings,
    "NoNoise" => resampling::NoiseSettings,
    "ResidualNoise" => resampling::NoiseSettings,
    "DistributionNoise" => resampling::NoiseSettings,
    "InformationMatchingEta" => smc::EtaCalibration,
    "BootstrapCoverageEta" => smc::EtaCalibration,
    "LossDistance" => smc::DistanceSettings,
    "SeriesDistance" => smc::DistanceSettings,
    "GibbsPosterior" => smc::GibbsPosteriorSettings,
    "BayesianPosterior" => smc::BayesianPosteriorSettings,
    "ABCSMC" => smc::AbcSmcSettings,
    "ABCRejection" => smc::AbcRejectionSettings,
    "Laplace" => local::LaplaceSettings,
    "Sandwich" => local::SandwichSettings,
    "LikelihoodTarget" => local::MixtureTarget,
    "GibbsTarget" => local::MixtureTarget,
    "LaplaceMixture" => local::LaplaceMixtureSettings,
    "SplitConformal" => conformal::SplitConformalSettings,
    "RollingConformal" => conformal::RollingConformalSettings,
    "JackknifePlus" => conformal::JackknifePlusSettings,
    "EnbPI" => conformal::EnbPiSettings,
    "AdaptiveConformal" => conformal::AdaptiveConformalSettings,
    "Threshold" => profiles::Threshold,
    "PredictionSpec" => profiles::PredictionSpec,
    "ProfileLikelihood" => profiles::ProfileLikelihoodSettings,
    "PredictionProfile" => profiles::PredictionProfileSettings,
    "StructuralRank" => identifiability::StructuralRankSettings,
    "Collinearity" => identifiability::CollinearitySettings,
    "SampleIdentifiability" => identifiability::SampleIdentifiabilitySettings,
    "StartsIdentifiability" => identifiability::StartsIdentifiabilitySettings,
    "InputSpace" => sensitivity::InputSpaceSettings,
    "Propagation" => sensitivity::PropagationSettings,
    "PRCC" => sensitivity::PrccSettings,
    "Morris" => sensitivity::MorrisSettings,
    "Sobol" => sensitivity::SobolSettings,
    "PolynomialChaos" => sensitivity::PolynomialChaosSettings,
}

/// The JSON of `text` as the engine reads an object of the Python class
/// `object`, written back
///
/// Args:
///     object: The name of the Python class, such as "CalibrationProblem"
///     text: The JSON of its `model_dump()`
///
/// Raises:
///     ValueError: If the engine reads no such class, or rejects the JSON;
///         the message names the path of the field
#[pyfunction]
fn as_read(object: &str, text: &str) -> PyResult<String> {
    read_object(object, text)
}

/// Check `text` as the engine reads an object of the Python class `object`:
/// every rule of the object, with no copy written back
///
/// Args:
///     object: The name of the Python class, such as "CalibrationProblem"
///     text: The JSON of its `model_dump()`
///
/// Raises:
///     ValueError: If the engine reads no such class, or rejects the JSON;
///         the message names the path of the field
#[pyfunction]
fn check(object: &str, text: &str) -> PyResult<()> {
    check_object(object, text)
}

/// The names of the Python classes whose JSON the engine reads
#[pyfunction]
fn objects() -> Vec<&'static str> {
    OBJECTS.to_vec()
}

/// Register the functions of the port.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(as_read, m)?)?;
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(objects, m)?)?;
    Ok(())
}
