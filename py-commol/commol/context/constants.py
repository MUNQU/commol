from typing import Final, Literal, get_args

# Calibration Parameter Types

CalibrationParameterType = Literal[
    "parameter",
    "initial_condition",
    "scale",
]
"""
Type of value being calibrated.

Values
------
"parameter"
    Model parameter
"initial_condition"
    Initial population in a compartment
"scale"
    Scaling factor for observed data
"""

_param_type_args = get_args(CalibrationParameterType)
# Model parameter type
PARAM_TYPE_PARAMETER: Final = _param_type_args[0]
# Initial population in a compartment
PARAM_TYPE_INITIAL_CONDITION: Final = _param_type_args[1]
# Scaling factor for observed data
PARAM_TYPE_SCALE: Final = _param_type_args[2]

# Loss Functions

LossFunction = Literal[
    "sse",
    "rmse",
    "mae",
    "weighted_sse",
]
"""
Available loss functions for calibration.

Values
------
"sse"
    Sum of Squared Errors (SSE)
"rmse"
    Root Mean Squared Error (RMSE)
"mae"
    Mean Absolute Error (MAE)
"weighted_sse"
    Weighted Sum of Squared Errors
"""

_loss_args = get_args(LossFunction)
# Sum of Squared Errors loss function
LOSS_SSE: Final = _loss_args[0]
# Root Mean Squared Error loss function
LOSS_RMSE: Final = _loss_args[1]
# Mean Absolute Error loss function
LOSS_MAE: Final = _loss_args[2]
# Weighted Sum of Squared Errors loss function
LOSS_WEIGHTED_SSE: Final = _loss_args[3]

# Optimization Algorithms

OptimizationAlgorithm = Literal["nelder_mead", "particle_swarm"]
"""
Available optimization algorithms.

Values
------
"nelder_mead"
    Nelder-Mead simplex algorithm
"particle_swarm"
    Particle Swarm Optimization
"""

_algo_args = get_args(OptimizationAlgorithm)
# Nelder-Mead simplex algorithm
OPT_ALG_NELDER_MEAD: Final = _algo_args[0]
# Particle Swarm Optimization algorithm
OPT_ALG_PARTICLE_SWARM: Final = _algo_args[1]
