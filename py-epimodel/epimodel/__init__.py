from epimodel import constants, context

try:
    import epimodel_rs
except ImportError as e:
    print(f"Error importing Rust extension: {e}")
    epimodel_rs = None

RUST_AVAILABLE = epimodel_rs is not None

from epimodel.api import ModelBuilder, ModelLoader  # noqa: E402

__all__ = [
    "ModelBuilder",
    "ModelLoader",
    "constants", 
    "context",
    "epimodel_rs",
    "RUST_AVAILABLE",
]