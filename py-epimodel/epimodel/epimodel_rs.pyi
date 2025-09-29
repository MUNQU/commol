from typing import Protocol


class RustModelProtocol(Protocol):
    """Rust-side Model class with JSON deserialization."""
    @staticmethod
    def from_json(json_string: str) -> "RustModelProtocol": ...


class DifferenceEquationsProtocol(Protocol):
    def __init__(self, model: RustModelProtocol) -> None: ...
    def step(self, dt: float) -> None: ...


class CoreModule(Protocol):
    Model: type[RustModelProtocol]


class DifferenceModule(Protocol):
    DifferenceEquations: type[DifferenceEquationsProtocol]


class RustEpiModelModule(Protocol):
    core: CoreModule
    difference: DifferenceModule


core: CoreModule
difference: DifferenceModule
rust_epimodel: RustEpiModelModule
