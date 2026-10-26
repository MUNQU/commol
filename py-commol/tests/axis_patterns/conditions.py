"""Conditions and stratified rates of axis-pattern schedules."""

from commol import AxisPattern, StratificationCondition, StratifiedRate


def condition(
    stratification: str, category: str, *, to: str | None = None
) -> StratificationCondition:
    """A condition on one category, optionally moving to another one."""
    return StratificationCondition(
        stratification=stratification, category=category, to=to
    )


def stratified_rates(rate: AxisPattern) -> list[StratifiedRate]:
    """The stratified rates a schedule hands to ``add_transition``."""
    rates = rate._builder_stratified_rates()
    assert rates is not None
    return rates


def rate_string(stratified_rate: StratifiedRate) -> str:
    """The rate of a stratified rate, which a schedule always writes as text."""
    rate = stratified_rate.rate
    assert isinstance(rate, str)
    return rate
