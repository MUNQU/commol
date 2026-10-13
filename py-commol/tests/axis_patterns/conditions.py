"""Conditions and stratified rates of axis-pattern schedules."""

from commol import AxisPattern
from commol.api.patterns import ConditionDict, StratifiedRateDict


def condition(
    stratification: str, category: str, *, to: str | None = None
) -> ConditionDict:
    """A condition on one category, optionally moving to another one."""
    result: ConditionDict = {"stratification": stratification, "category": category}
    if to is not None:
        result["to"] = to
    return result


def stratified_rates(rate: AxisPattern) -> list[StratifiedRateDict]:
    """The stratified rates a schedule hands to ``add_transition``."""
    rates = rate._builder_stratified_rates()
    assert rates is not None
    return rates


def rate_string(stratified_rate: StratifiedRateDict) -> str:
    """The rate of a stratified rate, which a schedule always writes as text."""
    rate = stratified_rate["rate"]
    assert isinstance(rate, str)
    return rate
