from __future__ import annotations

import pytest

from trader.mtf_strategies.gate import evaluate_all_or_nothing

_ALL_TRUE = {"a": True, "b": True, "c": True, "d": True, "e": True}


def test_all_conditions_true_passes():
    result = evaluate_all_or_nothing(dict(_ALL_TRUE))
    assert result.passed is True
    assert result.failed_condition is None


@pytest.mark.parametrize("missing", list(_ALL_TRUE.keys()))
def test_any_single_missing_condition_blocks_the_signal(missing):
    """Todo-o-nada: cada condición individual, faltando sola (todas las demás
    True), debe bloquear -- sin puntaje parcial que compense."""
    conditions = dict(_ALL_TRUE)
    conditions[missing] = False

    result = evaluate_all_or_nothing(conditions)

    assert result.passed is False
    assert result.failed_condition == missing


def test_reports_the_first_failing_condition_in_insertion_order():
    conditions = {"a": True, "b": False, "c": False}  # both b and c fail
    result = evaluate_all_or_nothing(conditions)
    assert result.failed_condition == "b"  # the FIRST one, not "c" or "both"


def test_empty_conditions_passes_vacuously():
    result = evaluate_all_or_nothing({})
    assert result.passed is True
