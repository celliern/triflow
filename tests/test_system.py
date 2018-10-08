#!/usr/bin/env python
# coding=utf-8

import string
from itertools import product

import numpy as np
import pytest
from sympy import Idx, IndexedBase, Symbol, Function, sympify, Min, Max
from triflow.core.system import (
    DependentVariable,
    IndependentVariable,
    PDEquation,
    _build_sympy_namespace,
    _apply_centered_scheme,
)


@pytest.mark.parametrize("ivar_name", ["x", "y", "z", "t", "xy", "u", "β"])
def test_independent_variable(ivar_name):
    if ivar_name not in string.ascii_letters or len(ivar_name) > 1:
        # only one character on ascii letters are accepted
        with pytest.raises(ValueError):
            ivar = IndependentVariable(ivar_name)
        return
    ivar = IndependentVariable(ivar_name)

    # indep var behaviour
    assert ivar.name == ivar_name
    assert ivar.symbol == Symbol(ivar_name)
    assert ivar.discrete == IndexedBase("%s_i" % ivar_name)
    assert ivar.N == Symbol("N_%s" % ivar_name, integer=True)
    assert ivar.idx == Idx("%s_idx" % ivar_name, ivar.N)
    assert ivar.step == Symbol("d%s" % ivar_name)
    assert ivar.step_value == (
        ivar.discrete[ivar.idx.upper] - ivar.discrete[ivar.idx.lower]
    ) / (ivar.N - 1)
    assert ivar.bound == (ivar.idx.lower, ivar.idx.upper)

    # should be hashable
    assert isinstance(hash(ivar), int)


@pytest.mark.parametrize("ivar_name", ["x", "y", "z", "t"])
def test_domain_independent_variable(ivar_name):
    ivar = IndependentVariable(ivar_name)
    assert ivar.domain(5) == "bulk"
    assert ivar.domain(0) == "bulk"
    assert ivar.domain(ivar.N - 1) == "bulk"
    assert ivar.domain(ivar.N) == "right"
    assert ivar.domain(-1) == "left"

    assert ivar.is_in_bulk(5)
    assert ivar.is_in_bulk(0)
    assert ivar.is_in_bulk(ivar.N - 1)
    assert not ivar.is_in_bulk(ivar.N)
    assert not ivar.is_in_bulk(-1)


@pytest.mark.parametrize("dvar_", ["U", "V", "T", "ϕ"])
@pytest.mark.parametrize("ivars", [tuple(), ("x",), ("x", "y")])
def test_dependent_variable(dvar_, ivars):
    dvar_name = "%s%s" % (dvar_, "(%s)" % ", ".join(ivars) if ivars else "")
    dvar = DependentVariable(dvar_name)
    ivars = tuple([IndependentVariable(ivar) for ivar in ivars])

    assert dvar.name == dvar_
    assert dvar.ivars == dvar.independent_variables
    assert dvar.ivars == ivars

    assert dvar.symbol == Function(dvar_) if ivars else Symbol(dvar_)
    assert dvar.discrete == IndexedBase(dvar_) if ivars else Symbol(dvar_)
    assert (
        dvar.discrete_i == IndexedBase(dvar_)[dvar.i_idxs] if ivars else Symbol(dvar_)
    )

    assert len(dvar) == len(ivars)

    assert dvar.i_symbols == tuple([ivar.symbol for ivar in ivars])
    assert dvar.i_steps == tuple([ivar.step for ivar in ivars])
    assert dvar.i_step_values == tuple([ivar.step_value for ivar in ivars])
    assert dvar.i_discs == tuple([ivar.discrete for ivar in ivars])
    assert dvar.i_idxs == tuple([ivar.idx for ivar in ivars])
    assert dvar.i_Ns == tuple([ivar.N for ivar in ivars])
    assert dvar.i_bounds == tuple([ivar.bound for ivar in ivars])

    # should be hashable
    assert isinstance(hash(dvar), int)


def test_domain_dependent_variable():
    dvar = DependentVariable("U(x, y)")

    N_x, N_y = dvar.i_Ns

    assert dvar.domains(-1, -1) == ("left", "left")
    assert dvar.domains(-1, 0) == ("left", "bulk")
    assert dvar.domains(3, -2) == ("bulk", "left")
    assert dvar.domains(0, 0) == ("bulk", "bulk")
    assert dvar.domains(5, 2) == ("bulk", "bulk")
    assert dvar.domains(N_x, -2) == ("right", "left")
    assert dvar.domains(5, N_y) == ("bulk", "right")
    assert dvar.domains(N_x - 1, N_y - 1) == ("bulk", "bulk")
    assert dvar.domains(N_x, N_y) == ("right", "right")

    assert not dvar.is_in_bulk(-1, -1)
    assert not dvar.is_in_bulk(-1, 0)
    assert not dvar.is_in_bulk(3, -2)
    assert dvar.is_in_bulk(0, 0)
    assert dvar.is_in_bulk(5, 2)
    assert not dvar.is_in_bulk(N_x, -2)
    assert not dvar.is_in_bulk(5, N_y)
    assert dvar.is_in_bulk(N_x - 1, N_y - 1)
    assert not dvar.is_in_bulk(N_x, N_y)


def test_pde_equation_1D():
    x = IndependentVariable("x")
    U = DependentVariable("U(x)")

    pde = PDEquation("dxU", ["U(x)"])
    assert (
        pde.fdiff - (U.discrete[x.idx + 1] - U.discrete[x.idx - 1]) / (2 * x.step)
    ).expand() == 0

    pde = PDEquation("dxU", ["U(x)"], scheme="left", accuracy_order=1)
    assert (
        pde.fdiff - (U.discrete[x.idx] - U.discrete[x.idx - 1]) / x.step
    ).expand() == 0

    pde = PDEquation("dxU", ["U(x)"], scheme="right", accuracy_order=1)
    assert (
        pde.fdiff - (U.discrete[x.idx + 1] - U.discrete[x.idx]) / x.step
    ).expand() == 0


def test_pde_equation_2D():
    x = IndependentVariable("x")
    y = IndependentVariable("y")
    U = DependentVariable("U(x, y)")

    pde = PDEquation("dxxU + dyyU", ["U(x, y)"])

    fdiff = (
        (
            U.discrete[x.idx + 1, y.idx]
            - 2 * U.discrete[x.idx, y.idx]
            + U.discrete[x.idx - 1, y.idx]
        )
        / x.step ** 2
        + (
            U.discrete[x.idx, y.idx + 1]
            - 2 * U.discrete[x.idx, y.idx]
            + U.discrete[x.idx, y.idx - 1]
        )
        / y.step ** 2
    )
    assert (pde.fdiff - fdiff).expand() == 0


def test_build_sympy_namespace():

    x = IndependentVariable("x")
    y = IndependentVariable("y")
    U = DependentVariable("U(x, y)")

    ns = _build_sympy_namespace("dxxU + dyyU", (x, y), (U,), [])

    assert "x" in ns.keys()
    assert "y" in ns.keys()
    assert "dxxU" in ns.keys()
    assert "dyyU" in ns.keys()

    ns = _build_sympy_namespace("dxxU + dy(U + dyU + dxU)", (x, y), (U,), [])

    assert "x" in ns.keys()
    assert "y" in ns.keys()
    assert "dxU" in ns.keys()
    assert "dyU" in ns.keys()
    assert "dxxU" in ns.keys()
    assert "dy" in ns.keys()


def test_apply_scheme():
    x = IndependentVariable("x")
    y = IndependentVariable("y")
    U = DependentVariable("U(x, y)")
    pde = PDEquation("dxxU + dyyU", ["U(x, y)"])

    x_applied = _apply_centered_scheme(
        2, x, U.symbol(x.symbol, y.symbol).diff(x.symbol, 2), 2, pde.symbolic_equation
    )
    y_applied = _apply_centered_scheme(
        2, y, U.symbol(x.symbol, y.symbol).diff(y.symbol, 2), 2, pde.symbolic_equation
    )

    assert (
        y_applied
        - sympify(
            "Derivative(U(x, y), (x, 2)) - 2*U(x, y)/dy**2 + "
            "U(x, -dy + y)/dy**2 + U(x, dy + y)/dy**2"
        )
    ).expand() == 0

    assert (
        x_applied
        - sympify(
            "Derivative(U(x, y), (y, 2)) - 2*U(x, y)/dx**2 + "
            "U(-dx + x, y)/dx**2 + U(dx + x, y)/dx**2"
        )
    ).expand() == 0

    xy_applied = _apply_centered_scheme(
        2, y, U.symbol(x.symbol, y.symbol).diff(y.symbol, 2), 2, x_applied
    )
    yx_applied = _apply_centered_scheme(
        2, x, U.symbol(x.symbol, y.symbol).diff(x.symbol, 2), 2, y_applied
    )
    assert (xy_applied - yx_applied).expand() == 0

def test_upwind():
    x = IndependentVariable("x")
    U = DependentVariable("U(x)")

    c = Symbol("c")
    ap = Max(c, 0)
    am = Min(c, 0)

    pde = PDEquation("upwind(c, U, x, 1)", ["U(x)"], parameters=["c"])
    up = (U.discrete[x.idx + 1] - U.discrete[x.idx]) / x.step
    um = (U.discrete[x.idx] - U.discrete[x.idx - 1]) / x.step
    assert (ap * up + am * um - pde.fdiff).expand() == 0

    pde = PDEquation("upwind(c, U, x, 2)", ["U(x)"], parameters=["c"])
    up = (-U.discrete[x.idx + 2] + 4 * U.discrete[x.idx + 1] - 3 * U.discrete[x.idx]) / (2 * x.step)
    um = (3 * U.discrete[x.idx] - 4 * U.discrete[x.idx - 1] + U.discrete[x.idx - 2]) / (2 * x.step)
    assert (ap * up + am * um - pde.fdiff).expand() == 0

    pde = PDEquation("upwind(c, U, x, 3)", ["U(x)"], parameters=["c"])
    up = (-U.discrete[x.idx + 2] + 6 * U.discrete[x.idx + 1] - 3 * U.discrete[x.idx] - 2 * U.discrete[x.idx - 1]) / (6 * x.step)
    um = (2 * U.discrete[x.idx + 1] + 3 * U.discrete[x.idx] - 6 * U.discrete[x.idx - 1] + U.discrete[x.idx - 2]) / (6 * x.step)
    assert (ap * up + am * um - pde.fdiff).expand() == 0