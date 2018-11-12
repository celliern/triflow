#!/usr/bin/env python
# coding=utf-8

import attr
from sympy import Derivative
from .variables import IndependentVariable
from itertools import product, chain


@attr.s()
class FiniteDifferenceScheme:
    scheme = attr.ib(type=str, default="centered")
    accuracy = attr.ib(type=int, default=2)
    offset = attr.ib(type=int, default=0)
    pattern = attr.ib(default=lambda derivative, wrt: True)

    def as_finite_diff(self, derivative, ivar):
        try:
            order = dict(derivative.variable_count)[ivar.symbol]
        except KeyError:
            return derivative
        if self.scheme == "centered":
            n = (order + 1) // 2 + self.accuracy - 2
            points = [
                ivar.symbol + i * ivar.step
                for i in range(-n + self.offset, n + 1 + self.offset)
            ]
        elif self.scheme == "right":
            n = self.accuracy + order
            points = [
                ivar.symbol + i * ivar.step
                for i in range(0 + self.offset, n + self.offset)
            ]
        elif self.scheme == "left":
            n = self.accuracy + order
            points = [
                ivar.symbol + i * ivar.step
                for i in range(-(n - 1) + self.offset, 1 + self.offset)
            ]
        else:
            raise NotImplementedError("scheme should be one of 'centered', 'left' or 'right'.")
        return derivative.as_finite_difference(points=points, wrt=ivar.symbol)

    def relevant_derivatives(self, expr):
        derivatives = expr.atoms(Derivative)
        return list(
            [
                (deriv, ivar)
                for deriv, ivar in chain(
                    *[
                        product([derivative], derivative.variables)
                        for derivative in derivatives
                    ]
                )
                if self.pattern(deriv, ivar)
            ]
        )

    def pop_derivative(self, expr):
        derivative, ivar = list(self.relevant_derivatives(expr))[0]
        ivar = IndependentVariable(str(ivar))
        return derivative, ivar

    def apply(self, expr):
        while self.relevant_derivatives(expr):
            derivative, ivar = self.pop_derivative(expr)
            discrete_derivative = self.as_finite_diff(derivative, ivar)
            expr = expr.replace(derivative, discrete_derivative)
        return expr


def chain_schemes(schemes, expr, default_scheme="centered", default_accuracy=2):
    for scheme in [*schemes, FiniteDifferenceScheme(default_scheme, default_accuracy)]:
        expr = scheme.apply(expr)
    return expr
