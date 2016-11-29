#!/usr/bin/env python3
# coding=utf8

import logging
from logging import info, debug

import numpy as np
import sympy as sp
from sympy.polys.orthopolys import chebyshevt_poly as cbpt


def model(Ny):
    """

    Parameters
    ----------
    Ny :


    Returns
    -------


    """

    Ts = sp.symbols('T:%i_i' % Ny)
    y = ((-np.cos(np.pi * np.arange(Ny) / (Ny - 1))) + 1) / 2
    ys = sp.Symbol('y')

    a = sp.symbols('a_i:%i' % Ny)

    approx_U = sum([a[n] * cbpt(n, ys * 2 - 1)
                    for n
                    in range(Ny)])
    residuals = [sp.N(approx_U.subs(ys, y[i]) - Ts[i])
                 for i in range(1, Ny - 1)]
    accoeff = sp.solve([res for res in residuals], a[2:])

    def solve_bdc(bdcs):
        """

        Parameters
        ----------
        bdcs :


        Returns
        -------


        """
        y = sp.Symbol('y')
        U = sp.Symbol('U')
        left = sp.Function('left')
        right = sp.Function('right')
        dx = sp.Function('dx')
        wild = sp.Wild('a')
        Uspec = (sum([a[n] * cbpt(n, y * 2 - 1)
                      for n
                      in range(Ny)])
                 .expand()
                 .subs(accoeff))
        bdc_sys = []
        for bdc in bdcs:
            bdc = sp.S(bdc).subs(U, Uspec)
            while bdc.find(dx(wild)) != set():
                for deriv in bdc.find(dx(wild)):
                    deriv = deriv.match(dx(wild))[wild]
                    bdc = bdc.subs(dx(deriv), deriv.diff(y))
            lefts_bdc = bdc.find(left(wild))
            for left_bdc in lefts_bdc:
                left_bdc = left_bdc.match(left(wild))[wild]
                bdc = bdc.subs(left(left_bdc), left_bdc.subs(y, 0))
            rights_bdc = bdc.find(right(wild))
            for right_bdc in rights_bdc:
                right_bdc = right_bdc.match(right(wild))[wild]
                bdc = bdc.subs(right(right_bdc), right_bdc.subs(y, 1))
            bdc_sys.append(bdc)
        return sp.solve(bdc_sys, a[:2])

    wphi = ("(1 + dxh**2) * right(dx(U))"
            "- h * dxh * dxTtop"
            "+ h*sqrt(1 + dxh**2) * B * right(U)")

    bdccoeff = solve_bdc([wphi, 'left(U) - 1'])
    # return bdccoeff

    info('mise en place des variables')

    dx = sp.symbols('dx')
    h, q = sp.symbols('h_i, q_i')
    hm1, hm2, hm3 = sp.symbols('h_im1:4')
    hp1, hp2, hp3 = sp.symbols('h_ip1:4')
    qm1, qm2, qm3 = sp.symbols('q_im1:4')
    qp1, qp2, qp3 = sp.symbols('q_ip1:4')

    s = 0
    dxs = 0
    dxxs = 0
    dxxxs = 0

    (Tsm1, Tsm2, Tsm3) = zip(*[sp.symbols('T%i_im1:4' % ids)
                               for ids in range(Ny)])
    (Tsp1, Tsp2, Tsp3) = zip(*[sp.symbols('T%i_ip1:4' % ids)
                               for ids in range(Ny)])

    info('Calcul des dérivés en x')
    dxh = (-.5 * hm1 + .5 * hp1) / dx
    dxq = (-.5 * qm1 + .5 * qp1) / dx

    dxxq = (1 * qm1 - 2 * q + 1 * qp1) / dx**2
    dxxh = (hm1 - 2 * h + hp1) / dx**2

    dxxxh = (-.5 * hm2 + hm1 - hp1 + .5 * hp2) / dx**3

    info('Calcul des polynomes de tchebychev')

    bdccoeff_new = {key: bdccoeff[key]
                    .subs(sp.Symbol('dxs'), dxs)
                    .subs(sp.Symbol('dxh'), dxh)
                    .subs(sp.Symbol('h'), h)
                    .subs(sp.Symbol('dxTtop'),
                          (Tsp1[-1] - Tsm1[-1]) / (2 * dx))
                    .subs(accoeff)
                    for key in a[:2]}
    accoeff_new = {key: accoeff[key].subs(bdccoeff_new) for key in a[2:]}

    ccoeff = {}
    ccoeff.update(bdccoeff_new)
    ccoeff.update(accoeff_new)

    Tchebm1 = [approx_U
               .subs(ys, yi)
               .subs(ccoeff)
               .subs(zip(Tsm1, Tsm2))
               .subs(zip(Ts, Tsm1))
               .subs(zip(Tsp1, Ts))
               .subs(hm1, hm2)
               .subs(h, hm1)
               .subs(hp1, h)
               for yi in y]

    Tchebp1 = [approx_U
               .subs(ys, yi)
               .subs(ccoeff)
               .subs(zip(Tsp1, Tsp2))
               .subs(zip(Ts, Tsp1))
               .subs(zip(Tsm1, Ts))
               .subs(hp1, hp2)
               .subs(h, hp1)
               .subs(hm1, h)
               for yi in y]

    info('T')
    Tcheb = [approx_U
             .subs(ys, yi)
             .subs(ccoeff)
             for yi in y]

    Tchebm1 = [approx_U
               .subs(ys, yi)
               .subs(ccoeff)
               .subs(zip(Tsm1, Tsm2))
               .subs(zip(Ts, Tsm1))
               .subs(zip(Tsp1, Ts))
               .subs(hm1, hm2)
               .subs(h, hm1)
               .subs(hp1, h)
               for yi in y]

    Tchebp1 = [approx_U
               .subs(ys, yi)
               .subs(ccoeff)
               .subs(zip(Tsp1, Tsp2))
               .subs(zip(Ts, Tsp1))
               .subs(zip(Tsm1, Ts))
               .subs(hp1, hp2)
               .subs(h, hp1)
               .subs(hm1, h)
               for yi in y]

    dyTchebm1 = [approx_U.diff(ys)
                 .subs(ys, yi)
                 .subs(ccoeff)
                 .subs(zip(Tsm1, Tsm2))
                 .subs(zip(Ts, Tsm1))
                 .subs(zip(Tsp1, Ts))
                 .subs(hm1, hm2)
                 .subs(h, hm1)
                 .subs(hp1, h)
                 for yi in y]

    dyTchebp1 = [approx_U.diff(ys)
                 .subs(ys, yi)
                 .subs(ccoeff)
                 .subs(zip(Tsp1, Tsp2))
                 .subs(zip(Ts, Tsp1))
                 .subs(zip(Tsm1, Ts))
                 .subs(hp1, hp2)
                 .subs(h, hp1)
                 .subs(hm1, h)
                 for yi in y]

    # Tcheb[0] = 1.

    info('dyT')
    dyTcheb = [approx_U
               .diff(ys)
               .subs(ys, yi)
               .subs(ccoeff)
               for yi in y]
    info('dyyT')
    dyyTcheb = [approx_U
                .diff(ys, ys)
                .subs(ys, yi)
                .subs(ccoeff)
                for yi in y]

    info('Calcul des dérivés en x, 2nd dim')
    dxT = [((.5 * Tchebp1[ids] - .5 * Tchebm1[ids]) / dx)
           for ids in range(Ny)]

    dxxT = [(Tchebm1[ids] - 2 * Tcheb[ids] + Tchebp1[ids]) /
            dx for ids in range(Ny)]

    dyT = dyTcheb
    dyyT = dyyTcheb

    info('Calcul des dérivés croisés')
    dxyT = [(dyTchebp1[ids] - dyTchebm1[ids]) / (2 * dx)
            for ids in range(Ny)]

    Re, We, Ct = sp.symbols('Re, We, Ct')
    Pe, B = sp.symbols('Pe, B')

    u = np.array([3 * q / h * (y[ids] - .5 * y[ids]**2) for ids in range(Ny)])

    v = np.array([((-3 * dxq * y[ids]**2) / 2. +
                   (3 * dxh * q * y[ids]**2) / h +
                   (dxq * y[ids]**3) / 2. -
                   (3 * dxh * q * y[ids]**3) / (2. * h)) for ids in range(Ny)])

    info('Calcul du membre de droite')
    Fh = -dxq
    Fq = ((378 * dxxq * h**2 +
           336 * dxh**2 * q - 210 * (q + 2 * dxs**2 * q) -
           9 * h * q * (56 * dxxh + 35 * dxxs + 68 * dxq * Re) -
           2 * dxh * (189 * dxq * h + 35 * Ct * h**3 +
                      3 * q * (35 * dxs - 54 * q * Re)) +
           70 * h**3 * (1 - Ct * dxs +
                        (dxxxh + dxxxs) * We)) / (252. * h**2 * Re))

    # dxh = 0 # little effect

    # lap_T = [(dxxT[ids] * h**2 -
    #           2 * dxh * dxyT[ids] * y[ids] / h +  # bug
    #           y[ids] / h * (2 / h * dxh**2 - dxxh) * dyT[ids] +
    #           ((y[ids] / h)**2 * dxh**2 + 1 / h**2) *
    #           dyyT[ids]) for ids in range(1, Ny - 1)]

    # vel_T = [(sp.Matrix([u[ids],
    #                      v[ids]]).T @
    #           sp.Matrix([dxT[ids] - y[ids] / h * dxh * dyT[ids],
    #                      dyT[ids] / h]))[0] for ids in range(1, Ny - 1)]

    FTbulk = [((2 * dxh * dxs * dyT[ids] + dyyT[ids] +
                dxs**2 * dyyT[ids] -
                2 * dxs * dxyT[ids] * h -
                dxxs * dyT[ids] * h + dxxT[ids] * h**2 +
                3 * dxs * dyT[ids] * h * Pe * u[ids] -
                3 * dxT[ids] * h**2 * Pe * u[ids] -
                3 * dyT[ids] * h * Pe * v[ids] +
                2 * dxh**2 * dyT[ids] * y[ids] +
                2 * dxh * dxs * dyyT[ids] * y[ids] -
                2 * dxh * dxyT[ids] * h * y[ids] -
                dxxh * dyT[ids] * h * y[ids] -
                3 * dxq * dyT[ids] * h * Pe * y[ids] +
                3 * dxh * dyT[ids] * h * Pe * u[ids] * y[ids] +
                dxh**2 * dyyT[ids] * y[ids]**2) / (3. * h**2 * Pe))
              for ids in range(1, Ny - 1)]

    FT = [0] + FTbulk + [0]

    F_therm = sp.Matrix([Fh, Fq, *FT])

    U_therm = np.array([hm2, qm2, *Tsm2,
                        hm1, qm1, *Tsm1,
                        h, q, *Ts,
                        hp1, qp1, *Tsp1,
                        hp2, qp2, *Tsp2]).reshape((5, -1))

    info('Calcul du jacobien')
    J_therm = sp.Matrix(np.array([F_therm.diff(u).tolist()
                                  for u in np.array(U_therm)
                                  .flatten()]
                                 ).squeeze()).T
    return (U_therm, F_therm, J_therm, (Re, We, Ct, Pe, B),
            (Tcheb[-1],
             dyTcheb[0]),)


if __name__ == '__main__':

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    model(4)
