#!/usr/bin/env python
# coding=utf8

from sym_dict import SymDict


def fluid():
    relations_template = SymDict()
    relations_template.add_relation('(nu**2 / (g * sin(bottom_angle)))'
                                    ' ** (1 / 3) - l_v')
    relations_template.add_relation('(nu / (g * sin(bottom_angle)) ** 2)'
                                    ' ** (1 / 3) - t_v')
    relations_template.add_relation('cot(bottom_angle) - Ct')
    relations_template.add_relation('nu / chi - Pr')
    relations_template.add_relation('sigma_inf / '
                                    '(rho * g ** (1 / 3) * nu ** (4/3))'
                                    ' - Ka_vert')
    # relations_template.add_relation('sigma_inf * l_v / (rho * nu ** 2) - Ka')
    relations_template.add_relation('Ka - '
                                    'Ka_vert / (sin(bottom_angle) ** (1/3))')
    relations_template.add_relation('alpha * l_v / lamb - Bi')
    # relations_template.add_relation('bhN / l_v - (3 * Re) ** (1 / 3)')
    relations_template.add_relation('Pe - Pr * Re')
    relations_template.add_relation('hN - bhN / l_v')
    relations_template.add_relation('hN - (3 * Re) ** (1 / 3)')
    relations_template.add_relation('We - Ka / ((3 * Re) ** (2 / 3))')
    relations_template.add_relation('M - Ma / ((3 * Re) ** (2 / 3))')
    relations_template.add_relation('B - Bi * (3 * Re) ** (1 / 3)')
    relations_template.add_relation('chi - lamb / (rho * cp)')
    relations_template.add_relation('t_factor - 1 / ((t_v * l_v) / bhN)')
    relations_template.add_relation('l_factor - 1 / bhN')
    relations_template.add_relation('theta_flat - 1 / (1 + B)')
    relations_template.add_relation('phi_flat + B * theta_flat')
    relations_template['g'] = 9.81
    return relations_template


def water():

    water = fluid()
    water['rho'] = 1000
    water['nu'] = 1E-6
    water['lamb'] = .6
    water['sigma_inf'] = 72E-3
    water['cp'] = 4.18E3
    return water
