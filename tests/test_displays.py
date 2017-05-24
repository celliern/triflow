#!/usr/bin/env python
# coding=utf8

import numpy as np
import pytest

from triflow import Model, Simulation, displays


def test_display_probe():
    model = Model("dxxU", "U")
    parameters = dict(periodic=False)

    x = np.linspace(0, 10, 50, endpoint=True)
    U = x ** 2

    fields = model.fields_template(x=x, U=U)
    simul = Simulation(model, 0, fields, parameters, dt=1, tmax=50, tol=1E-1)

    def std_probe(t, fields):
        return np.std(fields.U)

    with pytest.raises(AttributeError):
        display = displays.bokeh_probes_update(simul, {"std": std_probe},
                                               notebook=True)
        display(0, fields)


def test_display_fields():
    from triflow import Model, Simulation, displays
    import numpy as np

    model = Model("dxxU", "U")
    parameters = dict(periodic=False)

    x = np.linspace(0, 10, 50, endpoint=True)
    U = x ** 2

    fields = model.fields_template(x=x, U=U)
    simul = Simulation(model, 0, fields, parameters, dt=1, tmax=50, tol=1E-1,)

    with pytest.raises(AttributeError):
        display = displays.bokeh_fields_update(simul, "U", notebook=True)
        display(0, fields)
