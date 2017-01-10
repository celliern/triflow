#!/usr/bin/env python
# coding=utf8
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
Show use of SceneCanvas to display and update Image and Isocurve visuals using
ViewBox visual.
"""

import itertools as it
import sys

import datreant.core as dtr
import numpy as np
import pandas as pd
import moviepy.editor as mpy
from project_path import *
from triflow.triflow_helper.analyse import load_data
from triflow.triflow_helper.log import init_log
from vispy import app, scene
from vispy.visuals.transforms import STTransform
from matplotlib.cm import magma

init_log(log_dir, 'analysis', 'DEBUG', 'DEBUG')

samples_metadata = pd.read_csv(data_dir / 'samples.csv')

samples = dtr.Bundle(data_dir / 'runs/*')
phi_flat_data = load_data(data_dir / 'ref_flat',
                          'B21g08YYvky011ZL81o5Bd')['phi']
load_data = load_data(data_dir / 'runs')

finished_samples = [sample.name
                    for sample
                    in samples
                    if sample.categories['t'] == 10000]
samples_metadata = pd.read_csv(data_dir / 'samples.csv',
                               index_col=0).loc[finished_samples]


def init_canvas(**kwargs):
    canvas = scene.SceneCanvas(keys='interactive', **kwargs)
    grid = canvas.central_widget.add_grid(spacing=0)
    return canvas, grid


def plot(parameter, data, grid, by):
    x = data['x']
    # y = np.linspace(0, h.max(), 1000)
    N = x.size
    # T = computeT(data, parameter, y, i)
    # # T[np.where(y[np.newaxis, :] > h[i, :, np.newaxis])] = np.nan
    # Timg = magma(T.T)
    # Timg[:, :, 3][np.where((y[:, np.newaxis] >
    #                         h[i, np.newaxis, :]))] = 0

    title = scene.Label("freq = %4.1f" % parameter['freqd'], color='black')
    grid.add_widget(title, row=row * 2, col=0)
    h_plotbox = grid.add_view(row=row * 2 + 1, col=0, camera='panzoom')
    # Nu_plotbox = grid.add_view(row=row * 3 + 2, col=0, camera='panzoom')

    h_coords = np.zeros((N, 2), dtype=np.float32)
    h_coords[:, 0] = x
    h_coords[:, 1] = h[i, :]
    h_lim = [-1, h.max()]

    phi_coords = np.zeros((N, 2), dtype=np.float32)
    phi_coords[:, 0] = x
    phi_coords[:, 1] = (phi[i, :] - phi.min()) / (phi.max() - phi.min()) - 1

    x_lim = [0, x.max()]

    h_plotbox.camera.set_range(x_lim, h_lim)
    # Nu_plotbox.camera.set_range(x_lim, Nu_lim)

    h_line = scene.Line(pos=h_coords,
                        color='black',
                        parent=h_plotbox.scene, method='agg')
    phi_line = scene.Line(pos=phi_coords,
                          color='black',
                          parent=h_plotbox.scene, method='agg')
    T_surface = scene.Image(Timg, cmap='viridis', clim=(0, 1),
                            interpolation='bicubic',
                            parent=h_plotbox.scene)
    T_surface.transform = STTransform(scale=(x.max() / N,
                                             h.max() / y.size))
    return h_line, phi_line, T_surface


def update_plot(h_line, phi_line, T_surface, data, parameter, first_index, i):
    x = data['x']
    h = data['h']
    i = first_index + i % (h.shape[0] - first_index)
    y = np.linspace(0, h.max(), 1000)
    phi = data['phi']
    # Nu = phi / phi_flat_data[-1]
    N = x.size
    T = computeT(data, parameter, y, i)
    Timg = magma(T.T)
    Timg[:, :, 3][np.where((y[:, np.newaxis] >
                            h[i, np.newaxis, :]))] = 0

    h_coords = np.zeros((N, 2), dtype=np.float32)
    h_coords[:, 0] = x
    h_coords[:, 1] = h[i, :]

    phi_coords = np.zeros((N, 2), dtype=np.float32)
    phi_coords[:, 0] = x
    phi_coords[:, 1] = (phi[i, :] - phi.min()) / (phi.max() - phi.min()) - 1

    h_line.set_data(pos=h_coords)
    phi_line.set_data(pos=phi_coords)
    T_surface.set_data(Timg)
    T_surface.transform = STTransform(scale=(x.max() / N,
                                             h.max() / y.size))


def init_plot(canvas, parameters, datas, first_index):
    lines = []
    for row, (parameter, data) in enumerate(zip(parameters, datas)):
        lines.append(plot(parameter,
                          data, first_index, grid, row))
    return lines


def update(lines, first_index, index_step, datas, parameters):
    for i in it.count(0, index_step):
        for row, (parameter, data) in enumerate(zip(parameters, datas)):
            update_plot(*lines[row], data, parameter, first_index, i)
        yield


def animate(canvas, first_index, step, parameters, datas):
    lines = init_plot(grid, first_index, parameters, datas)
    canvas.show()
    timer = app.Timer()
    updater = update(lines, first_index, step, parameters, datas)
    timer.connect(lambda ev:
                  next(updater))
    timer.start(0)
    app.run()


def save(canvas, first_index, step):
    lines = init_plot(grid, first_index, parameters, datas)
    updater = update(lines, first_index, step, parameters, datas)
    for time in updater:
        canvas.update()
        yield canvas.render()
