#!/usr/bin/env python
# coding=utf8

import datreant.core as dtr
import pandas as pd
from project_path import *

from triflow.triflow_helper.analyse import load_data
from triflow.triflow_helper.log import init_log

import pylab as pl

init_log(log_dir, 'analysis', 'DEBUG', 'DEBUG')

samples_metadata = pd.read_csv(data_dir / 'samples.csv')

samples = dtr.Bundle(data_dir / 'runs/*')

if __name__ == '__main__':
    for sample in samples:
        data = load_data(sample.name)
        pl.figure()
        pl.plot(data['x'], data['h'][-1])
    pl.show()

# def absolute_error(x, field, ref):
#     err = np.sqrt(1 / max(x) * trapz((field - ref)**2, x))
#     return err


# def relative_error(x, field, ref):
#     err = np.sqrt(1 / max(x) * trapz((field - ref)**2, x))
#     err /= np.sqrt(1 / max(x) * trapz(ref**2, x))
#     return err


# def recentering_data(h):
#     maxindex = np.where(h == h.max())[0]
#     return lambda field: np.roll(field, int(maxindex))


# def compute_help_ff(data, parameters):
#     data = [np.array(data[field]) for field in helpers_ff.fields]
#     flat_data = helpers_ff.flatten_fields(*data)
#     phi, theta = helpers_ff.compute_Hs(flat_data, **dict(parameters))
#     return theta, phi


# def compute_err(sample, center=True):

#     path_4f = data_dir / 'runs_4f' / sample
#     path_ff = data_dir / 'runs_ff' / sample
#     parameters = dict(dtr.Treant(path_4f).categories)
#     parameters['id'] = sample

#     data_4f = np.load(path_4f / 'data.npz')
#     data_ff = np.load(path_ff / 'data.npz')

#     center_4f = recentering_data(data_4f['h'])
#     center_ff = recentering_data(data_ff['h'])

#     theta_ff, phi_ff = compute_help_ff(data_ff, parameters)
#     fields_ff = dict(data_ff)
#     fields_ff.update({'theta': theta_ff, 'phi': phi_ff})
#     fields_4f = dict(data_4f)
#     errs = {}
#     for fieldname in ['h', 'q', 'theta', 'phi']:
#         errs['rel_err_%s' % fieldname] = relative_error(
#             data_ff['x'],
#             center_4f(
#                 fields_4f[fieldname]),
#             center_ff(fields_ff[fieldname]))
#         errs['abs_err_%s' % fieldname] = absolute_error(
#             data_ff['x'],
#             center_4f(
#                 fields_4f[fieldname]),
#             center_ff(fields_ff[fieldname]))
#     parameters.update(errs)
#     return parameters


# def plot_comp(sample, field, center=True):
#     path_4f = data_dir / 'runs_4f' / sample
#     path_ff = data_dir / 'runs_ff' / sample
#     # parameters = dict(dtr.Treant(path_4f).categories)

#     data_4f = np.load(path_4f / 'data.npz')
#     data_ff = np.load(path_ff / 'data.npz')
#     if center:
#         center_4f = recentering_data(data_4f['h'])
#         center_ff = recentering_data(data_ff['h'])
#         field_4f = center_4f(data_4f[field])
#         field_ff = center_ff(data_ff[field])
#     else:
#         field_4f = data_4f[field]
#         field_ff = data_ff[field]
#     pl.plot(data_4f['x'], field_4f, color='blue', label=r"$\theta-\phi$ model")
#     pl.plot(data_ff['x'], field_ff, color='red', label="full Fourrier model")


# def plot_theta(sample, phase=False):
#     path_4f = data_dir / 'runs_4f' / sample
#     path_ff = data_dir / 'runs_ff' / sample
#     parameters = dict(dtr.Treant(path_4f).categories)

#     data_4f = np.load(path_4f / 'data.npz')
#     theta_4f = data_4f['theta']
#     data_ff = np.load(path_ff / 'data.npz')
#     theta_ff, phi_ff = compute_help_ff(data_ff, parameters)
#     center_4f = recentering_data(data_4f['h'])
#     center_ff = recentering_data(data_ff['h'])
#     field_4f = center_4f(theta_4f)
#     field_ff = center_ff(theta_ff)
#     x = data_4f['x'] if not phase else data_4f['h']
#     pl.plot(x, field_4f, label=r"$\theta-\phi$ model")
#     pl.plot(x, field_ff, label="full Fourrier model")
#     pl.plot(x, [parameters['theta_flat']] * parameters['Nx'],
#             label='Nusselt film theorical value')
#     pl.xlim(x.min(), x.max())


# def plot_phi(sample, phase=False):
#     path_4f = data_dir / 'runs_4f' / sample
#     path_ff = data_dir / 'runs_ff' / sample
#     parameters = dict(dtr.Treant(path_4f).categories)

#     data_4f = np.load(path_4f / 'data.npz')
#     phi_4f = data_4f['phi']
#     data_ff = np.load(path_ff / 'data.npz')
#     theta_ff, phi_ff = compute_help_ff(data_ff, parameters)
#     center_4f = recentering_data(data_4f['h'])
#     center_ff = recentering_data(data_ff['h'])
#     field_4f = center_4f(phi_4f)
#     field_ff = center_ff(phi_ff)
#     x = data_4f['x'] if not phase else data_4f['h']
#     pl.plot(x, field_4f, label=r"$\theta-\phi$ model")
#     pl.plot(x, field_ff, label="full Fourrier model")
#     pl.plot(x, [parameters['phi_flat']] * parameters['Nx'],
#             label='Nusselt film theorical value')
#     pl.xlim(x.min(), x.max())


# def plot_sample_distribution():
#     df = pd.read_csv(data_dir / 'samples.csv')
#     # sns.kdeplot(df.Pr, df.Bi)
#     pl.scatter(df.Pr, df.Bi, alpha=.2, marker='.', label='samples')
#     pl.scatter(df.Pr.mean(), df.Bi.mean(), color='black', marker='o',
#                label='samples average')
#     pl.scatter(7, 1, color='red', marker='o',
#                label='type sample')
#     pl.xscale('log')
#     pl.yscale('log')
#     pl.xlabel('Pr')
#     pl.ylabel('Bi')
#     pl.xlim(df.Pr.min() * .9, df.Pr.max() * 1.1)
#     pl.ylim(df.Bi.min() * .9, df.Bi.max() * 1.1)
#     pl.legend(loc='lower left')


# def errors_analysis(df):
#     sns.set_context('paper', font_scale=1.)
#     sns.set_style('whitegrid')
#     pl.figure(figsize=(5, 3))
#     plot_sample_distribution()
#     pl.tight_layout()
#     pl.savefig(figures_dir / 'sample_distribution.pdf')
#     g = sns.PairGrid(df,
#                      x_vars=["Bi", "Pr"],
#                      y_vars=["rel_err_theta", "abs_err_theta"], despine=True)
#     g.fig.set_size_inches((5, 3))
#     g.axes[0, 0].set_xscale('log')
#     g.axes[0, 1].set_xscale('log')
#     g.map(pl.scatter, marker='.', edgecolor='.2', alpha=.8)
#     g.axes[0, 0].yaxis.set_label_text('$\\theta$ relative error')
#     g.axes[1, 0].yaxis.set_label_text('$\\theta$ absolute error')
#     pl.tight_layout()
#     g.savefig(figures_dir / 'theta_errors.pdf')

#     g = sns.PairGrid(df,
#                      x_vars=["Bi", "Pr"],
#                      y_vars=["rel_err_phi", "abs_err_phi"], despine=True)
#     g.fig.set_size_inches((5, 3))
#     g.axes[0, 0].set_xscale('log')
#     g.axes[0, 1].set_xscale('log')
#     g.map(pl.scatter, marker='.', edgecolor='.2', alpha=.8)
#     g.axes[0, 0].yaxis.set_label_text('$\\phi$ relative error')
#     g.axes[1, 0].yaxis.set_label_text('$\\phi$ absolute error')
#     pl.tight_layout()
#     g.savefig(figures_dir / 'phi_errors.pdf')


# def plot_quartile(df):
#     sns.set_context('paper', font_scale=1.)
#     sns.set_style('whitegrid')
#     rel_quartile = df.quantile(np.linspace(0, 1, 5),
#                                interpolation='nearest').rel_err_phi
#     fig, plots = pl.subplots(5, 2, sharex=True, figsize=(15, 9))
#     sns.despine(bottom=True)
#     fig_phase, plots_phase = pl.subplots(5, 2, sharex=True, figsize=(15, 9))
#     sns.despine(bottom=True)
#     for i, rel_err_phi in enumerate(rel_quartile.values):
#         sample = df[df.rel_err_phi == rel_err_phi]
#         sample_id = sample.id.values[0]

#         pl.sca(plots[i][0])
#         plot_phi(sample_id)
#         pl.legend(loc='best')
#         pl.title(r"Bi = %.2e, Pr = %.2e" %
#                  (sample.Bi.values[0], sample.Pr.values[0]))
#         pl.ylabel(r'$\phi(x)$')

#         pl.sca(plots[i][1])
#         plot_theta(sample_id)
#         pl.title(r"Bi = %.2e, Pr = %.2e" %
#                  (sample.Bi.values[0], sample.Pr.values[0]))
#         pl.ylabel(r'$\theta(x)$')

#         pl.sca(plots_phase[i][0])
#         plot_phi(sample_id, phase=True)
#         pl.legend(loc='best')
#         pl.title(r"Bi = %.2e, Pr = %.2e" %
#                  (sample.Bi.values[0], sample.Pr.values[0]))
#         pl.ylabel(r'$\phi(x)$')

#         pl.sca(plots_phase[i][1])
#         plot_theta(sample_id, phase=True)
#         pl.title(r"Bi = %.2e, Pr = %.2e" %
#                  (sample.Bi.values[0], sample.Pr.values[0]))
#         pl.ylabel(r'$\theta(x)$')
#     plots[-1][0].set_xlabel('x')
#     plots[-1][1].set_xlabel('x')
#     plots_phase[-1][0].set_xlabel('h')
#     plots_phase[-1][1].set_xlabel('h')
#     pl.tight_layout()
#     fig.savefig(figures_dir / 'compare_quartile.pdf')
#     fig_phase.savefig(figures_dir / 'compare_quartile_phase.pdf')


# if __name__ == '__main__':

#     pool = mp.Pool(30)

#     samples_id = sorted(set([sample.name for sample in samples_ff]))
#     datas = pool.map(ft.partial(compute_err), samples_id)
#     df = pd.DataFrame(data=datas)

#     plot_quartile(df)
