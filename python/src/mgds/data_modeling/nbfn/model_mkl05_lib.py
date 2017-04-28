
import pandas as pd
import numpy as np
import tensorflow as tf
import edward as ed

import os
import plotly as plty
from plotly import graph_objs as go
from plotly.tools import make_subplots
from mgds.data_modeling import constants as model_constants
from ml.edward import models as ed_models
from mgds.data_modeling.nbfn import model_mkl05 as mklmodel

import logging
logger = logging.getLogger(__name__)


RES_DIR = '/Users/eczech/repos/mgds/python/notebook/data_modeling/kl_modeling/results/breast_cancer/cv_large_v5'
MODEL_DIR = '/Users/eczech/data/research/mgds/modeling/tkm/models'
SEED = model_constants.SEED


def rpath(filename):
    return os.path.join(RES_DIR, filename)


def mpath(modelname):
    return os.path.join(MODEL_DIR, modelname)


# Previous parameterization:
#     model = mklmodel.MTKLModel(
#         X_rppa.values, Y_rppa.values, gamma=None, print_kernel_stats=False,
#         h_scale=1., w_scale=1.,
#         rppa_scale=.05, rx_scale=.1, rppa_rx_scale=.05
#     )

def get_mkl_model_fn(model_dir_fmt, d_rppa):
    def get_model(i):
        tf.reset_default_graph()
        model = mklmodel.MTKLModel(
            d_rppa[0].values, d_rppa[1].values,
            gamma=None, print_kernel_stats=False,
            h_scale=1., w_scale=1.,
            rppa_scale=.05, rx_scale=.1, rppa_rx_scale=.1
        )
        est = ed_models.BayesianModelEstimator(
            model, n_collect=1, n_print_progress=500, random_state=SEED,
            fail_if_not_converged=False, max_steps=5000,
            inference_fn=ed.MAP
        )
        est.set_log_dir(model_dir_fmt.format(i))
        return est
    return get_model


def get_tcga_predictions(X_rppa, Y_rppa, X_drug, Y_drug, ests, model_dir_fmt, n_splits=10):
    from sklearn.model_selection import KFold
    from mgds.data_modeling.nbfn.mkl import stack_predictions

    d_pred_drug = []
    d_pred_rppa = []

    cv = KFold(n_splits=n_splits, random_state=SEED, shuffle=True).split(X_rppa)

    for i, (train, test) in enumerate(cv):
        logger.info('Training on fold {}'.format(i + 1))
        X_rppa_train, X_rppa_test = X_rppa.iloc[train, :], X_rppa.iloc[test, :]
        Y_rppa_train, Y_rppa_test = Y_rppa.iloc[train, :], Y_rppa.iloc[test, :]

        est = get_mkl_model_fn(model_dir_fmt, (X_rppa_train, Y_rppa_train))(i)
        est = est.fit(X_drug.values, Y_drug.values)

        # Make sensitivity predictions for held out TCGA samples
        logger.info('Generating sensitivity predictions for fold {}'.format(i + 1))
        Y_pred_drug = est.criticism_args_['pred_fn'](est.model.x_rppa_scaler.transform(X_rppa_test.values), scale=False)
        Y_pred_drug = pd.DataFrame(Y_pred_drug, index=Y_rppa_test.index, columns=Y_drug.columns)
        Y_pred_drug.columns.name = 'Task'
        Y_pred_drug.index.name = 'Sample'
        Y_pred_drug = Y_pred_drug.stack().rename('Predicted').reset_index().assign(Fold=i+1)
        d_pred_drug.append(Y_pred_drug)

        # Make RPPA predictions for held out TCGA samples as well
        logger.info('Generating kernel model RPPA predictions for fold {}'.format(i + 1))
        Y_pred_rppa = est.criticism_args_['pred_rppa_fn'](X_rppa_test.values)
        Y_pred_rppa = stack_predictions(Y_pred_rppa, Y_rppa_test, pred_name='Predicted')
        Y_pred_rppa = Y_pred_rppa.reset_index().assign(Fold=i+1, Model='mkl')
        d_pred_rppa.append(Y_pred_rppa)

        # Make RPPA predictions from reference models as well
        logger.info('Generating reference model RPPA predictions for fold {}'.format(i + 1))
        for est_name in ests:
            ref_est = ests[est_name](i)
            ref_est.fit(X_rppa_train, Y_rppa_train)
            Y_pred_ref = stack_predictions(ref_est.predict(X_rppa_test), Y_rppa_test, pred_name='Predicted')
            Y_pred_ref = Y_pred_ref.reset_index().assign(Fold=i+1, Model=est_name)
            d_pred_rppa.append(Y_pred_ref)

    d_pred_drug = pd.concat(d_pred_drug)
    d_pred_rppa = pd.concat(d_pred_rppa)
    return d_pred_drug, d_pred_rppa


# ### Visualizations ### #


def plot_performance_metric(d_score, filename=None, metric=None, plot_in_notebook=False):
    from ml.api.results import performance
    title = 'Performance Scores'
    if metric is not None:
        d_score = d_score[d_score['Metric'] == metric]
        title = metric

    fig = performance.visualize(d_score, separate_by=None, auto_plot=False, layout_kwargs={'title': title})['All']

    if filename is not None:
        filename = rpath('{}_{}.html'.format(filename, metric if metric else 'all'))
        print(filename)
        plty.offline.plt(fig, filename=filename)

    if plot_in_notebook:
        plty.offline.iplt(fig)


def plot_weights(W, cutoff, title, filename=None, height=600, save=True):
    layout = dict(
        title=title,
        width=1000, height=height,
        margin=dict(l=120)
    )
    fig = W.applymap(lambda v: np.nan if abs(v) <= cutoff else v)\
        .iplot(kind='heatmap', colorscale='Spectral', asFigure=True, layout=layout)
    fig['layout']['xaxis'].update(title='RPPA Gene')
    fig['layout']['yaxis'].update(title='Drug')

    if filename is not None:
        filename = rpath('{}.html'.format(filename))
        print(filename)
        plty.offline.plt(fig, filename=filename)

    plty.offline.iplt(fig)


def plot_weight_bars(W, filename=None, n_top=10, n_bottom=10, plot_in_notebook=True):
    from sklearn.preprocessing import LabelEncoder

    drugs = W.columns.tolist()
    rppas = W.index.tolist()

    n_row = len(drugs)
    height = n_row * 200
    fig = plty.tools.make_subplots(rows=n_row, cols=1, subplot_titles=drugs, print_grid=False)
    fig['layout'].update(height=height)

    colors = np.array(plty.colors.DEFAULT_PLOTLY_COLORS)
    cmap = dict(zip(rppas, colors[LabelEncoder().fit_transform(rppas) % len(colors)]))

    for i, c in enumerate(W):
        v = W[c].sort_values()
        v = pd.concat([v.head(n_bottom), v.tail(n_top)])
        trace = go.Bar(
            x=v.index.values, y=v,
            marker=dict(color=list(v.index.to_series().map(cmap))),
            showlegend=False
        )
        fig.append_trace(trace, i+1, 1)

    if filename is not None:
        filename = rpath('{}.html'.format(filename))
        print(filename)
        plty.offline.plt(fig, filename=filename)

    if plot_in_notebook:
        plty.offline.iplt(fig)


def plot_predictions_for_all_drugs(
    d_pred, top_drugs, model, title, filename=None,
    n_col=4, sens_thresh=-1, sens_ct=3,
    plot_in_notebook=True):

    d = d_pred[d_pred['Task'].isin(top_drugs)]
    d = d[d['Model'] == model]

    grps = d.groupby('Task')
    tasks = sorted(list(grps.groups.keys()))
    n_task = len(tasks)

    def get_rc(i):
        return (i // n_col) + 1, (i % n_col) + 1

    n_row = get_rc(n_task)[0]
    if n_task % n_col == 0:
        n_row -= 1

    fig = make_subplots(n_row, n_col, print_grid=False, subplot_titles=tasks)
    fig['layout'].update(width=1000, height=n_row*250)
    fig['layout'].update(title=title)
    fig['layout'].update(hovermode='closest')

    for i, k in enumerate(tasks):
        g = grps.get_group(k)
        text = g.index.get_level_values('CELL_LINE_ID:MGDS')

        is_sens = np.sum(g['Actual'] <= sens_thresh) >= sens_ct
        if is_sens:
            color = 'rgb(214, 39, 40)'
        else:
            color = 'rgb(31, 119, 180)'

        trace1 = go.Scatter(
            x=g['Predicted'],
            y=g['Actual'],
            name=k,
            text=text,
            mode='markers',
            marker=dict(color=color),
            showlegend=False
        )
        fig.append_trace(trace1, *get_rc(i))

        if is_sens:
            trace2 = go.Scatter(
                x=np.repeat(sens_thresh, 2),
                y=[min(g['Actual'].min(), sens_thresh), max(g['Actual'].max(), sens_thresh)],
                name='Sensitivity Threshold',
                mode='lines',
                line=dict(dash='dash'),
                marker=dict(color='rgb(44, 160, 44)'),
                hoverinfo='x+name',
                showlegend=False,
                opacity=.5
            )
            trace3 = go.Scatter(
                y=np.repeat(sens_thresh, 2),
                x=[min(g['Predicted'].min(), sens_thresh), max(g['Predicted'].max(), sens_thresh)],
                name='Sensitivity Threshold',
                mode='lines',
                line=dict(dash='dash'),
                marker=dict(color='rgb(44, 160, 44)'),
                hoverinfo='y+name',
                showlegend=False,
                opacity=.5
            )
            fig.append_trace(trace2, *get_rc(i))
            fig.append_trace(trace3, *get_rc(i))

    if filename is not None:
        filename = rpath('{}.html'.format(filename))
        print(filename)
        plty.offline.plt(fig, filename=filename)

    if plot_in_notebook:
        plty.offline.iplt(fig)

