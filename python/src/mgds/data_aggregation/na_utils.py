
import numpy as np
from matplotlib import pyplot as plt


def plot_dim_null_frequencies(d, bins=50):
    dp = d.isnull().astype(np.int64)
    fig, axs = plt.subplots(1, 2)
    m, n = len(dp.columns), len(dp)
    axs[0].set_title('Col-Wise Null Distribution (of {} columns)'.format(m))
    axs[0].set_xlabel('% Null Row Values')
    axs[1].set_title('Row-Wise Null Distribution (of {} rows)'.format(n))
    axs[1].set_xlabel('% Null Col Values')

    fig.set_size_inches((16, 4))
    (dp.mean(axis=0) * 100).hist(ax=axs[0], bins=bins)
    (dp.mean(axis=1) * 100).hist(ax=axs[1], bins=bins)


def get_null_stats(d):
    assert len(d.shape) == 2, 'Not supported for anything other than 2D datasets'
    n = d.shape[0] * d.shape[1]
    n_null = d.isnull().sum().sum()
    res = {
        'n_null': n_null,
        'n_non_null': d.notnull().sum().sum(),
        'n': n,
        'pct_null': 100. * n_null / n
    }
    assert res['n_null'] + res['n_non_null'] == res['n']
    return res