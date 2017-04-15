
import pandas as pd
import numpy as np
from mgds.data_aggregation import api
from mgds.data_aggregation import database as db
from mgds.data_aggregation import source as src
from mgds.data_aggregation import data_type as dtyp
from mgds.data_modeling import data as feature_data
from mgds.data_modeling.nbfn import drugs as nbfn_drugs
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error, r2_score
from py_utils import collection_utils
import logging
logger = logging.getLogger(__name__)


# Determine the sets of gene names that are shared across TCGA, CCLE, and GDSC
def get_gene_sets(df, tcga_dtypes, sources, data_types):

    gene_sets = {}
    for data_type in tcga_dtypes:
        gene_sets[('tcga', data_type)] = df[data_type].columns.tolist()

    for source in sources:
        for data_type in data_types:
            gene_sets[(source, data_type)] = api.get_raw_genomic_data(source, data_type)['GENE_ID:HGNC'].unique()

    return gene_sets


def get_tcga_modeling_data(cohort):
    # Load data from data_modeling/prep/tcga-prep.ipynb
    d = db.load_obj(src.TCGA_v1, db.PREP, 'raw-data-matrices')

    def prep(df, dt):
        df = df.loc[cohort]
        df.columns = pd.MultiIndex.from_tuples([(dt, c) for c in df])
        return df
    df = pd.concat([prep(df, dt) for dt, df in d.items()], axis=1)

    gene_sets = get_gene_sets(
        df,
        [dtyp.GENE_EXPRESSION, dtyp.GENE_RNA_SEQ],
        [src.CCLE_v1, src.GDSC_v2],
        [dtyp.GENE_EXPRESSION]
    )
    d_gene = pd.concat([pd.Series(np.repeat(1, len(v)), index=v).rename(k) for k, v in gene_sets.items()], axis=1)
    d_gene.head()

    shared_genes = d_gene.dropna().index.tolist()
    len(d_gene), len(d_gene.dropna())

    # Original Version - One feature set at a time
    #feature_typ = dtyp.GENE_RNA_SEQ
    feature_typ = dtyp.GENE_EXPRESSION
    X = df[[feature_typ, dtyp.GENE_RPPA]].dropna(how='all', axis=1).dropna(how='all', axis=0)
    X, Y = X[feature_typ], X[dtyp.GENE_RPPA]

    # Restrict X to only those genes in the "shared" set from other sources
    # (otherwise it will not be possible to make predictions using them)
    X = X.filter(items=shared_genes)

    mask = (X.isnull().all(axis=1)) | (Y.isnull().all(axis=1))
    X = X[~mask.values].dropna(how='all', axis=1)
    Y = Y[~mask.values].dropna(how='all', axis=1)

    # Scale X now to make sure that predictions are more likely to work when given inputs from other sources
    n_na = X.isnull().sum().sum()
    logger.info('Imputing {} X values of {}'.format(n_na, X.shape[0] * X.shape[1]))
    if n_na > 0:
        X = pd.DataFrame(Imputer().fit_transform(X), index=X.index, columns=X.columns)

    X = X.apply(lambda v: (v - v.mean()) / v.std())

    n_na = Y.isnull().sum().sum()
    logger.info('Imputing {} Y values of {}'.format(n_na, Y.shape[0] * Y.shape[1]))
    if n_na > 0:
        Y = pd.DataFrame(Imputer().fit_transform(Y), index=Y.index, columns=Y.columns)

    assert X.shape[1] == len(shared_genes)
    assert np.all(X.notnull())
    assert np.all(Y.notnull())

    return X, Y, df


# ### RX Modeling ### #

def gen_agg_data(d, src1, src2, data_type):
    d1 = d[(src1, data_type)]
    d2 = d[(src2, data_type)]

    shared_genes = list(np.intersect1d(d1.columns.tolist(), d2.columns.tolist()))

    def prep_scaling(df):
        dfv = df.stack()
        return (df - dfv.mean()) / dfv.std()
    d3 = (prep_scaling(d1[shared_genes]) + prep_scaling(d2[shared_genes]))/2
    return d3


def get_rx_modeling_data(drugs=nbfn_drugs.GDSC_PAPER_DRUGS, genes=None,
                         ge_src='agg', rx_src=src.GDSC_v2, na_y_thresh=.3, site='BREAST', verify=True):

    """
    Return X, Y dataset where:
    - X: Averaged CCLE and GDSC gene expression data for given gene set (or all if no filter given)
    - Y: GDSC drug sensitivity for cell lines

    All data is restricted to breast cancer samples at the moment
    """
    # Load all genomic data
    datasets = api.get_genomic_data_availability()
    d = db.cache_prep_operation(lambda: feature_data.get_feature_datasets(datasets), 'raw-features', overwrite=False)


    # #### Prep Features #### #
    if ge_src == 'agg':
        # Combine ccle and gdsc GE into one dataset
        d_agg_ge = gen_agg_data(d, src.CCLE_v1, src.GDSC_v2, dtyp.GENE_EXPRESSION)
        d_agg_ge.columns = pd.MultiIndex.from_tuples([('agg', dtyp.GENE_EXPRESSION, c) for c in d_agg_ge])
        d_agg_ge.head()

        # Concat aggregated data to original
        n_before = len(d)
        d = pd.concat([d, d_agg_ge], axis=1)
        assert len(d) == n_before

        X = d[('agg', dtyp.GENE_EXPRESSION)]
    else:
        X = d[(ge_src, dtyp.GENE_EXPRESSION)]

    # Restrict to shared genes
    if genes is not None:
        X = X[list(np.intersect1d(X.columns.tolist(), genes))]

    # #### Prep Responses #### #
    def select_drug(c):
        if drugs is None:
            return c[0] == rx_src and c[1] == dtyp.DRUG_SENSITIVITY
        return c[0] == rx_src and c[1] == dtyp.DRUG_SENSITIVITY and c[2] in drugs

    Y = d[[c for c in d if select_drug(c)]]
    Y.columns = [c[-1] for c in Y]

    # Remove records based on certain conditions
    # na_mask = X.isnull().all(axis=1)
    na_mask = X.isnull().any(axis=1)  # Should this be based on some threshold too?
    X = collection_utils.subset(X, lambda df: df[~na_mask], subset_op='Remove X records with null X values')
    Y = collection_utils.subset(Y, lambda df: df[~na_mask], subset_op='Remove Y records with null X values')

    na_mask = Y.isnull().all(axis=1)
    X = collection_utils.subset(X, lambda df: df[~na_mask], subset_op='Remove X records with null Y values')
    Y = collection_utils.subset(Y, lambda df: df[~na_mask], subset_op='Remove Y records with null Y values')

    site_mask = X.index.get_level_values('PRIMARY_SITE:MGDS') == site
    X, Y = X[site_mask], Y[site_mask]

    if len(X) == 0:
        raise ValueError('All data removed by null filters')

    # # Remove drugs with less than na_y_thresh non-null entries
    na_y = Y.notnull().sum(axis=0) / len(Y)
    keep_y = list(na_y[na_y >= na_y_thresh].index.values)
    n_y = Y.shape[1]
    if len(keep_y) != n_y:
        logger.info(
            'Removing {} drugs of {} due to having more than {}% null records'
            .format(n_y - len(keep_y), n_y, 100*na_y_thresh)
        )
        Y = Y.loc[:, keep_y]

    # X must be all non-null and while Y values can be null,
    # none can be entirely null
    if verify:
        assert np.all(X.notnull())
        assert not np.any(Y.isnull().all(axis=0))

    return X, Y


def stack_predictions(Y_pred, Y_test):
    Y_pred = pd.DataFrame(Y_pred, index=Y_test.index, columns=Y_test.columns)
    Y_pred.columns.name = 'Task'
    Y_pred.index.name = 'Sample'
    Y_pred = Y_pred.stack()
    Y_pred.name = 'Pred'

    Y_true = Y_test.copy()
    Y_true.columns.name = 'Task'
    Y_true.index.name = 'Sample'
    Y_true = Y_true.stack()
    Y_true.name = 'Actual'

    d_pred = pd.concat([Y_pred, Y_true], axis=1)
    return d_pred


SCORE_FN_MSE = lambda g: mean_squared_error(g['Actual'], g['Pred'])
SCORE_FN_R2 = lambda g: r2_score(g['Actual'], g['Pred'])
SCORE_FN_PEARSON = lambda g: g['Actual'].corr(g['Pred'], method='pearson')
SCORE_FN_SPEARMAN = lambda g: g['Actual'].corr(g['Pred'], method='spearman')


def score_predictions(d, score_fn):
    d_score = d.groupby(['Model', 'Fold', 'Task']).apply(score_fn).rename('Score').reset_index()
    return d_score.groupby(['Model', 'Task'])['Score'].describe().unstack().reset_index()