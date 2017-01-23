
import pandas as pd
import numpy as np
import os
from copy import deepcopy
from sklearn.model_selection import KFold
from py_utils import io_utils
import collections
import logging
logger = logging.getLogger(__name__)

TRAINING_RES_PATH = '/Users/eczech/data/research/mgds/modeling/rx/results/archive'


def _validate_fold(folds, i, j, k):
    assert len(folds[i]['test']) == \
        len(folds[i]['inner'][j]['train']) + len(folds[i]['inner'][j]['test'])
    assert len(folds[i]['inner'][j]['train']) == \
        len(folds[i]['inner'][j]['inner'][k]['train']) + len(folds[i]['inner'][j]['inner'][k]['test'])


def get_cv_folds(d, sites=None, n_split1=5, n_split2=3, random_state=None):
    idx = pd.Series(np.arange(len(d)))
    all_sites = d.index.get_level_values('PRIMARY_SITE:MGDS')

    if sites is None:
        sites = sorted(np.unique(all_sites))

    folds = collections.OrderedDict()
    for i, site in enumerate(sites):
        train_0 = idx.loc[all_sites != site].values
        test_0 = np.setdiff1d(idx, train_0)

        folds[i] = {
            'train': idx.iloc[train_0],
            'test': idx.iloc[test_0],
            'site': site,
            'inner': collections.OrderedDict()
        }
        cv_1 = KFold(n_splits=n_split1, random_state=random_state, shuffle=True)
        idx_1 = folds[i]['test']

        for j, (train_1, test_1) in enumerate(cv_1.split(idx_1)):
            folds[i]['inner'][j] = {
                'train': idx_1.iloc[train_1],
                'test': idx_1.iloc[test_1],
                'inner': collections.OrderedDict()
            }
            cv_2 = KFold(n_splits=n_split2, random_state=random_state, shuffle=True)
            idx_2 = folds[i]['inner'][j]['train']

            for k, (train_2, test_2) in enumerate(cv_2.split(idx_2)):
                folds[i]['inner'][j]['inner'][k] = {
                    'train': idx_2.iloc[train_2],
                    'test': idx_2.iloc[test_2]
                }
                _validate_fold(folds, i, j, k)
    return folds


def xy(d, cx, cy):
    assert len(set(cx).intersection(set(cy))) == 0
    assert len(cx) > 0
    assert len(cy) > 0
    return d[cx], d[cy]


def train_models(models, d_train, d_test=None, prefit_models=None, include_predictions=False):
    estimators = {}
    predictions = []

    responses = models['response_selector'](d_train)
    Y_test = None

    # Train each model individually
    for est_name in models['estimators']:
        est_def = models['estimators'][est_name]

        # Determine features and responses for this model
        features = est_def['feature_selector'](d_train)

        X_train, Y_train = xy(d_train, features, responses)

        X_test, Y_test = None, None
        if d_test is not None:
            X_test, Y_test = xy(d_test, features, responses)

        prefit_est = None
        if prefit_models is not None and est_name in prefit_models:
            prefit_est = deepcopy(prefit_models[est_name])

        print('Training ', est_name, X_train.shape, Y_train.shape, X_test.shape if X_test is not None else None)
        est, Y_pred = est_def['train'](X_train, Y_train, X_test, prefit_est)

        if Y_pred is not None:
            Y_pred.columns = pd.MultiIndex.from_tuples([(*c, est_name) for c in Y_pred])
            predictions.append(Y_pred)

        if est is not None:
            est.input_fields = X_train.columns
            est.output_fields = Y_train.columns
            estimators[est_name] = est

    # Concatenate predictions side by side noting that the predictions from the individual models
    # do not necessarily need to be for the same things
    Y_pred = pd.concat(predictions, axis=1) if len(predictions) > 0 else None

    if include_predictions:
        if Y_test is not None:
            Y_test.columns = pd.MultiIndex.from_tuples([(*c, 'Actual') for c in Y_test])
        return estimators, Y_pred, Y_test
    else:
        return estimators


def strip_actual_label(c):
    """ Removes unnecessary nested Actual label

    Nested columns at this point have a structure like:
        gdsc_v2, drug-sensitivity, NAVITOCLAX, Actual, [meta_rf|Actual]

    The second to last level is unnecessary, and should be removed here
    """
    assert len(c) == 5
    return tuple([v for i, v in enumerate(c) if i != 3])


def add_fold_id(d, fold_id):
    return d.assign(FOLD_ID=fold_id).set_index('FOLD_ID', append=True)


def partition(d, i_train, i_test, fold):
    d_train = d.iloc[i_train].copy()
    d_test = d.iloc[i_test].copy()
    return d_train, d_test


def predict(d_train, d_test, per_site_estimators, pan_site_estimators, prefit_models):
    # Create inner predictions from per-site models, which generally means
    # they will only be trained using the training data available in "d_train"
    per_site_models, Y_pred_1, Y_true_1 = train_models(
        per_site_estimators, d_train, d_test,
        include_predictions=True
    )

    # Create predictions from pan-site models, with training data generally from:
    # 1. Data not for this particular primary site
    # 2. The current inner fold (ie "d_train")
    # 3. Any combination of the above
    # 4. None of the above -- these models could have been trained on something external
    #    and only the predictions produced here are used as part of the meta model input
    pan_site_models, Y_pred_2, Y_true_2 = train_models(
        pan_site_estimators, d_train, d_test,
        prefit_models=prefit_models,
        include_predictions=True
    )
    assert Y_true_1.equals(Y_true_2)
    Y_true = Y_true_1

    #print('d_train = {}, d_test = {}, yp1 = {}, yp2 = {}'.format(d_train.shape, d_test.shape, Y_pred_1.shape, Y_pred_2.shape))
    Y_pred = pd.concat([Y_pred_1, Y_pred_2], axis=1)

    #return per_site_models, pan_site_models, Y_pred, Y_true
    return Y_pred, Y_true


def run_site_cv(d, site, cv, per_site_estimators, pan_site_estimators, meta_estimators, pan_site_prefit_models):
    # Run outer CV loop (this loop will be used to determine predictions
    # from the second-level "meta" model)

    prediction_data = []
    fold_results = collections.OrderedDict()
    for i, i_outer in enumerate(list(cv.keys())):
        d_train = d.iloc[cv[i_outer]['train']]
        d_test = d.iloc[cv[i_outer]['test']]

        logger.debug(
            '[fold {} of {}] Collecting predictions on outer fold from inner models '\
            '(meta estimator test data) [d_train.shape = {}, d_test.shape = {}]'
            .format(i+1, len(cv), d_train.shape, d_test.shape)
        )
        Y_pred_outer, Y_true_outer = predict(
                d_train, d_test, per_site_estimators,
                pan_site_estimators, pan_site_prefit_models
        )
        Y_feat_outer = pd.concat([Y_pred_outer, Y_true_outer], axis=1)

        logger.debug(
            '[fold {} of {}] Collecting predictions on inner fold from inner models (meta estimator training data)'
            .format(i+1, len(cv))
        )
        cv_inner = cv[i_outer]['inner']
        Y_feat_inner = []
        for i_inner in cv_inner:
            d_train = d.iloc[cv_inner[i_inner]['train']]
            d_test = d.iloc[cv_inner[i_inner]['test']]
            Y_pred, Y_true = predict(d_train, d_test, per_site_estimators, pan_site_estimators, pan_site_prefit_models)
            Y_feat_inner.append(pd.concat([Y_pred, Y_true], axis=1))
            del Y_pred, Y_true

        # Concatenate predictions from submodels into training set for meta model
        Y_feat_inner = pd.concat(Y_feat_inner)
        assert Y_feat_inner.columns.equals(Y_feat_outer.columns)

        logger.debug(
            '[fold {} of {}] Collecting predictions from meta estimators '\
            '[Y_feat_train.shape = {}, Y_feat_test.shape = {}]'
            .format(i+1, len(cv), Y_feat_inner.shape, Y_feat_outer.shape)
        )
        meta_models, Y_pred_meta, Y_true_meta = train_models(
            meta_estimators,
            Y_feat_inner,
            Y_feat_outer,
            include_predictions=True
        )

        Y_pred_meta.columns = [strip_actual_label(c) for c in Y_pred_meta]
        Y_true_meta.columns = [strip_actual_label(c) for c in Y_true_meta]

        assert Y_true_meta.equals(Y_true_outer)
        Y_pred = pd.concat([
            Y_pred_outer,
            Y_pred_meta,
            Y_true_meta
        ], axis=1)

        # print('Y pred size = {}, yp outer = {}, d_test size = {}'.format(len(Y_pred), len(Y_pred_outer), len(d_test)))
        prediction_data.append(add_fold_id(Y_pred, i_outer + 1))

    return {
        'prediction_data': pd.concat(prediction_data),
        'fold_results': fold_results
    }


def run_training(d, cv, pan_site_estimators, per_site_estimators, meta_estimators):

    results = {}

    # Fit pan-site models across entire dataset (all sites)
    pan_site_refit_models = train_models(pan_site_estimators, d)

    for i_site in cv:
        site = cv[i_site]['site']

        # Split into train/test based on primary site, where "training" data here
        # means any data NOT equal to this site, and test means all data for this site
        d_train = d.iloc[cv[i_site]['train']]
        d_test = d.iloc[cv[i_site]['test']]
        logger.info(
            'Beginning training for site "{}" [d_train.shape = {}, d_test.shape = {}]'
            .format(site, d_train.shape, d_test.shape)
        )

        # Train pan-site models on data excluding this primary site
        logger.info('Training pan site refit models ...')
        pan_site_prefit_models = train_models(pan_site_estimators, d_train)
        logger.info('Training for pan site refit models complete')

        # Run "refit", pre-site models on all data available for this site (ie the
        # "test" set for the site-based CV indexing)
        logger.info('Training per site refit models ...')
        per_site_refit_models = train_models(per_site_estimators, d_test)
        logger.info('Training for per site refit models complete')

        logger.info('Running inner cross validation for meta estimator data generation ...')
        cv_site = cv[i_site]['inner']
        site_results = run_site_cv(
            d, site, cv_site, per_site_estimators, pan_site_estimators,
            meta_estimators, pan_site_prefit_models
        )
        logger.info('Meta estimator data generation complete')

        logger.info('Training meta estimator models ...')
        meta_refit_models = train_models(meta_estimators, site_results['prediction_data'])
        site_results['models'] = {
            'pan_site': pan_site_refit_models,
            'per_site': per_site_refit_models,
            'meta': meta_refit_models
        }
        logger.info('Training for meta estimator models complete')

        # Ensure that a prediction was generated for every record with this primary site
        assert len(site_results['prediction_data']) == len(d_test)

        # Store results for this primary site
        results[site] = site_results

    return results


def get_result_description(notes):
    desc = 'RX Sensitivity Training Results Description\n*Created At {}\n'.format(pd.to_datetime('now'))
    return desc + notes


def save_training_results(results, version_number, description):

    model_res_path = os.path.join(TRAINING_RES_PATH, version_number)
    if not os.path.exists(model_res_path):
        os.mkdir(model_res_path)

    # Save text file with description of results
    model_desc_file = os.path.join(model_res_path, 'results.txt')
    with open(model_desc_file, 'w') as fd:
        fd.write(description)

    # Save actual result data
    model_res_file = os.path.join(model_res_path, 'results.pkl')
    io_utils.to_pickle(results, model_res_file, obj_name='RX Sensitivity Results')

    return model_res_path

def get_training_results(version_number):
    