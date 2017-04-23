
import edward as ed
import tensorflow as tf
import numpy as np
import pandas as pd
from edward.models import Normal, Gamma, Laplace, NormalWithSoftplusSigma, GammaWithSoftplusAlphaBeta
from edward.models import MultivariateNormalFull, MultivariateNormalDiag, MultivariateNormalDiagPlusVDVT, MultivariateNormalCholesky
from edward.models import PointMass, InverseGamma, Uniform, StudentT
from ml.edward import models as ed_models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances, cosine_distances
from edward.util import rbf as ed_rbf
from ml.tensorflow.utilities import pearson_correlation


def compute_kernel(X, Xs=None, gamma=None):
    return rbf_kernel(X, Y=Xs, gamma=gamma)


class MTKLModel(ed_models.BayesianModel):

    def __init__(self, X_rppa, Y_rppa, gamma=None, print_kernel_stats=True):
        self.initialize()
        self.gamma = gamma
        self.X_rppa = X_rppa
        self.Y_rppa = Y_rppa
        self.print_kernel_stats = print_kernel_stats

    def initialize(self):
        self.x_rppa_scaler = StandardScaler()
        self.x_drug_scaler = StandardScaler()
        self.y_rppa_scaler = StandardScaler()
        self.y_drug_scaler = StandardScaler()

    def inference_args(self, data):
        self.initialize()
        lv, tm = {}, {}

        dX_rppa, dY_rppa = self.X_rppa, self.Y_rppa
        dX_drug, dY_drug = data['X'], data['Y']

        self.x_rppa_scaler.fit(dX_rppa)
        self.x_drug_scaler.fit(dX_drug)
        dY_rppa = self.y_rppa_scaler.fit_transform(dY_rppa)
        dY_drug = self.y_drug_scaler.fit_transform(dY_drug)
        dX_rppa = self.x_rppa_scaler.transform(dX_rppa)
        dX_drug = self.x_drug_scaler.transform(dX_drug)

        tm['dX_rppa'] = dX_rppa
        dK_rppa = compute_kernel(dX_rppa, gamma=self.gamma)
        dK_drug = compute_kernel(dX_drug, dX_rppa, gamma=self.gamma)

        def kernel_stats(k):
            return pd.Series(k.ravel()).describe(percentiles=[.01, .1, .25, .50, .75, .90, .99])

        if self.print_kernel_stats:
            print('RPPA Kernel Stats: {}'.format(kernel_stats(dK_rppa)))
            print('Drug Kernel Stats: {}'.format(kernel_stats(dK_drug)))

        N1, P1 = dX_rppa.shape
        N2, P2 = dX_drug.shape
        T1, T2 = dY_rppa.shape[1], dY_drug.shape[1]
        assert P1 == P2

        assert dK_rppa.shape == (N1, N1)
        assert dK_drug.shape == (N2, N1)
        K_rppa = tf.constant(dK_rppa, dtype=tf.float32)
        K_drug = tf.constant(dK_drug, dtype=tf.float32)

        H = Normal(mu=tf.zeros([N1, T1]), sigma=1.*tf.ones([N1, T1])) # Original Model
        # H = Normal(mu=tf.zeros([N1, T1]), sigma=.5*tf.ones([N1, T1]))
        qH = PointMass(params=tf.Variable(tf.random_normal([N1, T1], stddev=.1), name='H'))
        tm['qH'] = qH.params
        lv[H] = qH
        # tf.summary.histogram('qH', tm['qH'])

        # W = Normal(mu=tf.zeros([T1, T2]), sigma=1.*tf.ones([T1, T2]))
        W = Laplace(loc=tf.zeros([T1, T2]), scale=1.*tf.ones([T1, T2]))
        qW = PointMass(params=tf.Variable(tf.random_normal([T1, T2], stddev=.1), name='W'))
        tm['qW'] = qW.params
        lv[W] = qW

        B = Normal(mu=tf.zeros([1, T2]), sigma=1.*tf.ones([1, T2]))
        qB = PointMass(params=tf.Variable(tf.random_normal([1, T2], stddev=.1), name='B'))
        tm['qB'] = qB.params
        lv[B] = qB


        # YR_sig = .1 # Original Model
        YR_sig = .1
        YR_mu = tf.matmul(K_rppa, H)  # N1 x T1
        YR = Normal(mu=YR_mu, sigma=YR_sig * tf.ones([N1, T1]))

        YRD_mu = tf.matmul(K_drug, H)  # N2 x T1
        YRD = Normal(mu=YRD_mu, sigma=YR_sig * tf.ones([N2, T1]))

        YD_mu = tf.matmul(YRD, W) + B  # N2 x T2
        # YD = Normal(mu=YD_mu, sigma=1.*tf.ones([N2, T2])) # Original Model
        YD = Normal(mu=YD_mu, sigma=1.*tf.ones([N2, T2]))

        qYR = tf.matmul(K_rppa, qH.params)
        assert qYR.get_shape().as_list() == [N1, T1]
        tm['qYR'] = qYR

        qYD = tf.matmul(tf.matmul(K_drug, qH.params), qW.params) + qB.params
        assert qYD.get_shape().as_list() == [N2, T2]
        tm['qYD'] = qYD

        mse_rppa = tf.reduce_mean(tf.square(qYR - tf.constant(dY_rppa, dtype=tf.float32)), axis=0)
        tf.summary.scalar('mse_rppa', tf.reduce_mean(mse_rppa))

        mse_drug = tf.reduce_mean(tf.square(qYD - tf.constant(dY_drug, dtype=tf.float32)), axis=0)
        tf.summary.scalar('mse_drug', tf.reduce_mean(mse_drug))

        def input_fn(d):
            return {YR: dY_rppa, YD: dY_drug}
            # return {YD: dY_drug}

        return input_fn, lv, tm

    def criticism_args(self, sess, tm):
        H = sess.run(tm['qH'])
        W = sess.run(tm['qW'])
        B = sess.run(tm['qB'])

        def drug_pred_fn(X):
            dK_drug = compute_kernel(self.x_drug_scaler.transform(X), tm['dX_rppa'], gamma=self.gamma)
            Y_rppa = np.matmul(dK_drug, H)
            Y_drug = np.matmul(Y_rppa, W) + B
            return self.y_drug_scaler.inverse_transform(Y_drug)

        def rppa_pred_fn(X):
            dK_rppa = compute_kernel(self.x_rppa_scaler.transform(X), tm['dX_rppa'], gamma=self.gamma)
            Y_rppa = np.matmul(dK_rppa, H)
            return self.y_rppa_scaler.inverse_transform(Y_rppa)

        def drug_pred_train_fn():
            return self.y_drug_scaler.inverse_transform(sess.run(tm['qYD']))

        def rppa_pred_train_fn():
            return self.y_rppa_scaler.inverse_transform(sess.run(tm['qYR']))

        return {
            'pred_fn': drug_pred_fn,
            'pred_train_fn': drug_pred_train_fn,
            'pred_rppa_train_fn': rppa_pred_train_fn,
            'pred_rppa_fn': rppa_pred_fn
        }

