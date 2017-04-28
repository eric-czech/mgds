
import edward as ed
import tensorflow as tf
import numpy as np
import pandas as pd
from edward.models import Normal, Gamma, Laplace, NormalWithSoftplusSigma, GammaWithSoftplusAlphaBeta
from edward.models import PointMass, InverseGamma, Uniform, StudentT, Beta
from ml.edward import models as ed_models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel


def compute_kernel(X, Xs=None, gamma=None):
    return rbf_kernel(X, Y=Xs, gamma=gamma)


class MTKLModel(ed_models.BayesianModel):

    def __init__(self, X_rppa, Y_rppa, gamma=None, print_kernel_stats=True,
                 h_scale=.5, rppa_scale=.3, rx_scale=.3, rppa_rx_scale=.3, w_scale=1.):
        self.initialize()
        self.X_rppa = X_rppa
        self.Y_rppa = Y_rppa
        self.print_kernel_stats = print_kernel_stats
        self.h_scale = h_scale
        self.rppa_scale = rppa_scale
        self.rx_scale = rx_scale
        self.w_scale = w_scale
        self.rppa_rx_scale = rppa_rx_scale
        self.gamma = gamma
        # if gamma is None:
        #     self.gamma = 1. / X_rppa.shape[0]
        # else:
        #     self.gamma = gamma

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

        # H = Laplace(loc=tf.zeros([N1, T1]), scale=self.h_scale * tf.ones([N1, T1]))
        H = Normal(mu=tf.zeros([N1, T1]), sigma=self.h_scale * tf.ones([N1, T1]))
        qH = PointMass(params=tf.Variable(tf.random_normal([N1, T1], stddev=.01), name='H'))
        tm['qH'] = qH.params
        lv[H] = qH
        # tf.summary.histogram('qH', tm['qH'])

        # W = Normal(mu=tf.zeros([T1, T2]), sigma=self.w_scale * tf.ones([T1, T2]))
        W = Laplace(loc=tf.zeros([T1, T2]), scale=self.w_scale * tf.ones([T1, T2]))
        qW = PointMass(params=tf.Variable(tf.random_normal([T1, T2], stddev=.01), name='W'))
        tm['qW'] = qW.params
        lv[W] = qW

        # B = Normal(mu=tf.zeros([1, T2]), sigma=1.*tf.ones([1, T2]))
        # qB = PointMass(params=tf.Variable(tf.random_normal([1, T2], stddev=.1), name='B'))
        # tm['qB'] = qB.params
        # lv[B] = qB

        # Stochastic Components
        YR_mu = tf.matmul(K_rppa, H)  # N1 x T1
        YR = Normal(mu=YR_mu, sigma=self.rppa_scale * tf.ones([N1, T1]))

        # YRD = tf.matmul(K_drug, H)  # N2 x T1
        YRD_mu = tf.matmul(K_drug, H)  # N2 x T1
        #YRD_sd = tf.reduce_mean(tf.square(YRD_mu - tf.reduce_mean(YRD_mu, axis=0)), axis=0)
        #tm['YRD_sd'] = YRD_sd

        # YRD = Normal(mu=YRD_mu, sigma=self.rppa_rx_scale * (YRD_sd * tf.ones([N2, T1], dtype=tf.float32)))
        YRD = Normal(mu=YRD_mu, sigma=self.rppa_rx_scale * tf.ones([N2, T1]))
        # tm['YRD_mu'] = YRD_mu

        YD_mu = tf.matmul(YRD, W)  # + B  # N2 x T2
        YD = Normal(mu=YD_mu, sigma=self.rx_scale * tf.ones([N2, T2]))

        # Static Components
        qYR = tf.matmul(K_rppa, qH.params)
        assert qYR.get_shape().as_list() == [N1, T1]
        tm['qYR'] = qYR

        qYRD = tf.matmul(K_drug, qH.params)
        qYD = tf.matmul(qYRD, qW.params)  # + qB.params
        assert qYD.get_shape().as_list() == [N2, T2]
        tm['qYD'] = qYD

        # Add scaled weight
        qYRD_sd = tf.reduce_mean(tf.square(qYRD - tf.reduce_mean(qYRD, axis=0)), axis=0, keep_dims=True)
        qW_scale = tf.transpose(qYRD_sd) * qW.params
        tm['qW_scale'] = qW_scale
        tm['qYRD_sd'] = qYRD_sd

        mse_rppa = tf.reduce_mean(tf.square(qYR - tf.constant(dY_rppa, dtype=tf.float32)), axis=0)
        tf.summary.scalar('mse_rppa', tf.reduce_mean(mse_rppa))

        mse_drug = tf.reduce_mean(tf.square(qYD - tf.constant(dY_drug, dtype=tf.float32)), axis=0)
        tf.summary.scalar('mse_drug', tf.reduce_mean(mse_drug))

        tf.summary.histogram('YRD', YRD)
        # tf.summary.histogram('qYRD_sd', self.rppa_rx_scale * qYRD_sd.params)
        tf.summary.histogram('qW_scale', qW_scale)

        def input_fn(d):
            return {YR: dY_rppa, YD: dY_drug}
            # return {YD: dY_drug}

        return input_fn, lv, tm

    def criticism_args(self, sess, tm):
        H = sess.run(tm['qH'])
        W = sess.run(tm['qW'])
        # YRD_sd = sess.run(tm['YRD_sd'])
        # B = sess.run(tm['qB'])

        def drug_pred_fn(X, scale=True):
            if scale:
                X = self.x_drug_scaler.transform(X)
            dK_drug = compute_kernel(X, tm['dX_rppa'], gamma=self.gamma)
            Y_rppa = np.matmul(dK_drug, H)
            Y_drug = np.matmul(Y_rppa, W)
            return self.y_drug_scaler.inverse_transform(Y_drug)

        def rppa_pred_fn(X, scale=True):
            if scale:
                X = self.x_rppa_scaler.transform(X)
            dK_rppa = compute_kernel(X, tm['dX_rppa'], gamma=self.gamma)
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


